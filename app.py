# ============================================================
# app.py — DenseNet121 Binary Classifier + Grad-CAM (final)
# - Training code와 동일한 입력 이름: "input_image"
# - model / grad_model 모두 dict 입력으로만 호출 (혼용 금지)
# - conv4_blockXX_concat 레이어 선택 가능
# ============================================================

import os, cv2, numpy as np, streamlit as st, tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import gdown, requests

# ----------------------- 기본 설정 -----------------------
st.set_page_config(page_title="CXR Pneumonia Grad-CAM (DenseNet121)", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
# 파일명은 네가 실제 저장한 모델명으로 맞춰줘
MODEL_LOCAL = MODEL_DIR / "densenet121_best.keras"

# 필요시 secrets 에 설정 가능
FILE_ID = st.secrets.get("MODEL_FILE_ID", "")
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")
TIMEOUT = 120


# ----------------------- 다운로드 유틸 -----------------------
def _http_download(url: str, out: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def ensure_model_file_cached(uploaded=None) -> str:
    """1) 로컬 캐시, 2) 업로드, 3) gdown, 4) http 순으로 확보"""
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)

    if uploaded is not None:
        tmp = MODEL_DIR / "_uploaded.keras"
        with open(tmp, "wb") as f:
            f.write(uploaded.read())
        tmp.replace(MODEL_LOCAL)
        return str(MODEL_LOCAL)

    if FILE_ID:
        try:
            gdown.download(id=FILE_ID, output=str(MODEL_LOCAL), quiet=False, fuzzy=True)
            if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
                return str(MODEL_LOCAL)
        except Exception:
            pass

    if HTTP_FALLBACK_URL:
        if _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL):
            return str(MODEL_LOCAL)

    raise RuntimeError("모델 파일을 확보하지 못했습니다. 업로드를 사용하세요.")


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    # Lambda(preprocess_input) 복원을 위해 custom_objects 등록
    return keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )


# ----------------------- 전처리 / 시각화 -----------------------
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)  # (1, H, W, 3)
    # 주의: 모델 그래프 안에 Lambda(preprocess_input)가 이미 있으므로 raw로 넣는다.
    return rgb_uint8, bchw_raw

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(np.clip(heatmap.astype(np.float32), 0.0, 1.0), (w, h))
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)


# ----------------------- 호출 방식 통일 (dict only) -----------------------
INPUT_KEY = "input_image"  # 네가 학습 때 지정한 이름

def call_model(model, x):
    """항상 dict 입력으로만 호출 (혼용 금지)"""
    return model({INPUT_KEY: x}, training=False)

def call_grad_model(grad_model, x):
    """항상 dict 입력으로만 호출"""
    return grad_model({INPUT_KEY: x}, training=False)


# ----------------------- 예측/Grad-CAM -----------------------
def predict_prob(model, bchw_raw) -> float:
    prob = call_model(model, bchw_raw)
    if isinstance(prob, (list, tuple)): prob = prob[0]
    return float(np.asarray(prob).squeeze())

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer, target_class: int = 1):
    """
    표준 Grad-CAM. 실패 시 SmoothGrad Saliency로 폴백.
    호출은 전구간 dict 입력(입력이름=INPUT_KEY)으로만 수행.
    """
    import numpy as np, tensorflow as tf, cv2
    from tensorflow import keras

    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)

    try:
        # 그래프 워밍업 (dict only)
        _ = call_model(model, x)

        # 중간출력 모델 (model.inputs 그대로 사용)
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = call_grad_model(grad_model, x)
            tape.watch(conv_out)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            # 이진(sigmoid) vs 다중(softmax)
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                class_channel = preds[:, target_class]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None or conv_out.shape.rank != 4:
            raise RuntimeError("no_grads_or_bad_rank")

        grads = tf.nn.relu(grads)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))   # [C]
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)

        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)

        # 살짝 대비 + 부드러운 블러
        p90 = float(np.percentile(cam.numpy(), 90.0))
        heat = np.clip(cam.numpy() / (p90 + 1e-6), 0, 1).astype(np.float32)
        heat = cv2.GaussianBlur(heat, (3, 3), 0)

        return heat, "gradcam", ""

    except Exception as e:
        # ---- SmoothGrad 폴백 (역시 dict only) ----
        note = f"gradcam_fallback({type(e).__name__})"
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = call_model(model, x)
            if isinstance(preds, (list, tuple)): preds = preds[0]
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                class_channel = preds[:, target_class]
        g = tape.gradient(class_channel, x)                 # [1,H,W,3]
        heat = tf.reduce_max(tf.abs(g), axis=-1)[0].numpy()
        heat = heat / (heat.max() + 1e-8)
        heat = cv2.GaussianBlur(heat.astype(np.float32), (3, 3), 0)
        return heat, "saliency", note


# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.40, 0.80, 0.69, 0.01)  # 네가 쓰던 0.69 기본
    st.caption("• 낮추면 민감도↑, 높이면 특이도↑ (너의 실험 기본값: 0.69)")

    st.divider()
    st.subheader("Grad-CAM")
    st.caption("conv4 블록 concat 레이어가 권장됩니다. (예: conv4_block24_concat)")

    st.divider()
    st.subheader("(선택) 폐 영역 마스크")
    use_mask = st.checkbox("Mask to lung area", value=True)
    cy_ratio = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx_ratio = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry_ratio = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap_ratio = st.slider("gap", 0.05, 0.20, 0.10, 0.01)
    thr_cut = st.slider("heatmap threshold", 0.00, 0.80, 0.00, 0.01)

    st.divider()
    st.subheader("모델 업로드(옵션)")
    uploaded_model = st.file_uploader("Upload .keras model", type=["keras"])


# ----------------------- Main -----------------------
st.title("🩻 Chest X-ray Pneumonia — DenseNet121 + Grad-CAM")
st.write("**주의:** 연구용 시각화 도구이며 의료기기가 아닙니다.")

# 1) 모델 확보 & 로드
try:
    model_path = ensure_model_file_cached(uploaded_model)
except Exception as e:
    st.error(f"모델 준비 실패: {e}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 2) DenseNet 서브모델 & CAM 후보 레이어
try:
    base = model.get_layer("densenet121")
except Exception:
    st.error("내부 DenseNet121 서브모델을 찾지 못했습니다. 학습 그래프를 확인하세요.")
    st.stop()

cam_names = [l.name for l in base.layers if ("concat" in l.name and "conv4_block" in l.name)]
cam_names.sort()
if not cam_names:
    st.error("conv4_blockXX_concat 레이어를 찾지 못했습니다. 모델을 확인하세요.")
    st.stop()

default_idx = cam_names.index("conv4_block24_concat") if "conv4_block24_concat" in cam_names else len(cam_names) - 1
chosen_name = st.sidebar.selectbox("Select CAM layer", cam_names, index=default_idx)
conv_layer = base.get_layer(chosen_name)

# 3) 업로드 & 추론
up = st.file_uploader("Upload X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)

    if st.button("Run Inference + Grad-CAM"):
        with st.spinner("Running..."):
            # 확률/라벨
            p_pneu = predict_prob(model, x_raw_bchw)
            label = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"
            conf = p_pneu if label == "PNEUMONIA" else (1 - p_pneu)
            target_class = 1 if label == "PNEUMONIA" else 0

            # Grad-CAM
            heatmap, method, note = make_gradcam_heatmap(x_raw_bchw, model, conv_layer, target_class)

            # (선택) 폐 마스크 + 임계값 컷
            if use_mask:
                h, w = heatmap.shape
                mask = np.zeros((h, w), np.uint8)
                cx, cy = w // 2, int(h * cy_ratio)
                rx, ry = int(w * rx_ratio), int(h * ry_ratio)
                gap = int(w * gap_ratio)
                cv2.ellipse(mask, (cx - gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                cv2.ellipse(mask, (cx + gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                heatmap = heatmap * (mask > 0).astype(np.float32)
                if thr_cut > 0:
                    heatmap = np.where(heatmap >= thr_cut, heatmap, 0.0)

            cam_img = overlay_heatmap(rgb_uint8, heatmap)

        with col2:
            st.image(cam_img, caption=f"CAM (layer: {conv_layer.name}) • method: {method}{' • '+note if note else ''}", use_column_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted", label)
        m2.metric("Prob. PNEUMONIA", f"{p_pneu*100:.2f}%")
        m3.metric("Confidence", f"{conf*100:.2f}%")

        if method != "gradcam":
            st.warning("지금은 **Saliency 폴백**입니다. 사이드바에서 **다른 conv4_blockXX_concat**을 선택해 보세요.")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
