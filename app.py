# ============================================================
# app.py — Grad-CAM integrated Streamlit (DenseNet121)
# Colab과 동일한 Grad-CAM 색감 / 구조로 표시
# ============================================================

import os, cv2, numpy as np, streamlit as st, tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import gdown, requests

# ----------------------- 기본 설정 -----------------------
st.set_page_config(page_title="CXR Pneumonia Grad-CAM Viewer", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOCAL = MODEL_DIR / "densenet121_best_9.keras"
FILE_ID = st.secrets.get("MODEL_FILE_ID", "1UPxtL1kx8a38z9fxlBRljNn8n6T4LL_l")
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")
TIMEOUT = 120

# ----------------------- 다운로드 유틸 -----------------------
def _http_download(url: str, out: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def ensure_model_file_cached() -> str:
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)
    try:
        gdown.download(id=FILE_ID, output=str(MODEL_LOCAL), quiet=False, fuzzy=True)
    except Exception:
        pass
    if HTTP_FALLBACK_URL:
        _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL)
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)
    raise RuntimeError("모델 다운로드 실패")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    # Lambda(preprocess_input) 복원을 위해 custom_objects 등록
    return keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

# ----------------------- 시각화/전처리 -----------------------
def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm = np.clip(hm, 0.0, 1.0)
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)

def prepare_inputs(pil_img):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)
    return rgb_uint8, bchw_raw

def predict_prob(model, bchw_raw):
    prob = model(bchw_raw, training=False)  # 모델이 단일 입력일 때 위치 인자 호출 OK
    if isinstance(prob, (list, tuple)):
        prob = prob[0]
    return float(np.asarray(prob).squeeze())

# ----------------------- Grad-CAM 본체 -----------------------
def make_gradcam_heatmap(img_bchw, model, conv_layer, target_class=1):
    """
    성공 시  : (heatmap, "gradcam", "")
    폴백 시  : (heatmap, "saliency", "gradcam_fallback(<Err>)")
    heatmap : float32 [0~1], shape=(Hc, Wc)
    """
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)
    grad_model = keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    try:
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            tape.watch(conv_out)

            # 이진(sigmoid) vs 다중(softmax) 둘 다 처리
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                class_channel = preds[:, target_class]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None or conv_out.shape.rank != 4:
            raise RuntimeError("no_grads_or_bad_rank")

        # ReLU(grads) 가중치 평균 → 채널별 가중합
        grads = tf.nn.relu(grads)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))      # [C]
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # [Hc,Wc]

        # 정규화 + 살짝 강도 보정(90퍼센타일 스케일) + 은은한 블러
        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        p90 = float(np.percentile(cam.numpy(), 90.0))
        heat = np.clip(cam / (p90 + 1e-6), 0, 1).astype(np.float32)
        heat = cv2.GaussianBlur(heat, (3, 3), 0)
        return heat, "gradcam", ""

    except Exception as e:
        # -------- SmoothGrad Saliency 폴백 --------
        note = f"gradcam_fallback({type(e).__name__})"
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x, training=False)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                class_channel = preds[:, target_class]
        g = tape.gradient(class_channel, x)                 # [1,H,W,3]
        heat = tf.reduce_max(tf.abs(g), axis=-1)[0].numpy() # [H,W]
        heat = heat / (heat.max() + 1e-8)
        heat = cv2.GaussianBlur(heat.astype(np.float32), (3, 3), 0)
        return heat, "saliency", note

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold", 0.40, 0.70, 0.50, 0.01)
    st.caption("• 낮출수록 민감도↑ • 높일수록 오탐↓")
    st.divider()
    st.subheader("Grad-CAM layer (conv4 권장)")
    use_mask = st.checkbox("Apply lung mask", value=True)
    cy_ratio = st.slider("Mask center Y", 0.35, 0.60, 0.48, 0.01)
    rx_ratio = st.slider("Radius X", 0.15, 0.35, 0.23, 0.01)
    ry_ratio = st.slider("Radius Y", 0.20, 0.45, 0.32, 0.01)
    gap_ratio = st.slider("Gap", 0.05, 0.20, 0.10, 0.01)

# ----------------------- Main Layout -----------------------
st.title("🩻 Chest X-ray Pneumonia Classifier — DenseNet121 + Grad-CAM")
st.caption("Upload an X-ray to visualize activation (Colab-like Grad-CAM). Not a medical device.")

# 모델 로드 & CAM 후보 레이어
model_path = ensure_model_file_cached()
model = load_model(model_path)
base = model.get_layer("densenet121")
layer_names = [l.name for l in base.layers if ("concat" in l.name and "conv4_block" in l.name)]
layer_names.sort()
default_idx = layer_names.index("conv4_block24_concat") if "conv4_block24_concat" in layer_names else (len(layer_names)-1 if layer_names else 0)
chosen = st.sidebar.selectbox("Select CAM layer", layer_names, index=default_idx if layer_names else 0)
conv_layer = base.get_layer(chosen) if layer_names else None

# ----------------------- Upload & Inference -----------------------
up = st.file_uploader("Upload X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up and conv_layer is not None:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)

    if st.button("Run Grad-CAM"):
        with st.spinner("Running model..."):
            p_pneu = predict_prob(model, x_raw_bchw)
            label = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"
            target_class = 1 if label == "PNEUMONIA" else 0

            # 🔹 Grad-CAM 생성
            heatmap, method, note = make_gradcam_heatmap(x_raw_bchw, model, conv_layer, target_class)

            # 🔹 폴백 여부 안내
            if method != "gradcam":
                st.warning(f"지금은 Saliency 폴백입니다. (원인: {note})  → 다른 conv4 블록을 선택해 보세요.")
            else:
                st.success(f"진짜 Grad-CAM 활성화됨 ({conv_layer.name})")

            # 🔹 (선택) 폐 마스크 적용
            if use_mask:
                h, w = heatmap.shape
                mask = np.zeros((h, w), np.uint8)
                cx, cy = w // 2, int(h * cy_ratio)
                rx, ry = int(w * rx_ratio), int(h * ry_ratio)
                gap = int(w * gap_ratio)
                cv2.ellipse(mask, (cx - gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                cv2.ellipse(mask, (cx + gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                heatmap = heatmap * (mask > 0).astype(np.float32)

            cam_img = overlay_heatmap(rgb_uint8, heatmap)

        with col2:
            st.image(cam_img, caption=f"Grad-CAM ({conv_layer.name}) • method: {method}", use_column_width=True)

        st.success(f"Prediction: {label} ({p_pneu*100:.2f}%)")
elif up and conv_layer is None:
    st.error("CAM용 4D 레이어를 찾지 못했습니다. 모델 구조를 확인하세요.")
else:
    st.info("⬆️ Upload an image file to start.")
