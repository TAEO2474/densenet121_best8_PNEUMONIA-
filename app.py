# app.py
import os
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import cv2
import gdown
import requests

# ============================= Page / Constants =============================
st.set_page_config(page_title="CXR Pneumonia Classifier (DenseNet121)", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

# ▶▶ Google Drive FILE ID (언더바 X). Secrets 우선 사용.
FILE_ID = st.secrets.get("MODEL_FILE_ID", "1UPxtL1kx8a38z9fxlBRljNn8n6T4LL_l")

MODEL_DIR = Path("models")
MODEL_LOCAL = MODEL_DIR / "densenet121_best_9.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ▶ (선택) Drive 막힐 때 직링크(HF/GitHub 등) – Secrets에 넣어두면 자동 폴백
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")
TIMEOUT = 120

# ============================= Utils: download & load model =============================
def _http_download(url: str, out: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        st.warning(f"HTTP download failed: {e}")
        return False

@st.cache_data(show_spinner=False)
def ensure_model_file_cached() -> str:
    """
    Try: (1) already exists → (2) gdown → (3) HTTP fallback.
    성공 시 모델 경로 문자열 반환. 실패 시 예외.
    """
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)

    # 1) gdown (공개 파일 전용, 쿠키 미사용)
    try:
        with st.spinner("📥 Downloading model from Google Drive (gdown)…"):
            gdown.download(
                id=FILE_ID,
                output=str(MODEL_LOCAL),
                quiet=False,
                use_cookies=False,
                fuzzy=True,  # 다양한 드라이브 URL 변형에 관대
            )
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
            return str(MODEL_LOCAL)
    except Exception as e:
        st.warning(f"gdown failed: {e}")

    # 2) HTTP 직링크 폴백
    if HTTP_FALLBACK_URL:
        with st.spinner("🌐 Downloading model via direct URL…"):
            if _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL):
                return str(MODEL_LOCAL)

    raise RuntimeError("다운로드 실패: Google Drive/직링크 모두 불가합니다.")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Lambda(preprocess_input) 역직렬화 대응.
    (모델 내부에 전처리가 포함되어 있다면 외부 중복 전처리 금지)
    """
    model = keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )
    return model

# ============================= Grad-CAM helpers =============================
def find_last_conv4d_layer(model: keras.Model):
    """
    출력 rank==4 (B,H,W,C) 인 마지막 레이어를 반환.
    conv/concat/relu로 이름 필터링하지 않고 '실제 4D 특징맵' 보장에 집중.
    """
    last = None
    for lyr in model.layers:
        shp = getattr(lyr, "output_shape", None)
        if shp is None:
            continue
        # multi-output 레이어일 수 있으므로 하나만 보정
        if isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], (list, tuple)):
            # shp가 [(None,H,W,C), ...] 같은 구조일 수 있음
            try:
                rank = len(shp[0])
            except Exception:
                continue
        else:
            try:
                rank = len(shp) if isinstance(shp, (list, tuple)) else len(tuple(shp))
            except Exception:
                continue
        if rank == 4:
            last = lyr
    return last or model.layers[-1]

@tf.function
def _normalize_heatmap(x):
    x = tf.maximum(x, 0.0)
    mx = tf.reduce_max(x)
    return tf.where(mx > 0, x / mx, x)

def make_gradcam_heatmap(img_bchw, model: keras.Model, last_conv_layer_name: str = None):
    """
    img_bchw: 모델 입력과 동일 스케일/shape (float32, [1,H,W,3])
    Returns: heatmap [Hc, Wc] (0~1 float32)
    """
    # 0) 입력 dtype 보정
    img_bchw = tf.convert_to_tensor(img_bchw)
    if img_bchw.dtype != tf.float32:
        img_bchw = tf.cast(img_bchw, tf.float32)

    # 1) 마지막 4D conv 레이어 결정
    conv_layer = None
    if last_conv_layer_name:
        try:
            lyr = model.get_layer(last_conv_layer_name)
            shp = getattr(lyr, "output_shape", None)
            try:
                rank = len(shp) if isinstance(shp, (list, tuple)) else len(tuple(shp))
            except Exception:
                rank = None
            if rank == 4:
                conv_layer = lyr
        except Exception:
            conv_layer = None
    if conv_layer is None:
        conv_layer = find_last_conv4d_layer(model)

    # 2) 같은 그래프에서 conv 출력과 최종 출력 동시 획득
    grad_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    # 3) 순전파 + 그래디언트
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_bchw, training=False)

        # 리스트/딕셔너리/튜플 출력 정규화
        def _to_tensor(x):
            if isinstance(x, dict):
                x = next(iter(x.values()))
            if isinstance(x, (list, tuple)):
                x = x[0]
            return tf.convert_to_tensor(x)
        conv_out = _to_tensor(conv_out)
        preds    = _to_tensor(preds)
        tape.watch(conv_out)

        # preds를 (N,C) 형태로 표준화
        if preds.shape.rank is None or preds.shape.rank == 0:
            preds = tf.reshape(preds, (-1, 1))
        elif preds.shape.rank == 1:
            preds = tf.expand_dims(preds, -1)

        # 이진(sigmoid) vs 다중(softmax)
        class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. 마지막 conv 레이어가 GAP/Flatten 이후거나 "
            "모델이 미분 불가능한 경로일 수 있습니다."
        )

    # 동적 축 평균 (마지막 채널축 제외)
    r = grads.shape.rank or tf.rank(grads)
    if isinstance(r, tf.Tensor):  # 그래프 모드 안전
        r = int(r.numpy())
    axes = tuple(range(0, max(1, r - 1)))  # 채널축 제외 모두 평균
    pooled_grads = tf.reduce_mean(grads, axis=axes)  # [C] 기대

    # conv_out: [N,Hc,Wc,C] 또는 [Hc,Wc,C] → [Hc,Wc,C]
    if conv_out.shape.rank == 4:
        conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # [Hc,Wc]
    heatmap = _normalize_heatmap(heatmap)
    return heatmap.numpy()

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.4):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    out = (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return out

# ============================= Inference =============================
def prepare_inputs(pil_img: Image.Image):
    """
    Return (img_uint8_rgb, bchw_raw, bchw_preprocessed)
    - 모델 내부에 preprocess Lambda가 있을 수 있으므로
      추론은 bchw_raw(0~255 float32) 사용.
    - Grad-CAM도 모델 입력과 동일 텐서를 사용해야 그래프 불일치가 없다.
    """
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)
    bchw_pp  = preprocess_input(bchw_raw.copy())  # 필요시 분석용
    return arr, bchw_raw, bchw_pp

def predict_pneumonia_prob(model, bchw_raw):
    """Sigmoid output for class=1 (PNEUMONIA)."""
    prob = model.predict(bchw_raw, verbose=0)
    # dict/list/array 케이스 정규화
    if isinstance(prob, dict):
        prob = next(iter(prob.values()))
    if isinstance(prob, (list, tuple)):
        prob = prob[0]
    prob = np.asarray(prob).squeeze()
    return float(prob.item() if hasattr(prob, "item") else prob)

# ============================= Sidebar =============================
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.50, 0.69, 0.50, 0.01)
    st.caption("• Lower = higher sensitivity for pneumonia\n• Higher = fewer false positives")

    st.divider()
    st.subheader("Model fallback")
    st.caption("If download fails, upload your .keras model here and it will be cached.")
    uploaded_model = st.file_uploader("Upload model (.keras)", type=["keras"])

# ============================= Main UI =============================
st.title("Chest X-ray Pneumonia Classifier (DenseNet121)")
st.write("Upload a chest X-ray, get a prediction and Grad-CAM visualization. **This is not a medical device.**")

# 1) 모델 확보: 캐시 다운로드 → 실패 시 업로드 유도
model_path = None
try:
    model_path = ensure_model_file_cached()
except Exception as e:
    st.warning(f"Auto-download failed: {e}")
    if uploaded_model is not None:
        tmp_path = MODEL_DIR / "uploaded_model.keras"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_model.read())
        MODEL_LOCAL.unlink(missing_ok=True)
        tmp_path.rename(MODEL_LOCAL)
        st.success("✅ Uploaded model saved.")
        model_path = str(MODEL_LOCAL)
    else:
        st.error("모델 자동 획득에 실패했습니다. 사이드바에서 .keras 모델을 업로드해 주세요.")
        st.stop()

# 2) 모델 로드
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# 3) 예측 UI
up = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw, x_pp_bchw = prepare_inputs(pil_img)

    colA, colB = st.columns([1, 1])
    with colA:
        st.image(rgb_uint8, caption="Input (Resized 224×224)", use_column_width=True)

    if st.button("Run inference"):
        with st.spinner("Running model..."):
            # 예측
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            # Grad-CAM: 마지막 4D conv 이름/레이어 확보 후 동일 입력으로 계산
            last_conv_layer = find_last_conv4d_layer(model)
            last_layer_name = last_conv_layer.name
            heatmap = make_gradcam_heatmap(x_raw_bchw, model, last_layer_name)
            cam_img = overlay_heatmap(rgb_uint8, heatmap, alpha=0.45)

        with colB:
            st.image(cam_img, caption=f"Grad-CAM (last: {last_layer_name})", use_column_width=True)

        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted class", pred_label)
        col2.metric("Prob. PNEUMONIA", f"{p_pneu*100:.2f}%")
        col3.metric("Confidence", f"{conf*100:.2f}%")

        st.info(
            "• Threshold tuning: set **0.50** to prioritize catching pneumonia (higher sensitivity), "
            "or **0.69** to reduce false positives and protect normals.\n"
            "• Use alongside clinical judgment and radiologist review."
        )
else:
    st.caption("Awaiting an image upload…")
