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

# -----------------------------
# Page / Constants
# -----------------------------
st.set_page_config(page_title="CXR Pneumonia Classifier (DenseNet121)", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

# ▶▶ Put your Google Drive FILE ID here (NO underscore)
FILE_ID = "1UPxtL1kx8a38z9fxlBRljNn8n6T4LL_l"   # ← 본인 ID (언더바 X)

MODEL_DIR = Path("models")
MODEL_LOCAL = MODEL_DIR / "densenet121_best_9.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ▶ (선택) 드라이브가 막히면 사용할 직링크(HF/GitHub 등)를 Secrets에 넣어두세요.
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")  # 예: https://huggingface.co/.../resolve/main/densenet121_best_9.keras
TIMEOUT = 120

# -----------------------------
# Utils: download & load model
# -----------------------------
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
    성공 시 모델 경로 문자열 반환. 실패 시 예외 발생.
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
                fuzzy=True,           # 링크 변형에도 관대하게 처리
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
    """Load Keras model with preprocess_input (Lambda) deserialization."""
    model = keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )
    return model

# -----------------------------
# Grad-CAM helpers
# -----------------------------
def find_base_model(m):
    """Try to locate DenseNet base (by name or nested Model)."""
    try:
        return m.get_layer("densenet121")
    except Exception:
        for lyr in m.layers[::-1]:
            if isinstance(lyr, keras.Model):
                return lyr
        return m

def find_last_conv_name(base):
    """Pick a suitable last conv/concat/relu-like layer for Grad-CAM."""
    candidates = []
    for lyr in base.layers:
        name = lyr.name.lower()
        if ("conv" in name) or ("concat" in name) or ("relu" in name):
            candidates.append(lyr)
    return (candidates[-1].name) if candidates else base.layers[-1].name

@tf.function
def _normalize_heatmap(x):
    x = tf.maximum(x, 0.0)
    mx = tf.reduce_max(x)
    return tf.where(mx > 0, x / mx, x)

def make_gradcam_heatmap(img_bchw, model, last_conv_layer_name: str):
    """
    img_bchw: 모델에 그대로 넣는 입력 (예: 0~255 float32, shape [1,H,W,3])
    Returns: heatmap [Hc, Wc] (0~1 float32)
    """
    # 1) top-level model에서 레이어를 바로 찾는다
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        # 이름이 안 맞으면 전체 모델에서 마지막 Conv 계열 자동 탐색
        candidates = []
        for lyr in model.layers:
            n = lyr.name.lower()
            if ("conv" in n) or ("concat" in n) or ("relu" in n):
                candidates.append(lyr)
        conv_layer = candidates[-1] if candidates else model.layers[-1]

    # 2) 같은 그래프에서 conv 출력과 최종 출력을 동시에 얻는 서브 모델
    grad_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    # 3) 순전파 + 그래디언트
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_bchw, training=False)
        # 출력 형태 자동 대응: (N,1) sigmoid 또는 (N,2) softmax
        if preds.shape[-1] == 1:
            class_channel = preds[:, 0]        # 양성 score
        else:
            class_channel = preds[:, 1]        # class 1 (PNEUMONIA)
        tape.watch(conv_out)

    grads = tape.gradient(class_channel, conv_out)           # d(score)/d(feature)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))     # GAP
    conv_out = conv_out[0]                                   # [Hc, Wc, C]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
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

# -----------------------------
# Inference
# -----------------------------
def prepare_inputs(pil_img: Image.Image):
    """Return (img_uint8_rgb, bchw_raw, bchw_preprocessed)."""
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)  # model 내부에 preprocess Lambda가 있어도 안전
    bchw_pp  = preprocess_input(bchw_raw.copy())               # Grad-CAM 경로에서 사용
    return arr, bchw_raw, bchw_pp

def predict_pneumonia_prob(model, bchw_raw):
    """Sigmoid output for class=1 (PNEUMONIA)."""
    prob = model.predict(bchw_raw, verbose=0).squeeze().item()
    return float(prob)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.50, 0.69, 0.50, 0.01)
    st.caption("• Lower = higher sensitivity for pneumonia\n• Higher = fewer false positives")

    st.divider()
    st.subheader("Model fallback")
    st.caption("If download fails, upload your .keras model here and it will be cached.")
    uploaded_model = st.file_uploader("Upload model (.keras)", type=["keras"])

# -----------------------------
# Main UI
# -----------------------------
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
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            last_layer_name = find_last_conv_name(model) 
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
