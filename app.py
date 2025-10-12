# app.py
import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import cv2
import gdown

# -----------------------------
# Page / Constants
# -----------------------------
st.set_page_config(page_title="CXR Pneumonia Classifier (DenseNet121)", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

# ▶▶ Put your Google Drive FILE ID here
FILE_ID = "PASTE_YOUR_DRIVE_FILE_ID_HERE"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

MODEL_DIR = "models"
MODEL_LOCAL = os.path.join(MODEL_DIR, "densenet121_best_9.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Utils: download & load model
# -----------------------------
@st.cache_resource
def ensure_model_file() -> str:
    """Download model from Google Drive if not exists."""
    if not os.path.exists(MODEL_LOCAL):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    return MODEL_LOCAL

@st.cache_resource
def load_model(model_path: str):
    """Load Keras model with Lambda(preprocess_input) deserialization."""
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False,
            compile=False,
        )
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

# -----------------------------
# Grad-CAM helpers
# -----------------------------
def find_base_model(m):
    """Try to locate DenseNet base (by name or type)."""
    try:
        return m.get_layer("densenet121")
    except Exception:
        # fallback: first Functional/Model inside, or last conv-like backbone
        for lyr in m.layers[::-1]:
            if isinstance(lyr, keras.Model):
                return lyr
        return m  # worst-case: use whole model

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

def make_gradcam_heatmap(img_preprocessed_bchw, model, last_conv_layer_name: str):
    """
    img_preprocessed_bchw: preprocessed, shape [1, H, W, 3]  (DenseNet rule)
    Returns: heatmap [Hc, Wc]
    """
    base = find_base_model(model)
    try:
        last_conv = base.get_layer(last_conv_layer_name)
    except Exception:
        last_conv_layer_name = find_last_conv_name(base)
        last_conv = base.get_layer(last_conv_layer_name)

    # 1) model: inputs -> last_conv feature
    last_conv_model = keras.Model(model.input, last_conv.output)

    # 2) tail model: last_conv feature -> prediction
    #    재연결: last_conv 이후의 레이어만 차례로 통과
    #    Trick: functional graph를 재사용하기 어렵다면, 직접 forward 구성
    #    여기서는 간단히 "중간출력 모델" + "전체 모델" 순전파에서 그래드만 가져오는 구조로 처리
    with tf.GradientTape() as tape:
        conv_out = last_conv_model(img_preprocessed_bchw)
        tape.watch(conv_out)
        # 전체 모델의 예측(스칼라 sigmoid) 취득
        preds = model(img_preprocessed_bchw, training=False)
        # 양성(폐렴, class=1) score에 대해 CAM 생성
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_out)                  # d(score)/d(feature)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))            # GAP over HWC
    conv_out = conv_out[0]                                          # [Hc, Wc, C]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)       # [Hc, Wc]
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
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)  # model 내부에 preprocess Lambda 존재
    bchw_pp  = preprocess_input(bchw_raw.copy())               # Grad-CAM 경로에 사용
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

# -----------------------------
# Main UI
# -----------------------------
st.title("Chest X-ray Pneumonia Classifier (DenseNet121)")
st.write("Upload a chest X-ray, get a prediction and Grad-CAM visualization. **This is not a medical device.**")

model_path = ensure_model_file()
model = load_model(model_path)

up = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw, x_pp_bchw = prepare_inputs(pil_img)

    colA, colB = st.columns([1, 1])
    with colA:
        st.image(rgb_uint8, caption="Input (Resized 224×224)", use_container_width=True)

    if st.button("Run inference"):
        with st.spinner("Running model..."):
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            # Grad-CAM
            base = find_base_model(model)
            last_layer_name = find_last_conv_name(base)
            heatmap = make_gradcam_heatmap(x_pp_bchw, model, last_layer_name)
            cam_img = overlay_heatmap(rgb_uint8, heatmap, alpha=0.45)

        with colB:
            st.image(cam_img, caption=f"Grad-CAM (last: {last_layer_name})", use_container_width=True)

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
