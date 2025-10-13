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
    if MODEL_LOCAL.exists():
        return str(MODEL_LOCAL)
    raise RuntimeError("모델 다운로드 실패")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

# ----------------------- Grad-CAM 함수 -----------------------
def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm = np.clip(hm, 0.0, 1.0)
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)

def make_gradcam_heatmap(img_bchw, model, conv_layer, target_class=1):
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)
    grad_model = keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    try:
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            tape.watch(conv_out)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        if preds.shape[-1] == 1:
            target = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
        else:
            target = preds[:, target_class]
        grads = tape.gradient(target, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.nn.relu(conv_out[0] * weights), axis=-1)
        cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
        p90 = np.percentile(cam.numpy(), 90.0)
        heat = np.clip(cam / (p90 + 1e-6), 0, 1)
        heat = cv2.GaussianBlur(heat.numpy().astype(np.float32), (3, 3), 0)
        return heat
    except Exception:
        # fallback (saliency)
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x, training=False)
            if isinstance(preds, (list, tuple)): preds = preds[0]
            cls = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
        g = tape.gradient(cls, x)
        heat = tf.reduce_max(tf.abs(g), axis=-1)[0].numpy()
        heat = heat / (heat.max() + 1e-8)
        heat = cv2.GaussianBlur(heat.astype(np.float32), (3, 3), 0)
        return heat

def prepare_inputs(pil_img):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)
    return rgb_uint8, bchw_raw

def predict_prob(model, bchw_raw):
    prob = model(bchw_raw, training=False)
    if isinstance(prob, (list, tuple)): prob = prob[0]
    return float(np.asarray(prob).squeeze())

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold", 0.4, 0.7, 0.5, 0.01)
    use_mask = st.checkbox("Apply lung mask", value=True)
    cy_ratio = st.slider("Mask center Y", 0.35, 0.6, 0.48, 0.01)
    rx_ratio = st.slider("Radius X", 0.15, 0.35, 0.23, 0.01)
    ry_ratio = st.slider("Radius Y", 0.2, 0.45, 0.32, 0.01)
    gap_ratio = st.slider("Gap", 0.05, 0.2, 0.1, 0.01)

# ----------------------- Main Layout -----------------------
st.title("🩻 Chest X-ray Pneumonia Classifier — DenseNet121 + Grad-CAM")
st.caption("Upload an X-ray to visualize activation regions (Colab-like Grad-CAM output).")

model_path = ensure_model_file_cached()
model = load_model(model_path)
base = model.get_layer("densenet121")
layer_names = [l.name for l in base.layers if "concat" in l.name and "conv4_block" in l.name]
layer_names.sort()
chosen = st.sidebar.selectbox("Select CAM layer", layer_names, index=len(layer_names)-1)
conv_layer = base.get_layer(chosen)

# ----------------------- Upload & Inference -----------------------
up = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)

    st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)
    if st.button("Run Grad-CAM"):
        with st.spinner("Running model..."):
            p_pneu = predict_prob(model, x_raw_bchw)
            label = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"
            target_class = 1 if label == "PNEUMONIA" else 0

            heatmap = make_gradcam_heatmap(x_raw_bchw, model, conv_layer, target_class)

            if use_mask:
                h, w = heatmap.shape
                mask = np.zeros((h, w), np.uint8)
                cx, cy = w // 2, int(h * cy_ratio)
                rx, ry = int(w * rx_ratio), int(h * ry_ratio)
                gap = int(w * gap_ratio)
                cv2.ellipse(mask, (cx-gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                cv2.ellipse(mask, (cx+gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                heatmap *= (mask > 0).astype(np.float32)

            cam_img = overlay_heatmap(rgb_uint8, heatmap)
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_uint8, caption="Input", use_column_width=True)
        with col2:
            st.image(cam_img, caption=f"Grad-CAM ({conv_layer.name})", use_column_width=True)

        st.success(f"Prediction: {label} ({p_pneu*100:.2f}%)")
else:
    st.info("⬆️ Upload an image file to start.")
