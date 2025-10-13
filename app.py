# app.py — Grad-CAM integrated version for DenseNet121
import os
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import cv2
import gdown
import requests

# ============================= Page / Constants =============================
st.set_page_config(page_title="CXR Pneumonia Classifier (DenseNet121)", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)
FILE_ID = st.secrets.get("MODEL_FILE_ID", "1UPxtL1kx8a38z9fxlBRljNn8n6T4LL_l")
MODEL_DIR = Path("models")
MODEL_LOCAL = MODEL_DIR / "densenet121_best_9.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")
TIMEOUT = 120

# ============================= Utils =============================
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
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)
    try:
        with st.spinner("📥 Downloading model from Google Drive (gdown)…"):
            gdown.download(id=FILE_ID, output=str(MODEL_LOCAL), quiet=False, use_cookies=False, fuzzy=True)
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
            return str(MODEL_LOCAL)
    except Exception as e:
        st.warning(f"gdown failed: {e}")
    if HTTP_FALLBACK_URL:
        with st.spinner("🌐 Downloading model via direct URL…"):
            if _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL):
                return str(MODEL_LOCAL)
    raise RuntimeError("다운로드 실패: Google Drive/직링크 모두 불가합니다.")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

# ============================= Grad-CAM helpers =============================
def find_base_model(m: keras.Model):
    try:
        return m.get_layer("densenet121")
    except Exception:
        for lyr in m.layers[::-1]:
            if isinstance(lyr, keras.Model):
                return lyr
        return m

def _is_4d(layer) -> bool:
    try:
        shp = K.int_shape(layer.output)
        return shp and len(shp) == 4
    except Exception:
        return False

def list_densenet_conv_names(base: keras.Model):
    names = [l.name for l in base.layers if _is_4d(l)]
    names.sort()
    return names

def get_densenet_feature_layer(base: keras.Model, preferred_name: str | None):
    names = list_densenet_conv_names(base)
    if preferred_name and preferred_name in names:
        return base.get_layer(preferred_name)
    if "conv4_block24_concat" in names:
        return base.get_layer("conv4_block24_concat")
    conv4 = [n for n in names if "conv4_block" in n and "concat" in n]
    if conv4:
        return base.get_layer(conv4[-1])
    conv5 = [n for n in names if "conv5_block" in n and "concat" in n]
    if conv5:
        return base.get_layer(conv5[-1])
    return base.get_layer(names[-1]) if names else None

def _gaussian_blur(np_map: np.ndarray, kmin=3):
    H, W = np_map.shape[:2]
    k = max(kmin, (min(H, W) // 4) | 1)
    return cv2.GaussianBlur(np_map, (k, k), 0)

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer, target_class: int = 1):
    if not isinstance(img_bchw, np.ndarray):
        img_bchw = np.array(img_bchw, dtype=np.float32)
    if img_bchw.dtype != np.float32:
        img_bchw = img_bchw.astype(np.float32)
    try:
        grad_model = keras.Model(inputs=model.input, outputs=[conv_layer.output, model.output])
        _ = model.predict(img_bchw, verbose=0)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_bchw, training=False)
            if isinstance(preds, dict):
                preds = next(iter(preds.values()))
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = tf.convert_to_tensor(preds)
            if isinstance(conv_out, (list, tuple)):
                conv_out = conv_out[0]
            conv_out = tf.convert_to_tensor(conv_out)
            tape.watch(conv_out)
            if preds.shape.rank == 1:
                preds = tf.expand_dims(preds, -1)
            if preds.shape[-1] == 1:
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                class_channel = preds[:, target_class]
        grads = tape.gradient(class_channel, conv_out)
        if grads is None or conv_out.shape.rank != 4:
            raise RuntimeError("no_grads_or_bad_rank")
        grads = tf.nn.relu(grads)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam = tf.where(tf.math.is_finite(cam), cam, 0.0)
        heat = tf.maximum(cam, 0.0)
        heat /= (tf.reduce_max(heat) + 1e-8)
        p95 = float(np.percentile(heat.numpy(), 95.0))
        heat = tf.clip_by_value(heat / (p95 + 1e-6), 0.0, 1.0)
        heat_np = _gaussian_blur(heat.numpy().astype(np.float32), kmin=3)
        return np.clip(heat_np, 0.0, 1.0), "gradcam", ""
    except Exception as e:
        note = f"gradcam_fallback({type(e).__name__})"
    # fallback (saliency)
    x = tf.convert_to_tensor(img_bchw)
    acc = 0
    for _ in range(12):
        noise = tf.random.normal(shape=tf.shape(x), stddev=0.1 * 255.0)
        xn = tf.clip_by_value(x + noise, 0.0, 255.0)
        with tf.GradientTape() as tape:
            tape.watch(xn)
            preds = model(xn, training=False)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
        g = tape.gradient(preds, xn)
        g = tf.reduce_max(tf.abs(g), axis=-1)[0]
        acc += g
    sal = acc / 12.0
    sal /= (tf.reduce_max(sal) + 1e-8)
    sal_np = np.power(_gaussian_blur(sal.numpy().astype(np.float32), 3), 1.2)
    return np.clip(sal_np, 0.0, 1.0), "saliency", note

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)

# ============================= Inference Helpers =============================
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)
    bchw_pp = preprocess_input(bchw_raw.copy())
    return arr, bchw_raw, bchw_pp

def predict_pneumonia_prob(model, bchw_raw):
    prob = model.predict(bchw_raw, verbose=0)
    if isinstance(prob, (list, tuple)):
        prob = prob[0]
    return float(np.asarray(prob).squeeze())

# ============================= Sidebar =============================
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold", 0.50, 0.69, 0.50, 0.01)
    st.divider()
    st.subheader("Grad-CAM")
    st.caption("기본: conv4_block24_concat (없으면 conv4 마지막 concat)")
    st.divider()
    use_mask = st.checkbox("Mask to lung area", value=True)
    cy_ratio = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx_ratio = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry_ratio = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap_ratio = st.slider("gap", 0.05, 0.20, 0.10, 0.01)
    thr_cut = st.slider("heatmap threshold", 0.00, 0.80, 0.00, 0.01)

# ============================= Main UI =============================
st.title("Chest X-ray Pneumonia Classifier (DenseNet121)")
st.write("Upload a chest X-ray to visualize Grad-CAM (not a medical device).")

# 1) Load model
model_path = ensure_model_file_cached()
model = load_model(model_path)
base = find_base_model(model)
cam_layer_names = list_densenet_conv_names(base)
default_index = cam_layer_names.index("conv4_block24_concat") if "conv4_block24_concat" in cam_layer_names else len(cam_layer_names) - 1
chosen_name = st.sidebar.selectbox("Select CAM layer", cam_layer_names, index=default_index)
conv_layer = get_densenet_feature_layer(base, chosen_name)

# 2) Upload image & inference
up = st.file_uploader("Upload X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw, _ = prepare_inputs(pil_img)
    colA, colB = st.columns(2)
    with colA:
        st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)
    if st.button("Run inference"):
        with st.spinner("Running Grad-CAM..."):
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)
            target_class = 1 if pred_label == "PNEUMONIA" else 0
            heatmap, method, note = make_gradcam_heatmap(x_raw_bchw, model, conv_layer, target_class)
            if use_mask:
                h, w = heatmap.shape[:2]
                mask = np.zeros((h, w), np.uint8)
                cx, cy = w // 2, int(h * cy_ratio)
                rx, ry = int(w * rx_ratio), int(h * ry_ratio)
                gap = int(w * gap_ratio)
                cv2.ellipse(mask, (cx - gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                cv2.ellipse(mask, (cx + gap, cy), (rx, ry), 0, 0, 360, 255, -1)
                heatmap *= (mask > 0).astype(np.float32)
            cam_img = overlay_heatmap(rgb_uint8, heatmap)
        with colB:
            st.image(cam_img, caption=f"CAM (layer: {conv_layer.name}, method: {method})", use_column_width=True)
        st.metric("Predicted", pred_label)
        st.metric("Prob. PNEUMONIA", f"{p_pneu*100:.2f}%")
else:
    st.caption("Awaiting image upload…")
