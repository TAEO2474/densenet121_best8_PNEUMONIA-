# ============================================================
# app.py — DenseNet121_BinaryClassifier 완전 안정형 Grad-CAM (Auto Input Safe)
# ============================================================

import os
from pathlib import Path
import re
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import cv2
import gdown, requests

# ----------------------- 기본 설정 -----------------------
st.set_page_config(page_title="CXR Pneumonia — DenseNet121 + Grad-CAM", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOCAL = MODEL_DIR / "densenet121_best.keras"
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
def ensure_model_file_cached() -> str:
    if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
        return str(MODEL_LOCAL)
    if FILE_ID:
        try:
            gdown.download(id=FILE_ID, output=str(MODEL_LOCAL), quiet=False, fuzzy=True)
        except Exception:
            pass
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
            return str(MODEL_LOCAL)
    if HTTP_FALLBACK_URL:
        _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL)
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
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

# ----------------------- 안전 입력 감지 -----------------------
def _forward_safe(model_or_fn, x, training=False):
    """입력 방식(tensor/dict)을 자동 감지해 재시도"""
    try:
        return model_or_fn(x, training=training)
    except Exception:
        in_name = model_or_fn.inputs[0].name.split(":")[0]
        return model_or_fn({in_name: x}, training=training)

# ----------------------- 전처리 -----------------------
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)
    return rgb_uint8, bchw_raw

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    hm = np.clip(hm, 0.0, 1.0)
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)

def ellipse_lung_mask(h, w, cy=0.48, rx=0.23, ry=0.32, gap=0.10):
    mask = np.zeros((h, w), np.uint8)
    cx = w // 2
    cy = int(h * cy)
    rx = int(w * rx)
    ry = int(h * ry)
    gap = int(w * gap)
    cv2.ellipse(mask, (cx - gap, cy), (rx, ry), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx + gap, cy), (rx, ry), 0, 0, 360, 255, -1)
    return (mask > 0).astype(np.float32)

# ----------------------- Grad-CAM (입력자동감지, Any Layer) -----------------------
def gradcam_from_any_layer(img_bchw, model, target_layer_name, target_class=1):
    base = model.get_layer("densenet121")
    try:
        target_tensor = base.get_layer(target_layer_name).output
    except Exception:
        fallback = "relu" if "relu" in [l.name for l in base.layers] else base.layers[-1].name
        target_tensor = base.get_layer(fallback).output
        target_layer_name = fallback

    cam_model = keras.Model(inputs=model.input, outputs=[target_tensor, model.output])
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_feat, preds = _forward_safe(cam_model, x, training=False)
        cls = preds[:, 0] if preds.shape[-1] == 1 else preds[:, target_class]

    grads = tape.gradient(cls, conv_feat)
    if grads is None:
        conv_feat += tf.random.normal(tf.shape(conv_feat), stddev=1e-8)
        with tf.GradientTape() as t2:
            _, preds2 = _forward_safe(cam_model, x, training=False)
            cls2 = preds2[:, 0] if preds2.shape[-1] == 1 else preds2[:, target_class]
        grads = t2.gradient(cls2, conv_feat)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.nn.relu(conv_feat[0] * weights), axis=-1)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    return cam.numpy().astype(np.float32), float(preds.numpy().squeeze()), target_layer_name

# ----------------------- 유틸: concat 선택 -----------------------
def _sorted_concats(names):
    concats = [n for n in names if n.endswith("_concat") and "block" in n]
    def key(n):
        m = re.search(r"conv(\d+)_block(\d+)_concat", n)
        return (int(m.group(1)), int(m.group(2))) if m else (0,0)
    return sorted(concats, key=key)

def pick_deep_and_prev(names):
    concats = _sorted_concats(names)
    if not concats:
        return None, None
    deep = concats[-1]
    prev = concats[-2] if len(concats) >= 2 else None
    return deep, prev

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.40, 0.75, 0.69, 0.01)
    st.caption("• 낮추면 민감도↑ • 높이면 정상 보호(오탐↓)")

    st.divider()
    st.subheader("Grad-CAM layer (DenseNet 내부)")
    st.caption("권장: 마지막 활성화 **relu**. 필요하면 concat 계열도 비교 가능.")

    st.divider()
    st.subheader("Lung mask (optional)")
    use_mask = st.checkbox("Apply ellipse lung mask", value=True)
    cy = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap = st.slider("gap", 0.05, 0.20, 0.10, 0.01)

    st.divider()
    st.subheader("CAM refine")
    use_multiscale = st.checkbox("Use multiscale (deep × prev)", value=True)
    fusion_gamma = st.slider("Prev exponent (γ)", 0.3, 1.5, 0.7, 0.1)
    cam_percentile = st.slider("Percentile clip", 80, 99, 97, 1)
    cam_blur = st.checkbox("Gaussian blur after fuse (3×3)", value=False)

# ----------------------- Main -----------------------
st.title("🩻 Chest X-ray Pneumonia — DenseNet121 + Grad-CAM (Separated)")
st.caption("의사용 장비가 아닙니다. 참고용 해석 도구입니다.")

try:
    model_path = ensure_model_file_cached()
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로딩 실패: {e}")
    st.stop()

base = model.get_layer("densenet121")
all_names = [l.name for l in base.layers]
cands = [n for n in all_names if ("relu" in n or ("_concat" in n and "block" in n))]
cands = sorted(set(cands), key=lambda s: (("relu" not in s), s))
default = "relu" if "relu" in all_names else (cands[-1] if cands else all_names[-1])
chosen = st.sidebar.selectbox("Select CAM target layer", cands or all_names,
                              index=(cands or all_names).index(default))

up = st.file_uploader("Upload an X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)

    if st.button("Run Grad-CAM"):
        with st.spinner("Running…"):
            try:
                deep, prev = pick_deep_and_prev(all_names)

                if use_multiscale and deep and prev:
                    cam5, p1, d_used = gradcam_from_any_layer(x_raw_bchw, model, deep, 1)
                    cam4, _, p_used  = gradcam_from_any_layer(x_raw_bchw, model, prev, 1)
                    cam5 /= cam5.max() + 1e-6
                    cam4 /= cam4.max() + 1e-6
                    heatmap = cam5 * (cam4 ** fusion_gamma)
                    label_layer = f"{d_used} × {p_used}^{fusion_gamma:.2f}"
                    prob = p1
                else:
                    heatmap, prob, l_used = gradcam_from_any_layer(x_raw_bchw, model, chosen, 1)
                    label_layer = l_used

                heatmap = np.clip(heatmap / (np.percentile(heatmap, cam_percentile) + 1e-6), 0, 1)
                if cam_blur:
                    heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
                if use_mask:
                    mh, mw = heatmap.shape
                    m = ellipse_lung_mask(mh, mw, cy, rx, ry, gap)
                    heatmap *= m

                label = "PNEUMONIA" if prob >= thresh else "NORMAL"
                cam_img = overlay_heatmap(rgb_uint8, heatmap)

                with col2:
                    st.image(cam_img, caption=f"Grad-CAM ({label_layer})", use_column_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", label)
                c2.metric("Prob. PNEUMONIA", f"{prob*100:.2f}%")
                c3.metric("Threshold", f"{thresh:.2f}")

            except Exception as e:
                st.error(f"Grad-CAM 실패: {type(e).__name__} — {e}")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
