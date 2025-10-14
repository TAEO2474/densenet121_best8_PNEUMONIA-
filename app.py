# ============================================================
# app.py — DenseNet121_BinaryClassifier (안정형 + 병변 국소화 최적화)
# ============================================================

import os
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import cv2
import gdown, requests

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="CXR Pneumonia — DenseNet121 + Grad-CAM", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOCAL = MODEL_DIR / "densenet121_best.keras"
FILE_ID = st.secrets.get("MODEL_FILE_ID", "")
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")
TIMEOUT = 120

# ============================================================
# 모델 다운로드 유틸
# ============================================================
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

# ============================================================
# 안전 입력 감지 (_forward_safe)
# ============================================================
def _forward_safe(model_or_fn, x, training=False):
    """
    입력 자동 감지 (KeyError 완전 방지)
    1️⃣ 단일 텐서 입력
    2️⃣ 단일 입력 이름 dict({exact_name: tensor})
    """
    try:
        return model_or_fn(x, training=training)
    except Exception:
        pass

    ins = getattr(model_or_fn, "inputs", None)
    if isinstance(ins, (list, tuple)) and len(ins) == 1:
        in_name = ins[0].name.split(":")[0]
        return model_or_fn({in_name: x}, training=training)

    raise RuntimeError("Unsupported input signature (expected single tensor or single-name dict).")

# ============================================================
# 전처리 / 시각화 유틸
# ============================================================
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

# ============================================================
# Grad-CAM 본체
# ============================================================
def gradcam_from_any_layer(img_bchw, model, layer_name, target_class=1):
    base = model.get_layer("densenet121")
    try:
        target_tensor = base.get_layer(layer_name).output
    except Exception:
        target_tensor = base.layers[-1].output
        layer_name = base.layers[-1].name

    cam_model = keras.Model(inputs=model.input, outputs=[target_tensor, model.output])
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_feat, preds = _forward_safe(cam_model, x, training=False)
        cls = preds[:, 0] if preds.shape[-1] == 1 else preds[:, target_class]

    grads = tape.gradient(cls, conv_feat)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.nn.relu(conv_feat[0] * weights), axis=-1)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    return cam.numpy().astype(np.float32), float(preds.numpy().squeeze()), layer_name

def pick_deep_and_prev(names):
    concats = [n for n in names if n.endswith("_concat") and "block" in n]
    if len(concats) >= 2:
        return concats[-1], concats[-2]
    elif len(concats) == 1:
        return concats[0], None
    return None, None

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.40, 0.75, 0.69, 0.01)

    st.divider()
    st.subheader("Grad-CAM layer")
    st.caption("권장: 마지막 활성화 ReLU 또는 마지막 concat")

    st.divider()
    st.subheader("Lung mask")
    use_mask = st.checkbox("Apply ellipse lung mask", value=True)
    cy = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap = st.slider("gap", 0.05, 0.20, 0.10, 0.01)

# ============================================================
# Main
# ============================================================
st.title("🩻 Chest X-ray Pneumonia — DenseNet121 + Grad-CAM (Optimized)")
st.caption("의사용 장비가 아닙니다. 참고용 해석 도구입니다.")

try:
    model_path = ensure_model_file_cached()
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로딩 실패: {e}")
    st.stop()

base = model.get_layer("densenet121")
all_names = [l.name for l in base.layers]
cands = [n for n in all_names if ("relu" in n or "concat" in n)]
cands = sorted(set(cands))
default_name = "relu" if "relu" in all_names else cands[-1]
chosen_name = st.sidebar.selectbox("Select CAM layer", cands, index=cands.index(default_name))

up = st.file_uploader("Upload an X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_uint8, caption="Input", use_column_width=True)

    if st.button("Run Grad-CAM"):
        with st.spinner("Running…"):
            try:
                # ✅ 멀티스케일 CAM 융합
                deep, prev = pick_deep_and_prev(all_names)
                if deep and prev:
                    cam5, p1, _ = gradcam_from_any_layer(x_raw_bchw, model, deep)
                    cam4, _, _ = gradcam_from_any_layer(x_raw_bchw, model, prev)
                    cam5 = cam5 / (cam5.max() + 1e-6)
                    cam4 = cam4 / (cam4.max() + 1e-6)
                    heatmap = cam5 * (cam4 ** 0.7)
                else:
                    heatmap, p1, _ = gradcam_from_any_layer(x_raw_bchw, model, chosen_name)

                # ✅ 퍼짐 최소화
                heatmap = np.clip(heatmap / (np.percentile(heatmap, 97) + 1e-6), 0, 1)

                # ✅ 마스크 적용
                if use_mask:
                    h, w = heatmap.shape
                    mask = ellipse_lung_mask(h, w, cy, rx, ry, gap)
                    heatmap *= mask

                cam_img = overlay_heatmap(rgb_uint8, heatmap)
                label = "PNEUMONIA" if p1 >= thresh else "NORMAL"

                with col2:
                    st.image(cam_img, caption=f"Grad-CAM ({chosen_name})", use_column_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", label)
                c2.metric("Prob. PNEUMONIA", f"{p1*100:.2f}%")
                c3.metric("Threshold", f"{thresh:.2f}")

            except Exception as e:
                st.error(f"Grad-CAM 실패: {type(e).__name__} — {e}")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
