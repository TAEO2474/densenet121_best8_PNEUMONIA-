# app.py
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

# Google Drive FILE ID (Secrets 우선)
FILE_ID = st.secrets.get("MODEL_FILE_ID", "1UPxtL1kx8a38z9fxlBRljNn8n6T4LL_l")
MODEL_DIR = Path("models")
MODEL_LOCAL = MODEL_DIR / "densenet121_best_9.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Drive 막힐 때 폴백 URL (HF/GitHub 등)
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
    """DenseNet 백본(서브모델) 또는 가장 안쪽 Model을 반환."""
    try:
        return m.get_layer("densenet121")
    except Exception:
        for lyr in m.layers[::-1]:
            if isinstance(lyr, keras.Model):
                return lyr
        return m

def _is_4d(layer) -> bool:
    """K.int_shape로 안전하게 rank==4 판정."""
    try:
        shp = K.int_shape(layer.output)  # (None,H,W,C)
        return (shp is not None) and (len(shp) == 4)
    except Exception:
        return False

def list_4d_layers(model_or_layer):
    if isinstance(model_or_layer, keras.Model):
        return [lyr for lyr in model_or_layer.layers if _is_4d(lyr)]
    return []

def find_last_conv4d_layer_recursive(layer_or_model):
    last = None
    if isinstance(layer_or_model, keras.Model):
        for lyr in layer_or_model.layers:
            cand = find_last_conv4d_layer_recursive(lyr)
            if cand is not None:
                last = cand
    else:
        if _is_4d(layer_or_model):
            last = layer_or_model
    return last

def get_densenet_feature_layer(base: keras.Model):
    """
    ✅ 우선순위로 DenseNet121의 마지막 conv 특성맵을 명시적으로 선택.
       1) conv5_block16_concat
       2) relu (일부 구현체)
       3) 이름에 concat/relu 포함된 4D 레이어
       4) 마지막 4D 레이어
    """
    preferred_names = ["conv5_block16_concat", "relu"]
    for name in preferred_names:
        try:
            lyr = base.get_layer(name)
            if _is_4d(lyr):
                return lyr
        except Exception:
            pass
    four_d = list_4d_layers(base)
    if four_d:
        pref = [l for l in four_d if ("concat" in l.name.lower()) or ("relu" in l.name.lower())]
        return pref[-1] if pref else four_d[-1]
    return None

@tf.function
def _normalize_heatmap(x):
    x = tf.maximum(x, 0.0)
    mx = tf.reduce_max(x)
    return tf.where(mx > 0, x / mx, x)

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer):
    """
    conv_layer: 레이어 '객체' (서브모델 내부 포함)
    img_bchw: float32, [1,H,W,3]
    실패 시 Saliency로 폴백.
    """
    # 입력 보장
    if not isinstance(img_bchw, np.ndarray):
        img_bchw = np.array(img_bchw)
    if img_bchw.dtype != np.float32:
        img_bchw = img_bchw.astype(np.float32)

    # ---------- 방법 A: 표준 Grad-CAM (같은 그래프에서 conv/출력) ----------
    try:
        grad_model = keras.Model(inputs=model.input, outputs=[conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_bchw, training=False)

            if isinstance(preds, dict): preds = next(iter(preds.values()))
            if isinstance(preds, (list, tuple)): preds = preds[0]
            preds = tf.convert_to_tensor(preds)

            if isinstance(conv_out, (list, tuple)): conv_out = conv_out[0]
            conv_out = tf.convert_to_tensor(conv_out)
            tape.watch(conv_out)

            if preds.shape.rank is None or preds.shape.rank == 0:
                preds = tf.reshape(preds, (-1, 1))
            elif preds.shape.rank == 1:
                preds = tf.expand_dims(preds, -1)

            class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None or conv_out.shape.rank != 4:
            raise RuntimeError("Grad-CAM A 실패")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # [C]
        conv_out = conv_out[0]                                # [Hc,Wc,C]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = _normalize_heatmap(heatmap)

        # 시각 강조(감마 보정)
        heatmap = tf.pow(heatmap, 2.0)
        return heatmap.numpy().astype(np.float32)
    except Exception:
        pass

    # ---------- 방법 B: Saliency(입력-그래디언트) 폴백 ----------
    x = tf.convert_to_tensor(img_bchw)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        if isinstance(preds, dict): preds = next(iter(preds.values()))
        if isinstance(preds, (list, tuple)): preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        if preds.shape.rank == 1:
            preds = preds[None, :]
        class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]
    grads = tape.gradient(class_channel, x)  # [1,H,W,3]
    sal = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # [H,W]
    sal = sal / (tf.reduce_max(sal) + 1e-8)
    return sal.numpy().astype(np.float32)

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.55):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    out = (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return out

# ============================= Inference =============================
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)  # 모델 입력과 동일 스케일
    bchw_pp  = preprocess_input(bchw_raw.copy())               # (분석용)
    return arr, bchw_raw, bchw_pp

def predict_pneumonia_prob(model, bchw_raw):
    prob = model.predict(bchw_raw, verbose=0)
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

# 1) 모델 확보
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
            # 그래프 빌드 고정
            _ = model(x_raw_bchw, training=False)

            # 예측
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            # ---- Grad-CAM: DenseNet 마지막 conv 특성맵 명시적 선택 ----
            base = find_base_model(model)
            conv_layer = get_densenet_feature_layer(base)
            if conv_layer is None:  # 드문 경우: 폴백
                conv_layer = find_last_conv4d_layer_recursive(base) or find_last_conv4d_layer_recursive(model)

            heatmap = make_gradcam_heatmap(x_raw_bchw, model, conv_layer)
            cam_img = overlay_heatmap(rgb_uint8, heatmap, alpha=0.55)

        with colB:
            st.image(cam_img, caption=f"Grad-CAM (layer: {conv_layer.name if conv_layer is not None else 'fallback'})", use_column_width=True)

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
