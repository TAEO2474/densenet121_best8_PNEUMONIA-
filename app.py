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
        shp = K.int_shape(layer.output)  # (None,H,W,C)
        return (shp is not None) and (len(shp) == 4)
    except Exception:
        return False

def list_4d_layers(model_or_layer):
    return [lyr for lyr in model_or_layer.layers if _is_4d(lyr)] if isinstance(model_or_layer, keras.Model) else []

def list_densenet_conv_names(base: keras.Model):
    """DenseNet 내부 4D conv 계열 레이어 이름 목록(사이드바 선택용)."""
    names = [l.name for l in base.layers if _is_4d(l)]
    # 뒤쪽 블록들이 위로 오도록 살짝 정렬 힌트
    names.sort(key=lambda x: (x.count("block5") == 0, x))
    return names

def get_densenet_feature_layer(base: keras.Model, preferred_name: str | None):
    """
    우선순위:
      - 사용자가 고른 preferred_name (있고 4D면)
      - conv5_block16_concat
      - relu
      - 이름에 concat/relu 포함된 4D
      - 마지막 4D
    """
    try:
        if preferred_name:
            lyr = base.get_layer(preferred_name)
            if _is_4d(lyr):
                return lyr
    except Exception:
        pass
    for name in ["conv5_block16_concat", "relu"]:
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
    return tf.where(mx > 0, x / (mx + 1e-8), x)

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer):
    """
    반환: (heatmap [Hc,Wc] float32(0~1), method 'gradcam' | 'saliency')
    """
    if not isinstance(img_bchw, np.ndarray):
        img_bchw = np.array(img_bchw, dtype=np.float32)
    if img_bchw.dtype != np.float32:
        img_bchw = img_bchw.astype(np.float32)

    # A) 표준 Grad-CAM
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

            # (N,C) 표준화
            if preds.shape.rank is None or preds.shape.rank == 0:
                preds = tf.reshape(preds, (-1, 1))
            elif preds.shape.rank == 1:
                preds = tf.expand_dims(preds, -1)
            # Grad-CAM 논문 방식: 양의 그라디언트만 사용
            class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None or conv_out.shape.rank != 4:
            raise RuntimeError("Grad-CAM path failed")

        grads = tf.nn.relu(grads)  # 양의 영향만
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # [C]
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # [Hc,Wc]
        heatmap = _normalize_heatmap(cam)
        # 강조(감마) + 약간의 스무딩
        heatmap = tf.pow(heatmap, 2.0)
        heatmap = tf.numpy_function(lambda m: cv2.GaussianBlur(m, (3, 3), 0), [heatmap], tf.float32)
        return np.clip(heatmap, 0.0, 1.0).astype(np.float32), "gradcam"
    except Exception:
        # B) 폴백: 입력-그라디언트 살리언시
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
        sal = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        sal = sal / (tf.reduce_max(sal) + 1e-8)
        sal = tf.pow(sal, 1.5)
        sal = tf.numpy_function(lambda m: cv2.GaussianBlur(m, (3, 3), 0), [sal], tf.float32)
        return np.clip(sal, 0.0, 1.0).astype(np.float32), "saliency"

def overlay_heatmap(rgb_uint8, heatmap, alpha=0.6):
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    out = (jet * alpha + rgb_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return out

# ============================= Inference =============================
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)  # 모델 입력 스케일
    bchw_pp  = preprocess_input(bchw_raw.copy())               # (분석용)
    return arr, bchw_raw, bchw_pp

def predict_pneumonia_prob(model, bchw_raw):
    prob = model.predict(bchw_raw, verbose=0)
    if isinstance(prob, dict): prob = next(iter(prob.values()))
    if isinstance(prob, (list, tuple)): prob = prob[0]
    prob = np.asarray(prob).squeeze()
    return float(prob.item() if hasattr(prob, "item") else prob)

# ============================= Sidebar =============================
with st.sidebar:
    st.header("Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.50, 0.69, 0.50, 0.01)
    st.caption("• Lower = higher sensitivity for pneumonia\n• Higher = fewer false positives")

    st.divider()
    st.subheader("Grad-CAM layer")
    st.caption("기본: conv5_block16_concat (없으면 자동 폴백). 필요시 다른 블록으로 비교해 보세요.")
    # 선택 가능한 레이어 목록은 모델 로드 후 채웁니다. (아래 본문에서 렌더)

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

# 사이드바: CAM 레이어 선택 옵션 채우기
base = find_base_model(model)
cam_layer_names = ["(auto) conv5_block16_concat"] + list_densenet_conv_names(base)
with st.sidebar:
    chosen_name = st.selectbox("Select CAM layer", cam_layer_names, index=0)

# 3) 예측 UI
up = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw, _ = prepare_inputs(pil_img)

    colA, colB = st.columns([1, 1])
    with colA:
        st.image(rgb_uint8, caption="Input (Resized 224×224)", use_column_width=True)

    if st.button("Run inference"):
        with st.spinner("Running model..."):
            _ = model(x_raw_bchw, training=False)  # 그래프 빌드 고정

            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            # CAM 레이어 선택
            preferred = None if chosen_name.startswith("(auto)") else chosen_name
            conv_layer = get_densenet_feature_layer(base, preferred)

            heatmap, method = make_gradcam_heatmap(x_raw_bchw, model, conv_layer)
            cam_img = overlay_heatmap(rgb_uint8, heatmap, alpha=0.6)

        with colB:
            cap = f"Grad-CAM (layer: {conv_layer.name if conv_layer is not None else 'N/A'}, method: {method})"
            st.image(cam_img, caption=cap, use_column_width=True)

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
