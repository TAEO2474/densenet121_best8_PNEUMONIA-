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
    names = [l.name for l in base.layers if _is_4d(l)]
    names.sort(key=lambda x: (x.count("block5") == 0, x))
    return names

def get_densenet_feature_layer(base: keras.Model, preferred_name: str | None):
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

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer, target_class: int = 1):
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

            # (N,C) 표준화 ...
            if preds.shape[-1] == 1:
                # sigmoid 이진: target_class==1(폐렴)이면 p, 0(정상)이면 (1-p)
                class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
                # softmax 다중: 예측(또는 지정) 클래스 인덱스 사용
                class_channel = preds[:, target_class]
                    grads = tape.gradient(class_channel, conv_out)
                    if grads is None or conv_out.shape.rank != 4:
                        raise RuntimeError("Grad-CAM path failed")

        grads = tf.nn.relu(grads)  # 양의 영향만
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # [C]
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # [Hc,Wc]
        heatmap = _normalize_heatmap(cam)
        # 대비 강화: 상위 퍼센타일로 스케일링
        p99 = float(np.percentile(heatmap.numpy(), 99.0))
        heatmap = tf.clip_by_value(heatmap / (p99 + 1e-6), 0.0, 1.0)
        heatmap = tf.pow(heatmap, 2.0)  # 감마 강조
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
            
                preds = preds[None, :]
            if preds.shape[-1] == 1:
            class_channel = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
            else:
            class_channel = preds[:, target_class]

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

# ========= Lung mask (ellipses) =========
def lung_mask_ellipses(h, w, cy_ratio=0.48, rx_ratio=0.23, ry_ratio=0.32, gap_ratio=0.10):
    """
    간단 타원 2개로 좌/우 폐 영역 근사 마스크 생성.
    파라미터는 사이드바에서 조정 가능.
    """
    mask = np.zeros((h, w), np.uint8)
    cx = w // 2
    cy = int(h * float(cy_ratio))
    rx = int(w * float(rx_ratio))
    ry = int(h * float(ry_ratio))
    gap = int(w * float(gap_ratio))
    left_center  = (cx - gap, cy)
    right_center = (cx + gap, cy)
    cv2.ellipse(mask, left_center,  (rx, ry), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, right_center, (rx, ry), 0, 0, 360, 255, -1)
    return (mask > 0).astype(np.float32)

def apply_lung_mask(heatmap, cy_ratio, rx_ratio, ry_ratio, gap_ratio, thr=None):
    """
    heatmap:[H,W] -> 타원 마스크를 곱하고(폐 바깥=0), 선택적으로 임계값 이하 컷(thr) 적용
    """
    h, w = heatmap.shape[:2]
    m = lung_mask_ellipses(h, w, cy_ratio, rx_ratio, ry_ratio, gap_ratio)
    masked = heatmap * m
    if thr is not None:
        masked = np.where(masked >= thr, masked, 0.0)
    return masked

# ============================= Inference =============================
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)
    bchw_pp  = preprocess_input(bchw_raw.copy())  # (분석용)
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
    # CAM 레이어 목록은 모델 로드 후 본문에서 채웁니다.

    st.divider()
    st.subheader("Lung mask (ellipses)")
    use_mask = st.checkbox("Mask to lung area only", value=True)
    with st.expander("Mask tuning (optional)"):
        cy_ratio  = st.slider("center y (ratio)", 0.35, 0.60, 0.48, 0.01)
        rx_ratio  = st.slider("radius x (ratio)", 0.15, 0.35, 0.23, 0.01)
        ry_ratio  = st.slider("radius y (ratio)", 0.20, 0.45, 0.32, 0.01)
        gap_ratio = st.slider("left/right gap (ratio)", 0.05, 0.20, 0.10, 0.01)
        thr_cut   = st.slider("heatmap threshold", 0.00, 0.80, 0.30, 0.01)

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
            target_class = 1 if p_pneu >= 0.5 else 0

            # CAM 레이어 선택
            preferred = None if chosen_name.startswith("(auto)") else chosen_name
            conv_layer = get_densenet_feature_layer(base, preferred)

            heatmap, method = make_gradcam_heatmap(x_raw_bchw, model, conv_layer, target_class=target_class)

            # === 여기서 폐 마스크 적용 ===
            if use_mask:
                heatmap = apply_lung_mask(
                    heatmap,
                    cy_ratio=cy_ratio,
                    rx_ratio=rx_ratio,
                    ry_ratio=ry_ratio,
                    gap_ratio=gap_ratio,
                    thr=thr_cut,
                )

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
