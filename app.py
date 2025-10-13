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
    """한 단계 자식 중 4D 출력 레이어 목록."""
    if isinstance(model_or_layer, keras.Model):
        return [lyr for lyr in model_or_layer.layers if _is_4d(lyr)]
    return []

def find_last_conv4d_layer_recursive(layer_or_model):
    """모델 트리를 DFS로 훑어 마지막 4D(B,H,W,C) 레이어를 반환."""
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

def pick_best_densenet_conv_layer(base: keras.Model):
    """
    DenseNet 내부 4D 레이어 중 이름에 'concat'/'relu' 포함 레이어를 우선.
    없으면 마지막 4D 레이어.
    """
    four_d_layers = list_4d_layers(base)
    if not four_d_layers:
        return None
    preferred = [l for l in four_d_layers if ("concat" in l.name.lower()) or ("relu" in l.name.lower())]
    return (preferred[-1] if preferred else four_d_layers[-1])

@tf.function
def _normalize_heatmap(x):
    x = tf.maximum(x, 0.0)
    mx = tf.reduce_max(x)
    return tf.where(mx > 0, x / mx, x)

def make_gradcam_heatmap(img_bchw, model: keras.Model, conv_layer):
    """
    conv_layer: 레이어 '객체' (서브모델 내부 포함)
    img_bchw: float32, [1,H,W,3]
    """
    # 입력 보장
    if not isinstance(img_bchw, np.ndarray):
        img_bchw = np.array(img_bchw)
    if img_bchw.dtype != np.float32:
        img_bchw = img_bchw.astype(np.float32)

    if conv_layer is None or not hasattr(conv_layer, "output"):
        raise ValueError("유효한 4D conv 레이어를 찾지 못했습니다.")

    # ---------- 방법 A: 서브모델 2번 호출 (가장 표준) ----------
    try:
        last_conv_model = keras.Model(inputs=model.input, outputs=conv_layer.output)
        with tf.GradientTape() as tape:
            conv_out = last_conv_model(img_bchw, training=False)      # [N,Hc,Wc,C]
            preds    = model(img_bchw,        training=False)          # [N,1] or [N,C]

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

            class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise RuntimeError("Gradients are None in method A.")

        if conv_out.shape.rank != 4:
            raise RuntimeError(f"Conv rank != 4 in method A: {conv_out.shape}")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # [C]
        conv_out = conv_out[0]                                # [Hc,Wc,C]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        return _normalize_heatmap(heatmap).numpy()
    except Exception as e_a:
        # st.warning(f"Grad-CAM A failed: {e_a}")
        pass

    # ---------- 방법 B: K.function 경로 (그래프 매칭 문제 우회) ----------
    try:
        # 학습/추론 모드 분기 없는 모델이면 learning_phase 필요 없음
        try:
            fetch_fn = K.function([model.input], [conv_layer.output, model.output])
            conv_out, preds = fetch_fn([img_bchw])
        except Exception:
            # 일부 환경에선 learning_phase가 필요
            fetch_fn = K.function([model.input, K.learning_phase()],
                                  [conv_layer.output, model.output])
            conv_out, preds = fetch_fn([img_bchw, 0])

        conv_out = np.asarray(conv_out)
        preds    = np.asarray(preds)

        if conv_out.ndim != 4:
            raise RuntimeError(f"Conv rank != 4 in method B: {conv_out.shape}")

        if preds.ndim == 0:
            preds = preds.reshape(1, 1)
        elif preds.ndim == 1:
            preds = preds[:, None]

        class_channel = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]

        # numpy로 그라디언트 못 구하므로, 간단한 CAM 가중치 근사 (채널별 GAP 가중)
        # -> 방법 B에서는 실제 그라디언트를 못 쓰니, 채널별 평균 활성으로 근사
        weights = conv_out.mean(axis=(1, 2), keepdims=False)[0]  # [C]
        fmap    = conv_out[0]                                    # [Hc,Wc,C]
        heatmap = np.tensordot(fmap, weights, axes=([2], [0]))  # [Hc,Wc]
        # 클래스 스코어의 부호를 반영 (양성일수록 강조)
        if class_channel.ndim:
            sign = 1.0 if float(class_channel.squeeze()) >= 0 else -1.0
            heatmap *= sign
        heatmap = np.maximum(heatmap, 0.0)
        mx = heatmap.max()
        if mx > 0:
            heatmap = heatmap / mx
        return heatmap.astype(np.float32)
    except Exception as e_b:
        # st.warning(f"Grad-CAM B failed: {e_b}")
        pass

    # ---------- 방법 C: Saliency(입력-그래디언트) 폴백 ----------
    # conv 레이어가 어떤 이유로든 연결 안될 때라도, 입력에 대한 민감도로 히트맵을 보여준다.
    try:
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
    except Exception as e_c:
        raise RuntimeError(f"Grad-CAM and saliency fallback all failed: {e_c}")

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
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(arr.astype(np.float32), axis=0)  # 모델 입력과 동일 스케일
    bchw_pp  = preprocess_input(bchw_raw.copy())               # 분석용
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
            # ------------------ 빌드 ------------------
            # 먼저 한 번 호출해 그래프/shape를 확정(빌드)한다.
            _ = model(x_raw_bchw, training=False)

            # 예측
            p_pneu = predict_pneumonia_prob(model, x_raw_bchw)
            pred_label = CLASS_NAMES[1] if p_pneu >= thresh else CLASS_NAMES[0]
            conf = p_pneu if pred_label == "PNEUMONIA" else (1 - p_pneu)

            # ------------------ 선택 ------------------
            # DenseNet 서브모델에서 '진짜 4D conv' 레이어를 고른다.
            base = find_base_model(model)
            conv_layer = pick_best_densenet_conv_layer(base)
            if conv_layer is None:
                conv_layer = find_last_conv4d_layer_recursive(base)
            if conv_layer is None:
                conv_layer = find_last_conv4d_layer_recursive(model)

            if conv_layer is None:
                st.error("4D conv 레이어를 찾지 못했습니다. 모델 구조를 확인해 주세요.")
                st.stop()

            last_layer_name = conv_layer.name
            st.write("conv shape:", tf.convert_to_tensor(keras.Model(model.input, conv_layer.output)(x_raw_bchw)).shape) # 디버그용 (원하면 제거)

            # ------------------ 실행 ------------------
            heatmap = make_gradcam_heatmap(x_raw_bchw, model, conv_layer)
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
