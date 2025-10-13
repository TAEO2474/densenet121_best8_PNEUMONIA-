# ============================================================
# app.py — DenseNet121_BinaryClassifier 전용, 분리형 Grad-CAM 확실 작동 버전
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

# ----------------------- 기본 설정 -----------------------
st.set_page_config(page_title="CXR Pneumonia — DenseNet121 + Grad-CAM", layout="wide")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_LOCAL = MODEL_DIR / "densenet121_best.keras"  # 파일명은 네가 실제 배포한 이름으로 맞추면 됨
FILE_ID = st.secrets.get("MODEL_FILE_ID", "")       # gdown File ID (선택)
HTTP_FALLBACK_URL = st.secrets.get("MODEL_DIRECT_URL", "")  # 직접 URL (선택)
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
    # 1) gdown
    if FILE_ID:
        try:
            gdown.download(id=FILE_ID, output=str(MODEL_LOCAL), quiet=False, fuzzy=True)
        except Exception:
            pass
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
            return str(MODEL_LOCAL)
    # 2) HTTP fallback
    if HTTP_FALLBACK_URL:
        _http_download(HTTP_FALLBACK_URL, MODEL_LOCAL)
        if MODEL_LOCAL.exists() and MODEL_LOCAL.stat().st_size > 0:
            return str(MODEL_LOCAL)
    raise RuntimeError("모델 다운로드 실패")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    # Lambda(preprocess_input) 복원 위해 custom_objects 등록
    return keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=False,
    )

# ----------------------- 전처리/시각화 -----------------------
def prepare_inputs(pil_img: Image.Image):
    pil = pil_img.convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)
    bchw_raw = np.expand_dims(rgb_uint8.astype(np.float32), axis=0)  # (1,H,W,3)
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

# ----------------------- 분리형 Grad-CAM -----------------------
def build_feature_and_classifier(model: keras.Model, last_conv_name: str):
    """
    네가 학습한 전체 모델을 아래처럼 분해:
    - preproc: input_image -> densenet_preprocess 출력
    - backbone: DenseNet121 (base_model) 의 'last_conv_name'까지
    - classifier: (GAP -> Dropout -> Dense) 머리부분
    """
    # 1) 서브그래프: 전처리
    pre_in = model.get_layer("input_image").input
    pre_out = model.get_layer("densenet_preprocess").output
    preproc = keras.Model(pre_in, pre_out, name="preprocessor")

    # 2) 서브그래프: DenseNet121의 마지막 conv/활성화 출력
    base = model.get_layer("densenet121")
    last_conv = base.get_layer(last_conv_name)  # 기본 'relu' 추천
    feature_extractor = keras.Model(base.input, last_conv.output, name="feature_extractor")

    # 3) 서브그래프: classifier 머리 (backbone 이후 레이어만 재구성)
    #    원래 모델 순서: [input_image, densenet_preprocess, densenet121, GAP, Dropout, Dense]
    #    → densenet121 이후 레이어들만 그대로 재사용
    classifier_in = keras.Input(shape=last_conv.output.shape[1:], name="cam_head_in")
    x = classifier_in
    for lyr in model.layers:
        if lyr.name in ["input_image", "densenet_preprocess", "densenet121"]:
            continue
        x = lyr(x)
    classifier = keras.Model(classifier_in, x, name="classifier_head")

    return preproc, feature_extractor, classifier

def gradcam_separated(img_bchw: np.ndarray, model: keras.Model, last_conv_name: str, target_class: int = 1):
    """
    분리형(권장) Grad-CAM:
    - x -> preproc(x) -> feature_extractor(relu까지) -> conv_feat
    - preds = classifier(conv_feat)
    - d(preds[:,target])/d(conv_feat) 로 CAM 생성
    """
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)

    # 분리 그래프 구성
    preproc, feat, head = build_feature_and_classifier(model, last_conv_name)

    # 전방통과 + Gradient 계산
    with tf.GradientTape() as tape:
        x_pp = preproc(x, training=False)
        conv_feat = feat(x_pp, training=False)
        tape.watch(conv_feat)

        preds = head(conv_feat, training=False)  # shape: (1,1) 이진
        if preds.shape[-1] == 1:
            cls = preds[:, 0] if target_class == 1 else (1.0 - preds[:, 0])
        else:
            cls = preds[:, target_class]

    grads = tape.gradient(cls, conv_feat)                 # [1,Hc,Wc,C]
    if grads is None:
        raise RuntimeError("Gradient is None — 레이어 이름을 'relu' 등으로 바꿔보세요.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))       # [C]
    cam = tf.reduce_sum(tf.nn.relu(conv_feat[0] * weights), axis=-1)  # [Hc,Wc]
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    cam_np = cam.numpy().astype(np.float32)

    # 살짝 블러 + 가벼운 대비
    p90 = float(np.percentile(cam_np, 90.0))
    cam_np = np.clip(cam_np / (p90 + 1e-6), 0, 1)
    cam_np = cv2.GaussianBlur(cam_np, (3, 3), 0)

    return cam_np, float(preds.numpy().squeeze())

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.40, 0.75, 0.69, 0.01)
    st.caption("• 낮추면 민감도↑ • 높이면 정상 보호(오탐↓)")

    st.divider()
    st.subheader("Grad-CAM layer")
    st.caption("권장: DenseNet121 내부 마지막 활성화 **relu**")
    # 필요시 conv4/conv5의 concat이나 relu를 선택해 비교할 수 있게 옵션 제공
    # 기본은 'relu' 로 두고, 목록은 로딩 후 채움

    st.divider()
    st.subheader("Lung mask (optional)")
    use_mask = st.checkbox("Apply ellipse lung mask", value=True)
    cy = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap = st.slider("gap", 0.05, 0.20, 0.10, 0.01)

# ----------------------- Main -----------------------
st.title("🩻 Chest X-ray Pneumonia — DenseNet121 + Grad-CAM (Separated)")
st.caption("Colab과 동일한 감으로 동작. 의사용 장비가 아닙니다.")

# 모델 로드
try:
    model_path = ensure_model_file_cached()
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로딩 실패: {e}")
    st.stop()

# DenseNet 내부 레이어 목록(선택 박스용)
base = model.get_layer("densenet121")
all_names = [l.name for l in base.layers]
# 마지막에 쓰기 좋은 후보들(먼저 'relu'가 있으면 그걸 기본값)
candidate_names = [n for n in all_names if ("relu" in n or "concat" in n or "conv5_block" in n)]
default_name = "relu" if "relu" in all_names else (candidate_names[-1] if candidate_names else all_names[-1])
chosen_name = st.sidebar.selectbox("Select CAM target layer", candidate_names or all_names, index=(candidate_names or all_names).index(default_name))

# 업로드
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
                # Grad-CAM (분리형)
                heatmap, p_pneu = gradcam_separated(x_raw_bchw, model, chosen_name, target_class=1)
                label = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"

                if use_mask:
                    mh, mw = heatmap.shape
                    m = ellipse_lung_mask(mh, mw, cy, rx, ry, gap)
                    heatmap = heatmap * m

                cam_img = overlay_heatmap(rgb_uint8, heatmap)

                with col2:
                    st.image(cam_img, caption=f"Grad-CAM ({chosen_name})", use_column_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted", label)
                m2.metric("Prob. PNEUMONIA", f"{p_pneu*100:.2f}%")
                m3.metric("Threshold", f"{thresh:.2f}")

            except Exception as e:
                st.error(f"Grad-CAM 실패: {type(e).__name__} — {e}")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
