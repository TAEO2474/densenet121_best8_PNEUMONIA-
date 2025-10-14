# ============================================================
# app.py — DenseNet121_BinaryClassifier 전용, 안정 동작 분리형 Grad-CAM (Any Layer, Dict-Input)
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
MODEL_LOCAL = MODEL_DIR / "densenet121_best.keras"     # ← 실제 파일명에 맞춰 수정 가능
FILE_ID = st.secrets.get("MODEL_FILE_ID", "")          # gdown File ID (선택)
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
    # Lambda(preprocess_input) 복원을 위해 custom_objects 등록
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

# ----------------------- Grad-CAM (원본 그래프 사용, Any Layer) -----------------------
def gradcam_from_any_layer(img_bchw: np.ndarray, model: keras.Model, target_layer_name: str, target_class: int = 1):
    """
    모델 전체 그래프를 그대로 사용해:
      - target_layer_name의 feature map
      - 최종 출력(model.output)
    을 한 번의 순전파로 받아 Grad-CAM을 계산한다.
    입력은 항상 dict({input_name: tensor})로 전달해 KeyError 방지.
    """
    # 입력 이름 확보 (이 모델은 dict 입력을 요구할 수 있음)
    input_name = model.inputs[0].name.split(":")[0]

    # DenseNet 서브모델에서 타깃 텐서 찾기
    base = model.get_layer("densenet121")
    try:
        target_tensor = base.get_layer(target_layer_name).output
    except Exception as e:
        # 타깃이 없으면 마지막 활성화(relu)로 폴백
        fallback_name = "relu" if "relu" in [l.name for l in base.layers] else base.layers[-1].name
        target_tensor = base.get_layer(fallback_name).output
        target_layer_name = fallback_name

    cam_model = keras.Model(inputs=model.input, outputs=[target_tensor, model.output])

    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_feat, preds = cam_model({input_name: x}, training=False)
        if preds.shape[-1] == 1:
            cls_score = preds[:, 0]
        else:
            cls_score = preds[:, target_class]

    grads = tape.gradient(cls_score, conv_feat)
    # 드물게 grads가 None이면 미세한 수치 이슈 → 작은 노이즈 더해 재시도
    if grads is None:
        conv_feat += tf.random.normal(tf.shape(conv_feat), stddev=1e-8)
        with tf.GradientTape() as tape2:
            preds2 = cam_model({input_name: x}, training=False)[1]
            cls_score2 = preds2[:, 0] if preds2.shape[-1] == 1 else preds2[:, target_class]
        grads = tape2.gradient(cls_score2, conv_feat)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # 채널별 α_k
    cam = tf.reduce_sum(tf.nn.relu(conv_feat[0] * weights), axis=-1)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    return cam.numpy().astype(np.float32), float(preds.numpy().squeeze()), target_layer_name

def predict_prob(model, bchw_raw):
    input_name = model.inputs[0].name.split(":")[0]
    prob = model({input_name: bchw_raw}, training=False)
    if isinstance(prob, (list, tuple)):
        prob = prob[0]
    return float(np.asarray(prob).squeeze())

# ----------------------- 유틸: concat 레이어 자동 선택 -----------------------
def _sorted_concats(base_layer_names):
    concats = [n for n in base_layer_names if n.endswith("_concat") and "block" in n]
    def depth_key(n):
        m = re.search(r"conv(\d+)_block(\d+)_concat", n)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return sorted(concats, key=depth_key)

def pick_deep_and_prev(base_layer_names):
    concats = _sorted_concats(base_layer_names)
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

# 모델 로드
try:
    model_path = ensure_model_file_cached()
    model = load_model(model_path)
except Exception as e:
    st.error(f"모델 로딩 실패: {e}")
    st.stop()

# DenseNet 내부 레이어 목록 & 기본값
base = model.get_layer("densenet121")
all_names = [l.name for l in base.layers]
# candidate: relu + concat 계열 위주
cands = [n for n in all_names if ("relu" in n or ("_concat" in n and "block" in n))]
cands = sorted(set(cands), key=lambda s: (("relu" not in s), s))
default_name = "relu" if "relu" in all_names else (cands[-1] if cands else all_names[-1])
chosen_name = st.sidebar.selectbox("Select CAM target layer", cands or all_names, index=(cands or all_names).index(default_name))

# 업로드 & 실행
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
                # 0) 가장 깊은 concat과 그 직전 concat 자동 선택
                deep_name, prev_name = pick_deep_and_prev(all_names)

                # 1) CAM 계산 (멀티스케일 or 단일)
                if use_multiscale and deep_name is not None and prev_name is not None:
                    cam_deep, p_pneu, deep_used = gradcam_from_any_layer(x_raw_bchw, model, deep_name, target_class=1)
                    cam_prev, _p2, prev_used   = gradcam_from_any_layer(x_raw_bchw, model, prev_name, target_class=1)

                    # 정규화
                    cam_deep = cam_deep / (cam_deep.max() + 1e-6)
                    cam_prev = cam_prev / (cam_prev.max() + 1e-6)

                    # 융합: 깊은층 × (앞층^γ)
                    heatmap = cam_deep * (cam_prev ** fusion_gamma)
                    layer_label = f"{deep_used} × {prev_used}^{fusion_gamma:.2f}"
                    p_show = p_pneu
                else:
                    heatmap, p_pneu, used = gradcam_from_any_layer(x_raw_bchw, model, chosen_name, target_class=1)
                    layer_label = f"{used}"
                    p_show = p_pneu

                # 2) 퍼짐 억제: Percentile clip
                heatmap = np.clip(heatmap / (np.percentile(heatmap, cam_percentile) + 1e-6), 0, 1)

                # 3) (옵션) 블러
                if cam_blur:
                    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (3, 3), 0)

                # 4) (옵션) 폐 마스크 적용
                if use_mask:
                    mh, mw = heatmap.shape
                    m = ellipse_lung_mask(mh, mw, cy, rx, ry, gap)
                    heatmap = heatmap * m

                # 5) 분류 라벨 결정
                label = "PNEUMONIA" if p_show >= thresh else "NORMAL"

                # 6) 오버레이 렌더
                cam_img = overlay_heatmap(rgb_uint8, heatmap)

                with col2:
                    st.image(cam_img, caption=f"Grad-CAM ({layer_label})", use_column_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", label)
                c2.metric("Prob. PNEUMONIA", f"{p_show*100:.2f}%")
                c3.metric("Threshold", f"{thresh:.2f}")

            except Exception as e:
                st.error(f"Grad-CAM 실패: {type(e).__name__} — {e}")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
