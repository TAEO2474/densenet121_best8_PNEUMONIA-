# ============================================================
# app.py — DenseNet121_BinaryClassifier 전용, 확실하게 동작하는 분리형 Grad-CAM
# (퍼짐 최소화: Top-K 채널, conv5×conv4^γ, Percentile clip, Top-area, Lung mask)
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

# ----------------------- 분리형 Grad-CAM 빌더 -----------------------
def build_feature_and_head(model: keras.Model, last_conv_name: str):
    preprocess_layer = model.get_layer("densenet_preprocess")
    base             = model.get_layer("densenet121")

    # DenseNet 내부 서브모델: base.input -> last_conv.output
    last_conv_tensor_inner = base.get_layer(last_conv_name).output
    feat_inner = keras.Model(base.input, last_conv_tensor_inner, name="feat_inner")

    # 새 입력으로 완전 새 경로 구성 (기존 심볼릭 텐서 재사용 금지!)
    cam_input = keras.Input(shape=model.input_shape[1:], name="cam_input")
    z = preprocess_layer(cam_input)
    z = feat_inner(z)
    feature_extractor = keras.Model(cam_input, z, name="feature_extractor")

    # densenet121 이후 헤드 재사용 (GAP/Dropout/Dense)
    head_input = keras.Input(shape=z.shape[1:], name="cam_head_in")
    x = head_input
    passed = False
    for lyr in model.layers:
        if lyr.name == "densenet121":
            passed = True
            continue
        if not passed:
            continue
        x = lyr(x)
    classifier_head = keras.Model(head_input, x, name="classifier_head")
    return feature_extractor, classifier_head

# ----------------------- Grad-CAM (분리형) -----------------------
def gradcam_separated(img_bchw: np.ndarray, model: keras.Model, last_conv_name: str,
                      target_class: int = 1, topk_channels: int = None, grad_relu: bool = True):
    x = tf.convert_to_tensor(img_bchw, dtype=tf.float32)
    feat, head = build_feature_and_head(model, last_conv_name)

    with tf.GradientTape() as tape:
        conv_feat = feat(x, training=False)     # (1,H,W,C)
        tape.watch(conv_feat)
        preds = head(conv_feat, training=False) # (1,num_classes or 1)
        cls = preds[:, 0] if preds.shape[-1] == 1 else preds[:, target_class]

    grads = tape.gradient(cls, conv_feat)       # (1,H,W,C)
    if grad_relu:
        grads = tf.nn.relu(grads)               # 음수 그래디언트 제거 → 퍼짐 감소

    # 채널 가중치 (H,W 평균) → (C,)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Top-K 채널만 사용 (퍼짐 억제)
    if topk_channels is not None:
        k = int(topk_channels)
        k = max(1, min(k, int(weights.shape[0])))
        topk = tf.math.top_k(tf.abs(weights), k=k, sorted=False).indices
        mask = tf.scatter_nd(indices=tf.expand_dims(topk, 1),
                             updates=tf.ones((k,), dtype=weights.dtype),
                             shape=tf.shape(weights))
        weights = weights * mask

    cam = tf.reduce_sum(tf.nn.relu(conv_feat[0] * weights), axis=-1)  # (H,W)
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
    cam_np = cam.numpy().astype(np.float32)
    return cam_np, float(preds.numpy().squeeze())

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    thresh = st.slider("Decision threshold (PNEUMONIA)", 0.40, 0.75, 0.69, 0.01)
    st.caption("• 낮추면 민감도↑ • 높이면 정상 보호(오탐↓)")

    st.divider()
    st.subheader("Grad-CAM layer (DenseNet 내부)")
    st.caption("권장: 마지막 활성화 **conv5_block16_concat** (없으면 relu). 필요 시 conv4/conv5 비교.")

    st.divider()
    st.subheader("CAM refine")
    use_multiscale = st.checkbox("Use multiscale (conv5 × conv4^γ)", value=True)
    fusion_gamma   = st.slider("γ (conv4 exponent)", 0.3, 1.5, 0.7, 0.1)
    topk_channels  = st.slider("Top-K channels (conv5)", 4, 128, 32, 4)
    percentile_cut = st.slider("Percentile clip", 90, 99, 97, 1)
    top_area_pct   = st.slider("Keep top area (%)", 5, 30, 15, 1)

    st.divider()
    st.subheader("Lung mask (optional)")
    use_mask = st.checkbox("Apply ellipse lung mask", value=True)
    cy = st.slider("center y", 0.35, 0.60, 0.48, 0.01)
    rx = st.slider("radius x", 0.15, 0.35, 0.23, 0.01)
    ry = st.slider("radius y", 0.20, 0.45, 0.32, 0.01)
    gap = st.slider("gap", 0.05, 0.20, 0.10, 0.01)

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
cands = [n for n in all_names if ("relu" in n or "concat" in n or "conv5_block" in n or "conv4_block" in n)]
cands = sorted(set(cands), key=lambda s: (("conv4" not in s, "conv5" not in s, "relu" not in s), s))
# conv5 concat을 기본으로 우선
if "conv5_block16_concat" in all_names:
    default_name = "conv5_block16_concat"
elif "relu" in all_names:
    default_name = "relu"
else:
    default_name = (cands[-1] if cands else all_names[-1])

chosen_name = st.sidebar.selectbox("Select CAM target layer", cands or all_names,
                                   index=(cands or all_names).index(default_name))

# 업로드 & 실행
up = st.file_uploader("Upload an X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up:
    pil_img = Image.open(up)
    rgb_uint8, x_raw_bchw = prepare_inputs(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_uint8, caption="Input (224×224)", use_column_width=True)

    # 버튼 실행부 (들여쓰기 주의)
    if st.button("Run Grad-CAM"):
        with st.spinner("Running…"):
            try:
                # 1) conv5 CAM (Top-K 채널만)
                target5 = "conv5_block16_concat" if "conv5_block16_concat" in all_names else chosen_name
                cam5, p_pneu = gradcam_separated(
                    x_raw_bchw, model, target5,
                    target_class=1, topk_channels=topk_channels, grad_relu=True
                )

                # 2) (옵션) conv4 CAM 구해 멀티스케일 융합
                if use_multiscale and any("conv4_block" in n and n.endswith("_concat") for n in all_names):
                    conv4_cands = [n for n in all_names if n.startswith("conv4_block") and n.endswith("_concat")]
                    # block 번호 기준 정렬 (conv4_block1_concat ... conv4_block32_concat)
                    conv4_cands.sort(key=lambda s: int(s.split("_block")[1].split("_")[0]))
                    conv4_last = conv4_cands[-1]
                    cam4, _ = gradcam_separated(
                        x_raw_bchw, model, conv4_last,
                        target_class=1, topk_channels=None, grad_relu=True
                    )
                    # 정규화 & 융합: 의미×위치
                    cam5 = cam5 / (cam5.max() + 1e-6)
                    cam4 = cam4 / (cam4.max() + 1e-6)
                    # ⬇️ 여기 추가(리사이즈로 해상도 맞추기)
                    h, w = cam5.shape
                    cam4 = cv2.resize(cam4, (w, h), interpolation=cv2.INTER_LINEAR)                    
                    heatmap = cam5 * (cam4 ** fusion_gamma)
                    layer_label = f"{target5} × {conv4_last}^{fusion_gamma:.2f}"
                else:
                    heatmap = cam5
                    layer_label = f"{target5}"

                # 3) 퍼짐 억제: 상위 퍼센타일로 클립 (기본 97)
                heatmap = np.clip(heatmap / (np.percentile(heatmap, percentile_cut) + 1e-6), 0, 1)

                # 4) (강력) Top-Area %만 남기기
                th = np.percentile(heatmap, 100 - top_area_pct)
                binary = (heatmap >= max(th, 1e-6)).astype(np.uint8)
                # (원하면 작은 잡음 제거)
                # kernel = np.ones((3,3), np.uint8)
                # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

                heatmap *= binary.astype(np.float32)

                # 5) (옵션) 폐 마스크
                if use_mask:
                    mh, mw = heatmap.shape
                    m = ellipse_lung_mask(mh, mw, cy, rx, ry, gap)
                    heatmap *= m

                # 6) 라벨 & 출력
                label = "PNEUMONIA" if p_pneu >= thresh else "NORMAL"
                cam_img = overlay_heatmap(rgb_uint8, heatmap)

                with col2:
                    st.image(cam_img, caption=f"Grad-CAM ({layer_label})", use_column_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted", label)
                c2.metric("Prob. PNEUMONIA", f"{p_pneu*100:.2f}%")
                c3.metric("Threshold", f"{thresh:.2f}")

            except Exception as e:
                st.error(f"Grad-CAM 실패: {type(e).__name__} — {e}")
else:
    st.info("⬆️ X-ray 이미지를 업로드하세요.")
