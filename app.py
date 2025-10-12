import os
import io
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# ================== Config ==================
st.set_page_config(page_title="CXR Classifier + Grad-CAM", layout="wide")

MODEL_PATH = "./models/densenet121_best_9.keras"
# DenseNet121 마지막 큰 피처맵(Grad-CAM에 적합)
LAST_CONV_LAYER_NAME = "conv5_block16_concat"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

# ================== Utils ==================
def load_image(file) -> np.ndarray:
    """Load file -> RGB numpy (H, W, 3) uint8."""
    pil = Image.open(file).convert("RGB")
    pil = pil.resize(IMG_SIZE)
    return np.array(pil)

def overlay_heatmap_on_image(img_uint8, heatmap, alpha=0.40):
    """img_uint8: (H,W,3) uint8, heatmap: (H,W) float [0,1]"""
    try:
        import cv2  # use OpenCV if available
        hm = (heatmap * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)      # (H,W,3) BGR uint8
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)          # to RGB
        out = (hm * alpha + img_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)
        return out
    except Exception:
        # Fallback without OpenCV (PIL)
        from PIL import ImageOps
        h = (heatmap * 255).astype(np.uint8)
        hm = Image.fromarray(h, mode="L").resize(img_uint8.shape[:2][::-1], Image.BILINEAR)
        hm = ImageOps.colorize(hm, black="blue", white="red")  # simple colormap
        hm = np.array(hm)
        out = (hm * alpha + img_uint8 * (1 - alpha)).clip(0, 255).astype(np.uint8)
        return out

# ================== Model ==================
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": densenet_preprocess},  # Lambda 포함 모델 복원용
            safe_mode=False
        )
        # Grad-CAM용 서브모델 (conv feature, 최종 출력 동시 반환)
        last_conv = model.get_layer("densenet121").get_layer(LAST_CONV_LAYER_NAME)
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )
        return model, grad_model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Check MODEL_PATH and make sure the .keras file exists in your repo.")
        return None, None

model, grad_model = load_model()

# ================== Grad-CAM ==================
def gradcam(img_batch):
    """
    img_batch: float32 (1, H, W, 3), **no manual preprocess** (model has Lambda)
    returns: heatmap (H, W) float [0,1], pred_prob float
    """
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        prob = preds[:, 0]  # sigmoid output for class=1 (PNEUMONIA)
        # target: class 1 (PNEUMONIA). If you want "predicted class", remove [:,0] selection.
    grads = tape.gradient(prob, conv_out)  # d(prob)/d(conv)
    if grads is None:
        return None, float(prob.numpy()[0])

    # Global-average-pool the gradients over H,W
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (1, H, W, C) -> (1,1,1,C)
    # Weight conv feature maps
    cam = tf.reduce_sum(pooled_grads * conv_out, axis=-1)  # (1,H,W)
    cam = cam[0]

    # ReLU + normalize to [0,1]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(prob.numpy()[0])

# ================== UI ==================
st.title("Chest X-ray Classifier (DenseNet121) + Grad-CAM")

left, right = st.columns([2, 1])
with right:
    thr = st.slider("Decision Threshold (PNEUMONIA)", min_value=0.50, max_value=0.69, value=0.50, step=0.01)
    alpha = st.slider("Heatmap Alpha", 0.1, 0.8, 0.40, 0.05)

st.markdown("---")
file = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if model is None or grad_model is None:
    st.stop()

if file is not None:
    img_uint8 = load_image(file)                 # (224,224,3) uint8
    img_batch = img_uint8.astype("float32")[None, ...]  # (1,224,224,3)

    run = st.button("Run Inference")
    if run:
        with st.spinner("Running inference and Grad-CAM..."):
            # model includes preprocess Lambda; feed raw float32 0..255
            heatmap, prob_pneumonia = gradcam(img_batch)

            # decision
            y_pred = 1 if prob_pneumonia >= thr else 0
            cls = CLASS_NAMES[y_pred]
            prob_show = prob_pneumonia if y_pred == 1 else (1 - prob_pneumonia)

            # viz
            if heatmap is not None:
                # resize heatmap to image size
                heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE, method="bilinear").numpy().squeeze()
                cam_img = overlay_heatmap_on_image(img_uint8, heatmap, alpha=alpha)
            else:
                cam_img = img_uint8

        # ===== Show results =====
        color_box = st.error if y_pred == 1 else st.success
        color_box(f"**Predicted: {cls}**  |  **Confidence:** {prob_show*100:.2f}%  |  **Threshold:** {thr:.2f}")
        st.caption("Class mapping: 0 = NORMAL, 1 = PNEUMONIA (probability is PNEUMONIA).")

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_uint8, caption="Input Image", use_column_width=True)
        with c2:
            st.image(cam_img, caption="Grad-CAM Overlay", use_column_width=True)

        st.markdown("**Raw probabilities**")
        st.json({"PNEUMONIA (class=1)": round(float(prob_pneumonia), 4),
                 "NORMAL (class=0)": round(float(1 - prob_pneumonia), 4)})

else:
    st.info("Please upload an image to begin.")
