"""Streamlit web demo for PneumoAI.

Inference matches training/validation exactly (see inference.py):
[-1, 1] normalisation, validation-tuned thresholds from configs/thresholds.json,
E4 classification gate, and the >= 64-pixel image-level rule.
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference import (
    cascade_decision,
    checkpoint_image_size,
    load_threshold_config,
    predict_tensor,
    preprocess,
    resolve_thresholds,
)
from models.factory import experiment_name, load_model_from_checkpoint

CHECKPOINT_DIR = Path("checkpoints")

# display name -> experiment key (= checkpoint stem without "_best.pt")
MODELS = {
    "E1 · U-Net": "e1_unet",
    "E2 · Attention U-Net": "e2_attention_unet",
    "E3 · Attention Multi-scale U-Net": "e3_attention_multiscale",
    "E4 · ResNet18 Multi-task": "e4_resnet18_multitask",
    "E4b · ResNet18 segmentation only": "e4b_resnet18_seg",
}
CASCADE = "E2 + E4 · screen then confirm"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(experiment: str) -> Path:
    return CHECKPOINT_DIR / f"{experiment}_best.pt"


@st.cache_resource
def load_model(experiment: str):
    path = checkpoint_path(experiment)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = get_device()
    model = load_model_from_checkpoint(checkpoint).to(device).eval()
    key = checkpoint.get("experiment") or experiment_name(
        checkpoint.get("model_name", "unet"), checkpoint_image_size(checkpoint))
    meta = {
        "image_size": checkpoint_image_size(checkpoint),
        "thresholds": resolve_thresholds(checkpoint, key),
    }
    return model, device, meta


def overlay_figure(image: np.ndarray, probabilities: np.ndarray, mask: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input X-ray")
    axes[1].imshow(probabilities, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Predicted probability")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(np.ma.masked_where(~mask, mask), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def mask_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()


def show_result(image, probabilities, mask, positive, details: dict):
    if positive:
        st.error(f"**Pneumothorax suspected** — {int(mask.sum())} pixels marked")
    else:
        st.success("**No pneumothorax detected** (fewer than the minimum pixels marked)")
    st.table({k: [v] for k, v in details.items()})
    fig = overlay_figure(image, probabilities, mask, "Final mask")
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Download predicted mask", data=mask_png(mask),
                       file_name="predicted_mask.png", mime="image/png")


st.set_page_config(page_title="PneumoAI", page_icon="🩻", layout="wide")
st.title("🩻 PneumoAI — Pneumothorax detection & segmentation")
st.write("Upload a chest X-ray, choose a model, and see the predicted pneumothorax region.")

with st.sidebar:
    st.header("Model")
    choice = st.selectbox("Select model", [*MODELS.keys(), CASCADE])
    available = {name for name, key in MODELS.items() if checkpoint_path(key).is_file()}
    if choice == CASCADE:
        st.caption("E2 marks suspicious images (high sensitivity); E4's classifier "
                   "must confirm before the E2 mask is shown (higher specificity).")
    elif choice not in available:
        st.warning(f"Missing file: {checkpoint_path(MODELS[choice])}")
    st.caption("Thresholds are the ones tuned on the validation split "
               "(configs/thresholds.json); the test split was never used for tuning.")

uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption=uploaded.name, width=400)

    if st.button("Predict", type="primary"):
        try:
            with st.spinner("Loading model and predicting..."):
                if choice == CASCADE:
                    config = load_threshold_config().get("cascade_e2_e4", {})
                    e2, device, e2_meta = load_model("e2_attention_unet")
                    e4, _, e4_meta = load_model("e4_resnet18_multitask")
                    e2_thr = float(config.get("e2_seg_threshold", e2_meta["thresholds"]["seg_threshold"]))
                    e4_thr = float(config.get("e4_cls_threshold", e4_meta["thresholds"]["cls_threshold"] or 0.5))
                    min_px = int(config.get("min_pred_pixels", 64))

                    shown, tensor = preprocess(image, e2_meta["image_size"])
                    screen = predict_tensor(e2, tensor, e2_thr, None, min_px)
                    _, tensor4 = preprocess(image, e4_meta["image_size"])
                    confirm = predict_tensor(e4, tensor4, 0.5, None, min_px)
                    mask, positive = cascade_decision(screen, confirm.cls_probability, e4_thr)
                    details = {
                        "E2 marked pixels": screen.positive_pixels,
                        "E2 says": "positive" if screen.is_positive else "negative",
                        "E4 classifier probability": f"{confirm.cls_probability:.2%}",
                        "E4 confirm threshold": f"{e4_thr:.2f}"
                        + ("" if config.get("tuned") else " (provisional)"),
                        "E2 threshold": f"{e2_thr:.2f}",
                    }
                    probabilities = screen.probabilities
                else:
                    model, device, meta = load_model(MODELS[choice])
                    thr = meta["thresholds"]
                    shown, tensor = preprocess(image, meta["image_size"])
                    pred = predict_tensor(model, tensor, thr["seg_threshold"],
                                          thr["cls_threshold"], thr["min_pred_pixels"])
                    mask, positive, probabilities = pred.mask, pred.is_positive, pred.probabilities
                    details = {
                        "Segmentation threshold": f"{thr['seg_threshold']:.2f}",
                        "Minimum pixels": thr["min_pred_pixels"],
                        "Marked pixels": pred.positive_pixels,
                        "Input size": f"{meta['image_size']}×{meta['image_size']}",
                        "Threshold source": thr["source"],
                    }
                    if pred.cls_probability is not None:
                        details["Classifier probability"] = f"{pred.cls_probability:.2%}"
                        details["Classifier gate"] = f"{thr['cls_threshold']:.2f}"
            st.caption(f"Device: {device}")
            show_result(shown, probabilities, mask, positive, details)
            st.warning("Research demonstration only. Predictions are not a medical diagnosis.")
        except Exception as exc:  # shown in the page instead of a stack trace
            st.error(f"Prediction failed: {exc}")
