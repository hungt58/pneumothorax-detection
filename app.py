
from pathlib import Path

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.factory import load_model_from_checkpoint


# =========================
# Configuration
# =========================

CHECKPOINTS = {
    "U-Net": "checkpoints/e1_unet_best.pt",
    "Attention U-Net": "checkpoints/e2_attention_unet_best.pt",
    "Attention Multi-scale U-Net":
        "checkpoints/e3_attention_multiscale_best.pt",
    "ResNet18 Multi-task":
        "checkpoints/e4_resnet18_multitask_best.pt",
}

IMAGE_SIZE = 256


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# Load model
# =========================

@st.cache_resource
def load_model(model_name):
    checkpoint_path = Path(CHECKPOINTS[model_name])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = load_model_from_checkpoint(
        checkpoint,
        map_location="cpu",
    )

    device = get_device()
    model = model.to(device).eval()

    threshold = float(checkpoint.get("threshold", 0.5))

    return model, device, threshold


# =========================
# Prediction
# =========================

def predict(model, image, device, threshold):
    # Keep the same preprocessing as the original script
    image = image.convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.asarray(
        image, dtype=np.float32
    ) / 255.0

    input_tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

        cls_output = None

        if isinstance(output, dict):
            logits = output.get("segmentation_logits")

            if logits is None:
                logits = output.get("segmentation")

            if logits is None:
                logits = output.get("mask")

            if logits is None:
                raise ValueError(
                    "Cannot find segmentation output. "
                    f"Keys: {list(output.keys())}"
                )

            cls_output = output.get("classification_logits")

            if cls_output is None:
                cls_output = output.get("classification")

        elif isinstance(output, (tuple, list)):
            logits = output[0]
            cls_output = output[1] if len(output) > 1 else None

        else:
            logits = output

        probabilities = torch.sigmoid(
            logits
        )[0, 0].detach().cpu().numpy()

        mask = probabilities >= threshold

        cls_probability = None

        if cls_output is not None:
            cls_probability = torch.sigmoid(
                cls_output.reshape(-1)[0]
            ).item()

    return (
        np.asarray(image, dtype=np.uint8),
        probabilities,
        mask,
        cls_probability,
    )


# =========================
# Web interface
# =========================

st.set_page_config(
    page_title="Pneumothorax AI",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 Pneumothorax AI Prediction")

st.write(
    "Upload a chest X-ray, select a trained model, "
    "and view its segmentation prediction."
)

with st.sidebar:
    st.header("Model settings")

    model_name = st.selectbox(
        "Select model",
        list(CHECKPOINTS.keys()),
    )

    st.caption(
        "Input size: 256 × 256 pixels"
    )

    st.caption(
        "The model uses its saved threshold, "
        "or 0.5 if no threshold is saved."
    )

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Input image")
    st.image(
        image,
        caption=uploaded_file.name,
        width=400,
    )

    if st.button("Predict", type="primary"):
        try:
            with st.spinner("Loading model and predicting..."):
                model, device, threshold = load_model(
                    model_name
                )

                (
                    original,
                    probabilities,
                    mask,
                    cls_probability,
                ) = predict(
                    model,
                    image,
                    device,
                    threshold,
                )

            st.success("Prediction completed!")

            st.write(f"**Model:** {model_name}")
            st.write(f"**Device:** {device}")
            st.write(f"**Threshold:** {threshold:.4f}")
            st.write(
                f"**Positive pixels:** {int(mask.sum())}"
            )

            # Display the three outputs
            fig, axes = plt.subplots(
                1, 3, figsize=(15, 5)
            )

            axes[0].imshow(original, cmap="gray")
            axes[0].set_title("Input X-ray")

            axes[1].imshow(
                probabilities,
                cmap="jet",
                vmin=0,
                vmax=1,
            )
            axes[1].set_title("Predicted probability")

            axes[2].imshow(original, cmap="gray")
            axes[2].imshow(
                np.ma.masked_where(~mask, mask),
                cmap="Reds",
                alpha=0.5,
                vmin=0,
                vmax=1,
            )
            axes[2].set_title("Segmentation overlay")

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Download prediction mask
            mask_image = Image.fromarray(
                mask.astype(np.uint8) * 255
            )

            import io

            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format="PNG")

            st.download_button(
                "Download predicted mask",
                data=mask_buffer.getvalue(),
                file_name="predicted_mask.png",
                mime="image/png",
            )

            if cls_probability is not None:
                st.metric(
                    "Classification probability",
                    f"{cls_probability:.2%}",
                )

            st.warning(
                "Research demonstration only. "
                "Predictions are not a medical diagnosis."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
