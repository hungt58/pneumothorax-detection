
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.factory import load_model_from_checkpoint


CHECKPOINTS = {
    "1": ("U-Net", "checkpoints/e1_unet_best.pt"),
    "2": ("Attention U-Net", "checkpoints/e2_attention_unet_best.pt"),
    "3": (
        "Attention Multi-scale U-Net",
        "checkpoints/e3_attention_multiscale_best.pt",
    ),
    "4": (
        "ResNet18 Multi-task",
        "checkpoints/e4_resnet18_multitask_best.pt",
    ),
}

OUTPUT_DIR = Path("predictions")
IMAGE_SIZE = 256


def main():
    print("\n=== Pneumothorax Prediction ===")
    for key, (name, _) in CHECKPOINTS.items():
        print(f"{key}. {name}")

    choice = input("\nChoose model (1-4): ").strip()

    if choice not in CHECKPOINTS:
        print("Invalid choice. Please choose 1, 2, 3 or 4.")
        return

    model_name, checkpoint_path = CHECKPOINTS[choice]
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    image_path = input("Enter X-ray image path: ").strip().strip("'\"")
    image_path = Path(image_path).expanduser()

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = load_model_from_checkpoint(checkpoint, map_location="cpu")
    model = model.to(device).eval()

    threshold = float(checkpoint.get("threshold", 0.5))

    # Read and preprocess the X-ray image
    image = Image.open(image_path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    input_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    print("Running prediction...")

   
    with torch.inference_mode():
        output = model(input_tensor)

        cls_output = None

        if isinstance(output, dict):
            # Hỗ trợ các tên khóa của cả 4 mô hình
            logits = (
                output.get("segmentation_logits")
                if output.get("segmentation_logits") is not None
                else output.get("segmentation")
            )

            if logits is None:
                logits = output.get("mask")

            if logits is None:
                raise ValueError(
                    f"Cannot find segmentation output. "
                    f"Keys: {list(output.keys())}"
                )

            cls_output = output.get(
                "classification_logits",
                output.get("classification"),
            )

        elif isinstance(output, (tuple, list)):
            logits = output[0]
            cls_output = output[1] if len(output) > 1 else None

        else:
            logits = output

        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = probabilities >= threshold

        cls_probability = None
        if cls_output is not None:
            cls_probability = torch.sigmoid(
                cls_output.reshape(-1)[0]
            ).item()


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"{image_path.stem}_{checkpoint_path.stem}"

    mask_path = OUTPUT_DIR / f"{output_stem}_mask.png"
    result_path = OUTPUT_DIR / f"{output_stem}_result.png"

    original = np.asarray(image, dtype=np.uint8)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Input X-ray")

    axes[1].imshow(probabilities, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Predicted probability")

    axes[2].imshow(original, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(~mask, mask),
        cmap="Reds",
        alpha=0.5,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(f"{model_name} - Mask overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\nPrediction completed!")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Positive pixels: {int(mask.sum())}")
    print(f"Mask saved: {mask_path}")
    print(f"Visualization saved: {result_path}")

    if cls_probability is not None:
        print(f"Classification probability: {cls_probability:.4f}")


if __name__ == "__main__":
    main()
