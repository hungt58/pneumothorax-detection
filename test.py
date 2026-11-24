
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from inference import checkpoint_image_size, predict_tensor, preprocess, resolve_thresholds
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
    "5": ("ResNet18 segmentation only (E4b)", "checkpoints/e4b_resnet18_seg_best.pt"),
}

OUTPUT_DIR = Path("predictions")
IMAGE_SIZE = 256


def main():
    print("\n=== Pneumothorax Prediction ===")
    for key, (name, _) in CHECKPOINTS.items():
        print(f"{key}. {name}")

    choice = input("\nChoose model (1-5): ").strip()

    if choice not in CHECKPOINTS:
        print("Invalid choice. Please choose 1-5.")
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

    # Same preprocessing + validation-tuned thresholds as training (inference.py)
    key = checkpoint_path.name.replace("_best.pt", "")
    thr = resolve_thresholds(checkpoint, checkpoint.get("experiment") or key)
    threshold = thr["seg_threshold"]
    image, input_tensor = preprocess(Image.open(image_path), checkpoint_image_size(checkpoint))

    print("Running prediction...")
    pred = predict_tensor(model, input_tensor, threshold, thr["cls_threshold"], thr["min_pred_pixels"])
    probabilities, mask, cls_probability = pred.probabilities, pred.mask, pred.cls_probability

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"{image_path.stem}_{checkpoint_path.stem}"

    mask_path = OUTPUT_DIR / f"{output_stem}_mask.png"
    result_path = OUTPUT_DIR / f"{output_stem}_result.png"

    original = image
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
    print(f"Positive pixels: {int(mask.sum())} (min {thr['min_pred_pixels']})")
    print("Result:", "PNEUMOTHORAX SUSPECTED" if pred.is_positive else "no pneumothorax detected")
    print(f"Mask saved: {mask_path}")
    print(f"Visualization saved: {result_path}")

    if cls_probability is not None:
        print(f"Classification probability: {cls_probability:.4f}")


if __name__ == "__main__":
    main()
