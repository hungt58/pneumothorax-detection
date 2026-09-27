"""Visualize U-Net predictions on the validation split only."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from build_dataset import build_dataloaders, build_file_list, split_data
from models import UNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/unet/visualizations"))
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def main():
    a = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    records = build_file_list(str(a.images), str(a.masks))
    train_records, val_records, test_records = split_data(records)
    _, val_loader, _ = build_dataloaders(
        train_records, val_records, test_records,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
    )
    print(
        f"Pairs: {len(records)} | train: {len(train_records)} | "
        f"val: {len(val_records)} | test: {len(test_records)}"
    )
    print("Visualization uses VALIDATION only; test set untouched.")

    ckpt = torch.load(a.checkpoint, map_location=device)
    features = tuple(ckpt.get("features", (32, 64, 128, 256)))
    model = UNet(features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    samples = []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].cpu()
        probs = torch.sigmoid(model(images)).cpu()
        preds = (probs >= a.threshold).float()
        images = images.cpu()

        for i in range(images.shape[0]):
            gt_pos = bool(masks[i].sum() > 0)
            pred_pos = bool(preds[i].sum() > 0)

            if gt_pos and pred_pos:
                category = "TP"
            elif gt_pos and not pred_pos:
                category = "FN"
            elif not gt_pos and pred_pos:
                category = "FP"
            else:
                category = "TN"

            samples.append({
                "image": images[i, 0].numpy(),
                "mask": masks[i, 0].numpy(),
                "prob": probs[i, 0].numpy(),
                "pred": preds[i, 0].numpy(),
                "category": category,
                "pred_pixels": int(preds[i].sum().item()),
                "max_prob": float(probs[i].max().item()),
            })

    counts = {c: sum(s["category"] == c for s in samples)
              for c in ("TP", "FN", "FP", "TN")}
    print("Validation image-level counts at threshold", a.threshold, counts)

    # Balanced selection where possible.
    per_category = max(1, a.num_samples // 4)
    selected = []
    for category in ("TP", "FN", "FP", "TN"):
        group = [s for s in samples if s["category"] == category]
        # For FP show the largest predicted masks first; otherwise deterministic.
        if category == "FP":
            group.sort(key=lambda x: x["pred_pixels"], reverse=True)
        selected.extend(group[:per_category])

    # Fill remaining slots if some categories are scarce.
    if len(selected) < a.num_samples:
        selected_ids = {id(s) for s in selected}
        for s in samples:
            if id(s) not in selected_ids:
                selected.append(s)
                if len(selected) >= a.num_samples:
                    break

    a.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, s in enumerate(selected[:a.num_samples], 1):
        # Dataset normalization is mean=.5/std=.5, so invert for display.
        image = np.clip(s["image"] * 0.5 + 0.5, 0, 1)

        fig = plt.figure(figsize=(15, 3.5))

        ax = fig.add_subplot(1, 5, 1)
        ax.imshow(image, cmap="gray")
        ax.set_title("X-ray")
        ax.axis("off")

        ax = fig.add_subplot(1, 5, 2)
        ax.imshow(s["mask"], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Ground Truth")
        ax.axis("off")

        ax = fig.add_subplot(1, 5, 3)
        im = ax.imshow(s["prob"], cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Probability")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(1, 5, 4)
        ax.imshow(s["pred"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Prediction t={a.threshold:.2f}")
        ax.axis("off")

        ax = fig.add_subplot(1, 5, 5)
        ax.imshow(image, cmap="gray")
        masked = np.ma.masked_where(s["pred"] <= 0, s["pred"])
        ax.imshow(masked, alpha=0.45, cmap="autumn", vmin=0, vmax=1)
        ax.set_title(
            f"{s['category']} | pixels={s['pred_pixels']}\n"
            f"maxP={s['max_prob']:.3f}"
        )
        ax.axis("off")

        fig.tight_layout()
        path = a.output_dir / f"{idx:02d}_{s['category']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {min(len(selected), a.num_samples)} images to: {a.output_dir}")
    print("Send me several TP/FN/FP/TN PNGs, especially FP images.")


if __name__ == "__main__":
    main()
