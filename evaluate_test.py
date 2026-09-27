"""Final TEST evaluation with a frozen validation-selected threshold."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from build_dataset import build_dataloaders, build_file_list, split_data
from models import UNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-pred-pixels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path,
                   default=Path("results/unet/final_test_metrics.csv"))
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
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    records = build_file_list(str(a.images), str(a.masks))
    train_records, val_records, test_records = split_data(records)
    _, _, test_loader = build_dataloaders(
        train_records, val_records, test_records,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        balanced_train=True,
        seed=42,
    )

    print(
        f"FINAL TEST | n={len(test_records)} | threshold={a.threshold:.4f} | "
        f"min_pred_pixels={a.min_pred_pixels}"
    )

    ckpt = torch.load(a.checkpoint, map_location=device)
    features = tuple(ckpt.get("features", (32, 64, 128, 256)))
    model = UNet(features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dice_sum = iou_sum = 0.0
    pos_count = 0
    tp = fn = fp = tn = 0

    for step, batch in enumerate(test_loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].float().to(device, non_blocking=True)

        probs = torch.sigmoid(model(images))
        pred = (probs >= a.threshold).float()

        pf = pred.flatten(1)
        mf = masks.flatten(1)
        pred_pixels = pf.sum(1)
        target_pixels = mf.sum(1)

        target_pos = target_pixels > 0
        target_neg = ~target_pos
        predicted_pos = pred_pixels >= a.min_pred_pixels

        inter = (pf * mf).sum(1)
        union = pred_pixels + target_pixels - inter
        dice = (2 * inter + 1.0) / (pred_pixels + target_pixels + 1.0)
        iou = (inter + 1.0) / (union + 1.0)

        if target_pos.any():
            dice_sum += dice[target_pos].sum().item()
            iou_sum += iou[target_pos].sum().item()
            pos_count += int(target_pos.sum().item())

        tp += int((predicted_pos & target_pos).sum().item())
        fn += int((~predicted_pos & target_pos).sum().item())
        fp += int((predicted_pos & target_neg).sum().item())
        tn += int((~predicted_pos & target_neg).sum().item())

        if step == 1 or step % 50 == 0:
            print(f"  test batch {step:04d}")

    metrics = {
        "threshold": a.threshold,
        "min_pred_pixels": a.min_pred_pixels,
        "dice_positive": dice_sum / max(pos_count, 1),
        "iou_positive": iou_sum / max(pos_count, 1),
        "image_sensitivity": tp / max(tp + fn, 1),
        "image_specificity": tn / max(tn + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "predicted_positive_rate": (tp + fp) / max(tp + fn + fp + tn, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }

    print("\nFINAL TEST METRICS")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print("Saved:", a.output)


if __name__ == "__main__":
    main()
