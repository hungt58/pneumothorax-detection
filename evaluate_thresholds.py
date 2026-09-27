"""Tune segmentation threshold on VALIDATION only.

Uses the same image-level rule as train.py:
predicted positive iff predicted foreground pixels >= min_pred_pixels.
The TEST split is never used here.
"""
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--min-pred-pixels", type=int, default=64)
    p.add_argument("--threshold-start", type=float, default=0.30)
    p.add_argument("--threshold-end", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
    p.add_argument("--output", type=Path,
                   default=Path("results/unet/threshold_tuning.csv"))
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balance_score(dice_positive, specificity):
    return 2.0 * dice_positive * specificity / (
        dice_positive + specificity + 1e-8
    )


@torch.inference_mode()
def main():
    a = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    records = build_file_list(str(a.images), str(a.masks))
    train_records, val_records, test_records = split_data(records)
    _, val_loader, _ = build_dataloaders(
        train_records, val_records, test_records,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        balanced_train=True,
        seed=42,
    )

    print(
        f"Pairs: {len(records)} | train: {len(train_records)} | "
        f"val: {len(val_records)} | test: {len(test_records)}"
    )
    print("Threshold tuning: VALIDATION ONLY; test remains untouched.")
    print(f"Image positive rule: predicted pixels >= {a.min_pred_pixels}")

    ckpt = torch.load(a.checkpoint, map_location=device)
    features = tuple(ckpt.get("features", (32, 64, 128, 256)))

    model = UNet(features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    probs_all, masks_all = [], []
    for step, batch in enumerate(val_loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        probs_all.append(torch.sigmoid(model(images)).cpu())
        masks_all.append(batch["mask"].float().cpu())
        if step == 1 or step % 50 == 0:
            print(f"  inference batch {step:04d}")

    probs = torch.cat(probs_all)
    masks = torch.cat(masks_all)
    mask_flat = masks.flatten(1)
    target_pixels = mask_flat.sum(1)
    target_positive = target_pixels > 0
    target_negative = ~target_positive

    thresholds = np.arange(
        a.threshold_start,
        a.threshold_end + a.threshold_step * 0.5,
        a.threshold_step,
    )

    rows = []
    print("\nthr   Dice+   IoU+    Sens     Spec     FPR      Pred+    Score")
    print("-" * 76)

    for thr in thresholds:
        thr = float(round(float(thr), 6))
        pred = (probs >= thr).float()
        pred_flat = pred.flatten(1)
        pred_pixels = pred_flat.sum(1)

        intersection = (pred_flat * mask_flat).sum(1)
        union = pred_pixels + target_pixels - intersection
        dice = (2 * intersection + 1.0) / (
            pred_pixels + target_pixels + 1.0
        )
        iou = (intersection + 1.0) / (union + 1.0)

        predicted_positive = pred_pixels >= a.min_pred_pixels
        tp = int((predicted_positive & target_positive).sum().item())
        fn = int((~predicted_positive & target_positive).sum().item())
        fp = int((predicted_positive & target_negative).sum().item())
        tn = int((~predicted_positive & target_negative).sum().item())

        dice_pos = float(dice[target_positive].mean().item())
        iou_pos = float(iou[target_positive].mean().item())
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        fpr = fp / max(fp + tn, 1)
        pred_rate = (tp + fp) / max(len(masks), 1)
        score = balance_score(dice_pos, specificity)

        row = {
            "threshold": thr,
            "dice_positive": dice_pos,
            "iou_positive": iou_pos,
            "image_sensitivity": sensitivity,
            "image_specificity": specificity,
            "false_positive_rate": fpr,
            "predicted_positive_rate": pred_rate,
            "selection_score": score,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }
        rows.append(row)

        print(
            f"{thr:0.2f}  {dice_pos:0.4f}  {iou_pos:0.4f}  "
            f"{sensitivity:0.4f}  {specificity:0.4f}  "
            f"{fpr:0.4f}  {pred_rate:0.4f}  {score:0.4f}"
        )

    best = max(rows, key=lambda r: r["selection_score"])

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nBEST VALIDATION THRESHOLD")
    for key, value in best.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved:", a.output)
    print("Next: visualize this checkpoint at the selected threshold.")


if __name__ == "__main__":
    main()
