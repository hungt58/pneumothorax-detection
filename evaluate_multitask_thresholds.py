"""Tune E4 thresholds on validation, or report one frozen pair on test."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from build_dataset import build_dataloaders, build_file_list, split_data
from evaluation import batch_metrics, summarize
from models.factory import load_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--seg-threshold", type=float, help="required for final test")
    p.add_argument("--cls-threshold", type=float, help="required for final test")
    p.add_argument("--min-pred-pixels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output", type=Path)
    return p.parse_args()


@torch.inference_mode()
def main():
    a = parse_args()
    if a.split == "test" and (a.seg_threshold is None or a.cls_threshold is None):
        raise ValueError("final test requires frozen --seg-threshold and --cls-threshold")
    if a.min_pred_pixels < 1:
        raise ValueError("--min-pred-pixels must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != "resnet18_multitask":
        raise ValueError("This evaluator is only for resnet18_multitask")
    model = load_model_from_checkpoint(checkpoint).to(device).eval()
    records = build_file_list(str(a.images), str(a.masks))
    train_records, val_records, test_records = split_data(records)
    _, val_loader, test_loader = build_dataloaders(train_records, val_records, test_records,
                                                   batch_size=a.batch_size, num_workers=a.num_workers,
                                                   balanced_train=True, seed=42)
    loader = val_loader if a.split == "val" else test_loader
    seg_probs, cls_probs, masks = [], [], []
    for batch in loader:
        outputs = model(batch["image"].to(device))
        seg_probs.append(torch.sigmoid(outputs["segmentation_logits"]).cpu())
        cls_probs.append(torch.sigmoid(outputs["classification_logits"]).cpu())
        masks.append(batch["mask"].cpu())
    seg_probs, cls_probs, masks = map(torch.cat, (seg_probs, cls_probs, masks))
    if a.split == "val":
        pairs = ((float(round(float(s), 6)), float(round(float(c), 6)))
                 for s in np.arange(.30, .901, .05) for c in np.arange(.30, .801, .05))
    else:
        pairs = [(a.seg_threshold, a.cls_threshold)]
    rows = []
    for seg_threshold, cls_threshold in pairs:
        pred = ((seg_probs >= seg_threshold) &
                (cls_probs >= cls_threshold).reshape(-1, 1, 1, 1)).float()
        summary = summarize([batch_metrics(pred, masks, a.min_pred_pixels)])
        rows.append({"seg_threshold": seg_threshold, "cls_threshold": cls_threshold,
                     "min_pred_pixels": a.min_pred_pixels, **summary})
    if a.output is None:
        filename = "seg_cls_threshold_tuning.csv" if a.split == "val" else "final_test_metrics.csv"
        a.output = Path("results/e4_resnet18_multitask") / filename
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["selection_score"])
    print("BEST VALIDATION PAIR" if a.split == "val" else "FINAL TEST METRICS")
    for key, value in best.items():
        print(f"{key}: {value}")
    print("Saved:", a.output)


if __name__ == "__main__":
    main()
