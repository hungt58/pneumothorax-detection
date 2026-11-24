"""E2 + E4 cascade: E2 screens (high sensitivity), E4's classifier confirms.

Final mask = E2 mask, kept only if E4 classification probability >= e4_cls_threshold.
Image is positive iff the final mask has >= min_pred_pixels pixels (same rule as E1-E4).

    # 1) tune both thresholds on VALIDATION only (writes the best pair to configs/thresholds.json)
    python evaluate_cascade.py --images ... --masks ... --split val --write-config
    # 2) report TEST once with the frozen pair from the config
    python evaluate_cascade.py --images ... --masks ... --split test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from build_dataset import build_dataloaders, build_file_list, split_data
from evaluation import batch_metrics, segmentation_logits, summarize
from inference import THRESHOLDS_FILE, load_threshold_config
from models.factory import load_model_from_checkpoint

E2_DEFAULT = Path("checkpoints/e2_attention_unet_best.pt")
E4_DEFAULT = Path("checkpoints/e4_resnet18_multitask_best.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--e2-checkpoint", type=Path, default=E2_DEFAULT)
    p.add_argument("--e4-checkpoint", type=Path, default=E4_DEFAULT)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--e2-threshold", type=float, help="test: frozen E2 threshold (default: config)")
    p.add_argument("--e4-cls-threshold", type=float, help="test: frozen E4 threshold (default: config)")
    p.add_argument("--min-pred-pixels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--write-config", action="store_true", help="val: save best pair to configs/thresholds.json")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def cascade_mask(e2_probs: torch.Tensor, e4_cls_probs: torch.Tensor,
                 e2_threshold: float, e4_cls_threshold: float) -> torch.Tensor:
    """[N,1,H,W] E2 probabilities + [N] E4 classifier probabilities -> final binary masks."""
    confirm = (e4_cls_probs.reshape(-1) >= e4_cls_threshold).reshape(-1, 1, 1, 1)
    return ((e2_probs >= e2_threshold) & confirm).float()


def threshold_grid():
    e2 = [round(float(t), 2) for t in np.arange(0.50, 0.951, 0.05)]
    e4 = [round(float(t), 2) for t in np.arange(0.05, 0.801, 0.05)]
    return [(a, b) for a in e2 for b in e4]


def _loader(records_split, image_size, a):
    train_records, val_records, test_records = records_split
    _, val_loader, test_loader = build_dataloaders(
        train_records, val_records, test_records, batch_size=a.batch_size,
        num_workers=a.num_workers, balanced_train=True, seed=42, image_size=image_size)
    return val_loader if a.split == "val" else test_loader


@torch.inference_mode()
def collect(a, device):
    e2_ckpt = torch.load(a.e2_checkpoint, map_location=device, weights_only=False)
    e4_ckpt = torch.load(a.e4_checkpoint, map_location=device, weights_only=False)
    if e4_ckpt.get("model_name") != "resnet18_multitask":
        raise ValueError("--e4-checkpoint must be the multi-task E4 (it needs the classifier)")
    e2 = load_model_from_checkpoint(e2_ckpt).to(device).eval()
    e4 = load_model_from_checkpoint(e4_ckpt).to(device).eval()

    records = build_file_list(str(a.images), str(a.masks))
    split = split_data(records)
    size2, size4 = int(e2_ckpt.get("image_size", 256)), int(e4_ckpt.get("image_size", 256))
    loader2 = _loader(split, size2, a)
    loader4 = loader2 if size4 == size2 else _loader(split, size4, a)

    e2_probs, e4_cls, masks = [], [], []
    for step, (b2, b4) in enumerate(zip(loader2, loader4, strict=True), 1):
        if list(b2["image_id"]) != list(b4["image_id"]):
            raise RuntimeError("E2/E4 loaders are out of order")
        e2_probs.append(torch.sigmoid(segmentation_logits(e2(b2["image"].to(device)))).half().cpu())
        e4_cls.append(torch.sigmoid(e4(b4["image"].to(device))["classification_logits"]).reshape(-1).cpu())
        masks.append(b2["mask"].to(torch.uint8))
        if step == 1 or step % 50 == 0:
            print(f"  inference batch {step:04d}")
    return torch.cat(e2_probs), torch.cat(e4_cls), torch.cat(masks)


def evaluate_pair(e2_probs, e4_cls, masks, e2_thr, e4_thr, min_px, chunk=256):
    rows = []
    for start in range(0, len(masks), chunk):
        sl = slice(start, start + chunk)
        pred = cascade_mask(e2_probs[sl].float(), e4_cls[sl], e2_thr, e4_thr)
        rows.append(batch_metrics(pred, masks[sl].float(), min_px))
    return summarize(rows)


def write_config(best: dict, path: Path = THRESHOLDS_FILE):
    config = load_threshold_config(path)
    config["cascade_e2_e4"] = {
        "e2_seg_threshold": best["e2_threshold"],
        "e4_cls_threshold": best["e4_cls_threshold"],
        "min_pred_pixels": best["min_pred_pixels"],
        "tuned": True,
        "_note": "Selected on VALIDATION by evaluate_cascade.py (max selection_score).",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated", path)


def main():
    a = parse_args()
    if a.min_pred_pixels < 1:
        raise ValueError("--min-pred-pixels must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| split:", a.split.upper())

    if a.split == "val":
        pairs = threshold_grid()
    else:
        cfg = load_threshold_config().get("cascade_e2_e4", {})
        e2_thr = a.e2_threshold if a.e2_threshold is not None else cfg.get("e2_seg_threshold")
        e4_thr = a.e4_cls_threshold if a.e4_cls_threshold is not None else cfg.get("e4_cls_threshold")
        if e2_thr is None or e4_thr is None:
            raise ValueError("test needs frozen thresholds: run --split val --write-config first")
        if not cfg.get("tuned") and a.e2_threshold is None:
            print("WARNING: config pair is provisional (tuned=false); tune on val first.")
        pairs = [(float(e2_thr), float(e4_thr))]

    e2_probs, e4_cls, masks = collect(a, device)
    print(f"Images: {len(masks)} | positives: {int((masks.flatten(1).sum(1) > 0).sum())}")

    rows = []
    for e2_thr, e4_thr in pairs:
        summary = evaluate_pair(e2_probs, e4_cls, masks, e2_thr, e4_thr, a.min_pred_pixels)
        rows.append({"e2_threshold": e2_thr, "e4_cls_threshold": e4_thr,
                     "min_pred_pixels": a.min_pred_pixels, **summary})

    # Reference: E2 alone at the same E2 threshold (E4 confirm disabled).
    best = max(rows, key=lambda r: r["selection_score"])
    e2_alone = evaluate_pair(e2_probs, e4_cls, masks, best["e2_threshold"], 0.0, a.min_pred_pixels)

    if a.output is None:
        a.output = Path("results/cascade_e2_e4") / (
            "threshold_tuning.csv" if a.split == "val" else "final_test_metrics.csv")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nBEST VALIDATION PAIR" if a.split == "val" else "\nFINAL TEST METRICS (cascade)")
    for key in ("e2_threshold", "e4_cls_threshold", "dice_positive", "iou_positive",
                "image_sensitivity", "image_specificity", "selection_score", "tp", "fn", "fp", "tn"):
        value = best[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nE2 alone @ {best['e2_threshold']:.2f}: sens {e2_alone['image_sensitivity']:.4f} | "
          f"spec {e2_alone['image_specificity']:.4f} | Dice+ {e2_alone['dice_positive']:.4f} | "
          f"FP {e2_alone['fp']}")
    print("Saved:", a.output)
    if a.split == "val" and a.write_config:
        write_config(best)


if __name__ == "__main__":
    main()
