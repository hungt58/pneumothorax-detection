"""Kaggle-style post-processing: tune on VAL, score TEST once (optionally an ensemble).

Kaggle SIIM-ACR metric = mean Dice over ALL images, where an empty prediction on an
empty mask scores 1 and any prediction on an empty mask scores 0. Because ~78% of
images are normal, false positives cost a lot, so the winning recipe is a triplet of
thresholds tuned on validation:
  1. pixel threshold on the segmentation probability,
  2. image gate: classification probability (multi-task models) or max pixel prob,
  3. minimum predicted area - smaller masks are removed (image called normal).
Horizontal-flip test-time augmentation (TTA) is averaged in. Several checkpoints can be
ensembled (probabilities averaged at the largest input size).

Example:
  python tune_kaggle.py --images ... --masks ... --name e5_resnet34_multitask_512 \
      --checkpoints checkpoints/e5_resnet34_multitask_512_best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_dataset import build_dataloaders, build_file_list, split_data
from models.factory import load_model_from_checkpoint

SEG_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.801, 0.05)]
GATE_THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
AREAS_AT_1024 = [0, 256, 512, 1024, 2048, 3072, 4096, 6144]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--name", required=True, help="key in configs/thresholds.json and results/<name>/")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--config", type=Path, default=Path("configs/thresholds.json"))
    return p.parse_args()


@torch.inference_mode()
def predict(model, loader, device, size, tta):
    """Return seg probs [N,1,size,size] float16, gate probs [N] (cls or None), masks (if native size)."""
    segs, clss, masks = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        seg_sum, cls_sum = 0, None
        for k, v in enumerate(views):
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(v)
            logits = out["segmentation_logits"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits.float())
            if k == 1:
                prob = torch.flip(prob, dims=[3])
            seg_sum = seg_sum + prob
            if isinstance(out, dict):
                c = torch.sigmoid(out["classification_logits"].float()).flatten()
                cls_sum = c if cls_sum is None else cls_sum + c
        prob = seg_sum / len(views)
        if prob.shape[-1] != size:
            prob = F.interpolate(prob, size=(size, size), mode="bilinear", align_corners=False)
        segs.append(prob.half().cpu())
        if cls_sum is not None:
            clss.append((cls_sum / len(views)).cpu())
        masks.append((batch["mask"] > 0.5).cpu())
    return torch.cat(segs), (torch.cat(clss) if clss else None), torch.cat(masks)


def run_split(models, records, args, device, split):
    size = max(m["size"] for m in models)
    seg_sum, gates, masks = None, [], None
    for m in models:
        loaders = build_dataloaders(records[0], records[1], records[2], batch_size=args.batch_size,
                                    num_workers=args.num_workers, balanced_train=True, seed=42,
                                    image_size=m["size"])
        loader = loaders[1] if split == "val" else loaders[2]
        seg, cls, msk = predict(m["model"], loader, device, size, not args.no_tta)
        seg_sum = seg.float() if seg_sum is None else seg_sum + seg.float()
        if cls is not None:
            gates.append(cls)
        if m["size"] == size and masks is None:
            masks = msk
        print(f"  [{split}] {m['path'].name}: {seg.shape[0]} images @ {m['size']} -> {size}", flush=True)
    seg = seg_sum / len(models)
    gate = torch.stack(gates).mean(0) if gates else seg.flatten(1).max(1).values
    return seg, gate.numpy(), masks, size, bool(gates)


@torch.inference_mode()
def pixel_stats(seg, masks, thresholds, device):
    """Per image and per pixel-threshold: predicted pixels P and intersection I. Shapes [T,N]."""
    n = seg.shape[0]
    P = np.zeros((len(thresholds), n), np.float64)
    I = np.zeros((len(thresholds), n), np.float64)
    for s in range(0, n, 64):
        p = seg[s:s + 64].to(device).float()
        m = masks[s:s + 64].to(device)
        for t, thr in enumerate(thresholds):
            b = p >= thr
            P[t, s:s + 64] = b.flatten(1).sum(1).cpu().numpy()
            I[t, s:s + 64] = (b & m).flatten(1).sum(1).cpu().numpy()
    T = masks.flatten(1).sum(1).numpy().astype(np.float64)
    return P, I, T


def score(P, I, T, gate, gate_thr, area):
    kept = (gate >= gate_thr) & (P >= max(area, 1))
    dice = np.where(kept, 2 * I / np.maximum(P + T, 1), (T == 0).astype(np.float64))
    pos, neg = T > 0, T == 0
    tp, fn = int((kept & pos).sum()), int((~kept & pos).sum())
    fp, tn = int((kept & neg).sum()), int((~kept & neg).sum())
    return {
        "kaggle_score": float(dice.mean()),
        "dice_positive": float(dice[pos].mean()),
        "image_sensitivity": tp / max(tp + fn, 1),
        "image_specificity": tn / max(tn + fp, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "all_empty_baseline": float(neg.mean()),
    }


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in a.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        models.append({"path": path, "size": int(ck.get("image_size", 256)),
                       "model": load_model_from_checkpoint(ck).to(device).eval()})
    recs = split_data(build_file_list(str(a.images), str(a.masks)))
    out_dir = Path("results") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    seg, gate, masks, size, has_cls = run_split(models, recs, a, device, "val")
    areas = sorted({int(round(x * (size / 1024) ** 2)) for x in AREAS_AT_1024})
    P, I, T = pixel_stats(seg, masks, SEG_THRESHOLDS, device)
    del seg
    rows = []
    for ti, st in enumerate(SEG_THRESHOLDS):
        for gt in GATE_THRESHOLDS:
            for area in areas:
                rows.append({"seg_threshold": st, "gate_threshold": gt, "min_area": area,
                             **score(P[ti], I[ti], T, gate, gt, area)})
    with (out_dir / "kaggle_tuning_val.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["kaggle_score"])
    print("BEST ON VAL:", json.dumps(best), flush=True)

    seg, gate, masks, _, _ = run_split(models, recs, a, device, "test")
    P, I, T = pixel_stats(seg, masks, [best["seg_threshold"]], device)
    test = score(P[0], I[0], T, gate, best["gate_threshold"], best["min_area"])
    print("TEST (scored once):", json.dumps(test), flush=True)

    thresholds = {"seg_threshold": best["seg_threshold"], "min_pred_pixels": max(best["min_area"], 1),
                  ("cls_threshold" if has_cls else "max_prob_threshold"): best["gate_threshold"],
                  "tta_hflip": not a.no_tta, "metric": "kaggle", "image_size": size}
    if len(models) > 1:
        thresholds["ensemble"] = [str(m["path"].name) for m in models]
    summary = {"key": a.name, "image_size": size, "thresholds": thresholds,
               "val": {k: best[k] for k in test}, "test": test}
    (out_dir / "kaggle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cfg = json.loads(a.config.read_text(encoding="utf-8")) if a.config.is_file() else {}
    cfg[a.name] = thresholds
    a.config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved", out_dir / "kaggle_summary.json")


if __name__ == "__main__":
    main()
