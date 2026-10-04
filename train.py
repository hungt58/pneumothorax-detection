"""Train E1-E4 with the fixed balanced-sampling protocol."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from build_dataset import build_dataloaders, build_file_list, split_data
from losses import MultiTaskLoss, SegmentationLoss
from models.factory import MODEL_NAMES, build_model
from evaluation import final_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--model", choices=MODEL_NAMES, default="unet")
    p.add_argument("--no-pretrained", action="store_true", help="E4 smoke tests without downloading weights")
    p.add_argument("--cls-threshold", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-pred-pixels", type=int, default=64)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--history", type=Path)
    p.add_argument("--patience", type=int, default=7)
    return p.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches, epoch):
    model.train()
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    batches = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks, batch["label"].to(device)) if isinstance(outputs, dict) else criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        if step == 1 or step % 50 == 0:
            positives = int(batch["label"].sum().item())
            print(f"  batch {step:04d} | loss {loss.item():.4f} | positives {positives}/{images.shape[0]}")
        if max_batches is not None and step >= max_batches:
            break
    return total_loss / max(batches, 1)


@torch.inference_mode()
def validate(model, loader, criterion, device, threshold, min_pred_pixels, max_batches, cls_threshold=0.5):
    model.eval()
    total_loss = 0.0
    batch_count = sample_count = 0
    positive_count = negative_count = 0
    positive_dice_sum = positive_iou_sum = 0.0
    tp_img = fp_img = tn_img = fn_img = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        total_loss += (criterion(outputs, masks, batch["label"].to(device)) if isinstance(outputs, dict)
                       else criterion(outputs, masks)).item()
        batch_count += 1
        sample_count += masks.shape[0]

        pred = final_mask(outputs, threshold, cls_threshold)
        pred_pixels = pred.flatten(1).sum(1)
        target_pixels = masks.flatten(1).sum(1)
        target_positive = target_pixels > 0
        target_negative = ~target_positive
        predicted_positive = pred_pixels >= min_pred_pixels

        intersection = (pred * masks).flatten(1).sum(1)
        denom = pred_pixels + target_pixels
        dice = (2 * intersection + 1.0) / (denom + 1.0)
        union = pred_pixels + target_pixels - intersection
        iou = (intersection + 1.0) / (union + 1.0)

        if target_positive.any():
            positive_dice_sum += dice[target_positive].sum().item()
            positive_iou_sum += iou[target_positive].sum().item()
            positive_count += int(target_positive.sum().item())
        negative_count += int(target_negative.sum().item())

        tp_img += int((predicted_positive & target_positive).sum().item())
        fn_img += int((~predicted_positive & target_positive).sum().item())
        fp_img += int((predicted_positive & target_negative).sum().item())
        tn_img += int((~predicted_positive & target_negative).sum().item())

        if max_batches is not None and step >= max_batches:
            break

    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("validation requires both positive and negative samples")

    dice_positive = positive_dice_sum / positive_count
    iou_positive = positive_iou_sum / positive_count
    sensitivity = tp_img / max(tp_img + fn_img, 1)
    specificity = tn_img / max(tn_img + fp_img, 1)
    fp_rate = fp_img / max(fp_img + tn_img, 1)
    pred_positive_rate = (tp_img + fp_img) / max(sample_count, 1)
    # Parameter-free balance: a model cannot score well by segmenting positives
    # while marking every negative image as positive (or vice versa).
    selection_score = (
        2.0 * dice_positive * specificity / (dice_positive + specificity + 1e-8)
    )

    return {
        "val_loss": total_loss / max(batch_count, 1),
        "val_dice_positive": dice_positive,
        "val_iou_positive": iou_positive,
        "image_sensitivity": sensitivity,
        "image_specificity": specificity,
        "false_positive_rate": fp_rate,
        "predicted_positive_rate": pred_positive_rate,
        "selection_score": selection_score,
    }


HISTORY_COLUMNS = [
    "epoch", "train_loss", "val_loss", "val_dice_positive", "val_iou_positive",
    "image_sensitivity", "image_specificity", "false_positive_rate",
    "predicted_positive_rate", "selection_score", "learning_rate",
]


def initialize_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HISTORY_COLUMNS)


def append_history(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row[c] for c in HISTORY_COLUMNS])


def main() -> None:
    args = parse_args()
    if args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be even for 50/50 balanced batches")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0,1)")
    if not args.images.is_dir() or not args.masks.is_dir():
        raise FileNotFoundError("--images and --masks must point to existing folders")

    experiment = {"unet": "e1_unet", "attention_unet": "e2_attention_unet",
                  "attention_multiscale_unet": "e3_attention_multiscale",
                  "resnet18_multitask": "e4_resnet18_multitask"}[args.model]
    if args.checkpoint is None:
        args.checkpoint = Path("checkpoints/unet_balanced_best.pt") if args.model == "unet" else Path(f"checkpoints/{experiment}_best.pt")
    if args.history is None:
        args.history = Path("results/unet/history_balanced.csv") if args.model == "unet" else Path(f"results/{experiment}/history.csv")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = build_file_list(str(args.images), str(args.masks))
    train_records, val_records, test_records = split_data(records)
    train_loader, val_loader, _ = build_dataloaders(
        train_records, val_records, test_records,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_train=True,
        seed=42,
    )

    train_pos = sum(r["HasDisease"] for r in train_records)
    print(
        f"Pairs: {len(records)} | train: {len(train_records)} "
        f"(positive={train_pos}, negative={len(train_records)-train_pos}) | "
        f"val: {len(val_records)} | test: {len(test_records)}"
    )
    print(f"Balanced TRAIN batches: {args.batch_size//2} positive + {args.batch_size//2} negative")
    loss_description = "BCEWithLogits(all images) + PositiveDice(positive masks only)"
    if args.model == "resnet18_multitask":
        loss_description += " + BCEWithLogits(image classification)"
    print(f"Loss: {loss_description}")

    model_kwargs = ({"in_channels": 1, "out_channels": 1, "pretrained": not args.no_pretrained}
                    if args.model == "resnet18_multitask" else
                    {"in_channels": 1, "out_channels": 1, "features": (32, 64, 128, 256)})
    model = build_model(args.model, **model_kwargs).to(device)
    criterion = (MultiTaskLoss() if args.model == "resnet18_multitask" else
                 SegmentationLoss(bce_weight=1.0, dice_weight=1.0)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    best_score = float("-inf")
    epochs_without_improvement = 0
    initialize_history(args.history)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.max_train_batches, epoch,
        )
        metrics = validate(
            model, val_loader, criterion, device,
            args.threshold, args.min_pred_pixels, args.max_val_batches, args.cls_threshold,
        )
        scheduler.step(metrics["selection_score"])
        lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "train_loss": train_loss, **metrics, "learning_rate": lr}
        append_history(args.history, row)
        print(
            f"  train_loss {train_loss:.4f} | val_loss {metrics['val_loss']:.4f} | "
            f"Dice+ {metrics['val_dice_positive']:.4f} | IoU+ {metrics['val_iou_positive']:.4f} | "
            f"sens {metrics['image_sensitivity']:.4f} | spec {metrics['image_specificity']:.4f} | "
            f"FPR {metrics['false_positive_rate']:.4f} | pred+ {metrics['predicted_positive_rate']:.4f} | "
            f"score {metrics['selection_score']:.4f} | lr {lr:.2e}"
        )

        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            epochs_without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "features": (32, 64, 128, 256),
                    "threshold": args.threshold,
                    "cls_threshold": args.cls_threshold if args.model == "resnet18_multitask" else None,
                    "min_pred_pixels": args.min_pred_pixels,
                    "balanced_train": True,
                    "loss": "BCE + positive-only Dice",
                    **metrics,
                },
                args.checkpoint,
            )
            print(f"  Saved checkpoint: {args.checkpoint}")
        else:
            epochs_without_improvement += 1
            print(f"  No selection_score improvement for {epochs_without_improvement}/{args.patience} epoch(s)")
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
