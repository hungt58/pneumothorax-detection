"""Train a baseline U-Net for binary pneumothorax segmentation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.optim import AdamW

from build_dataset import build_dataloaders, build_file_list, split_data
from losses import SegmentationLoss
from models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each epoch early after this many batches (useful for a smoke test).",
    )
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt")
    )
    parser.add_argument("--history", type=Path, default=Path("results/unet/history.csv"))
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches):
    model.train()
    total_loss = 0.0
    batch_count = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        if step == 1 or step % 20 == 0:
            print(f"  batch {step:04d} | loss {loss.item():.4f}")
        if max_batches is not None and step >= max_batches:
            break

    return total_loss / batch_count


@torch.inference_mode()
def validate(model, loader, criterion, device, max_batches):
    model.eval()
    total_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    positive_dice_total = 0.0
    positive_iou_total = 0.0
    positive_sample_count = 0
    predicted_positive_count = 0
    batch_count = 0
    sample_count = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        batch_count += 1
        sample_count += masks.shape[0]

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        intersection = (predictions * masks).flatten(1).sum(1)
        denominator = predictions.flatten(1).sum(1) + masks.flatten(1).sum(1)
        dice_per_sample = (2 * intersection + 1) / (denominator + 1)
        dice_total += dice_per_sample.sum().item()

        union = (
            predictions.flatten(1).sum(1)
            + masks.flatten(1).sum(1)
            - intersection
        )
        iou_per_sample = (intersection + 1) / (union + 1)
        iou_total += iou_per_sample.sum().item()

        # Positive-only segmentation metrics prevent the many empty masks from
        # dominating model selection. A sample is positive when its GT mask
        # contains at least one pneumothorax pixel.
        target_positive = masks.flatten(1).sum(1) > 0
        predicted_positive = predictions.flatten(1).sum(1) > 0
        predicted_positive_count += predicted_positive.sum().item()

        if target_positive.any():
            positive_dice_total += dice_per_sample[target_positive].sum().item()
            positive_iou_total += iou_per_sample[target_positive].sum().item()
            positive_sample_count += target_positive.sum().item()
        if max_batches is not None and step >= max_batches:
            break

    if positive_sample_count == 0:
        raise RuntimeError("Validation set contains no positive masks")

    return (
        total_loss / batch_count,
        dice_total / sample_count,
        iou_total / sample_count,
        positive_dice_total / positive_sample_count,
        positive_iou_total / positive_sample_count,
        predicted_positive_count / sample_count,
    )


def initialize_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch", "train_loss", "val_loss",
                "val_dice_all", "val_iou_all",
                "val_dice_positive", "val_iou_positive",
                "predicted_positive_rate", "learning_rate",
            ]
        )


def append_history(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_dice_all: float,
    val_iou_all: float,
    val_dice_positive: float,
    val_iou_positive: float,
    predicted_positive_rate: float,
    learning_rate: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                epoch, train_loss, val_loss,
                val_dice_all, val_iou_all,
                val_dice_positive, val_iou_positive,
                predicted_positive_rate, learning_rate,
            ]
        )



def calculate_pos_weight(train_records) -> tuple[float, int, int]:
    """Compute pixel imbalance from TRAIN masks and cap BCE pos_weight at 20."""
    positive_pixels = 0
    total_pixels = 0

    for record in train_records:
        mask = cv2.imread(str(record["MaskPath"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Could not read mask: {record["MaskPath"]}')
        positive_pixels += int(np.count_nonzero(mask > 0))
        total_pixels += int(mask.size)

    negative_pixels = total_pixels - positive_pixels
    if positive_pixels == 0:
        raise RuntimeError("Training split contains no positive mask pixels")

    raw_pos_weight = negative_pixels / positive_pixels
    pos_weight = min(raw_pos_weight, 20.0)

    print(f"Positive train pixels: {positive_pixels:,}")
    print(f"Negative train pixels: {negative_pixels:,}")
    print(f"Raw pos_weight (negative/positive): {raw_pos_weight:.4f}")
    print(f"Effective pos_weight used for training: {pos_weight:.4f}")

    return float(pos_weight), positive_pixels, negative_pixels


def main() -> None:
    args = parse_args()
    if not args.images.is_dir() or not args.masks.is_dir():
        raise FileNotFoundError("--images and --masks must point to existing folders")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = build_file_list(str(args.images), str(args.masks))
    if not records:
        raise ValueError("No matching PNG image-mask pairs were found")
    train_records, val_records, test_records = split_data(records)
    train_loader, val_loader, _ = build_dataloaders(
        train_records,
        val_records,
        test_records,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Pairs: {len(records)} | train: {len(train_records)} | "
        f"val: {len(val_records)} | test: {len(test_records)}"
    )

    pos_weight, positive_pixels, negative_pixels = calculate_pos_weight(train_records)

    model = UNet(features=(32, 64, 128, 256)).to(device)
    criterion = SegmentationLoss(pos_weight=pos_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_val_dice_positive = float("-inf")
    epochs_without_improvement = 0
    initialize_history(args.history)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.max_train_batches
        )
        (
            val_loss,
            val_dice_all,
            val_iou_all,
            val_dice_positive,
            val_iou_positive,
            predicted_positive_rate,
        ) = validate(model, val_loader, criterion, device, args.max_val_batches)
        learning_rate = optimizer.param_groups[0]["lr"]
        append_history(
            args.history,
            epoch,
            train_loss,
            val_loss,
            val_dice_all,
            val_iou_all,
            val_dice_positive,
            val_iou_positive,
            predicted_positive_rate,
            learning_rate,
        )
        print(
            f"  train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_dice_all {val_dice_all:.4f} | val_iou_all {val_iou_all:.4f} | "
            f"val_dice_positive {val_dice_positive:.4f} | "
            f"val_iou_positive {val_iou_positive:.4f} | "
            f"pred_positive_rate {predicted_positive_rate:.4f} | "
            f"learning_rate {learning_rate:.2e}"
        )

        if val_dice_positive > best_val_dice_positive:
            best_val_dice_positive = val_dice_positive
            epochs_without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice_all": val_dice_all,
                    "val_iou_all": val_iou_all,
                    "val_dice_positive": val_dice_positive,
                    "val_iou_positive": val_iou_positive,
                    "predicted_positive_rate": predicted_positive_rate,
                    "features": (32, 64, 128, 256),
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "pos_weight": pos_weight,
                    "positive_train_pixels": positive_pixels,
                    "negative_train_pixels": negative_pixels,
                },
                args.checkpoint,
            )
            print(f"  Saved checkpoint: {args.checkpoint}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No val_dice_positive improvement for "
                f"{epochs_without_improvement}/{args.patience} epoch(s)"
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
