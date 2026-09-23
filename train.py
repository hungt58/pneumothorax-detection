"""Train a baseline U-Net for binary pneumothorax segmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from build_dataset import build_dataloaders, build_file_list, split_data
from losses import SegmentationLoss
from models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each epoch early after this many batches (useful for a smoke test).",
    )
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("unet_best.pt"))
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
        dice_total += ((2 * intersection + 1) / (denominator + 1)).sum().item()
        if max_batches is not None and step >= max_batches:
            break

    return total_loss / batch_count, dice_total / sample_count


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

    # A narrower U-Net is fast enough for a first end-to-end Kaggle run.
    model = UNet(features=(16, 32, 64, 128)).to(device)
    criterion = SegmentationLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.max_train_batches
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, args.max_val_batches
        )
        print(
            f"  train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_dice {val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                },
                args.checkpoint,
            )
            print(f"  Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
