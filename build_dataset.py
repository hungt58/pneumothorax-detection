"""Dataset pipeline for SIIM pneumothorax segmentation.

Key rules used by the baseline experiment:
- deterministic stratified 70/15/15 split
- 256x256 grayscale input
- mild train-only augmentation
- balanced positive/negative TRAIN batches
- natural validation/test distribution
"""
from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader, Dataset

IMAGES_DIR = "images"
MASKS_DIR = "segmentations"
IMG_SIZE = 256
BATCH_SIZE = 16
RANDOM_SEED = 42


def build_file_list(images_dir: str, masks_dir: str) -> list[dict]:
    image_names = {
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.lower().endswith(".png")
    }
    mask_names = {
        os.path.splitext(f)[0]
        for f in os.listdir(masks_dir)
        if f.lower().endswith(".png")
    }
    matched = sorted(image_names & mask_names)

    records: list[dict] = []
    for name in matched:
        mask_path = os.path.join(masks_dir, name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        records.append(
            {
                "ImageId": name,
                "ImagePath": os.path.join(images_dir, name + ".png"),
                "MaskPath": mask_path,
                "HasDisease": int(np.max(mask) > 0),
            }
        )
    return records


def split_data(
    records: Sequence[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1")

    labels = [r["HasDisease"] for r in records]
    train_recs, temp_recs = train_test_split(
        list(records),
        test_size=1.0 - train_ratio,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    temp_labels = [r["HasDisease"] for r in temp_recs]
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_recs, test_recs = train_test_split(
        temp_recs,
        test_size=1.0 - relative_val,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )
    return train_recs, val_recs, test_recs


def get_transforms(mode: str = "train"):
    if mode == "train":
        # Deliberately mild. No synthetic noise in the clean baseline.
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.2,
                ),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )

    if mode not in {"val", "test"}:
        raise ValueError(f"unknown transform mode: {mode}")

    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


class PneumothoraxDataset(Dataset):
    def __init__(self, records: Sequence[dict], transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = cv2.imread(rec["ImagePath"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(rec["MaskPath"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Could not read image: {rec["ImagePath"]}')
        if mask is None:
            raise FileNotFoundError(f'Could not read mask: {rec["MaskPath"]}')

        mask = (mask > 0).astype(np.uint8)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return {
            "image": image.float(),
            "mask": mask.float(),
            "label": torch.tensor([rec["HasDisease"]], dtype=torch.float32),
            "image_id": rec["ImageId"],
        }


class BalancedBinaryBatchSampler(BatchSampler):
    """Create deterministic 50/50 positive-negative train batches.

    Positive examples are oversampled with replacement because the train split
    contains far fewer positive X-rays. Validation and test loaders are never
    balanced and keep the real prevalence.
    """

    def __init__(
        self,
        records: Sequence[dict],
        batch_size: int,
        seed: int = RANDOM_SEED,
        batches_per_epoch: int | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("balanced batching requires batch_size >= 2")
        if batch_size % 2 != 0:
            raise ValueError("use an even batch_size for exact 50/50 batches")

        self.positive_indices = [i for i, r in enumerate(records) if r["HasDisease"] == 1]
        self.negative_indices = [i for i, r in enumerate(records) if r["HasDisease"] == 0]
        if not self.positive_indices or not self.negative_indices:
            raise ValueError("balanced batching needs both positive and negative records")

        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = 0
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(records) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            pos = rng.sample(self.positive_indices, k=self.half)
            neg = rng.sample(self.negative_indices, k=self.half)
            batch = pos + neg
            rng.shuffle(batch)
            yield batch


def build_dataloaders(
    train_recs,
    val_recs,
    test_recs,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    balanced_train: bool = True,
    seed: int = RANDOM_SEED,
):
    train_ds = PneumothoraxDataset(train_recs, transform=get_transforms("train"))
    val_ds = PneumothoraxDataset(val_recs, transform=get_transforms("val"))
    test_ds = PneumothoraxDataset(test_recs, transform=get_transforms("test"))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())

    if balanced_train:
        batch_sampler = BalancedBinaryBatchSampler(train_recs, batch_size=batch_size, seed=seed)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def validate_batch(batch: dict) -> dict:
    images, masks, labels = batch["image"], batch["mask"], batch["label"]
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("images/masks must be [B,C,H,W]")
    if images.shape != masks.shape:
        raise ValueError(f"image/mask shapes differ: {images.shape} vs {masks.shape}")
    if images.shape[1:] != (1, IMG_SIZE, IMG_SIZE):
        raise ValueError(f"unexpected spatial shape: {tuple(images.shape)}")
    if labels.shape != (images.shape[0], 1):
        raise ValueError(f"unexpected label shape: {tuple(labels.shape)}")
    if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
        raise ValueError("batch contains non-finite values")
    if not set(torch.unique(masks).tolist()) <= {0.0, 1.0}:
        raise ValueError("masks must be binary")
    return batch


def smoke_test_dataloaders(train_loader, val_loader, test_loader):
    batches = {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        batches[name] = validate_batch(next(iter(loader)))
    return batches


def main():
    records = build_file_list(IMAGES_DIR, MASKS_DIR)
    train_recs, val_recs, test_recs = split_data(records)
    train_loader, val_loader, test_loader = build_dataloaders(train_recs, val_recs, test_recs)
    batches = smoke_test_dataloaders(train_loader, val_loader, test_loader)
    batch = batches["train"]
    positives = int(batch["label"].sum().item())
    print(
        f"OK | Total: {len(records)} | Train: {len(train_recs)} | Val: {len(val_recs)} | "
        f"Test: {len(test_recs)} | Batch: {tuple(batch['image'].shape)} | "
        f"Train batch positives: {positives}/{batch['label'].shape[0]}"
    )


if __name__ == "__main__":
    main()
