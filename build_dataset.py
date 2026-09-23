"""
=========================================================
PIPELINE GIAI ĐOẠN 2 (tiếp theo sau khi đã có images/ + segmentations/
và đã chạy check_match.py để xác nhận dữ liệu khớp/sạch)
=========================================================

Chia Train / Validation / Test
  -> Resize 256x256
  -> Normalize
  -> Data Augmentation
  -> PyTorch DataLoader

CÀI ĐẶT (chạy 1 lần):
    pip install torch torchvision albumentations opencv-python scikit-learn numpy

CHẠY:
    python build_dataset.py
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==== SỬA ĐƯỜNG DẪN CHO ĐÚNG VỚI MÁY BẠN ====
IMAGES_DIR = "images"
MASKS_DIR = "segmentations"

IMG_SIZE = 256
BATCH_SIZE = 16
RANDOM_SEED = 42


# =========================================================
# BƯỚC 1. LẤY DANH SÁCH CÁC CẶP ẢNH-MASK ĐÃ KHỚP + GÁN NHÃN
# =========================================================
def build_file_list(images_dir, masks_dir):
    image_names = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(".png")}
    mask_names = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith(".png")}
    matched = sorted(image_names & mask_names)

    records = []
    for name in matched:
        mask_path = os.path.join(masks_dir, name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        has_disease = int(np.max(mask) > 0)
        records.append({
            "ImageId": name,
            "ImagePath": os.path.join(images_dir, name + ".png"),
            "MaskPath": mask_path,
            "HasDisease": has_disease,
        })

    return records


# =========================================================
# BƯỚC 2. CHIA TRAIN / VALIDATION / TEST (stratified theo HasDisease)
# =========================================================
def split_data(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    labels = [r["HasDisease"] for r in records]

    train_recs, temp_recs = train_test_split(
        records, test_size=(1 - train_ratio),
        stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels = [r["HasDisease"] for r in temp_recs]
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_recs, test_recs = train_test_split(
        temp_recs, test_size=(1 - relative_val),
        stratify=temp_labels, random_state=RANDOM_SEED
    )

    return train_recs, val_recs, test_recs


# =========================================================
# BƯỚC 3-5. RESIZE + NORMALIZE + AUGMENTATION
# =========================================================
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])


# =========================================================
# BƯỚC 6. PYTORCH DATASET + DATALOADER
# (đọc thẳng file ảnh mask có sẵn, KHÔNG decode RLE lại)
# =========================================================
class PneumothoraxDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = cv2.imread(rec["ImagePath"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(rec["MaskPath"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # đưa về đúng 0/1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.float32)
        label = torch.tensor([rec["HasDisease"]], dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "image_id": rec["ImageId"],
        }


def build_dataloaders(train_recs, val_recs, test_recs,
                      batch_size=BATCH_SIZE, num_workers=2):
    train_ds = PneumothoraxDataset(train_recs, transform=get_transforms("train"))
    val_ds = PneumothoraxDataset(val_recs, transform=get_transforms("val"))
    test_ds = PneumothoraxDataset(test_recs, transform=get_transforms("test"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def validate_batch(batch):
    """Validate the tensor contract expected by segmentation models and losses."""
    required_keys = {"image", "mask", "label", "image_id"}
    missing_keys = required_keys - set(batch)
    if missing_keys:
        raise ValueError(f"Batch is missing keys: {sorted(missing_keys)}")

    image = batch["image"]
    mask = batch["mask"]
    label = batch["label"]

    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"Expected image [B,1,H,W], got {tuple(image.shape)}")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask [B,1,H,W], got {tuple(mask.shape)}")
    if image.shape[0] != mask.shape[0] or image.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Image and mask batch/spatial dimensions must match")
    if label.ndim != 2 or label.shape != (image.shape[0], 1):
        raise ValueError(f"Expected label [B,1], got {tuple(label.shape)}")
    if image.dtype != torch.float32 or mask.dtype != torch.float32:
        raise TypeError("Image and mask must use torch.float32")
    if label.dtype != torch.float32:
        raise TypeError("Label must use torch.float32")
    if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
        raise ValueError("Image or mask contains NaN/Inf")
    if not torch.isfinite(label).all():
        raise ValueError("Label contains NaN/Inf")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must contain binary values only")

    return batch


def smoke_test_dataloaders(train_loader, val_loader, test_loader):
    """Read and validate one batch from each data split."""
    batches = {}
    for split_name, loader in (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        try:
            batch = next(iter(loader))
        except StopIteration as exc:
            raise ValueError(f"{split_name} loader is empty") from exc
        batches[split_name] = validate_batch(batch)
    return batches


# =========================================================
# CHẠY TOÀN BỘ
# =========================================================
def main():
    records = build_file_list(IMAGES_DIR, MASKS_DIR)
    train_recs, val_recs, test_recs = split_data(records)
    train_loader, val_loader, test_loader = build_dataloaders(train_recs, val_recs, test_recs)

    batch = smoke_test_dataloaders(train_loader, val_loader, test_loader)["train"]
    print(f"✅ OK | Total: {len(records)} | Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)} "
          f"| Batch size: {BATCH_SIZE} | Image: {tuple(batch['image'].shape)} | Mask: {tuple(batch['mask'].shape)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
