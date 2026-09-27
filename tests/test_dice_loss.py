import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from build_dataset import PneumothoraxDataset, get_transforms, validate_batch
from losses import DiceLoss, PositiveDiceLoss, SegmentationLoss
from models import UNet


def test_dice_loss_is_near_zero_for_perfect_prediction():
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(targets == 1, 20.0, -20.0)
    assert DiceLoss()(logits, targets).item() < 1e-6


def test_positive_dice_ignores_empty_samples():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    loss = PositiveDiceLoss()(logits, targets)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_segmentation_loss_penalizes_false_positive_on_negative_image():
    targets = torch.zeros(1, 1, 16, 16)
    background_logits = torch.full_like(targets, -6.0)
    foreground_logits = torch.full_like(targets, 6.0)
    criterion = SegmentationLoss()
    assert criterion(background_logits, targets) < criterion(foreground_logits, targets)


def test_segmentation_loss_backward():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    loss = SegmentationLoss()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_dataloader_to_unet_loss_backward(tmp_path):
    records = []
    for index in range(2):
        image = np.full((48, 64), 40 + index * 40, dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8)
        if index:
            mask[10:35, 20:45] = 255
        image_path = tmp_path / f"image_{index}.png"
        mask_path = tmp_path / f"mask_{index}.png"
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        records.append({
            "ImageId": f"image_{index}", "ImagePath": str(image_path),
            "MaskPath": str(mask_path), "HasDisease": index,
        })

    dataset = PneumothoraxDataset(records, transform=get_transforms("val"))
    batch = validate_batch(next(iter(DataLoader(dataset, batch_size=2))))
    model = UNet(features=(8, 16, 32, 64))
    logits = model(batch["image"])
    loss = SegmentationLoss()(logits, batch["mask"])
    loss.backward()
    assert logits.shape == batch["mask"].shape == (2, 1, 256, 256)
    assert torch.isfinite(loss)
