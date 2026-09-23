import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from build_dataset import PneumothoraxDataset, get_transforms, validate_batch
from losses import DiceLoss, SegmentationLoss
from models import UNet


def test_dice_loss_is_near_zero_for_perfect_prediction():
    # Logit +/-20 tương ứng xác suất gần 1/0, tức prediction gần như hoàn hảo.
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(targets == 1, 20.0, -20.0)
    assert DiceLoss()(logits, targets).item() < 1e-6


def test_dice_loss_is_finite_for_empty_mask():
    # Negative sample (mask toàn 0) rất phổ biến trong bộ SIIM-ACR.
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    loss = DiceLoss()(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_segmentation_loss_backward():
    # Loss tổng không chỉ cần hữu hạn mà còn phải truyền gradient về model.
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    loss = SegmentationLoss()(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_dataloader_to_unet_loss_backward(tmp_path):
    # Tạo PNG thật trên đĩa để đi qua đúng cv2 + Dataset + transform hiện tại,
    # thay vì đưa tensor trực tiếp và vô tình bỏ qua lỗi tích hợp DataLoader.
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
        records.append(
            {
                "ImageId": f"image_{index}",
                "ImagePath": str(image_path),
                "MaskPath": str(mask_path),
                "HasDisease": index,
            }
        )

    # Dùng transform val để test xác định, không có augmentation ngẫu nhiên.
    dataset = PneumothoraxDataset(records, transform=get_transforms("val"))
    batch = validate_batch(next(iter(DataLoader(dataset, batch_size=2))))
    # Model hẹp giúp test chạy nhanh trên CPU nhưng giữ nguyên toàn bộ luồng U-Net.
    model = UNet(features=(8, 16, 32, 64))

    logits = model(batch["image"])
    loss = SegmentationLoss()(logits, batch["mask"])
    loss.backward()

    assert logits.shape == batch["mask"].shape == (2, 1, 256, 256)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
