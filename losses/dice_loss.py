"""Losses for binary pneumothorax segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss tính trực tiếp từ logits phân đoạn nhị phân.

    Dice được tính riêng cho từng sample/channel rồi mới lấy trung bình.
    ``smooth`` tránh chia cho 0 khi ground-truth rỗng và vẫn thưởng cho dự đoán
    rỗng khi cả prediction lẫn target đều không có vùng bệnh.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        # smooth phải dương thì mới bảo vệ được trường hợp mẫu và mẫu số đều 0.
        if smooth <= 0:
            raise ValueError("smooth must be greater than zero")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        # Chỉ sigmoid ở nhánh Dice. Nhánh BCE bên dưới vẫn nhận logits thô.
        probabilities = torch.sigmoid(logits)
        # Giữ B và C, gộp mọi chiều không gian để tính overlap toàn mask.
        probabilities = probabilities.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)

        intersection = (probabilities * targets).sum(dim=2)
        denominator = probabilities.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Tối ưu loss nên đổi Dice score (càng cao càng tốt) thành 1 - Dice.
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Tổng có trọng số của BCE-with-logits và soft Dice loss.

    BCE giám sát từng pixel; Dice nhấn mạnh độ chồng lấp vùng bệnh và hỗ trợ dữ
    liệu mất cân bằng khi vùng tràn khí thường nhỏ hơn nền.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("at least one loss weight must be positive")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # BCEWithLogitsLoss gộp sigmoid + BCE trong một phép tính ổn định số học.
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> None:
    """Báo lỗi contract rõ ràng trước khi PyTorch broadcast nhầm tensor."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.ndim < 3:
        raise ValueError("expected tensors shaped [B,C,...spatial dimensions]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
