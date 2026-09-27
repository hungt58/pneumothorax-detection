"""Losses for binary pneumothorax segmentation."""
from __future__ import annotations
import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be greater than zero")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        probabilities = torch.sigmoid(logits).flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)
        intersection = (probabilities * targets).sum(dim=2)
        denominator = probabilities.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

class SegmentationLoss(nn.Module):
    """Weighted BCE-with-logits + soft Dice loss."""
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if pos_weight <= 0:
            raise ValueError("pos_weight must be greater than zero")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer(
            "pos_weight",
            torch.tensor([float(pos_weight)], dtype=torch.float32),
        )
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.ndim < 3:
        raise ValueError("expected tensors shaped [B,C,...spatial dimensions]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
