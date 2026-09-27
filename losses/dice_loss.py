"""Losses for binary pneumothorax segmentation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Standard soft Dice loss kept for compatibility and unit tests."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be > 0")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(1) + targets.sum(1) + self.smooth
        )
        return 1.0 - dice.mean()


class PositiveDiceLoss(nn.Module):
    """Soft Dice only on samples whose ground-truth mask is non-empty."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be > 0")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        positive = targets.flatten(1).sum(1) > 0
        if not positive.any():
            return logits.sum() * 0.0

        probs = torch.sigmoid(logits[positive]).flatten(1)
        positive_targets = targets[positive].flatten(1)
        intersection = (probs * positive_targets).sum(1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(1) + positive_targets.sum(1) + self.smooth
        )
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Clean baseline objective: BCE over all images + Dice over positive masks.

    Image-level imbalance is handled by the balanced train sampler, not by
    pos_weight. Empty-mask images still contribute BCE and therefore teach
    the model to suppress false positives.
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
        self.positive_dice = PositiveDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_inputs(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.positive_dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def _validate_inputs(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.ndim < 3:
        raise ValueError("expected [B,C,...spatial] tensors")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
