"""Các loss dùng cho model segmentation và multi-task về sau."""

# Export hai loss tại package level để code train có import ngắn, ổn định.
from .dice_loss import DiceLoss, SegmentationLoss

__all__ = ["DiceLoss", "SegmentationLoss"]
