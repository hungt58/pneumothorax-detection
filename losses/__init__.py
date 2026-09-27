"""Các loss dùng cho model segmentation và multi-task về sau."""

# Export các loss tại package level để train.py có thể:
# from losses import SegmentationLoss
from .dice_loss import PositiveDiceLoss, NegativeTopKLoss, SegmentationLoss

__all__ = [
    "PositiveDiceLoss",
    "NegativeTopKLoss",
    "SegmentationLoss",
]
