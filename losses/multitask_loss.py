"""Segmentation and image classification objective for E4."""
import torch
from torch import nn

from .dice_loss import SegmentationLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_seg: float = 1.0, lambda_cls: float = 1.0):
        super().__init__()
        if lambda_seg < 0 or lambda_cls < 0 or lambda_seg + lambda_cls == 0:
            raise ValueError("loss weights must be nonnegative and at least one positive")
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.segmentation = SegmentationLoss()
        self.classification = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict[str, torch.Tensor], masks: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        return (self.lambda_seg * self.segmentation(outputs["segmentation_logits"], masks)
                + self.lambda_cls * self.classification(outputs["classification_logits"], labels))
