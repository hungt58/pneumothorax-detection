"""Residual multi-scale context fusion for E3 and E4."""
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleContextFusion(nn.Module):
    def __init__(self, skip_channels: Sequence[int], bottleneck_channels: int, fusion_channels: int = 64):
        super().__init__()
        if not skip_channels or min(*skip_channels, bottleneck_channels, fusion_channels) <= 0:
            raise ValueError("channel widths must be positive")
        self.projections = nn.ModuleList(nn.Conv2d(channels, fusion_channels, 1) for channels in skip_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(len(skip_channels) * fusion_channels, bottleneck_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True),
        )

    def forward(self, skips: list[torch.Tensor], bottleneck: torch.Tensor) -> torch.Tensor:
        if len(skips) != len(self.projections):
            raise ValueError("skip count does not match fusion projections")
        size = bottleneck.shape[-2:]
        projected = [F.interpolate(conv(skip), size=size, mode="bilinear", align_corners=False)
                     for conv, skip in zip(self.projections, skips)]
        return bottleneck + self.fuse(torch.cat(projected, dim=1))
