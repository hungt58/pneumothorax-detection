"""E3: E2 with residual multi-scale context at the bottleneck."""
from collections.abc import Sequence
import torch

from .attention_unet import AttentionUNet
from .multiscale import MultiScaleContextFusion


class AttentionMultiScaleUNet(AttentionUNet):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: Sequence[int] = (32, 64, 128, 256), *,
                 bilinear: bool = False, fusion_channels: int = 64):
        super().__init__(in_channels, out_channels, features, bilinear=bilinear)
        self.context = MultiScaleContextFusion(features, features[-1] * 2, fusion_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, bottleneck = self.encode(x)
        return self.decode(self.context(skips, bottleneck), skips)
