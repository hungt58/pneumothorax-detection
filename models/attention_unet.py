"""E2: baseline encoder and bottleneck with attention decoder."""
from collections.abc import Sequence
import torch
from torch import nn

from .attention_blocks import AttentionUpBlock
from .blocks import DoubleConv, DownBlock


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: Sequence[int] = (32, 64, 128, 256), *, bilinear: bool = False):
        super().__init__()
        if not features or any(width <= 0 for width in features):
            raise ValueError("features must contain positive channel widths")
        widths = tuple(features)
        self.input_block = DoubleConv(in_channels, widths[0])
        self.down_blocks = nn.ModuleList(DownBlock(widths[i - 1], widths[i]) for i in range(1, len(widths)))
        self.bottleneck = DownBlock(widths[-1], widths[-1] * 2)
        self.up_blocks = nn.ModuleList()
        decoder_in = widths[-1] * 2
        for skip_channels in reversed(widths):
            self.up_blocks.append(AttentionUpBlock(decoder_in, skip_channels, skip_channels, bilinear=bilinear))
            decoder_in = skip_channels
        self.output_conv = nn.Conv2d(widths[0], out_channels, 1)

    def encode(self, x: torch.Tensor):
        skips = [self.input_block(x)]
        for down in self.down_blocks:
            skips.append(down(skips[-1]))
        return skips, self.bottleneck(skips[-1])

    def decode(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        return self.output_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, bottleneck = self.encode(x)
        return self.decode(bottleneck, skips)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
