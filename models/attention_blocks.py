"""Attention-gated decoder blocks."""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DoubleConv


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int | None = None):
        super().__init__()
        inter_channels = inter_channels or max(1, min(skip_channels, gating_channels) // 2)
        self.theta_x = nn.Sequential(nn.Conv2d(skip_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.phi_g = nn.Sequential(nn.Conv2d(gating_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def attention(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        if gating.shape[-2:] != skip.shape[-2:]:
            gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.psi(F.relu(self.theta_x(skip) + self.phi_g(gating)))

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        return skip * self.attention(skip, gating)


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(in_channels, out_channels, 1, bias=False))
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.gate = AttentionGate(skip_channels, out_channels)
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((self.gate(skip, x), x), dim=1))
