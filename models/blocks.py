"""Reusable convolutional blocks used by U-Net variants."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Hai lớp Conv-BatchNorm-ReLU liên tiếp, đúng building block của U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # padding=1 giúp convolution 3x3 giữ nguyên chiều cao và chiều rộng.
        # Conv không cần bias vì BatchNorm ngay sau đó đã có tham số dịch riêng.
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """Giảm H, W một nửa rồi học đặc trưng ở mức sâu hơn."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Pooling làm vùng nhìn của tầng sau lớn hơn và giảm chi phí tính toán.
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    """Phóng to decoder, ghép skip encoder rồi tinh lọc bằng ``DoubleConv``."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if bilinear:
            # Bilinear không có tham số; Conv 1x1 phía sau đổi số channel về mức
            # decoder cần trước khi ghép với skip connection.
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            # Transposed convolution vừa tăng H, W gấp đôi vừa học cách upsample.
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        # Sau torch.cat, số channel bằng decoder + encoder skip.
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Ảnh có kích thước lẻ có thể lệch 1 pixel sau nhiều lần pool/upsample.
        # Ép decoder khớp chính xác skip giúp U-Net không chỉ chạy với 256x256.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        # Ghép theo chiều channel để decoder nhận cả ngữ nghĩa tầng sâu (x) và
        # chi tiết không gian độ phân giải cao từ encoder (skip).
        return self.conv(torch.cat((skip, x), dim=1))
