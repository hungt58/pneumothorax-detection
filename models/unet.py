"""Baseline U-Net for binary pneumothorax segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    """U-Net đối xứng trả về raw logits cho bài toán phân đoạn nhị phân.

    Args:
        in_channels: Số channel ảnh; X-quang grayscale dùng 1.
        out_channels: Số channel logits; phân đoạn nhị phân dùng 1.
        features: Số feature channel của encoder từ nông đến sâu.
        bilinear: Dùng nội suy bilinear thay cho transposed convolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        *,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        # Chặn cấu hình rỗng/sai sớm để lỗi dễ hiểu hơn lỗi Conv2d phía sau.
        if not features or any(width <= 0 for width in features):
            raise ValueError("features must contain positive channel widths")

        widths = tuple(features)
        # Stage đầu giữ nguyên 256x256 để tạo skip có độ phân giải cao nhất.
        self.input_block = DoubleConv(in_channels, widths[0])
        # Mỗi DownBlock giảm H, W một nửa và tăng số feature channel.
        self.down_blocks = nn.ModuleList(
            DownBlock(widths[index - 1], widths[index])
            for index in range(1, len(widths))
        )
        # Bottleneck là biểu diễn sâu nhất, nằm giữa encoder và decoder.
        bottleneck_channels = widths[-1] * 2
        self.bottleneck = DownBlock(widths[-1], bottleneck_channels)

        decoder_in = bottleneck_channels
        self.up_blocks = nn.ModuleList()
        # Duyệt widths ngược để decoder đối xứng với encoder và dùng đúng skip.
        for skip_channels in reversed(widths):
            self.up_blocks.append(
                UpBlock(
                    decoder_in,
                    skip_channels,
                    skip_channels,
                    bilinear=bilinear,
                )
            )
            decoder_in = skip_channels

        # Conv 1x1 ánh xạ feature cuối thành một logit cho mỗi pixel.
        # Không sigmoid tại đây: BCEWithLogitsLoss cần nhận logits thô để ổn định số.
        self.output_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lưu output từng encoder stage để decoder phục hồi biên và chi tiết ảnh.
        skips = [self.input_block(x)]
        for down in self.down_blocks:
            skips.append(down(skips[-1]))

        x = self.bottleneck(skips[-1])
        # Skip sâu nhất ghép trước, sau đó lần lượt quay về độ phân giải ban đầu.
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        return self.output_conv(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Đếm tham số để báo cáo và so sánh công bằng giữa các model."""
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
