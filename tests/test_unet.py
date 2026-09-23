import pytest
import torch

from models.blocks import DoubleConv, DownBlock, UpBlock
from models.unet import UNet


def test_double_conv_preserves_spatial_shape():
    # DoubleConv chỉ đổi channel; padding phải giữ nguyên H, W.
    output = DoubleConv(1, 8)(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 8, 64, 80)


def test_down_block_halves_spatial_shape():
    # MaxPool 2x2 phải giảm đúng một nửa kích thước không gian.
    output = DownBlock(8, 16)(torch.randn(2, 8, 64, 80))
    assert output.shape == (2, 16, 32, 40)


@pytest.mark.parametrize("bilinear", [False, True])
def test_up_block_restores_skip_shape_and_backpropagates(bilinear):
    # Kiểm tra cả hai chiến lược upsample để option bilinear không bị bỏ quên.
    decoder = torch.randn(2, 32, 16, 20, requires_grad=True)
    skip = torch.randn(2, 16, 32, 40, requires_grad=True)
    output = UpBlock(32, 16, 16, bilinear=bilinear)(decoder, skip)

    assert output.shape == skip.shape
    # Gradient phải đi qua cả nhánh decoder lẫn skip connection.
    output.mean().backward()
    assert decoder.grad is not None
    assert skip.grad is not None


def test_unet_output_shape_parameter_count_and_backward():
    # Dùng channel nhỏ để smoke test CPU nhanh; cấu hình thật vẫn mặc định 64-512.
    model = UNet(features=(8, 16, 32, 64))
    image = torch.randn(2, 1, 256, 256, requires_grad=True)
    logits = model(image)

    assert logits.shape == (2, 1, 256, 256)
    assert model.count_parameters() > 0

    logits.mean().backward()
    assert image.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_unet_returns_logits_not_probabilities():
    # Nếu forward có sigmoid sai thiết kế, bias=2 sẽ cho ~0.88 thay vì đúng 2.0.
    model = UNet(features=(4, 8))
    model.output_conv.weight.data.zero_()
    model.output_conv.bias.data.fill_(2.0)

    output = model(torch.randn(1, 1, 32, 32))
    assert torch.all(output == 2.0)
