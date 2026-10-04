import torch
from models.attention_unet import AttentionUNet


def test_attention_unet_raw_logits_and_backward():
    model = AttentionUNet(features=(4, 8, 16, 32))
    image = torch.randn(2, 1, 64, 80, requires_grad=True)
    logits = model(image)
    assert logits.shape == (2, 1, 64, 80)
    assert model.count_parameters() > 0
    logits.mean().backward()
    assert image.grad is not None
    model.output_conv.weight.data.zero_()
    model.output_conv.bias.data.fill_(2)
    assert torch.all(model(image.detach()) == 2)
