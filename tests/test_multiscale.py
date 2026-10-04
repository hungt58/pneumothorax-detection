import torch
from models.multiscale import MultiScaleContextFusion
from models.attention_multiscale_unet import AttentionMultiScaleUNet


def test_fusion_and_e3_shapes():
    fusion = MultiScaleContextFusion((4, 8, 16), 32, fusion_channels=4)
    skips = [torch.randn(2, 4, 32, 40), torch.randn(2, 8, 16, 20), torch.randn(2, 16, 8, 10)]
    bottleneck = torch.randn(2, 32, 4, 5)
    assert fusion(skips, bottleneck).shape == bottleneck.shape
    assert AttentionMultiScaleUNet(features=(4, 8, 16))(torch.randn(2, 1, 64, 80)).shape == (2, 1, 64, 80)
