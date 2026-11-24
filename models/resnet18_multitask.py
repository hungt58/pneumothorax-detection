"""E4: pretrained grayscale ResNet18, context fusion, attention, two heads."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

# names resolved at call time so tests can monkeypatch resnet18/resnet34
ENCODERS = {"resnet18": ("resnet18", ResNet18_Weights), "resnet34": ("resnet34", ResNet34_Weights)}

from .attention_blocks import AttentionUpBlock
from .multiscale import MultiScaleContextFusion


class ResNet18MultiTask(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 fusion_channels: int = 64, pretrained: bool = True,
                 with_classification: bool = True, encoder: str = "resnet18"):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {tuple(ENCODERS)}")
        self.with_classification = with_classification
        if in_channels != 1:
            raise ValueError("E4 requires grayscale input (in_channels=1)")
        if out_channels != 1:
            raise ValueError("E4 requires one segmentation logit channel")
        build_name, weights = ENCODERS[encoder]
        backbone = globals()[build_name](weights=weights.DEFAULT if pretrained else None)
        original = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, original.out_channels, original.kernel_size,
                                    original.stride, original.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.context = MultiScaleContextFusion((64, 64, 128, 256), 512, fusion_channels)
        self.decoder = nn.ModuleList((
            AttentionUpBlock(512, 256, 256),
            AttentionUpBlock(256, 128, 128),
            AttentionUpBlock(128, 64, 64),
            AttentionUpBlock(64, 64, 64),
        ))
        self.segmentation_head = nn.Conv2d(64, 1, 1)
        self.classification_head = (nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                                  nn.Dropout(), nn.Linear(512, 1))
                                    if with_classification else None)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        stem = self.stem(x)
        l1 = self.layer1(self.pool(stem))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        fused = self.context([stem, l1, l2, l3], l4)
        seg = fused
        for up, skip in zip(self.decoder, (l3, l2, l1, stem)):
            seg = up(seg, skip)
        seg = F.interpolate(self.segmentation_head(seg), size=input_size, mode="bilinear", align_corners=False)
        if not self.with_classification:
            # E4b: same backbone/decoder, segmentation only (plain tensor like E1-E3).
            return seg
        return {"segmentation_logits": seg, "classification_logits": self.classification_head(fused)}

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)


class ResNet18Segmentation(ResNet18MultiTask):
    """E4b ablation: E4 without the classification head and gate."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 fusion_channels: int = 64, pretrained: bool = True):
        super().__init__(in_channels, out_channels, fusion_channels, pretrained,
                         with_classification=False)



class ResNet34MultiTask(ResNet18MultiTask):
    """E5: same E4 design with a deeper pretrained ResNet34 encoder (same channel widths)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 fusion_channels: int = 64, pretrained: bool = True):
        super().__init__(in_channels, out_channels, fusion_channels, pretrained,
                         with_classification=True, encoder="resnet34")
