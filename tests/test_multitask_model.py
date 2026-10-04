import torch
from torchvision.models import resnet18
from losses import MultiTaskLoss
from models.resnet18_multitask import ResNet18MultiTask


def test_e4_two_heads_and_backward_without_weight_download():
    model = ResNet18MultiTask(pretrained=False, fusion_channels=8)
    assert model.stem[0].in_channels == 1
    image = torch.randn(2, 1, 64, 64)
    outputs = model(image)
    assert outputs["segmentation_logits"].shape == (2, 1, 64, 64)
    assert outputs["classification_logits"].shape == (2, 1)
    masks = torch.zeros(2, 1, 64, 64)
    masks[0, :, 8:16, 8:16] = 1
    loss = MultiTaskLoss()(outputs, masks, torch.tensor([[1.], [0.]]))
    loss.backward()
    assert model.segmentation_head.weight.grad is not None
    assert model.classification_head[-1].weight.grad is not None


def test_pretrained_rgb_conv_is_averaged_for_grayscale(monkeypatch):
    import models.resnet18_multitask as module

    def fake_resnet18(weights):
        backbone = resnet18(weights=None)
        with torch.no_grad():
            backbone.conv1.weight[:, 0].fill_(1)
            backbone.conv1.weight[:, 1].fill_(2)
            backbone.conv1.weight[:, 2].fill_(3)
        return backbone

    monkeypatch.setattr(module, "resnet18", fake_resnet18)
    model = module.ResNet18MultiTask(pretrained=True, fusion_channels=8)
    assert model.stem[0].in_channels == 1
    assert torch.all(model.stem[0].weight == 2)
