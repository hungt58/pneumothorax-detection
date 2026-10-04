"""Build models and restore their architecture from checkpoint metadata."""
from collections.abc import Mapping
import torch

from .unet import UNet
from .attention_unet import AttentionUNet
from .attention_multiscale_unet import AttentionMultiScaleUNet
from .resnet18_multitask import ResNet18MultiTask

MODEL_NAMES = ("unet", "attention_unet", "attention_multiscale_unet", "resnet18_multitask")
MODEL_CLASSES = dict(zip(MODEL_NAMES, (UNet, AttentionUNet, AttentionMultiScaleUNet, ResNet18MultiTask)))


def build_model(name: str, **kwargs):
    if name not in MODEL_CLASSES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return MODEL_CLASSES[name](**kwargs)


def checkpoint_model_spec(checkpoint: Mapping) -> tuple[str, dict]:
    name = checkpoint.get("model_name", "unet")
    if name not in MODEL_CLASSES:
        raise ValueError(f"unknown checkpoint model {name!r}")
    if "model_kwargs" in checkpoint:
        kwargs = dict(checkpoint["model_kwargs"])
    else:
        kwargs = {"features": tuple(checkpoint.get("features", (32, 64, 128, 256)))}
    if name == "resnet18_multitask":
        # Loading a checkpoint must never download pretrained weights.
        kwargs["pretrained"] = False
    return name, kwargs


def load_model_from_checkpoint(checkpoint: Mapping | str, map_location="cpu"):
    if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__"):
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
    name, kwargs = checkpoint_model_spec(checkpoint)
    model = build_model(name, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
