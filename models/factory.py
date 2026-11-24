"""Build models and restore their architecture from checkpoint metadata."""
from collections.abc import Mapping
import torch

from .unet import UNet
from .attention_unet import AttentionUNet
from .attention_multiscale_unet import AttentionMultiScaleUNet
from .resnet18_multitask import ResNet18MultiTask, ResNet18Segmentation

MODEL_NAMES = ("unet", "attention_unet", "attention_multiscale_unet", "resnet18_multitask", "resnet18_seg")
MODEL_CLASSES = dict(zip(MODEL_NAMES, (UNet, AttentionUNet, AttentionMultiScaleUNet,
                                       ResNet18MultiTask, ResNet18Segmentation)))
RESNET_MODELS = ("resnet18_multitask", "resnet18_seg")

EXPERIMENT_NAMES = {
    "unet": "e1_unet",
    "attention_unet": "e2_attention_unet",
    "attention_multiscale_unet": "e3_attention_multiscale",
    "resnet18_multitask": "e4_resnet18_multitask",
    "resnet18_seg": "e4b_resnet18_seg",
}


def experiment_name(model_name: str, image_size: int = 256) -> str:
    """Folder/checkpoint stem; 512 runs get a suffix so 256 results are never overwritten."""
    base = EXPERIMENT_NAMES[model_name]
    return base if int(image_size) == 256 else f"{base}_{int(image_size)}"


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
    if name in RESNET_MODELS:
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
