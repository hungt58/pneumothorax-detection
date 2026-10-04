import torch
from models.factory import build_model, load_model_from_checkpoint, MODEL_NAMES


def test_factory_and_legacy_checkpoint():
    for name in MODEL_NAMES:
        kwargs = {"pretrained": False, "fusion_channels": 8} if name == "resnet18_multitask" else {"features": (4, 8)}
        assert build_model(name, **kwargs) is not None
    model = build_model("unet", features=(4, 8))
    legacy = {"features": (4, 8), "model_state_dict": model.state_dict()}
    assert type(load_model_from_checkpoint(legacy)) is type(model)
    e2 = build_model("attention_unet", features=(4, 8))
    checkpoint = {"model_name": "attention_unet", "model_kwargs": {"features": [4, 8]},
                  "model_state_dict": e2.state_dict()}
    assert type(load_model_from_checkpoint(checkpoint)) is type(e2)
    e4 = build_model("resnet18_multitask", pretrained=False, fusion_channels=8)
    checkpoint = {"model_name": "resnet18_multitask",
                  "model_kwargs": {"pretrained": True, "fusion_channels": 8},
                  "model_state_dict": e4.state_dict()}
    assert type(load_model_from_checkpoint(checkpoint)) is type(e4)
