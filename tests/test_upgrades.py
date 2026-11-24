"""Tests for the second-half upgrades: demo preprocessing, cascade, E4b, 512 input, leakage."""
import json

import cv2
import numpy as np
import torch
from PIL import Image

from build_dataset import get_transforms
from check_split_leakage import image_hashes, near_duplicates, patient_overlap
from evaluate_cascade import cascade_mask, threshold_grid
from inference import (Prediction, cascade_decision, predict_tensor, preprocess,
                       resolve_thresholds)
from models.factory import build_model, experiment_name, load_model_from_checkpoint


def test_demo_preprocessing_matches_validation_transform():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(300, 280), dtype=np.uint8)
    expected = get_transforms("val", 256)(image=raw, mask=np.zeros_like(raw))["image"]
    _, tensor = preprocess(Image.fromarray(raw), 256)
    assert tensor.shape == (1, 1, 256, 256)
    assert torch.allclose(tensor[0, 0], expected.float().reshape(256, 256), atol=1e-5)
    assert tensor.min() >= -1 and tensor.max() <= 1


def test_transforms_follow_image_size():
    raw = np.zeros((300, 300), dtype=np.uint8)
    out = get_transforms("train", 512)(image=raw, mask=raw)
    assert out["image"].shape[-2:] == (512, 512)
    assert out["mask"].shape[-2:] == (512, 512)


def test_thresholds_come_from_config_then_checkpoint():
    config = {"e2_attention_unet": {"seg_threshold": 0.95, "min_pred_pixels": 64}}
    thr = resolve_thresholds({"threshold": 0.5}, "e2_attention_unet", config)
    assert thr["seg_threshold"] == 0.95 and thr["source"] == "configs/thresholds.json"
    fallback = resolve_thresholds({"threshold": 0.5, "cls_threshold": 0.4}, "unknown", config)
    assert fallback["seg_threshold"] == 0.5 and fallback["cls_threshold"] == 0.4


def test_repo_threshold_config_is_valid():
    config = json.load(open("configs/thresholds.json", encoding="utf-8"))
    for key in ("e1_unet", "e2_attention_unet", "e3_attention_multiscale", "e4_resnet18_multitask"):
        assert 0 < config[key]["seg_threshold"] < 1
    assert 0 < config["e4_resnet18_multitask"]["cls_threshold"] < 1


def test_e4_gate_empties_mask_when_classifier_rejects():
    model = build_model("resnet18_multitask", pretrained=False, fusion_channels=8).eval()
    with torch.no_grad():
        model.segmentation_head.weight.zero_()
        model.segmentation_head.bias.fill_(10.0)        # every pixel very positive
        model.classification_head[-1].weight.zero_()
        model.classification_head[-1].bias.fill_(-10.0)  # classifier says "no"
    pred = predict_tensor(model, torch.zeros(1, 1, 64, 64), 0.5, cls_threshold=0.3, min_pred_pixels=1)
    assert pred.positive_pixels == 0 and not pred.is_positive
    open_gate = predict_tensor(model, torch.zeros(1, 1, 64, 64), 0.5, cls_threshold=None, min_pred_pixels=1)
    assert open_gate.positive_pixels == 64 * 64


def test_cascade_requires_both_models():
    probs = torch.zeros(3, 1, 8, 8)
    probs[:2, :, :4, :4] = 0.99
    cls = torch.tensor([0.9, 0.1, 0.9])
    pred = cascade_mask(probs, cls, 0.5, 0.5)
    assert pred[0].sum() == 16 and pred[1].sum() == 0 and pred[2].sum() == 0
    screen = Prediction(np.ones((4, 4)), np.ones((4, 4), bool), None, 16, True)
    assert cascade_decision(screen, 0.8, 0.5)[1] is True
    assert cascade_decision(screen, 0.2, 0.5)[1] is False
    assert len(threshold_grid()) == 10 * 16


def test_e4b_is_segmentation_only_and_reloads():
    model = build_model("resnet18_seg", pretrained=False, fusion_channels=8)
    out = model(torch.randn(2, 1, 64, 64))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 1, 64, 64)
    assert model.classification_head is None
    ckpt = {"model_name": "resnet18_seg", "model_kwargs": {"pretrained": True, "fusion_channels": 8},
            "model_state_dict": model.state_dict()}
    assert type(load_model_from_checkpoint(ckpt)) is type(model)


def test_experiment_names_keep_256_results():
    assert experiment_name("attention_unet", 256) == "e2_attention_unet"
    assert experiment_name("attention_unet", 512) == "e2_attention_unet_512"
    assert experiment_name("resnet18_seg") == "e4b_resnet18_seg"


def test_leakage_finds_exact_and_near_copies(tmp_path):
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((64, 64)) * 255).astype(np.uint8), (9, 9), 3)
    other = cv2.GaussianBlur((rng.random((64, 64)) * 255).astype(np.uint8), (9, 9), 3)
    for name, img in {"a": base, "b": base, "c": np.clip(base.astype(int) + 3, 0, 255), "d": other}.items():
        cv2.imwrite(str(tmp_path / f"{name}.png"), np.asarray(img, dtype=np.uint8))
    h = {k: image_hashes(str(tmp_path / f"{k}.png")) for k in "abcd"}
    assert h["a"][0] == h["b"][0]
    matches = {j for _, j, _ in near_duplicates([h["a"][1]], [h["b"][1], h["c"][1], h["d"][1]], 4)}
    assert matches == {0, 1}
    info = patient_overlap({"train": [{"ImageId": "x"}], "val": [{"ImageId": "y"}], "test": []},
                           {"x": "p1", "y": "p1"})
    assert info["shared_patients"] == 1


def test_resnet34_multitask_builds_and_runs():
    import torch
    from models.factory import build_model, experiment_name
    model = build_model("resnet34_multitask", pretrained=False, fusion_channels=8).eval()
    out = model(torch.zeros(1, 1, 64, 64))
    assert out["segmentation_logits"].shape == (1, 1, 64, 64)
    assert out["classification_logits"].shape == (1, 1)
    assert experiment_name("resnet34_multitask", 512) == "e5_resnet34_multitask_512"
