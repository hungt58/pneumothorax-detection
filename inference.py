"""Shared single-image inference used by the web demo and test.py.

Keeps inference identical to training/validation:
- grayscale, resize with the same interpolation as albumentations.Resize
- Normalize(mean=0.5, std=0.5) -> pixel range [-1, 1]
- thresholds frozen from VALIDATION tuning (configs/thresholds.json)
- E4: segmentation mask is kept only if the classification head passes
- image is called positive only if mask has >= min_pred_pixels pixels
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation import segmentation_logits

THRESHOLDS_FILE = Path(__file__).resolve().parent / "configs" / "thresholds.json"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MIN_PRED_PIXELS = 64


def load_threshold_config(path: Path | str = THRESHOLDS_FILE) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_thresholds(checkpoint: dict, experiment: str | None = None,
                       config: dict | None = None) -> dict:
    """Pick the validation-tuned thresholds for one checkpoint.

    Priority: configs/thresholds.json[experiment] > values stored in the
    checkpoint by tuning (tuned_*) > training defaults stored in checkpoint.
    """
    config = load_threshold_config() if config is None else config
    entry = dict(config.get(experiment or "", {}))
    seg = entry.get("seg_threshold",
                    checkpoint.get("tuned_seg_threshold", checkpoint.get("threshold", 0.5)))
    cls = entry.get("cls_threshold",
                    checkpoint.get("tuned_cls_threshold", checkpoint.get("cls_threshold")))
    min_px = entry.get("min_pred_pixels", checkpoint.get("min_pred_pixels", DEFAULT_MIN_PRED_PIXELS))
    return {
        "seg_threshold": float(seg),
        "cls_threshold": None if cls is None else float(cls),
        "min_pred_pixels": int(min_px),
        "source": "configs/thresholds.json" if entry else "checkpoint",
    }


def checkpoint_image_size(checkpoint: dict) -> int:
    return int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))


def preprocess(image, image_size: int = DEFAULT_IMAGE_SIZE) -> tuple[np.ndarray, torch.Tensor]:
    """PIL image or uint8 array -> (resized uint8 image, [1,1,H,W] tensor in [-1,1])."""
    if hasattr(image, "convert"):
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        array = array.astype(np.uint8)
    resized = cv2.resize(array, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(normalized)[None, None]
    return resized, tensor


@dataclass
class Prediction:
    probabilities: np.ndarray          # [H,W] segmentation probabilities
    mask: np.ndarray                   # [H,W] bool, after gating
    cls_probability: float | None      # E4 only
    positive_pixels: int
    is_positive: bool                  # image-level decision (>= min_pred_pixels)


@torch.inference_mode()
def predict_tensor(model, tensor: torch.Tensor, seg_threshold: float,
                   cls_threshold: float | None = None,
                   min_pred_pixels: int = DEFAULT_MIN_PRED_PIXELS) -> Prediction:
    device = next(model.parameters()).device
    outputs = model(tensor.to(device))
    probs = torch.sigmoid(segmentation_logits(outputs))[0, 0].float().cpu().numpy()
    mask = probs >= seg_threshold
    cls_probability = None
    if isinstance(outputs, dict) and "classification_logits" in outputs:
        cls_probability = float(torch.sigmoid(outputs["classification_logits"].reshape(-1)[0]))
        if cls_threshold is not None and cls_probability < cls_threshold:
            mask = np.zeros_like(mask)
    pixels = int(mask.sum())
    return Prediction(probs, mask, cls_probability, pixels, pixels >= min_pred_pixels)


def cascade_decision(screen: Prediction, confirm_cls_probability: float,
                     confirm_threshold: float) -> tuple[np.ndarray, bool]:
    """E2 screens (high sensitivity), E4 classification confirms (high specificity).

    The final mask is the E2 mask, kept only when E4 confirms the image.
    """
    confirmed = screen.is_positive and confirm_cls_probability >= confirm_threshold
    mask = screen.mask if confirmed else np.zeros_like(screen.mask)
    return mask, bool(confirmed)
