"""Shared prediction and image-level metrics across train, tuning and test."""
import torch


def segmentation_logits(outputs):
    return outputs["segmentation_logits"] if isinstance(outputs, dict) else outputs


def final_mask(outputs, seg_threshold: float, cls_threshold: float | None = None):
    pred = torch.sigmoid(segmentation_logits(outputs)) >= seg_threshold
    if isinstance(outputs, dict) and cls_threshold is not None:
        passes = torch.sigmoid(outputs["classification_logits"]).flatten(1)[:, 0] >= cls_threshold
        pred = pred & passes[:, None, None, None]
    return pred.float()


def image_categories(pred: torch.Tensor, masks: torch.Tensor, min_pred_pixels: int = 64):
    if min_pred_pixels < 1:
        raise ValueError("min_pred_pixels must be at least 1")
    predicted_positive = pred.flatten(1).sum(1) >= min_pred_pixels
    target_positive = masks.flatten(1).sum(1) > 0
    return predicted_positive, target_positive


def batch_metrics(pred: torch.Tensor, masks: torch.Tensor, min_pred_pixels: int = 64):
    predicted_positive, target_positive = image_categories(pred, masks, min_pred_pixels)
    pf, mf = pred.flatten(1), masks.flatten(1)
    pp, mp = pf.sum(1), mf.sum(1)
    intersection = (pf * mf).sum(1)
    dice = (2 * intersection + 1) / (pp + mp + 1)
    iou = (intersection + 1) / (pp + mp - intersection + 1)
    tp = int((predicted_positive & target_positive).sum())
    fn = int((~predicted_positive & target_positive).sum())
    fp = int((predicted_positive & ~target_positive).sum())
    tn = int((~predicted_positive & ~target_positive).sum())
    return {"dice_sum": float(dice[target_positive].sum()), "iou_sum": float(iou[target_positive].sum()),
            "positive_count": int(target_positive.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def summarize(metrics: list[dict]):
    totals = {key: sum(row[key] for row in metrics) for key in metrics[0]}
    pos = totals["positive_count"]
    tp, fn, fp, tn = (totals[key] for key in ("tp", "fn", "fp", "tn"))
    dice = totals["dice_sum"] / max(pos, 1)
    specificity = tn / max(tn + fp, 1)
    return {"dice_positive": dice, "iou_positive": totals["iou_sum"] / max(pos, 1),
            "image_sensitivity": tp / max(tp + fn, 1), "image_specificity": specificity,
            "false_positive_rate": fp / max(fp + tn, 1),
            "predicted_positive_rate": (tp + fp) / max(tp + fn + fp + tn, 1),
            "selection_score": 2 * dice * specificity / (dice + specificity + 1e-8),
            **{key: totals[key] for key in ("tp", "fn", "fp", "tn")}}
