import torch
from evaluation import batch_metrics, image_categories


def test_visual_categories_use_same_minimum_as_metrics():
    masks = torch.zeros(2, 1, 8, 8)
    masks[0, :, :2, :2] = 1
    pred = torch.zeros_like(masks)
    pred[0, :, :2, :2] = 1
    pred[1, :, :1, :1] = 1
    predicted, target = image_categories(pred, masks, min_pred_pixels=4)
    assert predicted.tolist() == [True, False]
    assert target.tolist() == [True, False]
    metrics = batch_metrics(pred, masks, min_pred_pixels=4)
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 0, 0, 1)
