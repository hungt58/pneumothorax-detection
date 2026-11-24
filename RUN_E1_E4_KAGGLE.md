# Kaggle protocol for E1-E4

Run every experiment on the same SIIM image and mask directories. The code keeps the fixed stratified 70/15/15 split (seed 42), 256x256 grayscale preprocessing, train augmentation, balanced 4+4 batches, AdamW and validation selection score. All models (E1 included) now save to `checkpoints/<experiment>_best.pt`; rename an old `unet_balanced_best.pt` to `e1_unet_best.pt`. Second-half experiments: see RUN_NUA_SAU_KAGGLE.md. E4 downloads torchvision's default ResNet18 weights on first training run.

```python
%cd /kaggle/working/pneumothorax-detection
!pip install -q -r requirements.txt
!python -m pytest -q
```

Set the paths once in the notebook:

```python
IMAGES = "/kaggle/input/datasets/vaillant/siim-ptx-png/images"
MASKS = "/kaggle/input/datasets/vaillant/siim-ptx-png/segmentations"
```

Run the shell commands below with those paths substituted literally. E2 smoke test:

```bash
python train.py --model attention_unet --images IMAGES --masks MASKS \
  --epochs 1 --batch-size 8 --max-train-batches 20 --max-val-batches 10 \
  --checkpoint checkpoints/e2_attention_smoke.pt --history results/e2_attention_unet/smoke.csv
```

Train E1-E4 separately. Omit `--model` for E1 (U-Net). The remaining models use separate default paths:

```bash
python train.py --model attention_unet --images IMAGES --masks MASKS --epochs 15 --batch-size 8
python train.py --model attention_multiscale_unet --images IMAGES --masks MASKS --epochs 15 --batch-size 8
python train.py --model resnet18_multitask --images IMAGES --masks MASKS --epochs 15 --batch-size 8
```

For each E1-E3 checkpoint, tune on validation, inspect validation predictions, then run final test once with the selected threshold:

```bash
python evaluate_thresholds.py --images IMAGES --masks MASKS --checkpoint CHECKPOINT
python visualize_predictions.py --images IMAGES --masks MASKS --checkpoint CHECKPOINT --threshold SELECTED_SEG_THRESHOLD
python evaluate_test.py --images IMAGES --masks MASKS --checkpoint CHECKPOINT --threshold SELECTED_SEG_THRESHOLD
```

For E4, tune both thresholds on validation. Use the reported best pair for validation visualization and the one final test. The test mode requires both frozen thresholds and does no search:

```bash
python evaluate_multitask_thresholds.py --images IMAGES --masks MASKS --checkpoint checkpoints/e4_resnet18_multitask_best.pt
python visualize_predictions.py --images IMAGES --masks MASKS --checkpoint checkpoints/e4_resnet18_multitask_best.pt --threshold SELECTED_SEG_THRESHOLD --cls-threshold SELECTED_CLS_THRESHOLD
python evaluate_multitask_thresholds.py --split test --images IMAGES --masks MASKS --checkpoint checkpoints/e4_resnet18_multitask_best.pt --seg-threshold SELECTED_SEG_THRESHOLD --cls-threshold SELECTED_CLS_THRESHOLD
```

All image-level TP/FN/FP/TN categories require at least 64 predicted foreground pixels. Dice and IoU are averaged only over ground-truth positive images. Thresholds and checkpoint selection use validation only. Do not use the test result to change thresholds or hyperparameters.
