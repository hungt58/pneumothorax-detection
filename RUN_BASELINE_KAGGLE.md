# Clean balanced U-Net baseline

This run is the diagnostic baseline before moving to a pretrained encoder.

```python
%cd /kaggle/working/pneumothorax-detection
!git fetch origin
!git reset --hard origin/ChuongDinh
!pip install -q -r requirements.txt
!pytest -q
```

Smoke test (optional, 20 train + 10 val batches):

```python
!python train.py \
  --images "/kaggle/input/datasets/vaillant/siim-ptx-png/images" \
  --masks "/kaggle/input/datasets/vaillant/siim-ptx-png/segmentations" \
  --epochs 1 --batch-size 8 --max-train-batches 20 --max-val-batches 10 \
  --checkpoint "/kaggle/working/checkpoints/unet_balanced_smoke.pt" \
  --history "/kaggle/working/results/unet/history_balanced_smoke.csv"
```

Main diagnostic run:

```python
!python train.py \
  --images "/kaggle/input/datasets/vaillant/siim-ptx-png/images" \
  --masks "/kaggle/input/datasets/vaillant/siim-ptx-png/segmentations" \
  --epochs 15 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --patience 7 \
  --checkpoint "/kaggle/working/checkpoints/unet_balanced_best.pt" \
  --history "/kaggle/working/results/unet/history_balanced.csv"
```

Decision after this run:
- If predictions localize pneumothorax and Dice+ improves without specificity collapsing, keep this as the vanilla U-Net baseline.
- If the model still learns fixed anatomical shortcuts, stop tuning this baseline and move to a pretrained ResNet18 encoder.
