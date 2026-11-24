# PneumoAI – kết quả cuối kỳ (Test 1.602 ảnh, ngưỡng chọn trên Val)

| Model | Size | Thresholds (VAL) | Dice+ | IoU+ | Sens | Spec | Score | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 U-Net | 256 | seg_threshold=0.97 | 0.354 | 0.265 | 0.801 | 0.643 | 0.457 | 286 | 71 | 444 | 801 |
| E2 Attention U-Net | 256 | seg_threshold=0.95 | 0.326 | 0.237 | 0.821 | 0.639 | 0.432 | 293 | 64 | 449 | 796 |
| E3 Attention + Multi-scale | 256 | seg_threshold=0.96 | 0.358 | 0.264 | 0.852 | 0.573 | 0.441 | 304 | 53 | 531 | 714 |
| E4 ResNet18 multi-task | 256 | seg_threshold=0.3, cls_threshold=0.3 | 0.365 | 0.275 | 0.810 | 0.840 | 0.509 | 289 | 68 | 199 | 1046 |
| E2 + E4 cascade | 256 | e2_seg_threshold=0.5, e4_cls_threshold=0.15 | 0.330 | 0.241 | 0.815 | 0.812 | 0.469 | 291 | 66 | 234 | 1011 |
| E4b ResNet18 seg-only | 256 | seg_threshold=0.86 | 0.369 | 0.274 | 0.852 | 0.702 | 0.484 | 304 | 53 | 371 | 874 |
| E2 Attention U-Net @512 | 512 | seg_threshold=0.97 | 0.383 | 0.286 | 0.891 | 0.586 | 0.463 | 318 | 39 | 515 | 730 |

Leakage check:
```
train-val: exact copies 0 | near copies (dHash<= 4) 57991 | val images affected 1126/1601
train-test: exact copies 0 | near copies (dHash<= 4) 59313 | test images affected 1117/1602
patients: no patient ids given (PNG release) - duplicate-image check only
VERDICT: no exact copies across splits; review near copies in duplicates.csv
```

Ghi chú: E2 (256) dùng lại checkpoint giữa kỳ; các mô hình khác huấn luyện lại trên Kaggle (GPU T4). Số 'near copies' là do dHash 8×9 quá thô với X-quang ngực, không phải rò rỉ.
