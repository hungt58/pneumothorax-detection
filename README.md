# SIIM-ACR Pneumothorax — Data Pipeline

Pipeline xử lý dữ liệu cho đề tài phân đoạn/phát hiện tràn khí màng phổi từ ảnh X-quang ngực.
Dataset gốc: [SIIM-ACR Pneumothorax Segmentation (Kaggle)](https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data)

## Cấu trúc thư mục

```
.
├── images/                          # Ảnh X-quang gốc (.png), convert từ DICOM
├── segmentations/                   # Mask tương ứng (.png), decode từ RLE trong train-rle.csv
├── checkdata/                       # Output của check_match.py
│   ├── check_match_report.txt
│   └── disease_statistics_chart.png
├── check_match.py                   # Kiểm tra khớp file + thống kê có bệnh/không bệnh
├── build_dataset.py                 # Chia train/val/test, resize, normalize, augment, tạo DataLoader
└── README.md
```

## Cài đặt

```bash
pip install torch torchvision albumentations opencv-python scikit-learn numpy matplotlib
```

## Thứ tự chạy

### 1. Kiểm tra dữ liệu

```bash
mkdir -p checkdata
python check_match.py
```

Kiểm tra:
- Số ảnh trong `images/` và `segmentations/` có khớp tên nhau không
- Thống kê số ảnh **có bệnh** / **không bệnh** dựa trên mask (mask toàn đen = không bệnh)

Output: `checkdata/check_match_report.txt`, `checkdata/disease_statistics_chart.png`

### 2. Build dataset + DataLoader

```bash
python build_dataset.py
```

Thực hiện:
- Chia dữ liệu Train (70%) / Validation (15%) / Test (15%), stratified theo nhãn có bệnh/không bệnh
- Resize ảnh + mask về 256×256
- Normalize pixel về khoảng chuẩn (mean=0.5, std=0.5)
- Data Augmentation (chỉ áp dụng cho tập Train): flip ngang, xoay nhẹ, chỉnh sáng/tương phản, nhiễu Gauss
- Tạo `train_loader`, `val_loader`, `test_loader` (PyTorch DataLoader, batch size = 16)

Chạy xong sẽ in ra 1 dòng xác nhận thông số, ví dụ:
```
✅ OK | Total: 12047 | Train: 8432 | Val: 1807 | Test: 1808 | Batch size: 16 | Image: (16, 1, 256, 256) | Mask: (16, 256, 256)
```

## Cấu hình có thể chỉnh

Trong `build_dataset.py`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `IMG_SIZE` | 256 | Kích thước resize ảnh/mask |
| `BATCH_SIZE` | 16 | Số ảnh mỗi batch |
| `RANDOM_SEED` | 42 | Seed để chia train/val/test cố định, tái lặp được kết quả |

## Bước tiếp theo (không nằm trong phần này)

`train_loader` / `val_loader` / `test_loader` sẵn sàng để đưa vào định nghĩa model (U-Net), viết training loop, đánh giá bằng Dice score / IoU, và trực quan hóa kết quả dự đoán.
