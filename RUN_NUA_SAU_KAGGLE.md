# Nửa sau học kỳ — chạy trên Kaggle

**Cách nhanh nhất:** mở `kaggle/PneumoAI_cuoi_ky.ipynb` trên Kaggle (GPU + Internet bật), bấm Run All.
Notebook gọi `run_final.py`, chạy hết các bước dưới đây theo thứ tự, dừng nếu phát hiện ảnh trùng
exact hoặc patient giao giữa các split, tự dò ngưỡng trên Val, chấm Test 1 lần, rồi gom kết quả,
checkpoint và cấu hình vào `/kaggle/working/final_results.zip`
(`results/final_summary.md` là bảng cuối kỳ). Chạy lại thì bước nào có kết quả rồi sẽ được bỏ qua.

Các lệnh lẻ nếu muốn chạy tay:

Ba việc trong slide 9.1, theo thứ tự nên chạy. Mọi lệnh dùng cùng split 70/15/15 (seed 42),
chỉ dò ngưỡng trên **Val**, chấm **Test đúng 1 lần** với ngưỡng đã khóa.

```python
%cd /kaggle/working/pneumothorax-detection
!git fetch origin && git reset --hard origin/ChuongDinh
!pip install -q -r requirements.txt
!python -m pytest -q          # phải xanh hết trước khi train
IMAGES = "/kaggle/input/datasets/vaillant/siim-ptx-png/images"
MASKS  = "/kaggle/input/datasets/vaillant/siim-ptx-png/segmentations"
```

Checkpoint E1–E4 cũ (256×256) đặt vào `checkpoints/` với tên `e1_unet_best.pt`,
`e2_attention_unet_best.pt`, `e3_attention_multiscale_best.pt`, `e4_resnet18_multitask_best.pt`
(E1 cũ tên `unet_balanced_best.pt` → đổi tên thành `e1_unet_best.pt`).

## 0. Kiểm tra rò rỉ dữ liệu (làm trước tiên, ~5 phút, không cần GPU)

```bash
python check_split_leakage.py --images $IMAGES --masks $MASKS
```

Ghi `results/leakage/report.txt` + `duplicates.csv`: ảnh trùng y hệt (MD5) và gần trùng (dHash)
giữa Train với Val/Test. Bản PNG của SIIM không có mã bệnh nhân nên phần "patient" chỉ chạy khi có
`--metadata ImageId,PatientID.csv` hoặc `--dicom-dir` (bộ DICOM gốc, cần `pip install pydicom`).
Nếu VERDICT báo **LEAK FOUND** thì pipeline dừng. Nếu có near-duplicates, xem từng cặp trong CSV;
chỉ sau khi kiểm tra và chấp nhận mới chạy lại `run_final.py` với
`--allow-reviewed-near-duplicates`.

## 1. Web demo đã sửa (không cần train lại)

`app.py` và `test.py` giờ dùng `inference.py`: chuẩn hóa [-1, 1] giống lúc train, ngưỡng lấy từ
`configs/thresholds.json` (các giá trị đang có là cấu hình hiện tại; chỉ coi là đã xác nhận khi có
CSV dò ngưỡng Val đi kèm), có cổng lọc của E4 và luật ≥ 64 pixel. Chạy máy cá nhân:

```bash
pip install streamlit matplotlib
streamlit run app.py
```

## 2. Kết hợp E2 + E4 (dùng checkpoint có sẵn, ~10 phút GPU)

E2 sàng lọc (nhạy), E4 xác nhận bằng nhánh phân loại (đặc hiệu). Mask cuối = mask của E2,
chỉ giữ khi xác suất phân loại của E4 ≥ ngưỡng.

```bash
# dò cặp ngưỡng trên VAL (10 × 16 cặp) và ghi cặp tốt nhất vào configs/thresholds.json
python evaluate_cascade.py --images $IMAGES --masks $MASKS --split val --write-config
# chấm TEST 1 lần với cặp vừa khóa
python evaluate_cascade.py --images $IMAGES --masks $MASKS --split test
```

Kết quả: `results/cascade_e2_e4/threshold_tuning.csv`, `final_test_metrics.csv`.
Nhớ tải `configs/thresholds.json` mới về và commit, demo sẽ tự dùng cặp đã dò.

## 3. Nâng cấp mô hình (cần train lại)

**E4b — ResNet18 chỉ phân đoạn** (bỏ nhánh phân loại, để biết nhánh phân loại có thật sự giúp không):

```bash
python train.py --model resnet18_seg --images $IMAGES --masks $MASKS --epochs 15 --batch-size 8
python evaluate_thresholds.py --images $IMAGES --masks $MASKS --checkpoint checkpoints/e4b_resnet18_seg_best.pt
python evaluate_test.py --images $IMAGES --masks $MASKS --checkpoint checkpoints/e4b_resnet18_seg_best.pt --threshold <NGƯỠNG_VAL>
```

**Ảnh 512×512** (bắt đầu với E2 vì đang nhạy nhất; batch 4 cho vừa GPU T4/P100):

```bash
python train.py --model attention_unet --image-size 512 --images $IMAGES --masks $MASKS --epochs 15 --batch-size 4
python evaluate_thresholds.py --images $IMAGES --masks $MASKS --checkpoint checkpoints/e2_attention_unet_512_best.pt
python evaluate_test.py --images $IMAGES --masks $MASKS --checkpoint checkpoints/e2_attention_unet_512_best.pt --threshold <NGƯỠNG_VAL>
```

Kích thước ảnh được lưu trong checkpoint (`image_size`), nên các script đánh giá, `test.py` và web demo
tự resize đúng. Kết quả 512 nằm ở thư mục riêng (`..._512`), không ghi đè kết quả 256.
Sau khi có ngưỡng Val, thêm vào `configs/thresholds.json`, ví dụ
`"e4b_resnet18_seg": {"seg_threshold": 0.9, "min_pred_pixels": 64}`.

## Điền vào bảng cuối kỳ

| Thí nghiệm | Lệnh lấy số Test |
|---|---|
| E1–E4 (256) | đã có từ giữa kỳ |
| E2 + E4 | `evaluate_cascade.py --split test` |
| E4b | `evaluate_test.py` với checkpoint E4b |
| E2 @512 | `evaluate_test.py` với checkpoint `_512` |
| Rò rỉ dữ liệu | `results/leakage/report.txt` |
