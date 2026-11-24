# Pneumothorax Detection with Deep Learning

A deep learning project for pneumothorax segmentation from chest X-ray images using four PyTorch models. The project provides a local web interface built with Streamlit for running predictions and visualizing results.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used as a substitute for professional medical diagnosis.

## Kết quả cuối kỳ (tập Test 1.602 ảnh, ngưỡng chọn trên Val, chấm Test 1 lần)

| Mô hình | Dice+ | Độ nhạy | Đặc hiệu | Báo nhầm |
|---|---|---|---|---|
| E1 U-Net | 35,4% | 80,1% | 64,3% | 444 |
| E2 Attention U-Net | 32,6% | 82,1% | 63,9% | 449 |
| E3 Attention + đa tầng | 35,8% | 85,2% | 57,3% | 531 |
| **E4 ResNet18 đa nhiệm** | 36,5% | 81,0% | **84,0%** | **199** |
| E2 + E4 kết hợp | 33,0% | 81,5% | 81,2% | 234 |
| E4b ResNet18 chỉ phân đoạn | 36,9% | 85,2% | 70,2% | 371 |
| E2 @ 512 | **38,3%** | **89,1%** | 58,6% | 515 |

**Benchmark (điểm Dice trên mọi ảnh, cách chấm của Kaggle SIIM-ACR):** E4 + hậu xử lý kiểu Kaggle
(`tune_kaggle.py`: TTA lật ngang, ngưỡng pixel 0,65, ngưỡng phân loại 0,85, bỏ vùng < 256 pixel) đạt **0,800**,
vượt U-Net gốc (0,795, Frontiers in Radiology 2024) và mốc đoán toàn "không bệnh" (0,777); top Kaggle ~0,84–0,87.

Chi tiết: `results/final_summary.md`, `results/k_e4/kaggle_summary.json`.
Checkpoint (.pt) không có trong repo vì nặng – tải từ Output của notebook Kaggle và đặt vào `checkpoints/`.

## Features

- Upload chest X-ray images through a web interface.
- Select one of four trained models:
  - U-Net
  - Attention U-Net
  - Attention Multi-scale U-Net
  - ResNet18 Multi-task
- Visualize the input X-ray, predicted probability map, and segmentation overlay.
- Display classification probability when supported by the selected model.
- Download the predicted segmentation mask as a PNG image.
- Run locally on your computer using Streamlit.

## Project Structure

```text
SIIM-ACR Pneumothorax Training/
├── app.py
├── models/
│   └── factory.py
├── checkpoints/
│   ├── e1_unet_best.pt
│   ├── e2_attention_unet_best.pt
│   ├── e3_attention_multiscale_best.pt
│   └── e4_resnet18_multitask_best.pt
├── predictions/
├── requirements.txt
└── README.md
```

The structure above is an example. Keep any additional source files and configuration files required by your project.

## Requirements

- Python 3.10 recommended.
- Git.
- PyTorch.
- Streamlit.
- NumPy, Pillow, and Matplotlib.
- The trained model checkpoints.

## Installation

### 1. Clone the repository

Replace `<REPOSITORY_URL>` with the actual GitHub repository URL.

```bash
git clone <REPOSITORY_URL>
cd "SIIM-ACR Pneumothorax Training"
```

If Git clones the repository into a differently named directory, navigate to that directory instead.

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

If the repository includes a `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

If dependencies are not fully specified in that file, install the packages required by the application:

```bash
python -m pip install streamlit numpy pillow matplotlib
```

Install a compatible PyTorch version by following the official instructions:

https://pytorch.org/get-started/locally/

Additional dependencies may be required by the model implementations under `models/`.

### 4. Prepare model checkpoints

The application expects the following files inside the `checkpoints/` directory:

| Model | Checkpoint |
|---|---|
| U-Net | `e1_unet_best.pt` |
| Attention U-Net | `e2_attention_unet_best.pt` |
| Attention Multi-scale U-Net | `e3_attention_multiscale_best.pt` |
| ResNet18 Multi-task | `e4_resnet18_multitask_best.pt` |
| ResNet18 segmentation only (E4b, optional) | `e4b_resnet18_seg_best.pt` |

If the checkpoints are not included in the repository, obtain them from the project owner and place them in the expected directory.

**Important:** The application cannot run predictions for a model whose checkpoint is missing.

Verify the files:

```bash
ls checkpoints
```

## Run the Application

From the project root, with the virtual environment activated, execute:

```bash
python -m streamlit run app.py
```

Do not use `python app.py` to launch the Streamlit interface.

When the server starts, Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open this address in your web browser.

Keep the terminal running while using the application. Press `Ctrl+C` in the terminal to stop the server.

## How to Use

1. Open the application in your browser.
2. Select a model from the sidebar.
3. Upload a chest X-ray image in PNG, JPG, JPEG, or BMP format.
4. Click **Predict**.
5. View the input image, predicted probability map, and segmentation overlay.
6. Download the predicted segmentation mask as a PNG file.
7. Review the classification probability if the selected model provides classification output.
8. Choose **E2 + E4 · screen then confirm** to combine the two models.

## Prediction Pipeline

The application (and `test.py`) uses `inference.py`, which mirrors training exactly:

1. Load the selected checkpoint (its `image_size` is read from the checkpoint; default 256).
2. Convert the input image to grayscale and resize it with the same interpolation as training.
3. Normalize with mean 0.5 / std 0.5, i.e. pixel values in `[-1, 1]` (same as `build_dataset.py`).
4. Run inference and apply sigmoid to the segmentation logits.
5. Apply the model threshold from `configs/thresholds.json`. Treat checked-in values as provisional
   until the matching validation tuning CSV is available; `run_final.py` writes the tuned values.
6. For E4, clear the mask when the classification probability is below the gate.
7. Call the image positive only if at least 64 pixels remain (same rule as the reported metrics).

**E2 + E4 cascade** (sidebar option): E2 screens for suspicious images, E4's classifier must confirm
before the E2 mask is shown. The checked-in cascade thresholds are provisional (`tuned: false`);
run `evaluate_cascade.py` on validation before using it for final reporting.

## Training and Final Evaluation

Training and end-of-term experiments require the SIIM image/mask dataset and a GPU. Use
[`RUN_NUA_SAU_KAGGLE.md`](RUN_NUA_SAU_KAGGLE.md) and run
[`kaggle/PneumoAI_cuoi_ky.ipynb`](kaggle/PneumoAI_cuoi_ky.ipynb) on Kaggle. The notebook produces
`final_results.zip` with metrics, validation prediction figures, tuned thresholds, and checkpoints.
Do not report final Test metrics until the data-integrity check passes and the Val thresholds are
locked.

## Hardware

The application uses Apple's MPS backend when available; otherwise, it falls back to CPU.

On other systems, GPU support depends on the installed PyTorch build and the device-selection logic in `app.py`.

## Troubleshooting

### Checkpoint not found

Ensure the required `.pt` file exists in the `checkpoints/` directory and that its filename matches the configuration in `app.py`.

### ModuleNotFoundError

Activate the correct virtual environment and install the missing dependency. Check whether the model implementation requires additional packages.

### Streamlit ScriptRunContext warnings

Launch the application using:

```bash
python -m streamlit run app.py
```

Running `python app.py` directly does not start Streamlit correctly.

### Prediction errors

Check that the checkpoint matches the selected model architecture. Confirm that the output structure and input dimensions are compatible with the inference code.

### Port already in use

Run the application on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## Limitations

- Predictions depend on the quality and distribution of the training data.
- The preprocessing pipeline must be consistent with training.
- Segmentation masks and classification probabilities are model outputs, not confirmed clinical findings.
- Model checkpoints and datasets may need to be downloaded separately.

## License

Add the license applicable to this project before distributing or publishing the repository.
