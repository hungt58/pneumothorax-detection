# Pneumothorax Detection with Deep Learning

A deep learning project for pneumothorax segmentation from chest X-ray images using four PyTorch models. The project provides a local web interface built with Streamlit for running predictions and visualizing results.

> **Disclaimer:** This project is intended for research and educational purposes only. It is not a medical device and must not be used as a substitute for professional medical diagnosis.

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

## Prediction Pipeline

The application follows this workflow:

1. Load the selected model checkpoint.
2. Convert the input image to grayscale.
3. Resize the image to `256 × 256` pixels.
4. Normalize pixel values to the range `[0, 1]`.
5. Run inference using PyTorch.
6. Apply sigmoid to the segmentation logits.
7. Apply the model's saved threshold, or `0.5` if no threshold is available.
8. Generate the segmentation mask and visualization.

**Note:** The preprocessing used during inference must match the preprocessing used during training. The current pipeline assumes that all four models accept a single-channel grayscale input of size `256 × 256`.

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
