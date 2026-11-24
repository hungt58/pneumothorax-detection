"""Streamlit web demo for PneumoAI (giao diện tiếng Việt).

Inference matches training/validation exactly (see inference.py):
[-1, 1] normalisation, validation-tuned thresholds from configs/thresholds.json,
E4 classification gate, and the >= 64-pixel image-level rule.
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference import (
    cascade_decision,
    checkpoint_image_size,
    load_threshold_config,
    predict_tensor,
    preprocess,
    resolve_thresholds,
)
from models.factory import experiment_name, load_model_from_checkpoint

CHECKPOINT_DIR = Path("checkpoints")

# tên hiển thị -> (tên checkpoint, khóa ngưỡng trong configs/thresholds.json)
MODELS = {
    "E1 – U-Net": ("e1_unet", "e1_unet"),
    "E2 – Attention U-Net": ("e2_attention_unet", "e2_attention_unet"),
    "E3 – Attention U-Net gom đặc trưng đa tầng": ("e3_attention_multiscale", "e3_attention_multiscale"),
    "E4 – ResNet18 đa nhiệm · chế độ sàng lọc": ("e4_resnet18_multitask", "e4_resnet18_multitask"),
    "E4 – ResNet18 đa nhiệm · chế độ đặc hiệu cao": ("e4_resnet18_multitask", "k_e4"),
    "E4b – ResNet18 chỉ phân đoạn": ("e4b_resnet18_seg", "e4b_resnet18_seg"),
    "E2 – Attention U-Net, ảnh 512 × 512": ("e2_attention_unet_512", "e2_attention_unet_512"),
    "E5 – ResNet34 đa nhiệm, ảnh 512 × 512": ("e5_resnet34_multitask_512", "e5_resnet34_multitask_512"),
}
CASCADE = "Tổ hợp E2 + E4 – E2 sàng lọc, E4 xác nhận"
MODE_NOTE = {
    "E4 – ResNet18 đa nhiệm · chế độ sàng lọc": "Ưu tiên bắt bệnh: độ nhạy 81,0%, độ đặc hiệu 84,0% trên tập kiểm tra.",
    "E4 – ResNet18 đa nhiệm · chế độ đặc hiệu cao": "Ưu tiên tránh báo nhầm: lật ngang khi dự đoán, độ đặc hiệu 96,9%, "
                                                   "điểm Dice kiểu Kaggle 0,800; độ nhạy giảm còn 43,4%.",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(experiment: str) -> Path:
    return CHECKPOINT_DIR / f"{experiment}_best.pt"


@st.cache_resource
def load_model(experiment: str, config_key: str | None = None):
    path = checkpoint_path(experiment)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp trọng số: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = get_device()
    model = load_model_from_checkpoint(checkpoint).to(device).eval()
    key = config_key or checkpoint.get("experiment") or experiment_name(
        checkpoint.get("model_name", "unet"), checkpoint_image_size(checkpoint))
    meta = {
        "image_size": checkpoint_image_size(checkpoint),
        "thresholds": resolve_thresholds(checkpoint, key),
    }
    return model, device, meta


def overlay_figure(image: np.ndarray, probabilities: np.ndarray, mask: np.ndarray, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Ảnh X-quang đầu vào")
    axes[1].imshow(probabilities, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Bản đồ xác suất")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(np.ma.masked_where(~mask, mask), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def mask_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()


def show_result(image, probabilities, mask, positive, details: dict):
    if not positive:                       # vùng nhỏ hơn diện tích tối thiểu bị loại, giống lúc đánh giá
        mask = np.zeros_like(mask)
    if positive:
        st.error(f"**Nghi tràn khí màng phổi** – vùng khoanh {int(mask.sum())} điểm ảnh")
    else:
        st.success("**Không phát hiện tràn khí màng phổi**")
    st.table({"Thông số": list(details.keys()), "Giá trị": [str(v) for v in details.values()]})
    fig = overlay_figure(image, probabilities, mask, "Vùng khoanh cuối cùng")
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Tải mặt nạ dự đoán", data=mask_png(mask),
                       file_name="mat_na_du_doan.png", mime="image/png")


st.set_page_config(page_title="PneumoAI", page_icon="🩻", layout="wide")
st.title("🩻 PneumoAI – Phát hiện và phân đoạn tràn khí màng phổi")
st.write("Tải ảnh X-quang ngực, chọn mô hình và chế độ vận hành để xem vùng nghi tràn khí.")

with st.sidebar:
    st.header("Mô hình")
    choice = st.selectbox("Chọn mô hình", [*MODELS.keys(), CASCADE])
    if choice == CASCADE:
        st.caption("E2 khoanh vùng nghi ngờ với độ nhạy cao; nhánh phân loại của E4 phải xác nhận "
                   "thì vùng khoanh của E2 mới được giữ lại.")
    else:
        if not checkpoint_path(MODELS[choice][0]).is_file():
            st.warning(f"Thiếu tệp trọng số: {checkpoint_path(MODELS[choice][0])}")
        if choice in MODE_NOTE:
            st.info(MODE_NOTE[choice])
    st.caption("Mọi ngưỡng được dò trên tập kiểm định và lưu trong configs/thresholds.json; "
               "tập kiểm tra không được dùng để chọn ngưỡng.")

uploaded = st.file_uploader("Tải ảnh X-quang ngực", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption=uploaded.name, width=400)
    if st.button("Dự đoán", type="primary"):
        try:
            with st.spinner("Đang nạp mô hình và dự đoán..."):
                if choice == CASCADE:
                    config = load_threshold_config().get("cascade_e2_e4", {})
                    e2, device, e2_meta = load_model("e2_attention_unet", "e2_attention_unet")
                    e4, _, e4_meta = load_model("e4_resnet18_multitask", "e4_resnet18_multitask")
                    e2_thr = float(config.get("e2_seg_threshold", e2_meta["thresholds"]["seg_threshold"]))
                    e4_thr = float(config.get("e4_cls_threshold", e4_meta["thresholds"]["cls_threshold"] or 0.5))
                    min_px = int(config.get("min_pred_pixels", 64))
                    shown, tensor = preprocess(image, e2_meta["image_size"])
                    screen = predict_tensor(e2, tensor, e2_thr, None, min_px)
                    _, tensor4 = preprocess(image, e4_meta["image_size"])
                    confirm = predict_tensor(e4, tensor4, 0.5, None, min_px)
                    mask, positive = cascade_decision(screen, confirm.cls_probability, e4_thr)
                    details = {
                        "Số điểm ảnh E2 khoanh": screen.positive_pixels,
                        "Kết luận của E2": "dương tính" if screen.is_positive else "âm tính",
                        "Xác suất phân loại của E4": f"{confirm.cls_probability:.2%}",
                        "Ngưỡng xác nhận của E4": f"{e4_thr:.2f}",
                        "Ngưỡng điểm ảnh của E2": f"{e2_thr:.2f}",
                    }
                    probabilities = screen.probabilities
                else:
                    experiment, config_key = MODELS[choice]
                    model, device, meta = load_model(experiment, config_key)
                    thr = meta["thresholds"]
                    shown, tensor = preprocess(image, meta["image_size"])
                    pred = predict_tensor(model, tensor, thr["seg_threshold"], thr["cls_threshold"],
                                          thr["min_pred_pixels"], tta=thr.get("tta", False))
                    mask, positive, probabilities = pred.mask, pred.is_positive, pred.probabilities
                    details = {
                        "Ngưỡng điểm ảnh": f"{thr['seg_threshold']:.2f}",
                        "Diện tích tối thiểu": f"{thr['min_pred_pixels']} điểm ảnh",
                        "Số điểm ảnh được khoanh": pred.positive_pixels,
                        "Kích thước đầu vào": f"{meta['image_size']} × {meta['image_size']}",
                        "Lật ngang khi dự đoán": "có" if thr.get("tta") else "không",
                    }
                    if pred.cls_probability is not None:
                        details["Xác suất phân loại"] = f"{pred.cls_probability:.2%}"
                        details["Ngưỡng cổng lọc"] = f"{thr['cls_threshold']:.2f}"
            st.caption(f"Thiết bị tính toán: {device}")
            show_result(shown, probabilities, mask, positive, details)
            st.warning("Công cụ phục vụ nghiên cứu, không thay thế chẩn đoán của bác sĩ.")
        except Exception as exc:
            st.error(f"Dự đoán thất bại: {exc}")
