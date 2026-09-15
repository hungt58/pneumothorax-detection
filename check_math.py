"""
Kiểm tra xem tên file trong 2 thư mục images/ và segmentations/
có khớp nhau không (mỗi ảnh X-quang phải có đúng 1 mask cùng tên),
đồng thời thống kê luôn số ảnh CÓ BỆNH / KHÔNG BỆNH dựa trên mask.

Chạy:
    python check_match.py
"""

import os
import cv2
import numpy as np

# ==== SỬA 2 ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI MÁY BẠN ====
IMAGES_DIR = "images"
MASKS_DIR = "segmentations"


def get_filenames(folder):
    """Lấy set tên file (không phân biệt đuôi .png) trong 1 thư mục."""
    names = set()
    for f in os.listdir(folder):
        if f.lower().endswith(".png"):
            name_no_ext = os.path.splitext(f)[0]
            names.add(name_no_ext)
    return names


def check_match():
    image_names = get_filenames(IMAGES_DIR)
    mask_names = get_filenames(MASKS_DIR)

    missing_masks = sorted(image_names - mask_names)
    missing_images = sorted(mask_names - image_names)
    matched = sorted(image_names & mask_names)

    return matched, missing_masks, missing_images


def check_statistics(matched_names):
    total = len(matched_names)
    n_positive = 0
    n_negative = 0
    unreadable = []
    positive_list = []
    negative_list = []

    for name in matched_names:
        mask_path = os.path.join(MASKS_DIR, name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            unreadable.append(name)
            continue

        if np.max(mask) > 0:
            n_positive += 1
            positive_list.append(name)
        else:
            n_negative += 1
            negative_list.append(name)

    return n_positive, n_negative, unreadable, positive_list, negative_list


def save_report(matched, missing_masks, missing_images,
                 n_positive, n_negative, unreadable,
                 positive_list, negative_list):
    total = len(matched)
    with open("checkdata/check_match_report.txt", "w", encoding="utf-8") as f:
        f.write("===== KIEM TRA KHOP FILE =====\n")
        f.write(f"So cap khop: {len(matched)}\n")
        f.write(f"Anh thieu mask: {len(missing_masks)}\n")
        f.write("\n".join(missing_masks))
        f.write(f"\n\nMask thieu anh: {len(missing_images)}\n")
        f.write("\n".join(missing_images))

        f.write("\n\n===== THONG KE CO BENH / KHONG BENH =====\n")
        f.write(f"Tong so anh: {total}\n")
        if total > 0:
            f.write(f"Co benh: {n_positive} ({n_positive/total*100:.2f}%)\n")
            f.write(f"Khong benh: {n_negative} ({n_negative/total*100:.2f}%)\n")
        f.write(f"File loi: {len(unreadable)}\n\n")

        f.write("=== Danh sach anh CO BENH ===\n")
        f.write("\n".join(positive_list))

        f.write("\n\n=== Danh sach anh KHONG BENH ===\n")
        f.write("\n".join(negative_list))


def plot_chart(n_positive, n_negative):
    try:
        import matplotlib.pyplot as plt

        labels = ["Có bệnh", "Không bệnh"]
        counts = [n_positive, n_negative]

        plt.figure(figsize=(5, 4))
        plt.bar(labels, counts, color=["#e74c3c", "#2ecc71"])
        plt.title("Thống kê Có bệnh / Không bệnh")
        plt.ylabel("Số lượng ảnh")
        for i, v in enumerate(counts):
            plt.text(i, v + max(counts) * 0.01, str(v), ha="center")
        plt.tight_layout()
        plt.savefig("checkdata/disease_statistics_chart.png")
    except ImportError:
        pass


def main():
    matched, missing_masks, missing_images = check_match()
    n_positive, n_negative, unreadable, positive_list, negative_list = check_statistics(matched)
    save_report(matched, missing_masks, missing_images,
                n_positive, n_negative, unreadable,
                positive_list, negative_list)
    plot_chart(n_positive, n_negative)


if __name__ == "__main__":
    main()