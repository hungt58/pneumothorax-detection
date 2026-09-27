"""Audit image-mask matching and train augmentation for SIIM PTX pipeline."""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from build_dataset import build_file_list, split_data, get_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/data_audit"))
    p.add_argument("--num-positive", type=int, default=12)
    p.add_argument("--augmentations-per-image", type=int, default=2)
    return p.parse_args()


def read_gray(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise RuntimeError(f"Cannot read: {path}")
    return x


def bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    a = parse_args()
    random.seed(42)
    np.random.seed(42)

    records = build_file_list(str(a.images), str(a.masks))
    train, val, test = split_data(records)

    print(f"Pairs: {len(records)} | train: {len(train)} | val: {len(val)} | test: {len(test)}")

    # Basic pairing audit.
    bad_stems = []
    unreadable = []
    for r in records:
        ip, mp = Path(r["ImagePath"]), Path(r["MaskPath"])
        if ip.stem != mp.stem:
            bad_stems.append((ip.name, mp.name))
        if not ip.exists() or not mp.exists():
            unreadable.append((str(ip), str(mp)))

    print("Stem mismatches:", len(bad_stems))
    print("Missing paths:", len(unreadable))
    if bad_stems[:5]:
        print("Examples:", bad_stems[:5])

    positives = [r for r in train if int(r["HasDisease"]) == 1]
    negatives = [r for r in train if int(r["HasDisease"]) == 0]
    print(f"Train positive images: {len(positives)} | negative images: {len(negatives)}")

    # Deterministic spread across positive examples.
    random.shuffle(positives)
    chosen = positives[:min(a.num_positive, len(positives))]

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train_tf = get_transforms("train")

    summary = []

    for idx, r in enumerate(chosen, 1):
        image = read_gray(r["ImagePath"])
        mask = (read_gray(r["MaskPath"]) > 0).astype(np.uint8)

        orig_box = bbox(mask)
        orig_pixels = int(mask.sum())
        summary.append((Path(r["ImagePath"]).name, orig_pixels, orig_box))

        # Original image + original mask + original overlay.
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(image, cmap="gray")
        ax.set_title("Original X-ray")
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Original mask | px={orig_pixels}")
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(image, cmap="gray")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=.45)
        ax.set_title(f"Original overlay | bbox={orig_box}")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(out / f"{idx:02d}_original.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Show transformed image/mask pairs exactly through the training transform.
        for j in range(1, a.augmentations_per_image + 1):
            transformed = train_tf(image=image, mask=mask)
            aug_img = transformed["image"]
            aug_mask = transformed["mask"]

            # tensors -> numpy
            if hasattr(aug_img, "cpu"):
                aug_img = aug_img.cpu().numpy()
            if hasattr(aug_mask, "cpu"):
                aug_mask = aug_mask.cpu().numpy()

            if aug_img.ndim == 3:
                aug_img = aug_img[0]
            if aug_mask.ndim == 3:
                aug_mask = aug_mask[0]

            aug_mask = (aug_mask > 0.5).astype(np.uint8)
            display_img = np.clip(aug_img * 0.5 + 0.5, 0, 1)

            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(display_img, cmap="gray")
            ax.set_title("Augmented X-ray")
            ax.axis("off")

            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(aug_mask, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Augmented mask | px={int(aug_mask.sum())}")
            ax.axis("off")

            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(display_img, cmap="gray")
            ax.imshow(np.ma.masked_where(aug_mask == 0, aug_mask),
                      cmap="autumn", alpha=.45)
            ax.set_title(f"Augmented overlay | bbox={bbox(aug_mask)}")
            ax.axis("off")

            fig.tight_layout()
            fig.savefig(out / f"{idx:02d}_aug{j}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print("\nSelected positive masks:")
    for name, pixels, box in summary:
        print(f"{name} | mask_pixels={pixels} | bbox={box}")

    print(f"\nSaved audit images to: {out}")
    print("Inspect *_original.png first, then matching *_aug1.png / *_aug2.png.")
    print("If original overlays are wrong -> pairing/data issue.")
    print("If originals are right but augmented overlays separate -> augmentation issue.")
    print("If both stay aligned -> data pipeline is likely not the cause.")


if __name__ == "__main__":
    main()
