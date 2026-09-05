"""
Data Augmentation Pipeline
===========================
Multiplies the real + synthetic dataset using Albumentations.
Includes orientation augmentation (0/90/180/270°) for microgravity invariance.

Usage:
    python data_generation/augmentation.py --input dataset/annotated --output dataset/augmented
"""

import os
import sys
import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import albumentations as A
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False
    print("[WARNING] albumentations not installed. Run: pip install albumentations")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_augmentation_pipeline() -> "A.Compose":
    """Build the augmentation pipeline for HAR training data."""
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.Perspective(scale=(0.02, 0.06), p=0.3),

        # ── Color / Brightness ────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.RandomShadow(p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),  # simulates lens haze

        # ── Noise / Blur ──────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),

        # ── Compression artifacts (edge-AI camera simulation) ─
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
    ])


def orientation_augment(image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """
    Return image rotated at 0°, 90°, 180°, 270°.
    Critical for orientation-agnostic training (microgravity).
    """
    results = [
        (image, "0deg"),
        (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "90deg"),
        (cv2.rotate(image, cv2.ROTATE_180), "180deg"),
        (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "270deg"),
    ]
    return results


def augment_yolo_dataset(input_dir: str, output_dir: str, augmentation_factor: int = 8):
    """
    Augment a YOLO-format dataset (images + labels in parallel folders).
    
    Expected structure:
        input_dir/
          images/
            step_01_frame0001.jpg
            ...
          labels/
            step_01_frame0001.txt  (YOLO format)
    """
    if not ALB_AVAILABLE:
        logger.error("albumentations not installed. Run: pip install albumentations")
        return

    images_in = Path(input_dir) / "images"
    labels_in = Path(input_dir) / "labels"
    images_out = Path(output_dir) / "images"
    labels_out = Path(output_dir) / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    transform = build_augmentation_pipeline()
    image_files = list(images_in.glob("*.jpg")) + list(images_in.glob("*.png"))

    logger.info("Augmenting %d images × %d factor + 4 orientations",
                len(image_files), augmentation_factor)

    total_out = 0
    for img_path in image_files:
        label_path = labels_in / (img_path.stem + ".txt")
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Read label (copy as-is for now — bounding boxes are approximate post-rotation)
        label_content = ""
        if label_path.exists():
            label_content = label_path.read_text()

        # ── Orientation augmentation (save all 4 rotations of original) ──
        for rot_img, rot_tag in orientation_augment(image):
            fname = f"{img_path.stem}__{rot_tag}.jpg"
            cv2.imwrite(str(images_out / fname), rot_img)
            (labels_out / (img_path.stem + f"__{rot_tag}.txt")).write_text(label_content)
            total_out += 1

        # ── Random augmentation variants ──────────────────────
        for aug_idx in range(augmentation_factor):
            try:
                augmented = transform(image=image)["image"]
                fname = f"{img_path.stem}__aug{aug_idx:02d}.jpg"
                cv2.imwrite(str(images_out / fname), augmented)
                (labels_out / (img_path.stem + f"__aug{aug_idx:02d}.txt")).write_text(label_content)
                total_out += 1
            except Exception as e:
                logger.warning("Augmentation failed for %s: %s", img_path.name, e)

    logger.info("Augmentation complete: %d output images → %s", total_out, output_dir)


def augment_frames_folder(input_dir: str, output_dir: str, augmentation_factor: int = 8):
    """
    Augment a flat folder of frames (for skeleton-sequence training data).
    No label files — just image augmentation.
    """
    if not ALB_AVAILABLE:
        logger.error("albumentations not installed.")
        return

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    transform = build_augmentation_pipeline()
    files = list(in_path.glob("*.jpg")) + list(in_path.glob("*.png"))

    logger.info("Augmenting %d frames in %s", len(files), input_dir)
    total = 0

    for fpath in files:
        image = cv2.imread(str(fpath))
        if image is None:
            continue

        # Orientation variants
        for rot_img, rot_tag in orientation_augment(image):
            fname = f"{fpath.stem}__{rot_tag}.jpg"
            cv2.imwrite(str(out_path / fname), rot_img)
            total += 1

        # Augmentation variants
        for aug_idx in range(augmentation_factor):
            try:
                aug = transform(image=image)["image"]
                fname = f"{fpath.stem}__aug{aug_idx:02d}.jpg"
                cv2.imwrite(str(out_path / fname), aug)
                total += 1
            except Exception as e:
                logger.warning("Error: %s", e)

    logger.info("Done: %d augmented frames → %s", total, output_dir)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation Pipeline")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--factor", type=int, default=8, help="Augmentation multiplier")
    parser.add_argument("--mode", choices=["yolo", "frames"], default="frames",
                        help="'yolo' for annotated dataset, 'frames' for raw frames")
    args = parser.parse_args()

    if args.mode == "yolo":
        augment_yolo_dataset(args.input, args.output, args.factor)
    else:
        augment_frames_folder(args.input, args.output, args.factor)
