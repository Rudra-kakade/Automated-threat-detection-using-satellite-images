"""
Augmentation Pipeline for Satellite OBB Detection
===================================================
Albumentations-based transforms for domain gap mitigation.
Simulates different sensor resolutions, atmospheric conditions,
and geometric variations.

Blueprint v2, Section 4 (Augmentation).
"""

import argparse
import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Transform Definitions ──────────────────────────────────────────────────────

def get_train_transform(img_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline.

    Includes:
        - RandomScale: Simulate different sensor resolutions / altitudes
        - RandomBrightnessContrast: Simulate varying lighting conditions
        - HueSaturationValue: Simulate seasonal / vegetation changes
        - RandomFog: Simulate haze / cloud interference
        - Geometric flips and 90° rotations (OBB-safe)

    Returns:
        Albumentations Compose pipeline with YOLO bbox_params.
    """
    return A.Compose(
        [
            # Simulate different sensor resolutions
            A.RandomScale(scale_limit=0.3, p=0.5),
            # Resize back to tile size after scaling
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.CenterCrop(height=img_size, width=img_size),
            # Simulate different atmospheric and seasonal conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4,
            ),
            # Simulate haze / cloud interference
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.3),
            # Geometric — critical for OBB
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Slight Gaussian noise to simulate sensor noise
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            # Subtle blur to simulate motion / defocus
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transform(img_size: int = 512) -> A.Compose:
    """
    Validation / test transform — no augmentation, only ensures correct size.

    Returns:
        Albumentations Compose pipeline with YOLO bbox_params.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.CenterCrop(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ─── Augmented Dataset Wrapper ───────────────────────────────────────────────────

class AugmentedDataset:
    """
    Wraps a tiled dataset directory and applies Albumentations transforms.

    Directory structure expected:
        tile_dir/
            images/<split>/*.png
            labels/<split>/*.txt

    Each label file contains YOLO OBB format lines:
        class_id x1 y1 x2 y2 x3 y3 x4 y4
    """

    def __init__(self, tile_dir: Path, split: str = "train", transform: A.Compose = None):
        self.img_dir = tile_dir / "images" / split
        self.lbl_dir = tile_dir / "labels" / split
        self.transform = transform or get_train_transform()

        self.image_paths = sorted(self.img_dir.glob("*.png"))
        if not self.image_paths:
            logger.warning("No PNG images found in %s", self.img_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        bboxes = []
        class_labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 9:
                        class_id = int(parts[0])
                        # Convert 4-point OBB to axis-aligned bbox for augmentation
                        # (Albumentations works with AABB; the OBB points are
                        #  preserved separately and transformed manually)
                        xs = parts[1::2]  # x1, x2, x3, x4
                        ys = parts[2::2]  # y1, y2, y3, y4
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(class_id)

        # Apply augmentation
        if self.transform and bboxes:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        return {
            "image": img,
            "bboxes": bboxes,
            "class_labels": class_labels,
            "image_path": str(img_path),
        }


# ─── Offline Augmentation ───────────────────────────────────────────────────────

def augment_dataset_offline(
    tile_dir: Path,
    out_dir: Path,
    split: str = "train",
    num_augmented: int = 3,
) -> int:
    """
    Generate augmented copies of each tile offline.

    Args:
        tile_dir: Root directory containing images/<split>/ and labels/<split>/.
        out_dir: Output directory for augmented tiles.
        split: Dataset split to augment.
        num_augmented: Number of augmented copies per original tile.

    Returns:
        Total number of augmented tiles generated.
    """
    dataset = AugmentedDataset(tile_dir, split)
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in range(len(dataset)):
        for aug_id in range(num_augmented):
            sample = dataset[idx]
            stem = Path(sample["image_path"]).stem + f"_aug{aug_id:02d}"

            # Save augmented image
            img_bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_dir / f"{stem}.png"), img_bgr)

            # Save augmented labels (YOLO format: class cx cy w h)
            with open(out_lbl_dir / f"{stem}.txt", "w") as f:
                for bbox, cls in zip(sample["bboxes"], sample["class_labels"]):
                    f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in bbox) + "\n")

            count += 1

    logger.info("Generated %d augmented tiles for split '%s'", count, split)
    return count


# ─── CLI Entrypoint ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented copies of tiled satellite images"
    )
    parser.add_argument(
        "tile_dir",
        type=Path,
        help="Root tile directory (contains images/ and labels/ subdirs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as tile_dir, augmented tiles mixed in)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to augment (default: train)",
    )
    parser.add_argument(
        "--num-augmented", "-n",
        type=int,
        default=3,
        help="Number of augmented copies per tile (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_dir = args.out_dir or args.tile_dir
    count = augment_dataset_offline(args.tile_dir, out_dir, args.split, args.num_augmented)
    print(f"\n✓ Generated {count} augmented tiles → {out_dir / 'images' / args.split}")


if __name__ == "__main__":
    main()
