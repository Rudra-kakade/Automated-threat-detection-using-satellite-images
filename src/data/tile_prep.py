"""
Tile Preparation Pipeline for Satellite OBB Detection
======================================================
Slices large satellite images into 512×512 tiles with 128px overlap.
Transforms OBB annotations into per-tile YOLO OBB format.

Blueprint v2, Section 4.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

# ─── Constants ──────────────────────────────────────────────────────────────────
TILE_SIZE = 512
OVERLAP = 128  # 25% of 512
MIN_VISIBILITY = 0.30

# Class name → class ID mapping (must match dataset.yaml)
CLASS_NAME_TO_ID = {
    "ship": 0,
    "aircraft": 1,
    "vehicle": 2,
    "storage-tank": 3,
    # DOTA-specific aliases
    "small-vehicle": 2,
    "large-vehicle": 2,
    "plane": 1,
    "harbor": -1,           # ignored
    "bridge": -1,
    "helicopter": 1,
    "roundabout": -1,
    "soccer-ball-field": -1,
    "swimming-pool": -1,
    "ground-track-field": -1,
    "baseball-diamond": -1,
    "tennis-court": -1,
    "basketball-court": -1,
    "container-crane": -1,
    "helipad": -1,
}

logger = logging.getLogger(__name__)


# ─── Annotation I/O ────────────────────────────────────────────────────────────

def load_obb_annotations(image_path: Path) -> list:
    """
    Load OBB annotations for a source image.

    Expects a DOTA-format .txt file alongside the image:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty

    Returns:
        List of (class_id, x1, y1, x2, y2, x3, y3, x4, y4) tuples.
        Annotations for classes not in CLASS_NAME_TO_ID (or mapped to -1)
        are skipped.
    """
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        # Try parallel labels/ directory
        label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")

    if not label_path.exists():
        logger.warning("No annotation file found for %s", image_path)
        return []

    annotations = []
    with open(label_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                # Skip DOTA header lines (first two lines are metadata)
                continue

            try:
                coords = list(map(float, parts[:8]))
                class_name = parts[8].lower()
            except (ValueError, IndexError):
                logger.warning("Skipping malformed line %d in %s", line_no, label_path)
                continue

            class_id = CLASS_NAME_TO_ID.get(class_name)
            if class_id is None or class_id < 0:
                continue  # Class not in our target set

            annotations.append((class_id, *coords))

    return annotations


def save_tile_image(tile_data: np.ndarray, out_path: Path) -> None:
    """
    Save a tile (C, H, W) numpy array as a PNG image.

    Args:
        tile_data: Shape (C, H, W), dtype uint8 or uint16.
        out_path: Output PNG path.
    """
    if tile_data.ndim == 3 and tile_data.shape[0] in (1, 3, 4):
        # (C, H, W) → (H, W, C)
        img = np.transpose(tile_data, (1, 2, 0))
    else:
        img = tile_data

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() > 255:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Convert RGB → BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(out_path), img)


# ─── Core Tiling ────────────────────────────────────────────────────────────────

def tile_image(src_path: Path, out_dir: Path, split: str = "train") -> list:
    """
    Slice a large satellite image into 512×512 tiles with 128px overlap.
    Transforms OBB annotations into per-tile YOLO OBB format.

    Args:
        src_path: Path to the source satellite image (GeoTIFF, PNG, etc.).
        out_dir: Root output directory (will create images/<split>/ and labels/<split>/).
        split: Dataset split name ("train", "val", "test").

    Returns:
        List of (tile_img_path, tile_lbl_path, (col_offset, row_offset)) tuples.
    """
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stride = TILE_SIZE - OVERLAP
    records = []

    # Read image — try rasterio first (supports GeoTIFF), fallback to OpenCV
    try:
        with rasterio.open(src_path) as src:
            W, H = src.width, src.height
            data = src.read()  # shape: (C, H, W)
    except rasterio.errors.RasterioIOError:
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {src_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = np.transpose(img_rgb, (2, 0, 1))  # (H, W, C) → (C, H, W)
        H, W = img_rgb.shape[:2]

    annotations = load_obb_annotations(src_path)

    tile_id = 0
    for row_start in range(0, H - TILE_SIZE + 1, stride):
        for col_start in range(0, W - TILE_SIZE + 1, stride):
            row_end = row_start + TILE_SIZE
            col_end = col_start + TILE_SIZE

            tile_region = shapely_box(col_start, row_start, col_end, row_end)
            tile_img = data[:, row_start:row_end, col_start:col_end]

            tile_labels = []
            for ann in annotations:
                class_id = ann[0]
                pts = np.array(ann[1:]).reshape(4, 2)
                obj_poly = Polygon(pts)

                if not obj_poly.is_valid:
                    obj_poly = obj_poly.buffer(0)

                intersection = obj_poly.intersection(tile_region)
                visibility = intersection.area / obj_poly.area if obj_poly.area > 0 else 0

                if visibility < MIN_VISIBILITY:
                    continue  # Skip near-invisible truncated objects

                # Transform global coords → tile-local coords → YOLO normalized
                local_pts = pts - np.array([col_start, row_start])
                norm_pts = local_pts / TILE_SIZE
                norm_pts = np.clip(norm_pts, 0.0, 1.0)

                # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                tile_labels.append((class_id, *norm_pts.flatten()))

            stem = f"{src_path.stem}_tile_{tile_id:05d}"
            img_path = out_img_dir / f"{stem}.png"
            lbl_path = out_lbl_dir / f"{stem}.txt"

            save_tile_image(tile_img, img_path)
            with open(lbl_path, "w") as f:
                for lbl in tile_labels:
                    f.write(" ".join(map(str, lbl)) + "\n")

            records.append((img_path, lbl_path, (col_start, row_start)))
            tile_id += 1

    logger.info(
        "Tiled %s → %d tiles (%d with annotations)",
        src_path.name,
        tile_id,
        sum(1 for _, lp, _ in records if lp.stat().st_size > 0),
    )
    return records


# ─── Batch Processing ───────────────────────────────────────────────────────────

def tile_dataset(
    src_dir: Path,
    out_dir: Path,
    split: str = "train",
    extensions: tuple = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> list:
    """
    Tile all images in a source directory.

    Args:
        src_dir: Directory containing large satellite images.
        out_dir: Root output directory for tiles.
        split: Dataset split name.
        extensions: Accepted image file extensions.

    Returns:
        Aggregated list of all tile records.
    """
    all_records = []
    image_files = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_files:
        logger.warning("No images found in %s with extensions %s", src_dir, extensions)
        return []

    logger.info("Processing %d images from %s", len(image_files), src_dir)

    for img_path in image_files:
        try:
            records = tile_image(img_path, out_dir, split)
            all_records.extend(records)
        except Exception as e:
            logger.error("Failed to tile %s: %s", img_path, e)
            continue

    logger.info("Total: %d tiles generated for split '%s'", len(all_records), split)
    return all_records


# ─── CLI Entrypoint ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tile satellite images for YOLOv8-OBB training"
    )
    parser.add_argument(
        "src_dir",
        type=Path,
        help="Directory containing source satellite images with DOTA-format annotations",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/satellite_obb"),
        help="Output directory for tiled dataset (default: ./data/satellite_obb)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    records = tile_dataset(args.src_dir, args.out_dir, args.split)
    print(f"\n✓ Generated {len(records)} tiles → {args.out_dir / 'images' / args.split}")


if __name__ == "__main__":
    main()
