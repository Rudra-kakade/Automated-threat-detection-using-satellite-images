"""
Prediction Visualisation for Satellite OBB Detection
======================================================
Renders OBB predictions overlaid on raw tiles.
Shows predicted OBBs in red with class labels and confidence.

Blueprint v2, Section 7.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship", "aircraft", "vehicle", "storage-tank"]

# Color palette for each class (BGR for OpenCV)
CLASS_COLORS = {
    0: (220, 60, 60),     # ship — red
    1: (60, 180, 60),     # aircraft — green
    2: (60, 60, 220),     # vehicle — blue
    3: (220, 180, 60),    # storage-tank — cyan
}


def visualise_predictions(
    image_path: str,
    model_path: str,
    conf: float = 0.3,
    out_path: str = "prediction_overlay.png",
    gt_label_path: str = None,
) -> str:
    """
    Renders OBB predictions overlaid on the raw tile.
    Optionally shows GT boxes in green alongside predicted OBBs.

    Args:
        image_path: Path to the input image/tile.
        model_path: Path to trained best.pt.
        conf: Minimum confidence threshold.
        out_path: Output path for the overlay image.
        gt_label_path: Optional path to GT label file (YOLO OBB format).

    Returns:
        Path to the saved overlay image.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf, verbose=False)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # Draw GT boxes if provided (in green)
    if gt_label_path and Path(gt_label_path).exists():
        with open(gt_label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 9:
                    cls_id = int(parts[0])
                    pts = np.array(parts[1:9]).reshape(4, 2)
                    # Denormalize
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(int).reshape(-1, 1, 2)
                    cv2.polylines(img, [pts], isClosed=True,
                                  color=(0, 200, 0), thickness=2)
                    label = f"GT:{CLASS_NAMES[cls_id]}"
                    cv2.putText(img, label, tuple(pts[0][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 200, 0), 1)

    # Draw predictions
    for result in results:
        if result.obb is None:
            continue
        for obb in result.obb:
            pts = obb.xyxyxyxy.cpu().numpy().astype(int).reshape(-1, 1, 2)
            cls = int(obb.cls)
            conf_val = float(obb.conf)
            color = CLASS_COLORS.get(cls, (0, 0, 220))

            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            label = f"{CLASS_NAMES[cls]} {conf_val:.2f}"
            # Position label slightly above the first corner
            label_pos = (pts[0][0][0], max(pts[0][0][1] - 5, 10))
            cv2.putText(img, label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(out_path, img)
    print(f"✓ Prediction overlay saved to {out_path}")
    return out_path


def visualise_batch(
    image_dir: str,
    model_path: str,
    out_dir: str = "./vis_output",
    conf: float = 0.3,
    max_images: int = 20,
) -> list:
    """
    Visualise predictions on a batch of images.

    Args:
        image_dir: Directory containing images.
        model_path: Path to trained best.pt.
        out_dir: Output directory for overlays.
        conf: Confidence threshold.
        max_images: Maximum number of images to process.

    Returns:
        List of output image paths.
    """
    img_dir = Path(image_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in extensions
    )[:max_images]

    outputs = []
    for img_path in images:
        out_path = str(out / f"{img_path.stem}_overlay.png")
        gt_path = str(img_path).replace("/images/", "/labels/").replace(
            img_path.suffix, ".txt"
        )
        try:
            result = visualise_predictions(
                str(img_path), model_path, conf, out_path,
                gt_label_path=gt_path if Path(gt_path).exists() else None,
            )
            outputs.append(result)
        except Exception as e:
            logger.error("Failed to visualise %s: %s", img_path, e)

    print(f"\n✓ Visualised {len(outputs)} images → {out_dir}")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Visualise OBB predictions on satellite tiles"
    )
    parser.add_argument("image", type=str, help="Image path or directory")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", action="store_true",
                        help="Process directory of images")
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.batch:
        visualise_batch(args.image, args.model, args.output or "./vis_output",
                        args.conf, args.max_images)
    else:
        out = args.output or Path(args.image).stem + "_overlay.png"
        visualise_predictions(args.image, args.model, args.conf, out)


if __name__ == "__main__":
    main()
