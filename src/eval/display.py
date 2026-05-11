"""
Detection Display Module — Satellite OBB Detection
=====================================================
Draws bounding boxes on inferenced images with class labels,
confidence scores, and a detection summary panel.

Supports:
  - JSON detections from SAHI inference (infer_sahi.py output)
  - Direct YOLO model inference with OBB visualization
  - Interactive display with OpenCV or headless save-to-file

Usage:
    python -m src.eval.display <image> --detections <json>
    python -m src.eval.display <image> --model <best.pt>
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Visual Style ────────────────────────────────────────────────────────────

# Extended color palette (BGR) for all DOTA classes
CLASS_COLORS_BGR = {
    "ship":          (60, 60, 220),      # red
    "aircraft":      (60, 200, 60),      # green
    "plane":         (60, 200, 60),
    "vehicle":       (220, 140, 40),     # orange-blue
    "small-vehicle": (220, 140, 40),
    "large-vehicle": (180, 100, 40),
    "storage-tank":  (40, 200, 220),     # yellow
    "harbor":        (180, 60, 180),     # purple
    "bridge":        (100, 200, 200),    # light yellow
    "helicopter":    (60, 255, 160),     # lime
    "roundabout":    (200, 200, 60),     # cyan
    "container-crane": (80, 80, 200),
    "swimming-pool": (255, 160, 60),     # light blue
}

DEFAULT_COLOR = (200, 200, 200)  # gray fallback

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.50
FONT_SCALE_PANEL = 0.45
FONT_THICKNESS = 1
BOX_THICKNESS = 2


def _get_color(class_name: str) -> tuple:
    """Get BGR color for a class name."""
    return CLASS_COLORS_BGR.get(class_name.lower(), DEFAULT_COLOR)


def _draw_label(img, text, origin, color, bg_alpha=0.7):
    """Draw a text label with a semi-transparent background."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
    x, y = int(origin[0]), int(origin[1])
    # Background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - th - 6), (x + tw + 6, y + 2), color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    # Text (white on colored bg)
    cv2.putText(img, text, (x + 3, y - 3), FONT, FONT_SCALE_LABEL,
                (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def _draw_summary_panel(img, detections, panel_width=260):
    """Draw a detection summary panel on the right side of the image."""
    h, w = img.shape[:2]

    # Count detections per class
    class_counts = {}
    for det in detections:
        name = det["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    # Panel background
    panel_h = max(180, 50 + len(class_counts) * 30 + 60)
    panel_x = w - panel_width - 15
    panel_y = 15
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    # Title
    cv2.putText(img, "DETECTION SUMMARY", (panel_x + 12, panel_y + 28),
                FONT, FONT_SCALE_PANEL + 0.05, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (panel_x + 12, panel_y + 35),
             (panel_x + panel_width - 12, panel_y + 35), (100, 100, 100), 1)

    # Total count
    y_offset = panel_y + 55
    cv2.putText(img, f"Total: {len(detections)}", (panel_x + 12, y_offset),
                FONT, FONT_SCALE_PANEL, (200, 200, 200), 1, cv2.LINE_AA)
    y_offset += 30

    # Per-class counts with colored dots
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        color = _get_color(cls_name)
        cv2.circle(img, (panel_x + 20, y_offset - 5), 6, color, -1)
        cv2.putText(img, f"{cls_name}: {count}", (panel_x + 34, y_offset),
                    FONT, FONT_SCALE_PANEL, (220, 220, 220), 1, cv2.LINE_AA)
        y_offset += 28

    # Confidence range
    if detections:
        confs = [d["confidence"] for d in detections]
        y_offset += 5
        cv2.putText(img, f"Conf: {min(confs):.2f} - {max(confs):.2f}",
                    (panel_x + 12, y_offset),
                    FONT, FONT_SCALE_PANEL, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Core Display Functions ─────────────────────────────────────────────────

def display_detections_from_json(
    image_path: str,
    json_path: str,
    output_path: str = None,
    show: bool = True,
    conf_threshold: float = 0.0,
    show_panel: bool = True,
) -> str:
    """
    Draw bounding boxes on an image from a SAHI JSON detections file.

    Args:
        image_path: Path to the source image.
        json_path: Path to detections JSON (from infer_sahi.py).
        output_path: Where to save the annotated image. Auto-generated if None.
        show: Whether to display the image in an OpenCV window.
        conf_threshold: Only draw detections above this confidence.
        show_panel: Whether to draw the summary panel.

    Returns:
        Path to the saved annotated image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    with open(json_path) as f:
        detections = json.load(f)

    # Filter by confidence
    detections = [d for d in detections if d["confidence"] >= conf_threshold]

    for det in detections:
        bbox = det["bbox_xyxy"]
        x1, y1, x2, y2 = map(int, bbox)
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = _get_color(cls_name)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw corner accents (small L-shaped marks at corners)
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, BOX_THICKNESS + 1)
            cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, BOX_THICKNESS + 1)

        # Draw centroid dot
        cx, cy = det.get("centroid", [(x1 + x2) / 2, (y1 + y2) / 2])
        cv2.circle(img, (int(cx), int(cy)), 3, color, -1)

        # Label
        label = f"{cls_name} {conf:.2f}"
        _draw_label(img, label, (x1, y1 - 2), color)

    # Summary panel
    if show_panel and detections:
        _draw_summary_panel(img, detections)

    # Save
    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_display.png"
    cv2.imwrite(output_path, img)
    print(f"✓ Display image saved to {output_path}")

    # Show
    if show:
        try:
            cv2.imshow("Satellite Detection Results", img)
            print("  Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            logger.warning("Cannot open display window (headless mode).")

    return output_path


def display_detections_from_model(
    image_path: str,
    model_path: str,
    output_path: str = None,
    show: bool = True,
    conf_threshold: float = 0.25,
    show_panel: bool = True,
) -> str:
    """
    Run YOLO OBB inference and draw OBB polygons on the image.

    Args:
        image_path: Path to the source image.
        model_path: Path to trained best.pt.
        output_path: Where to save. Auto-generated if None.
        show: Whether to display interactively.
        conf_threshold: Minimum confidence.
        show_panel: Whether to draw summary panel.

    Returns:
        Path to the saved annotated image.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)
    class_names = model.names  # {0: 'ship', 1: 'aircraft', ...}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detections = []

    for result in results:
        if result.obb is None:
            continue
        for obb in result.obb:
            cls_id = int(obb.cls)
            conf_val = float(obb.conf)
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            color = _get_color(cls_name)

            # OBB polygon points
            pts = obb.xyxyxyxy.cpu().numpy().astype(int).reshape(-1, 1, 2)

            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

            # Draw polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=BOX_THICKNESS)

            # Centroid
            cx = int(pts[:, 0, 0].mean())
            cy = int(pts[:, 0, 1].mean())
            cv2.circle(img, (cx, cy), 3, color, -1)

            # Label at topmost point
            top_idx = pts[:, 0, 1].argmin()
            label_pos = (pts[top_idx][0][0], pts[top_idx][0][1] - 2)
            _draw_label(img, f"{cls_name} {conf_val:.2f}", label_pos, color)

            # Collect for panel
            bbox = [int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min()),
                    int(pts[:, 0, 0].max()), int(pts[:, 0, 1].max())]
            detections.append({
                "class_name": cls_name,
                "confidence": conf_val,
                "bbox_xyxy": bbox,
                "centroid": [cx, cy],
            })

    if show_panel and detections:
        _draw_summary_panel(img, detections)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_obb_display.png"
    cv2.imwrite(output_path, img)
    print(f"✓ OBB display image saved to {output_path}")

    if show:
        try:
            cv2.imshow("Satellite OBB Detection Results", img)
            print("  Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            logger.warning("Cannot open display window (headless mode).")

    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Display satellite detection results with bounding boxes"
    )
    parser.add_argument("image", type=str, help="Path to the source image")
    parser.add_argument("--detections", type=str, default=None,
                        help="Path to SAHI detections JSON")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to best.pt for direct OBB inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open display window")
    parser.add_argument("--no-panel", action="store_true",
                        help="Don't draw summary panel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.detections:
        display_detections_from_json(
            args.image, args.detections, args.output,
            show=not args.no_show, conf_threshold=args.conf,
            show_panel=not args.no_panel,
        )
    elif args.model:
        display_detections_from_model(
            args.image, args.model, args.output,
            show=not args.no_show, conf_threshold=args.conf,
            show_panel=not args.no_panel,
        )
    else:
        parser.error("Either --detections or --model must be provided.")


if __name__ == "__main__":
    main()
