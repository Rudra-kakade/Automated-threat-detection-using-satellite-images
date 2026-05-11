"""
SAHI Inference Pipeline for Satellite Object Detection
=======================================================
Slicing Aided Hyper Inference on large satellite images.

Blueprint v2, Section 6.
"""

import argparse
import json
import logging
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src.inference.class_nms import DEFAULT_CLASS_IOU_THRESHOLDS, apply_class_aware_nms

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship", "aircraft", "vehicle", "storage-tank"]


def run_inference(
    image_path: str,
    model_path: str,
    confidence_threshold: float = 0.25,
    device: str = "cuda:0",
    slice_size: int = 512,
    overlap_ratio: float = 0.25,
    class_iou_thresholds: dict = None,
) -> list:
    """
    Run full-image inference on a large satellite image using SAHI.

    Args:
        image_path: Path to the satellite image.
        model_path: Path to trained best.pt weights.
        confidence_threshold: Pre-NMS confidence filter.
        device: CUDA device string.
        slice_size: Tile size for sliced inference (must match training).
        overlap_ratio: Overlap ratio (must match training: 0.25).
        class_iou_thresholds: Per-class IoU thresholds for class-aware NMS.

    Returns:
        List of SAHI ObjectPrediction objects with global coordinates.
    """
    if class_iou_thresholds is None:
        class_iou_thresholds = DEFAULT_CLASS_IOU_THRESHOLDS

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    # Apply class-aware IoU filtering as a post-step
    filtered = apply_class_aware_nms(
        result.object_prediction_list, class_iou_thresholds
    )
    return filtered


def predictions_to_json(predictions: list) -> list:
    """Convert SAHI predictions to serialisable dicts."""
    results = []
    for pred in predictions:
        results.append({
            "class_id": pred.category.id,
            "class_name": pred.category.name,
            "confidence": round(pred.score.value, 4),
            "bbox_xyxy": list(pred.bbox.to_xyxy()),
            "centroid": [
                (pred.bbox.minx + pred.bbox.maxx) / 2,
                (pred.bbox.miny + pred.bbox.maxy) / 2,
            ],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SAHI inference on satellite images")
    parser.add_argument("image", type=str, help="Path to satellite image")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    predictions = run_inference(args.image, args.model, args.conf, args.device)
    results = predictions_to_json(predictions)

    out_path = args.output or Path(args.image).stem + "_detections.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ {len(results)} detections saved to {out_path}")
    for r in results[:10]:
        print(f"   {r['class_name']:15s} conf={r['confidence']:.3f}  @ {r['centroid']}")


if __name__ == "__main__":
    main()
