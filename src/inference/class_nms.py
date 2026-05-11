"""
Class-Aware NMS for Satellite Object Detection
================================================
Per-class IoU thresholds for Non-Maximum Suppression.

Blueprint v2, Section 6.
"""

from collections import defaultdict

import torch
import torchvision.ops as ops


# Per-class IoU thresholds — tuned for satellite object characteristics
DEFAULT_CLASS_IOU_THRESHOLDS = {
    0: 0.6,   # ship — large, rarely truly overlapping
    1: 0.5,   # aircraft — medium, near other aircraft at airfields
    2: 0.4,   # vehicle — small, densely packed
    3: 0.55,  # storage-tank — medium, clustered in depots
}


def apply_class_aware_nms(
    predictions: list,
    class_iou_map: dict = None,
) -> list:
    """
    Secondary NMS pass with per-class IoU thresholds.

    SAHI's global NMS may be too permissive for dense classes (vehicles)
    or too strict for large sparse ones (ships). This applies class-specific
    IoU thresholds after SAHI's initial postprocessing.

    Args:
        predictions: List of SAHI ObjectPrediction objects.
        class_iou_map: Dict mapping class_id → IoU threshold.
                       Defaults to DEFAULT_CLASS_IOU_THRESHOLDS.

    Returns:
        Filtered list of ObjectPrediction objects.
    """
    if class_iou_map is None:
        class_iou_map = DEFAULT_CLASS_IOU_THRESHOLDS

    if not predictions:
        return []

    # Group predictions by class
    by_class = defaultdict(list)
    for pred in predictions:
        by_class[pred.category.id].append(pred)

    kept = []
    for class_id, preds in by_class.items():
        threshold = class_iou_map.get(class_id, 0.5)

        # Extract boxes and scores for torchvision NMS
        boxes = torch.tensor(
            [[*p.bbox.to_xyxy()] for p in preds], dtype=torch.float32
        )
        scores = torch.tensor(
            [p.score.value for p in preds], dtype=torch.float32
        )

        keep_idx = ops.nms(boxes, scores, threshold)
        kept.extend([preds[i] for i in keep_idx])

    return kept
