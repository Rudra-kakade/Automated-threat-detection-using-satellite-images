"""
SAHI Coordinate Verification
==============================
Spot-checks SAHI global coordinate output against known GT boxes.

Blueprint v2, Section 6 (Coordinate Verification).
"""

import argparse
import json
import logging
from pathlib import Path

from src.inference.infer_sahi import run_inference

logger = logging.getLogger(__name__)


def verify_sahi_global_coords(
    image_path: str,
    ground_truth_boxes: list,
    model_path: str,
    max_drift_px: float = 20.0,
) -> dict:
    """
    Quick sanity check: run SAHI on an image with known GT boxes.
    Warns if any predicted box centroid is more than max_drift_px
    from its GT match.

    Args:
        image_path: Path to the test image.
        ground_truth_boxes: List of dicts with 'cx', 'cy', 'class_id' keys.
        model_path: Path to trained best.pt.
        max_drift_px: Maximum acceptable centroid drift in pixels.

    Returns:
        Dict with 'passed', 'total', 'drifts' keys.
    """
    predictions = run_inference(image_path, model_path)

    results = {"passed": 0, "total": len(ground_truth_boxes), "drifts": []}

    for gt in ground_truth_boxes:
        gt_cx, gt_cy = gt["cx"], gt["cy"]

        if not predictions:
            drift = float("inf")
            print(f"⚠ No predictions for GT object at ({gt_cx}, {gt_cy})")
            results["drifts"].append({"gt": [gt_cx, gt_cy], "drift": drift, "ok": False})
            continue

        closest = min(
            predictions,
            key=lambda p: (
                ((p.bbox.minx + p.bbox.maxx) / 2 - gt_cx) ** 2
                + ((p.bbox.miny + p.bbox.maxy) / 2 - gt_cy) ** 2
            ),
        )
        pred_cx = (closest.bbox.minx + closest.bbox.maxx) / 2
        pred_cy = (closest.bbox.miny + closest.bbox.maxy) / 2
        drift = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5

        ok = drift <= max_drift_px
        if ok:
            results["passed"] += 1
            print(
                f"✓ GT ({gt_cx}, {gt_cy}) → pred ({pred_cx:.0f}, {pred_cy:.0f}), "
                f"drift={drift:.1f}px"
            )
        else:
            print(
                f"⚠ Coord drift {drift:.1f}px for GT object at ({gt_cx}, {gt_cy}). "
                f"Closest pred at ({pred_cx:.0f}, {pred_cy:.0f})."
            )

        results["drifts"].append({
            "gt": [gt_cx, gt_cy],
            "pred": [round(pred_cx, 1), round(pred_cy, 1)],
            "drift": round(drift, 1),
            "ok": ok,
        })

    print(f"\n{'='*50}")
    print(f"Passed: {results['passed']}/{results['total']} (max drift: {max_drift_px}px)")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify SAHI global coordinates vs GT")
    parser.add_argument("image", type=str, help="Path to test image")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--gt-json", type=str, required=True,
                        help="JSON file with GT boxes: [{cx, cy, class_id}, ...]")
    parser.add_argument("--max-drift", type=float, default=20.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.gt_json) as f:
        gt_boxes = json.load(f)

    verify_sahi_global_coords(args.image, gt_boxes, args.model, args.max_drift)


if __name__ == "__main__":
    main()
