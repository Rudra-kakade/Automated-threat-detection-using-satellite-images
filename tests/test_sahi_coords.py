"""
SAHI Coordinate Verification Test
====================================
Test wrapper for verify_sahi_global_coords().

This test requires a trained model and is intended as an integration
test — it is skipped if no model checkpoint is available.

Blueprint v2, Section 6 (Coordinate Verification).
"""

import json
from pathlib import Path

import pytest

# Mark the entire module as requiring a trained model
MODEL_PATH = Path("runs/yolov8s_obb_run1/weights/best.pt")
TEST_IMAGE = Path("data/satellite_obb/images/test")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Trained model not found at {MODEL_PATH}. Train first.",
)


@pytest.fixture
def sample_gt_boxes():
    """
    Sample ground truth boxes for coordinate verification.
    Replace with actual GT from your test set.
    """
    return [
        {"cx": 256, "cy": 256, "class_id": 0},
        {"cx": 400, "cy": 300, "class_id": 1},
    ]


def test_sahi_coord_drift(sample_gt_boxes):
    """
    Verify SAHI global coordinates are within 20px of GT centroids.
    """
    from src.inference.verify_coords import verify_sahi_global_coords

    # Find a test image
    test_images = list(TEST_IMAGE.glob("*.png"))
    if not test_images:
        pytest.skip("No test images found")

    result = verify_sahi_global_coords(
        image_path=str(test_images[0]),
        ground_truth_boxes=sample_gt_boxes,
        model_path=str(MODEL_PATH),
        max_drift_px=20.0,
    )

    assert result["passed"] == result["total"], (
        f"Coordinate drift too large: {result['passed']}/{result['total']} passed. "
        f"Drifts: {result['drifts']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
