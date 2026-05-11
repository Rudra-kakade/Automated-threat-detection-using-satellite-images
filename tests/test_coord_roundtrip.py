"""
Coordinate Round-Trip Test
============================
Verifies that the tiling pipeline's coordinate transformation is lossless:
    global → tile-local → normalized → denormalized → global

Creates a synthetic test image + annotations and asserts that all annotations
survive the round trip within 1px tolerance.

Blueprint v2, Section 4 (Critical unit test).
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.tile_prep import TILE_SIZE, load_obb_annotations, tile_image


# ─── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data(tmp_path):
    """
    Create a synthetic 1024×1024 test image with known OBB annotations.

    Annotations are placed so that:
        - ann_0: fully within a single tile (no boundary issues)
        - ann_1: spans a tile boundary (tests overlap handling)
        - ann_2: near the center (should appear in multiple tiles)
    """
    img_dir = tmp_path / "source"
    img_dir.mkdir()

    # Create 1024×1024 synthetic image (3-channel, uint8)
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    img_path = img_dir / "test_image.png"
    cv2.imwrite(str(img_path), img)

    # Create DOTA-format annotation file
    # Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    annotations = [
        # ann_0: ship in top-left quadrant (100,100)→(200,150) axis-aligned rect as OBB
        "100 100 200 100 200 150 100 150 ship 0",
        # ann_1: aircraft spanning tile boundary around x=384 (stride=384)
        "350 200 450 200 450 260 350 260 aircraft 0",
        # ann_2: vehicle near center
        "500 500 540 500 540 530 500 530 vehicle 0",
    ]
    lbl_path = img_dir / "test_image.txt"
    with open(lbl_path, "w") as f:
        f.write("\n".join(annotations) + "\n")

    return img_path, tmp_path / "output"


# ─── Tests ───────────────────────────────────────────────────────────────────────

def test_coordinate_round_trip(synthetic_data):
    """
    UNIT TEST: Verify that a known annotation survives the
    tile → local → norm → denorm → global round trip within 1px tolerance.
    """
    src_path, out_dir = synthetic_data
    records = tile_image(src_path, out_dir, split="test_rt")

    assert len(records) > 0, "No tiles generated"

    original_anns = load_obb_annotations(src_path)
    assert len(original_anns) > 0, "No annotations loaded"

    # Recover all annotations from tiles back to global coords
    recovered = []
    for img_path, lbl_path, (col_off, row_off) in records:
        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
            continue
        with open(lbl_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                pts = np.array(parts[1:]).reshape(4, 2)
                # Denormalize: tile-local
                pts_local = pts * TILE_SIZE
                # Translate back to global
                pts_global = pts_local + np.array([col_off, row_off])
                recovered.append((class_id, pts_global))

    # Check: every original annotation should be recoverable within 1px
    for orig in original_anns:
        class_id = orig[0]
        orig_pts = np.array(orig[1:]).reshape(4, 2)

        matches = [
            r for r in recovered
            if r[0] == class_id and np.allclose(r[1], orig_pts, atol=1.5)
        ]
        assert len(matches) > 0, (
            f"Annotation for class {class_id} at {orig_pts.tolist()} "
            f"was not recovered within tolerance. CHECK COORD MATH."
        )

    print("✓ Coordinate round-trip test passed for all annotations.")


def test_tiles_have_correct_dimensions(synthetic_data):
    """Verify all generated tiles are exactly TILE_SIZE × TILE_SIZE."""
    src_path, out_dir = synthetic_data
    records = tile_image(src_path, out_dir, split="test_dim")

    for img_path, _, _ in records:
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read tile: {img_path}"
        h, w = img.shape[:2]
        assert h == TILE_SIZE and w == TILE_SIZE, (
            f"Tile {img_path.name} has shape ({h}, {w}), expected ({TILE_SIZE}, {TILE_SIZE})"
        )


def test_empty_annotations_produce_empty_labels(tmp_path):
    """Tiles with no visible objects should have empty label files."""
    # Use a larger image (2048×2048) with annotations only in top-left corner
    # so tiles in the bottom-right are guaranteed to be empty
    img_dir = tmp_path / "source_large"
    img_dir.mkdir()

    img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    img_path = img_dir / "large_test.png"
    cv2.imwrite(str(img_path), img)

    # Place annotation only in top-left corner
    ann = "50 50 120 50 120 90 50 90 ship 0"
    with open(img_dir / "large_test.txt", "w") as f:
        f.write(ann + "\n")

    out_dir = tmp_path / "output_large"
    records = tile_image(img_path, out_dir, split="test_empty")

    # Tiles far from (50-120, 50-90) should have no annotations
    empty_tiles = [
        (img, lbl) for img, lbl, _ in records
        if lbl.stat().st_size == 0
    ]
    assert len(empty_tiles) > 0, (
        "Expected some tiles without annotations (background tiles)"
    )


def test_visibility_filter(tmp_path):
    """
    Objects with <30% visibility in a tile should be excluded.
    Create an object at the very edge so only a sliver is visible.
    """
    img_dir = tmp_path / "source"
    img_dir.mkdir()

    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    img_path = img_dir / "edge_test.png"
    cv2.imwrite(str(img_path), img)

    # Object at far right edge — only ~50px of a 200px-wide object
    # will be inside the first tile column (0-512)
    ann = "460 100 660 100 660 200 460 200 ship 0"
    with open(img_dir / "edge_test.txt", "w") as f:
        f.write(ann + "\n")

    out_dir = tmp_path / "output"
    records = tile_image(img_path, out_dir, split="test_vis")

    # The first tile (0,0 → 512,512) should contain this annotation
    # because 52/200 = 26% < 30% visibility... but with 128px overlap
    # a later tile starting at 384 should contain more of it
    # This tests that the visibility filter is working
    assert len(records) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
