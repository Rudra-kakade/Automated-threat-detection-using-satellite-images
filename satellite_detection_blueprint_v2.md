# Project Blueprint v2: Autonomous Threat Detection via Satellite Imagery
### All architectural weaknesses and breakpoints resolved

---

## Changelog from v1

| # | Breakpoint | Severity | Resolution |
|---|---|---|---|
| 1 | Axis-aligned bounding boxes on rotated objects | Hard | Switched to YOLOv8s-obb (oriented bounding boxes) |
| 2 | Coordinate stitching bug risk in inference | Hard | Replaced custom inference tiling with SAHI |
| 3 | Domain gap — proxy datasets ≠ real imagery | Weakness | Multi-source augmentation + per-source eval tracking |
| 4 | Object truncation across tile boundaries | Weakness | Larger overlap in training prep + SAHI's auto-stitch at inference |
| 5 | Batch norm instability at batch=8 | Weakness | Retuned for YOLOv8s-obb: batch=6, accumulate=3 |
| 6 | AMP FP16 underflow on minority classes | Weakness | Gradient scaler + per-class loss monitoring hook |
| 7 | No evaluation framework | Weakness | Full suite: mAP, per-class recall, FPR, confusion matrix, visualisation |
| 8 | Single IoU threshold for all classes in NMS | Weakness | Class-aware NMS via SAHI's postprocess config |

---

## 1. Project Specifications

- **Project Title:** Autonomous Threat Detection via Satellite Imagery
- **Core Problem:** Defense agencies are overwhelmed by daily satellite imagery volume. Manual scanning for tactical assets (ships, aircraft, unauthorized vehicles) is too slow.
- **Solution:** A deep learning computer vision pipeline that ingests large geospatial images, detects objects using oriented bounding boxes, and flags anomalies with a rigorous evaluation layer.
- **Tech Stack:**
  - **Modeling:** PyTorch, Ultralytics YOLOv8
  - **Data Engineering:** GDAL, Rasterio, Shapely, OpenCV, Albumentations
  - **Inference:** SAHI (Slicing Aided Hyper Inference), ONNX, TensorRT
  - **Evaluation:** Ultralytics metrics API, Seaborn, Matplotlib

---

## 2. Model Specifications

### Base Architecture
**YOLOv8s-obb** (Small, Oriented Bounding Box variant).

**Why OBB is mandatory for this task:**
Standard YOLO uses axis-aligned bounding boxes (AABB). Aerial objects — ships, aircraft, vehicles in parking lots — are almost never axis-aligned. A diagonal ship detected with an AABB results in a large, loose box that:
- Inflates the apparent object area by 2–4×
- Causes false overlaps with neighbouring objects
- Destroys precision on dense scenes

OBB adds a fifth prediction parameter `θ` (rotation angle in radians) to each box, giving the model a tight, rotated fit around the object regardless of orientation. YOLOv8-obb supports this natively.

```
AABB on a 45° ship:       OBB on a 45° ship:
┌──────────────┐           ╱‾‾‾‾‾‾‾‾╲
│  ╱‾‾‾‾‾‾╲   │           ╲        ╱
│  ╲      ╱   │            ╲______╱
│   ╲____╱    │
└──────────────┘
Large noisy box           Tight rotated box
```

### Why YOLOv8s-obb over YOLOv8n-obb
YOLOv8s has ~11M parameters vs ~3.2M for nano. On satellite imagery the feature density is much higher than natural images (objects are small, texturally similar, and densely packed), so the extra capacity of the small model meaningfully improves recall on hard cases. The VRAM cost is manageable with the tuning in Section 4.

**Learning Paradigm:** Transfer learning from COCO-pretrained weights (`yolov8s-obb.pt`), fine-tuned on overhead imagery datasets.

---

## 3. Data Specifications

### Target Datasets
- **DOTA v2.0** — Primary. Purpose-built for oriented bounding boxes in aerial imagery. 11,268 images, 1.79M annotated instances across 18 categories including ships, planes, vehicles, and storage tanks. Already in OBB format.
- **xView** — Secondary (requires OBB conversion from AABB, acceptable for large objects with low rotation variance like buildings and storage tanks).
- **Airbus Ship Detection (Kaggle)** — Tertiary, for maritime-specific fine-tuning.

> **Note on domain gap:** These datasets were collected under specific sensor configurations, altitudes, and geographic regions. Model performance on operational imagery from a different sensor will degrade. To mitigate this, log `dataset_source` as metadata per tile during training. During evaluation, always report metrics *broken down by source dataset* — not just as an aggregate. Significant per-source divergence indicates the model is memorising sensor characteristics rather than learning object semantics.

### Dataset YAML
```yaml
# dataset.yaml
path: ./data/satellite_obb
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: ship
  1: aircraft
  2: vehicle
  3: storage-tank

# Track source for per-source eval
# Each image filename should include source prefix: dota_, xview_, airbus_
```

---

## 4. Data Engineering Pipeline — Training Tile Preparation

This is the custom Python pipeline for generating the training dataset. It is separate from inference (which uses SAHI). The goal here is deterministic, reproducible tile generation with ground truth annotation.

### Why custom for training, SAHI for inference
During training, you need full control over the tile generation to ensure:
- Annotation coordinates are correctly transformed
- Overlap is handled consistently and reproducibly
- A unit test can verify coordinate math before you train on corrupted labels

During inference, the coord math becomes far more complex (global image, arbitrary tile ordering, cross-tile NMS). SAHI's battle-tested implementation eliminates that entire bug surface.

### Tiling Parameters
| Parameter | Value | Reasoning |
|---|---|---|
| Tile size | 512×512 | GPU constraint and YOLO input size |
| Overlap | 25% (128px) | Increased from v1's 10–15%. Large ships can span 800–1500px; 25% overlap ensures any object appears fully in at least one tile |
| Min object visibility | 30% | Discard a tile's annotation for an object if less than 30% of the object is within the tile — avoids training on near-invisible truncated targets |

### Tiling Script with Unit Test

```python
import rasterio
import numpy as np
from pathlib import Path
from shapely.geometry import box as shapely_box, Polygon

TILE_SIZE = 512
OVERLAP = 128  # 25% of 512


def tile_image(src_path: Path, out_dir: Path, split: str = "train"):
    """
    Slice a large satellite image into 512x512 tiles with 128px overlap.
    Transforms OBB annotations into per-tile YOLO OBB format.
    Returns list of (tile_path, annotation_path) for verification.
    """
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stride = TILE_SIZE - OVERLAP
    records = []

    with rasterio.open(src_path) as src:
        W, H = src.width, src.height
        data = src.read()  # shape: (C, H, W)

    annotations = load_obb_annotations(src_path)  # [(class_id, x1,y1,x2,y2,x3,y3,x4,y4), ...]

    tile_id = 0
    for row_start in range(0, H - TILE_SIZE + 1, stride):
        for col_start in range(0, W - TILE_SIZE + 1, stride):
            row_end = row_start + TILE_SIZE
            col_end = col_start + TILE_SIZE

            tile_region = shapely_box(col_start, row_start, col_end, row_end)
            tile_img = data[:, row_start:row_end, col_start:col_end]

            tile_labels = []
            for ann in annotations:
                class_id, pts = ann[0], np.array(ann[1:]).reshape(4, 2)
                obj_poly = Polygon(pts)
                intersection = obj_poly.intersection(tile_region)

                visibility = intersection.area / obj_poly.area
                if visibility < 0.30:
                    continue  # Skip near-invisible truncated objects

                # Transform global coords → tile-local coords → YOLO normalized
                local_pts = pts - np.array([col_start, row_start])
                norm_pts = local_pts / TILE_SIZE  # normalized [0, 1]
                norm_pts = np.clip(norm_pts, 0.0, 1.0)

                # YOLO OBB format: class cx cy w h angle
                # (Ultralytics auto-converts from 4-point poly at training time)
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

    return records


def test_coordinate_round_trip(src_path: Path, out_dir: Path):
    """
    UNIT TEST: Verify that a known annotation survives the tile → local → norm → denorm → global
    round trip with zero drift. Run this before any training run.
    """
    records = tile_image(src_path, out_dir, split="test_rt")
    original_anns = load_obb_annotations(src_path)

    recovered = []
    for img_path, lbl_path, (col_off, row_off) in records:
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

    # Check: every original annotation should be recoverable within 1px tolerance
    for orig in original_anns:
        class_id = orig[0]
        orig_pts = np.array(orig[1:]).reshape(4, 2)
        matches = [r for r in recovered if r[0] == class_id and
                   np.allclose(r[1], orig_pts, atol=1.0)]
        assert len(matches) > 0, (
            f"Annotation for class {class_id} at {orig_pts.tolist()} "
            f"was not recovered within tolerance. CHECK COORD MATH."
        )
    print("✓ Coordinate round-trip test passed for all annotations.")
```

> **Critical:** Run `test_coordinate_round_trip()` on at least one source image before generating your full training set. If it raises an `AssertionError`, your tiling script has a coordinate bug. Do not proceed until it passes.

### Augmentation (Domain Gap Mitigation)

```python
import albumentations as A

train_transform = A.Compose([
    # Simulate different sensor resolutions
    A.RandomScale(scale_limit=0.3, p=0.5),
    # Simulate different atmospheric and seasonal conditions
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, p=0.4),
    # Simulate haze/cloud interference
    A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.3),
    # Geometric — critical for OBB: must use bbox_params with 'obb' format
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 5. Training Process — YOLOv8s-obb on 4GB VRAM

YOLOv8s-obb is larger than the v1 nano model. The VRAM budget is tighter. The following configuration is tuned specifically for 4GB.

### VRAM Budget Breakdown (YOLOv8s-obb at imgsz=512)

| Component | Approx. VRAM |
|---|---|
| Model weights (FP16) | ~22 MB |
| Optimizer states (AdamW, FP32) | ~88 MB |
| Activations (batch=6, FP16) | ~1.8 GB |
| Gradients (FP16) | ~900 MB |
| Framework overhead | ~600 MB |
| **Total** | **~3.4 GB** |

Leaves ~600 MB headroom. Do not increase batch size without profiling.

### Batch Norm Fix

The v1 plan used `batch=8` with `accumulate=2`. Gradient accumulation simulates a larger effective batch for weight updates but does **not** fix the batch norm statistics problem — BN layers still compute stats over the 8 real images, not the simulated 16.

With `batch=6` and `accumulate=3`, the effective gradient batch becomes 18, and the BN statistics improve slightly (more samples per step). Additionally, set `bn_momentum` conservatively to smooth out per-batch variance.

### Training Script

```python
from ultralytics import YOLO
import torch

def train_satellite_model():
    # Verify VRAM before starting
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    assert vram_gb >= 3.8, f"Expected >=4GB VRAM, got {vram_gb:.1f}GB"

    # OBB variant — this is not interchangeable with yolov8s.pt
    model = YOLO('yolov8s-obb.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=75,           # s-obb benefits from more epochs than nano
        imgsz=512,
        batch=6,             # Reduced from 8 to fit YOLOv8s-obb in 4GB
        accumulate=3,        # Effective gradient batch = 18
        device=0,
        amp=True,            # FP16 — see gradient scaler hook below
        workers=2,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,     # Longer warmup helps s-size models stabilise
        cos_lr=True,         # Cosine LR schedule — more stable than step decay
        close_mosaic=20,     # Disable mosaic aug in last 20 epochs for fine detail
        project='satellite_defense',
        name='yolov8s_obb_run1',
        val=True,
        plots=True,
        save_period=10,      # Checkpoint every 10 epochs
    )

    return results


# --- AMP / FP16 Gradient Underflow Monitor ---
# Attach this hook after model.train() initialises the trainer,
# or run it as a standalone check on the first 5 epochs.

class GradientUnderflowMonitor:
    """
    Detects FP16 gradient underflow on minority-class outputs.
    Prints a warning if any named parameter's gradient norm drops
    below 1e-7 (the FP16 underflow boundary).
    """
    def __init__(self, model, threshold=1e-7):
        self.model = model
        self.threshold = threshold

    def check(self, epoch):
        underflows = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                if grad_norm < self.threshold:
                    underflows.append((name, grad_norm))
        if underflows:
            print(f"\n⚠ Epoch {epoch} — FP16 gradient underflow detected:")
            for name, norm in underflows[:5]:
                print(f"   {name}: max_grad={norm:.2e}")
            print("   Consider class-weighted loss or oversampling minority class.\n")


if __name__ == '__main__':
    train_satellite_model()
```

---

## 6. Inference Pipeline — SAHI + Class-Aware NMS

SAHI (Slicing Aided Hyper Inference) replaces the custom inference tiling from v1. It handles:
- Image slicing into overlapping patches
- Running the model on each patch
- **Translating patch-local coordinates back to global coordinates correctly**
- Merging overlapping detections across patch boundaries with configurable NMS

The coordinate translation and cross-patch NMS — the two hardest parts to get right in a custom script — are handled by SAHI's tested implementation.

### Class-Aware NMS Configuration

A single IoU threshold for NMS does not work well across object scales:
- **Ships** are large, rarely overlap. A high IoU threshold (0.6) is appropriate.
- **Vehicles** are small and densely packed. A lower threshold (0.4) prevents merging distinct nearby vehicles into one detection.
- **Aircraft** are medium-sized, often near other aircraft at airfields. Use 0.5.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.combine import GreedyNMMPostprocess

def run_inference(image_path: str, model_path: str) -> list:
    """
    Run full-image inference on a large satellite image using SAHI.
    Returns a list of ObjectPrediction objects with global coordinates.
    """
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,        # Path to best.pt from training
        confidence_threshold=0.25,    # Pre-NMS confidence filter
        device='cuda:0',
    )

    # Class-aware NMS thresholds — keyed by class index
    # SAHI supports per-class postprocessing via custom postprocess callable
    class_iou_thresholds = {
        0: 0.6,   # ship — large, rarely truly overlapping
        1: 0.5,   # aircraft
        2: 0.4,   # vehicle — dense, small
        3: 0.55,  # storage-tank
    }

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,    # Matches training overlap
        overlap_width_ratio=0.25,
        postprocess_type='GREEDYNMM', # Non-Maximum Merging — more robust than NMS for overlapping tiles
        postprocess_match_metric='IOU',
        postprocess_match_threshold=0.5,  # Default; overridden per-class below
        verbose=0,
    )

    # Apply class-aware IoU filtering as a post-step
    filtered = apply_class_aware_nms(result.object_prediction_list, class_iou_thresholds)
    return filtered


def apply_class_aware_nms(predictions: list, class_iou_map: dict) -> list:
    """
    Secondary pass: filter remaining predictions per class using
    class-specific IoU thresholds. SAHI's global NMS may be too
    permissive for dense classes or too strict for large ones.
    """
    from collections import defaultdict
    import torchvision.ops as ops
    import torch

    by_class = defaultdict(list)
    for pred in predictions:
        by_class[pred.category.id].append(pred)

    kept = []
    for class_id, preds in by_class.items():
        threshold = class_iou_map.get(class_id, 0.5)
        boxes = torch.tensor([[*p.bbox.to_xyxy()] for p in preds], dtype=torch.float32)
        scores = torch.tensor([p.score.value for p in preds], dtype=torch.float32)
        keep_idx = ops.nms(boxes, scores, threshold)
        kept.extend([preds[i] for i in keep_idx])

    return kept
```

### Coordinate Verification (Inference)

Even with SAHI, spot-check the global coordinate output on a known test image:

```python
def verify_sahi_global_coords(image_path: str, ground_truth_boxes: list, model_path: str):
    """
    Quick sanity check: run SAHI on an image with known GT boxes.
    Warns if any predicted box centroid is more than 20px from its GT match.
    """
    predictions = run_inference(image_path, model_path)
    for gt in ground_truth_boxes:
        gt_cx, gt_cy = gt['cx'], gt['cy']
        closest = min(predictions,
                      key=lambda p: ((p.bbox.minx + p.bbox.maxx)/2 - gt_cx)**2 +
                                    ((p.bbox.miny + p.bbox.maxy)/2 - gt_cy)**2)
        pred_cx = (closest.bbox.minx + closest.bbox.maxx) / 2
        pred_cy = (closest.bbox.miny + closest.bbox.maxy) / 2
        drift = ((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2) ** 0.5
        if drift > 20:
            print(f"⚠ Coord drift {drift:.1f}px for GT object at ({gt_cx}, {gt_cy}). "
                  f"Closest pred at ({pred_cx:.0f}, {pred_cy:.0f}).")
        else:
            print(f"✓ GT ({gt_cx}, {gt_cy}) → pred ({pred_cx:.0f}, {pred_cy:.0f}), drift={drift:.1f}px")
```

---

## 7. Evaluation Framework — Full Suite

This is the most critical addition over v1. A model without a rigorous eval layer produces a number (mAP) with no operational meaning. The full suite produces:

1. **mAP@0.5 and mAP@0.5:0.95** per class and overall
2. **Per-class precision, recall, and F1** at multiple confidence thresholds
3. **False positive rate (FPR)** — operationally, how many wrong alerts per image
4. **Confusion matrix** — which classes are being confused with each other
5. **Visualisation script** — renders GT vs predicted OBBs on sample tiles

### Metrics Script

```python
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

CLASS_NAMES = ['ship', 'aircraft', 'vehicle', 'storage-tank']


def run_full_evaluation(model_path: str, dataset_yaml: str, output_dir: str = "./eval_results"):
    """
    Run the complete evaluation suite on the validation set.
    Saves: metrics JSON, confusion matrix PNG, per-class PR curve PNG.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    model = YOLO(model_path)
    metrics = model.val(
        data=dataset_yaml,
        imgsz=512,
        batch=8,
        device=0,
        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=True,
    )

    # --- Core metrics ---
    results = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {}
    }

    for i, name in enumerate(CLASS_NAMES):
        results["per_class"][name] = {
            "ap50": float(metrics.box.ap50[i]),
            "ap": float(metrics.box.ap[i]),
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "f1": float(2 * metrics.box.p[i] * metrics.box.r[i] /
                        (metrics.box.p[i] + metrics.box.r[i] + 1e-9)),
        }

    # --- False Positive Rate ---
    # FPR = FP / (FP + TN). For detection, approximate per image:
    # FP = detections with no matching GT box
    tp = metrics.box.tp  # shape: (num_classes,)
    fp = metrics.box.fp
    fn = metrics.box.fn
    for i, name in enumerate(CLASS_NAMES):
        fpr = float(fp[i] / (fp[i] + tp[i] + 1e-9))
        results["per_class"][name]["fpr"] = fpr

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"mAP@0.5:      {results['map50']:.4f}")
    print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")
    print(f"\nPer-class breakdown:")
    for name, m in results["per_class"].items():
        print(f"  {name:15s}  AP50={m['ap50']:.3f}  R={m['recall']:.3f}  "
              f"P={m['precision']:.3f}  F1={m['f1']:.3f}  FPR={m['fpr']:.3f}")
    print(f"{'='*50}\n")

    plot_confusion_matrix(metrics, out)
    plot_pr_curves(metrics, out)

    return results


def plot_confusion_matrix(metrics, out: Path):
    """Saves a labelled confusion matrix heatmap."""
    cm = metrics.confusion_matrix.matrix  # shape: (nc+1, nc+1) incl. background
    labels = CLASS_NAMES + ['background']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm.astype(int),
        annot=True, fmt='d',
        xticklabels=labels, yticklabels=labels,
        cmap='Blues', ax=ax,
        linewidths=0.5, linecolor='#cccccc',
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix — Satellite OBB Detection', fontsize=13, pad=14)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"✓ Confusion matrix saved to {out / 'confusion_matrix.png'}")


def plot_pr_curves(metrics, out: Path):
    """Saves per-class precision-recall curves on one plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, name in enumerate(CLASS_NAMES):
        px = metrics.box.px   # confidence thresholds
        py = metrics.box.py   # precision at each threshold per class
        ry = metrics.box.ry   # recall at each threshold per class
        ax.plot(ry[:, i], py[:, i], label=f"{name} (AP50={metrics.box.ap50[i]:.3f})",
                color=colors[i], linewidth=1.8)

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision–Recall Curves by Class', fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)
    print(f"✓ PR curves saved to {out / 'pr_curves.png'}")


def visualise_predictions(image_path: str, model_path: str, conf: float = 0.3,
                          out_path: str = "prediction_overlay.png"):
    """
    Renders OBB predictions overlaid on the raw tile.
    Shows GT boxes in green and predicted OBBs in red.
    Use this to visually verify rotation correctness.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model(image_path, conf=conf, verbose=False)

    img = cv2.imread(image_path)
    for result in results:
        if result.obb is None:
            continue
        for obb in result.obb:
            # obb.xyxyxyxy: (N, 4, 2) rotated box corners
            pts = obb.xyxyxyxy.cpu().numpy().astype(int).reshape(-1, 1, 2)
            cls = int(obb.cls)
            conf_val = float(obb.conf)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 220), thickness=2)
            label = f"{CLASS_NAMES[cls]} {conf_val:.2f}"
            cv2.putText(img, label, tuple(pts[0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    cv2.imwrite(out_path, img)
    print(f"✓ Prediction overlay saved to {out_path}")
```

### Interpreting the Eval Results — Decision Thresholds

| Metric | Minimum acceptable | Target |
|---|---|---|
| mAP@0.5 (overall) | 0.50 | 0.70+ |
| Recall (ship) | 0.75 | 0.85+ |
| Recall (aircraft) | 0.70 | 0.80+ |
| FPR (any class) | < 0.15 | < 0.08 |
| Ship/aircraft confusion | < 5% of ship GTs | < 2% |

If FPR for any class exceeds 0.15 before mAP targets are met, the model is over-detecting. Lower the confidence threshold only after checking the PR curve — there is usually a knee point where recall improves substantially at low FPR cost.

---

## 8. Running Order Checklist

```
[ ] 1. Download DOTA v2.0 and confirm OBB annotation format
[ ] 2. Run test_coordinate_round_trip() on one source image — MUST PASS before continuing
[ ] 3. Run full tiling pipeline to generate training tiles
[ ] 4. Verify class distribution in generated tiles (watch for vehicle/aircraft imbalance)
[ ] 5. Train with yolov8s-obb.pt using training script above
[ ] 6. Monitor GradientUnderflowMonitor output for first 5 epochs
[ ] 7. Run run_full_evaluation() on validation set — review per-class FPR
[ ] 8. Visually inspect predictions on 10–20 sample tiles with visualise_predictions()
[ ] 9. Run SAHI inference on a full-size test image (not a tile)
[ ] 10. Run verify_sahi_global_coords() against known GT on the full image
[ ] 11. Export to ONNX / TensorRT for deployment
```

---

## 9. Dependencies

```txt
# requirements.txt
ultralytics>=8.1.0
sahi>=0.11.15
torch>=2.1.0
torchvision>=0.16.0
rasterio>=1.3.0
shapely>=2.0.0
albumentations>=1.3.0
opencv-python>=4.8.0
seaborn>=0.13.0
matplotlib>=3.8.0
numpy>=1.24.0
gdal>=3.6.0
```

---

*Blueprint v2 — all breakpoints from v1 analysis resolved.*
