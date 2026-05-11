"""
Full Evaluation Suite for Satellite OBB Detection
====================================================
Runs mAP, per-class metrics, FPR, confusion matrix, and PR curves.

Blueprint v2, Section 7.
"""

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

from src.eval.confusion_matrix import plot_confusion_matrix
from src.eval.pr_curves import plot_pr_curves

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship", "aircraft", "vehicle", "storage-tank"]

# Decision thresholds from blueprint Section 7
DECISION_THRESHOLDS = {
    "map50_min": 0.50,
    "map50_target": 0.70,
    "recall_ship_min": 0.75,
    "recall_aircraft_min": 0.70,
    "fpr_max": 0.15,
    "fpr_target": 0.08,
}


def run_full_evaluation(
    model_path: str,
    dataset_yaml: str,
    output_dir: str = "./eval_results",
    imgsz: int = 512,
    batch: int = 8,
    device: int = 0,
    conf: float = 0.25,
    iou: float = 0.5,
) -> dict:
    """
    Run the complete evaluation suite on the validation set.

    Saves:
        - metrics.json: Core metrics + per-class breakdown + FPR
        - confusion_matrix.png: Labelled heatmap
        - pr_curves.png: Per-class precision-recall curves

    Args:
        model_path: Path to trained best.pt.
        dataset_yaml: Path to dataset.yaml.
        output_dir: Directory to save results.
        imgsz: Image size for validation.
        batch: Batch size for validation.
        device: CUDA device index.
        conf: Confidence threshold.
        iou: IoU threshold for matching.

    Returns:
        Dict with full metrics breakdown.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    metrics = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
    )

    # ─── Core metrics ───────────────────────────────────────────────
    results = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {},
    }

    for i, name in enumerate(CLASS_NAMES):
        p = float(metrics.box.p[i])
        r = float(metrics.box.r[i])
        f1 = float(2 * p * r / (p + r + 1e-9))

        results["per_class"][name] = {
            "ap50": float(metrics.box.ap50[i]),
            "ap": float(metrics.box.ap[i]),
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    # ─── False Positive Rate ────────────────────────────────────────
    tp = metrics.box.tp
    fp = metrics.box.fp
    for i, name in enumerate(CLASS_NAMES):
        fpr = float(fp[i] / (fp[i] + tp[i] + 1e-9))
        results["per_class"][name]["fpr"] = fpr

    # ─── Decision threshold check ──────────────────────────────────
    results["threshold_check"] = _check_thresholds(results)

    # ─── Save metrics JSON ─────────────────────────────────────────
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Print summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  mAP@0.5:      {results['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['map50_95']:.4f}")
    print(f"\n  Per-class breakdown:")
    for name, m in results["per_class"].items():
        print(
            f"    {name:15s}  AP50={m['ap50']:.3f}  R={m['recall']:.3f}  "
            f"P={m['precision']:.3f}  F1={m['f1']:.3f}  FPR={m['fpr']:.3f}"
        )

    # Print threshold warnings
    for warning in results["threshold_check"].get("warnings", []):
        print(f"\n  ⚠ {warning}")

    print(f"{'='*60}\n")

    # ─── Generate plots ────────────────────────────────────────────
    plot_confusion_matrix(metrics, out)
    plot_pr_curves(metrics, out)

    logger.info("Full evaluation saved to %s", out)
    return results


def _check_thresholds(results: dict) -> dict:
    """Check results against operational decision thresholds."""
    check = {"pass": True, "warnings": []}

    if results["map50"] < DECISION_THRESHOLDS["map50_min"]:
        check["pass"] = False
        check["warnings"].append(
            f"mAP@0.5 = {results['map50']:.3f} < minimum {DECISION_THRESHOLDS['map50_min']}"
        )

    for name, m in results["per_class"].items():
        if m["fpr"] > DECISION_THRESHOLDS["fpr_max"]:
            check["pass"] = False
            check["warnings"].append(
                f"{name} FPR = {m['fpr']:.3f} > maximum {DECISION_THRESHOLDS['fpr_max']}"
            )

    ship_recall = results["per_class"].get("ship", {}).get("recall", 0)
    if ship_recall < DECISION_THRESHOLDS["recall_ship_min"]:
        check["warnings"].append(
            f"Ship recall = {ship_recall:.3f} < minimum {DECISION_THRESHOLDS['recall_ship_min']}"
        )

    aircraft_recall = results["per_class"].get("aircraft", {}).get("recall", 0)
    if aircraft_recall < DECISION_THRESHOLDS["recall_aircraft_min"]:
        check["warnings"].append(
            f"Aircraft recall = {aircraft_recall:.3f} < minimum {DECISION_THRESHOLDS['recall_aircraft_min']}"
        )

    return check


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation suite")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", type=str, default="./eval_results")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_full_evaluation(args.model, args.data, args.output, args.imgsz, args.batch, args.device, args.conf)


if __name__ == "__main__":
    main()
