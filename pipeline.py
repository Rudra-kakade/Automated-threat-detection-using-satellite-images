"""
Pipeline Orchestrator — Satellite Defence
===========================================
Chains all stages: ingest → augment → train → evaluate → infer → visualize → export.

Usage:
    python pipeline.py --stage all
    python pipeline.py --stage train evaluate
    python pipeline.py --stage all --dry-run
    python pipeline.py --config my_config.yaml
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

STAGES_ORDER = ["ingest", "augment", "train", "evaluate", "infer", "visualize", "export"]
DEFAULT_CONFIG = "pipeline_config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("pipeline")


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config %s not found, using defaults.", config_path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_best_weights(cfg: dict) -> str:
    train_cfg = cfg.get("training", {})
    project = train_cfg.get("project", "runs")
    name = train_cfg.get("name", "yolov8s_obb_pipeline")
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        return str(best_pt)
    run_dir = Path(project)
    if run_dir.exists():
        candidates = sorted(run_dir.glob(f"{name}*/weights/best.pt"))
        if candidates:
            return str(candidates[-1])
    for fallback in ["best.pt", "yolov8s-obb.pt"]:
        if Path(fallback).exists():
            return fallback
    raise FileNotFoundError(f"No weights found. Expected {best_pt}. Train first.")


# ─── Stages ──────────────────────────────────────────────────────────────────

def stage_ingest(cfg, dry_run=False):
    data_cfg = cfg.get("data", {})
    src_dir = Path(data_cfg.get("source_dir", "./data/raw"))
    out_dir = Path(data_cfg.get("output_dir", "./data/satellite_obb"))
    splits = data_cfg.get("splits", ["train", "val", "test"])
    if dry_run:
        logger.info("[DRY] Would tile %s → %s", src_dir, out_dir)
        return {"status": "dry_run"}
    from src.data.tile_prep import tile_dataset
    total = 0
    for split in splits:
        split_src = src_dir / split
        if not split_src.exists():
            logger.warning("Split dir %s missing, skipping.", split_src)
            continue
        records = tile_dataset(split_src, out_dir, split=split)
        total += len(records)
    return {"status": "success", "total_tiles": total}


def stage_augment(cfg, dry_run=False):
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return {"status": "skipped"}
    data_cfg = cfg.get("data", {})
    tile_dir = Path(data_cfg.get("output_dir", "./data/satellite_obb"))
    n = aug_cfg.get("num_copies", 3)
    split = aug_cfg.get("split", "train")
    if dry_run:
        logger.info("[DRY] Would augment %s/%s ×%d", tile_dir, split, n)
        return {"status": "dry_run"}
    from src.data.augmentation import augment_dataset_offline
    count = augment_dataset_offline(tile_dir, tile_dir, split=split, num_augmented=n)
    return {"status": "success", "augmented_tiles": count}


def stage_train(cfg, dry_run=False):
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    dataset_yaml = data_cfg.get("dataset_yaml", "./src/data/dataset.yaml")
    if dry_run:
        logger.info("[DRY] Would train epochs=%d batch=%d", train_cfg.get("epochs", 75), train_cfg.get("batch", 6))
        return {"status": "dry_run"}
    from src.train.train import train_satellite_model
    train_satellite_model(
        data=dataset_yaml,
        weights=train_cfg.get("weights", "yolov8s-obb.pt"),
        epochs=train_cfg.get("epochs", 75),
        batch=train_cfg.get("batch", 6),
        imgsz=train_cfg.get("imgsz", 512),
        device=train_cfg.get("device", 0),
        project=train_cfg.get("project", "runs"),
        name=train_cfg.get("name", "yolov8s_obb_pipeline"),
        resume=train_cfg.get("resume", False),
    )
    best = _find_best_weights(cfg)
    return {"status": "success", "best_weights": best}


def stage_evaluate(cfg, dry_run=False):
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    gate_cfg = cfg.get("quality_gate", {})
    dataset_yaml = data_cfg.get("dataset_yaml", "./src/data/dataset.yaml")
    output_dir = eval_cfg.get("output_dir", "./eval_results")
    if dry_run:
        logger.info("[DRY] Would evaluate on %s", dataset_yaml)
        return {"status": "dry_run", "quality_gate_passed": True, "quality_gate_warnings": []}
    model_path = _find_best_weights(cfg)
    from src.eval.evaluate import run_full_evaluation
    results = run_full_evaluation(
        model_path=model_path, dataset_yaml=dataset_yaml,
        output_dir=output_dir, imgsz=eval_cfg.get("imgsz", 512),
        batch=eval_cfg.get("batch", 8),
        device=cfg.get("training", {}).get("device", 0),
        conf=eval_cfg.get("conf", 0.25), iou=eval_cfg.get("iou", 0.5),
    )
    passed, warnings = True, []
    if gate_cfg.get("enabled", True):
        if results.get("map50", 0) < gate_cfg.get("map50_min", 0.50):
            passed = False
            warnings.append(f"mAP@0.5={results['map50']:.3f} < {gate_cfg.get('map50_min')}")
        for name, m in results.get("per_class", {}).items():
            if m.get("fpr", 0) > gate_cfg.get("fpr_max", 0.15):
                passed = False
                warnings.append(f"{name} FPR={m['fpr']:.3f} > {gate_cfg.get('fpr_max')}")
    return {"status": "success", "metrics": results, "quality_gate_passed": passed, "quality_gate_warnings": warnings}


def stage_infer(cfg, dry_run=False):
    infer_cfg = cfg.get("inference", {})
    test_dir = Path(infer_cfg.get("test_images_dir", "./data/test_images"))
    out_dir = Path(infer_cfg.get("output_dir", "./inference_results"))
    if dry_run:
        logger.info("[DRY] Would infer on %s", test_dir)
        return {"status": "dry_run"}
    if not test_dir.exists():
        return {"status": "skipped", "reason": "test_images_dir not found"}
    model_path = _find_best_weights(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.inference.infer_sahi import run_inference, predictions_to_json
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)
    total = 0
    for img in images:
        preds = run_inference(str(img), model_path, infer_cfg.get("confidence", 0.25),
                              f"cuda:{cfg.get('training',{}).get('device',0)}")
        rj = predictions_to_json(preds)
        total += len(rj)
        with open(out_dir / f"{img.stem}_detections.json", "w") as f:
            json.dump(rj, f, indent=2)
    return {"status": "success", "images": len(images), "detections": total}


def stage_visualize(cfg, dry_run=False):
    vis_cfg = cfg.get("visualization", {})
    train_cfg = cfg.get("training", {})
    output_dir = vis_cfg.get("output_dir", "./eval_results/plots")
    if dry_run:
        logger.info("[DRY] Would generate visualizations → %s", output_dir)
        return {"status": "dry_run"}
    result = {"status": "success", "plots": {}}
    project = train_cfg.get("project", "runs")
    name = train_cfg.get("name", "yolov8s_obb_pipeline")
    run_dir = Path(project) / name
    if not run_dir.exists():
        cands = sorted(Path(project).glob(f"{name}*"))
        run_dir = cands[-1] if cands else None
    if run_dir and (run_dir / "results.csv").exists():
        from src.eval.dashboard import generate_all_plots
        result["plots"]["dashboard"] = generate_all_plots(str(run_dir), output_dir)
    test_dir = cfg.get("inference", {}).get("test_images_dir", "./data/test_images")
    if Path(test_dir).exists():
        try:
            from src.eval.visualise import visualise_batch
            model_path = _find_best_weights(cfg)
            overlay_out = Path(output_dir) / "overlays"
            visualise_batch(test_dir, model_path, str(overlay_out),
                            vis_cfg.get("conf_threshold", 0.3), vis_cfg.get("max_overlay_images", 20))
        except Exception as e:
            logger.error("Overlay failed: %s", e)
    return result


def stage_export(cfg, dry_run=False):
    export_cfg = cfg.get("export", {})
    out_dir = Path(export_cfg.get("output_dir", "./exports"))
    fmt = export_cfg.get("format", "onnx")
    if dry_run:
        logger.info("[DRY] Would export to %s", fmt)
        return {"status": "dry_run"}
    import shutil
    from ultralytics import YOLO
    model_path = _find_best_weights(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    ep = model.export(format=fmt, imgsz=export_cfg.get("imgsz", 512), simplify=export_cfg.get("simplify", True))
    if ep and Path(ep).exists():
        dest = out_dir / Path(ep).name
        shutil.copy2(ep, dest)
        return {"status": "success", "path": str(dest)}
    return {"status": "failed"}


STAGE_FNS = {
    "ingest": stage_ingest, "augment": stage_augment, "train": stage_train,
    "evaluate": stage_evaluate, "infer": stage_infer, "visualize": stage_visualize, "export": stage_export,
}


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_pipeline(stages, cfg, dry_run=False):
    t0 = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
               "dry_run": dry_run, "stages": {}}
    logger.info("═" * 60)
    logger.info("SATELLITE DEFENCE PIPELINE — %s", run_id)
    logger.info("Stages: %s  %s", " → ".join(stages), "(DRY RUN)" if dry_run else "")
    logger.info("═" * 60)
    failed = False
    for s in stages:
        logger.info("─── ▶ %s ───", s.upper())
        st = time.time()
        try:
            r = STAGE_FNS[s](cfg, dry_run)
            r["duration_s"] = round(time.time() - st, 2)
            results["stages"][s] = r
            logger.info("✓ %s [%.1fs]", s, r["duration_s"])
            if s == "evaluate" and not r.get("quality_gate_passed", True):
                for w in r.get("quality_gate_warnings", []):
                    logger.warning("  ⚠ %s", w)
                if cfg.get("quality_gate", {}).get("enabled", True) and not dry_run:
                    logger.error("Pipeline halted: quality gate failed.")
                    failed = True
                    break
        except Exception as e:
            logger.error("✗ %s failed: %s", s, e, exc_info=True)
            results["stages"][s] = {"status": "error", "error": str(e), "duration_s": round(time.time() - st, 2)}
            failed = True
            break
    results["total_duration_s"] = round(time.time() - t0, 2)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["success"] = not failed
    out_dir = PROJECT_ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rf = out_dir / f"pipeline_run_{run_id}.json"
    with open(rf, "w") as f:
        json.dump(results, f, indent=2, default=str)
    try:
        from src.eval.report import generate_report
        generate_report(results, cfg, str(out_dir))
    except Exception as e:
        logger.warning("Report gen failed: %s", e)
    logger.info("═" * 60)
    logger.info("PIPELINE %s — %.1fs", "COMPLETE" if not failed else "FAILED", results["total_duration_s"])
    logger.info("═" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Satellite Defence Pipeline")
    parser.add_argument("--stage", nargs="+", default=["all"], choices=STAGES_ORDER + ["all"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    cfg = load_config(args.config)
    stages = STAGES_ORDER if "all" in args.stage else args.stage
    results = run_pipeline(stages, cfg, args.dry_run)
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
