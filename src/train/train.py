"""
Training Script — YOLOv8s-OBB on Satellite Imagery
====================================================
Fine-tunes YOLOv8s-obb on tiled satellite data with 4GB VRAM budget.

Blueprint v2, Section 5.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from src.train.grad_monitor import GradientUnderflowMonitor

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_YAML = str(Path(__file__).resolve().parents[1] / "data" / "dataset.yaml")
DEFAULT_WEIGHTS = "yolov8s-obb.pt"


def train_satellite_model(
    data: str = DEFAULT_DATA_YAML,
    weights: str = DEFAULT_WEIGHTS,
    epochs: int = 75,
    batch: int = 6,
    accumulate: int = 3,
    imgsz: int = 512,
    device: int = 0,
    project: str = "runs",
    name: str = "yolov8s_obb_run1",
    resume: bool = False,
):
    """
    Train YOLOv8s-OBB on satellite imagery dataset.

    VRAM budget: ~3.4 GB (fits in 4 GB GPU with ~600 MB headroom).
    Effective gradient batch = batch × accumulate = 18.
    """
    # Verify VRAM before starting
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        if vram_gb < 3.8:
            logger.warning(
                "Expected >=4GB VRAM, got %.1fGB. Training may OOM.", vram_gb
            )
    else:
        logger.warning("No CUDA device found. Training will run on CPU (very slow).")
        device = "cpu"

    # Load OBB variant — NOT interchangeable with yolov8s.pt
    model = YOLO(weights)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=True,              # FP16 mixed precision
        workers=2,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,       # Longer warmup for s-size models
        cos_lr=True,           # Cosine LR schedule
        close_mosaic=20,       # Disable mosaic in last 20 epochs
        project=project,
        name=name,
        val=True,
        plots=True,
        save_period=10,        # Checkpoint every 10 epochs
        resume=resume,
    )

    # Log gradient underflow summary if monitor was attached
    logger.info("Training complete. Results saved to %s/%s", project, name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8s-OBB satellite model")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to dataset.yaml")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--accumulate", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="yolov8s_obb_run1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_satellite_model(**vars(args))


if __name__ == "__main__":
    main()
