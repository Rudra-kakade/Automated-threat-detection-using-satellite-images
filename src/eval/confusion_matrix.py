"""
Confusion Matrix Plotting for Satellite OBB Detection
=======================================================
Saves a labelled confusion matrix heatmap.

Blueprint v2, Section 7.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship", "aircraft", "vehicle", "storage-tank"]


def plot_confusion_matrix(metrics, out: Path, class_names: list = None) -> Path:
    """
    Generate and save a labelled confusion matrix heatmap.

    Args:
        metrics: Ultralytics validation metrics object (from model.val()).
        out: Output directory for the PNG.
        class_names: List of class names. Defaults to CLASS_NAMES.

    Returns:
        Path to the saved confusion matrix PNG.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Shape: (nc+1, nc+1) including background class
    cm = metrics.confusion_matrix.matrix
    labels = class_names + ["background"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm.astype(int),
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="#cccccc",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(
        "Confusion Matrix — Satellite OBB Detection", fontsize=13, pad=14
    )
    plt.tight_layout()

    save_path = out / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Confusion matrix saved to {save_path}")
    return save_path
