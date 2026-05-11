"""
Precision-Recall Curve Plotting for Satellite OBB Detection
=============================================================
Saves per-class PR curves on one plot.

Blueprint v2, Section 7.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

CLASS_NAMES = ["ship", "aircraft", "vehicle", "storage-tank"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_pr_curves(metrics, out: Path, class_names: list = None) -> Path:
    """
    Generate and save per-class precision-recall curves on one plot.

    Args:
        metrics: Ultralytics validation metrics object (from model.val()).
        out: Output directory for the PNG.
        class_names: List of class names. Defaults to CLASS_NAMES.

    Returns:
        Path to the saved PR curves PNG.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(class_names):
        color = COLORS[i % len(COLORS)]
        # px = confidence thresholds, py = precision, ry = recall
        py = metrics.box.py  # precision at each threshold per class
        ry = metrics.box.ry  # recall at each threshold per class
        ap50 = metrics.box.ap50[i]

        ax.plot(
            ry[:, i],
            py[:, i],
            label=f"{name} (AP50={ap50:.3f})",
            color=color,
            linewidth=1.8,
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Curves by Class", fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = out / "pr_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ PR curves saved to {save_path}")
    return save_path
