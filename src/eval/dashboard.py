"""
Training Dashboard — Satellite OBB Detection
==============================================
Generates comprehensive training visualizations from YOLO results.csv:
  - Loss curves (train + val)
  - mAP progression
  - Precision & Recall over epochs
  - Learning rate schedule
  - Combined performance dashboard

Usage:
    python -m src.eval.dashboard <results_dir>
    python -m src.eval.dashboard runs/obb/runs/yolov8s_obb_run1-2
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Style ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family": "sans-serif",
    "font.size": 10,
})

# Color palette (GitHub-inspired)
C_BLUE = "#58a6ff"
C_GREEN = "#3fb950"
C_ORANGE = "#d29922"
C_RED = "#f85149"
C_PURPLE = "#bc8cff"
C_CYAN = "#39d2c0"
C_PINK = "#f778ba"
C_GRAY = "#8b949e"


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results.csv from a YOLO training run directory."""
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv found in {results_dir}")

    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (YOLO adds spaces)
    df.columns = df.columns.str.strip()
    return df


# ─── Individual Plot Functions ───────────────────────────────────────────────

def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training & Validation Losses", fontsize=16, fontweight="bold",
                 color="#e6edf3", y=0.98)

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss", C_BLUE, C_RED),
        ("train/cls_loss", "val/cls_loss", "Classification Loss", C_GREEN, C_ORANGE),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss", C_PURPLE, C_PINK),
        ("train/angle_loss", "val/angle_loss", "Angle Loss (OBB)", C_CYAN, C_RED),
    ]

    for ax, (train_col, val_col, title, tc, vc) in zip(axes.flatten(), loss_pairs):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], color=tc, linewidth=1.8,
                    label="Train", alpha=0.9)
        if val_col in df.columns:
            ax.plot(df["epoch"], df[val_col], color=vc, linewidth=1.8,
                    label="Val", linestyle="--", alpha=0.9)

        ax.set_title(title, fontsize=12, color="#e6edf3")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)

        # Mark best (lowest) validation loss
        if val_col in df.columns:
            best_idx = df[val_col].idxmin()
            ax.axvline(df["epoch"][best_idx], color=C_GRAY, linestyle=":",
                       alpha=0.5, linewidth=0.8)
            ax.scatter([df["epoch"][best_idx]], [df[val_col][best_idx]],
                       color=vc, s=40, zorder=5, edgecolors="white", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = out_dir / "loss_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Loss curves → {save_path}")
    return save_path


def plot_map_progression(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot mAP@50 and mAP@50-95 over training epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    map50_col = "metrics/mAP50(B)"
    map5095_col = "metrics/mAP50-95(B)"

    if map50_col in df.columns:
        ax.plot(df["epoch"], df[map50_col], color=C_BLUE, linewidth=2.2,
                label="mAP@50", marker="o", markersize=3, alpha=0.9)
        # Fill area under curve
        ax.fill_between(df["epoch"], 0, df[map50_col], color=C_BLUE, alpha=0.08)

        best_idx = df[map50_col].idxmax()
        best_val = df[map50_col][best_idx]
        best_epoch = df["epoch"][best_idx]
        ax.annotate(f"Best: {best_val:.4f}\n(epoch {best_epoch})",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + len(df) * 0.05, best_val - 0.05),
                    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2),
                    fontsize=10, color=C_BLUE, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                              edgecolor=C_BLUE, alpha=0.9))

    if map5095_col in df.columns:
        ax.plot(df["epoch"], df[map5095_col], color=C_GREEN, linewidth=2.2,
                label="mAP@50-95", marker="s", markersize=3, alpha=0.9)
        ax.fill_between(df["epoch"], 0, df[map5095_col], color=C_GREEN, alpha=0.06)

    ax.set_title("Mean Average Precision Over Training", fontsize=14,
                 fontweight="bold", color="#e6edf3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = out_dir / "map_progression.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ mAP progression → {save_path}")
    return save_path


def plot_precision_recall(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot precision and recall over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    prec_col = "metrics/precision(B)"
    rec_col = "metrics/recall(B)"

    if prec_col in df.columns:
        ax.plot(df["epoch"], df[prec_col], color=C_ORANGE, linewidth=2,
                label="Precision", alpha=0.9)
    if rec_col in df.columns:
        ax.plot(df["epoch"], df[rec_col], color=C_PURPLE, linewidth=2,
                label="Recall", alpha=0.9)

    # F1 approximation
    if prec_col in df.columns and rec_col in df.columns:
        p = df[prec_col].values
        r = df[rec_col].values
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
        ax.plot(df["epoch"], f1, color=C_CYAN, linewidth=2,
                label="F1 Score", linestyle="--", alpha=0.8)

    ax.set_title("Precision, Recall & F1 Over Training", fontsize=14,
                 fontweight="bold", color="#e6edf3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = out_dir / "precision_recall.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Precision/Recall → {save_path}")
    return save_path


def plot_learning_rate(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=(12, 4))

    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    colors = [C_BLUE, C_GREEN, C_ORANGE]

    for i, col in enumerate(lr_cols):
        label = col.replace("lr/", "LR ")
        ax.plot(df["epoch"], df[col], color=colors[i % len(colors)],
                linewidth=1.8, label=label, alpha=0.9)

    ax.set_title("Learning Rate Schedule", fontsize=14,
                 fontweight="bold", color="#e6edf3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    plt.tight_layout()
    save_path = out_dir / "lr_schedule.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ LR schedule → {save_path}")
    return save_path


def plot_combined_dashboard(df: pd.DataFrame, out_dir: Path) -> Path:
    """Generate a single combined dashboard image with all key metrics."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("SATELLITE OBB DETECTION — TRAINING DASHBOARD",
                 fontsize=18, fontweight="bold", color="#e6edf3", y=0.98)

    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ── Row 1: Loss curves (4 panels) ──
    loss_configs = [
        ("train/box_loss", "val/box_loss", "Box Loss", C_BLUE, C_RED),
        ("train/cls_loss", "val/cls_loss", "Cls Loss", C_GREEN, C_ORANGE),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss", C_PURPLE, C_PINK),
        ("train/angle_loss", "val/angle_loss", "Angle Loss", C_CYAN, C_RED),
    ]
    for i, (tc, vc, title, color_t, color_v) in enumerate(loss_configs):
        ax = fig.add_subplot(gs[0, i])
        if tc in df.columns:
            ax.plot(df["epoch"], df[tc], color=color_t, lw=1.5, label="Train")
        if vc in df.columns:
            ax.plot(df["epoch"], df[vc], color=color_v, lw=1.5, ls="--", label="Val")
        ax.set_title(title, fontsize=10, color="#e6edf3")
        ax.legend(fontsize=7)
        if i == 0:
            ax.set_ylabel("Loss", fontsize=9)

    # ── Row 2: mAP + Precision/Recall ──
    ax_map = fig.add_subplot(gs[1, :2])
    for col, color, label in [
        ("metrics/mAP50(B)", C_BLUE, "mAP@50"),
        ("metrics/mAP50-95(B)", C_GREEN, "mAP@50-95"),
    ]:
        if col in df.columns:
            ax_map.plot(df["epoch"], df[col], color=color, lw=2, label=label)
            ax_map.fill_between(df["epoch"], 0, df[col], color=color, alpha=0.06)
    ax_map.set_title("Mean Average Precision", fontsize=11, color="#e6edf3")
    ax_map.set_ylabel("mAP")
    ax_map.legend(fontsize=9)
    ax_map.set_ylim(bottom=0)

    ax_pr = fig.add_subplot(gs[1, 2:])
    for col, color, label in [
        ("metrics/precision(B)", C_ORANGE, "Precision"),
        ("metrics/recall(B)", C_PURPLE, "Recall"),
    ]:
        if col in df.columns:
            ax_pr.plot(df["epoch"], df[col], color=color, lw=2, label=label)
    # F1
    if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
        p = df["metrics/precision(B)"].values
        r = df["metrics/recall(B)"].values
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
        ax_pr.plot(df["epoch"], f1, color=C_CYAN, lw=1.8, ls="--", label="F1")
    ax_pr.set_title("Precision / Recall / F1", fontsize=11, color="#e6edf3")
    ax_pr.set_ylabel("Score")
    ax_pr.legend(fontsize=9)
    ax_pr.set_ylim(0, 1.05)

    # ── Row 3: LR schedule + Summary stats ──
    ax_lr = fig.add_subplot(gs[2, :2])
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    for j, col in enumerate(lr_cols):
        ax_lr.plot(df["epoch"], df[col], color=[C_BLUE, C_GREEN, C_ORANGE][j % 3],
                   lw=1.5, label=col.replace("lr/", ""))
    ax_lr.set_title("Learning Rate Schedule", fontsize=11, color="#e6edf3")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.legend(fontsize=8)
    ax_lr.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    # Summary statistics panel
    ax_stats = fig.add_subplot(gs[2, 2:])
    ax_stats.axis("off")

    stats_lines = []
    stats_lines.append(("TRAINING SUMMARY", "", "#e6edf3", 14))
    stats_lines.append(("", "", None, 4))  # spacer line
    stats_lines.append((f"Total Epochs:", f"{int(df['epoch'].max())}", C_BLUE, 12))

    if "metrics/mAP50(B)" in df.columns:
        best_map50 = df["metrics/mAP50(B)"].max()
        best_epoch = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])
        stats_lines.append((f"Best mAP@50:", f"{best_map50:.4f} (ep {best_epoch})", C_GREEN, 12))

    if "metrics/mAP50-95(B)" in df.columns:
        best_map5095 = df["metrics/mAP50-95(B)"].max()
        stats_lines.append((f"Best mAP@50-95:", f"{best_map5095:.4f}", C_CYAN, 12))

    if "metrics/precision(B)" in df.columns:
        final_p = df["metrics/precision(B)"].iloc[-1]
        stats_lines.append((f"Final Precision:", f"{final_p:.4f}", C_ORANGE, 12))

    if "metrics/recall(B)" in df.columns:
        final_r = df["metrics/recall(B)"].iloc[-1]
        stats_lines.append((f"Final Recall:", f"{final_r:.4f}", C_PURPLE, 12))

    if "val/box_loss" in df.columns:
        final_loss = df["val/box_loss"].iloc[-1]
        stats_lines.append((f"Final Val Box Loss:", f"{final_loss:.4f}", C_RED, 12))

    y_pos = 0.92
    for label, value, color, size in stats_lines:
        if color is None:
            # Spacer line
            y_pos -= 0.06
            continue
        if value:
            ax_stats.text(0.05, y_pos, label, fontsize=size, color=C_GRAY,
                         transform=ax_stats.transAxes, fontweight="normal")
            ax_stats.text(0.55, y_pos, value, fontsize=size, color=color,
                         transform=ax_stats.transAxes, fontweight="bold")
        else:
            ax_stats.text(0.05, y_pos, label, fontsize=size, color=color,
                         transform=ax_stats.transAxes, fontweight="bold")
        y_pos -= 0.13

    save_path = out_dir / "training_dashboard.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Combined dashboard → {save_path}")
    return save_path


# ─── Main Entry ──────────────────────────────────────────────────────────────

def generate_all_plots(results_dir: str, out_dir: str = None) -> dict:
    """
    Generate all training visualization plots.

    Args:
        results_dir: Path to YOLO training run directory (containing results.csv).
        out_dir: Output directory. Defaults to <results_dir>/plots/.

    Returns:
        Dict of plot name → file path.
    """
    results_dir = Path(results_dir)
    if out_dir is None:
        out_path = results_dir / "plots"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    print(f"\n📊 Generating training plots from {results_dir / 'results.csv'}")
    print(f"   Epochs: {int(df['epoch'].max())} | Columns: {len(df.columns)}\n")

    plots = {}
    plots["loss_curves"] = str(plot_loss_curves(df, out_path))
    plots["map_progression"] = str(plot_map_progression(df, out_path))
    plots["precision_recall"] = str(plot_precision_recall(df, out_path))
    plots["lr_schedule"] = str(plot_learning_rate(df, out_path))
    plots["dashboard"] = str(plot_combined_dashboard(df, out_path))

    print(f"\n✓ All plots saved to {out_path}")
    return plots


def main():
    parser = argparse.ArgumentParser(
        description="Generate training visualization dashboard from YOLO results"
    )
    parser.add_argument("results_dir", type=str,
                        help="Path to YOLO training run directory (with results.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    generate_all_plots(args.results_dir, args.output)


if __name__ == "__main__":
    main()
