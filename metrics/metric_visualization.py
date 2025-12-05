import csv
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from ultralytics.utils.metrics import ConfusionMatrix
from utils.echo import log, log_success


def plot_all_metrics(confusion_matrix: ConfusionMatrix, base_dir: Path):
    """Generate and save all metric plots."""
    csv_path = base_dir / "metrics.csv"
    save_dir = base_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(confusion_matrix, save_dir)
    plot_csv_results(csv_path, save_dir)

    log_success(f"Metrics saved to {save_dir}")
    log(f"   - Results CSV: {csv_path}")
    log(f"   - Plots: {save_dir}/*.png")


def plot_confusion_matrix(confusion_matrix: ConfusionMatrix, save_dir: Path):
    """Generate final plots and save metrics."""
    confusion_matrix.plot(normalize=True, save_dir=str(save_dir))
    confusion_matrix.plot(normalize=False, save_dir=str(save_dir))


def plot_csv_results(csv_path: Path, save_dir: Path):
    """Generate comprehensive results plot with all training metrics."""
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for header in headers:
            data[header] = []

        for row in reader:
            for i, value in enumerate(row):
                try:
                    data[headers[i]].append(float(value))
                except:
                    data[headers[i]].append(0)

    for key in data:
        data[key] = np.array(data[key])

    epochs = data["epoch"]

    # Define plot layout: losses on left, metrics on right
    plot_configs = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Class Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("train/spatial_loss", "Train Spatial Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Class Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
        ("val/spatial_loss", "Val Spatial Loss"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("lr/pg0", "Learning Rate"),
    ]

    n_plots = len(plot_configs)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, (key, title) in enumerate(plot_configs):
        ax = axes[idx]

        if key in data and len(data[key]) > 0:
            y = data[key]

            ax.plot(epochs, y, marker="o", markersize=3, linewidth=1.5, label="Actual")

            # Plot smoothed curve if enough data points
            if len(y) > 3:
                y_smooth = gaussian_filter1d(y, sigma=2)
                ax.plot(
                    epochs,
                    y_smooth,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label="Smooth",
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title, fontweight="bold")

    # Hide extra subplots
    for idx in range(len(plot_configs), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=200, bbox_inches="tight")
    plt.close()
