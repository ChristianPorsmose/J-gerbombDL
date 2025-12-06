from typing import Dict, List
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from analysis.utils import total_loss
from analysis.plotting.utils import (
    save_plot, set_percentile_ylim, make_grid, hide_unused_axes
)
from analysis.globals import Color, PhaseName

def _plot_epoch(df : dict, val_total : float, label : str, color_idx : int, alpha : int = 0.8, linewidth : float = 2.5):
    color_idx = color_idx % len(Color.colors)
    plt.plot(
        df["epoch"],
        val_total,
        label=label,
        linewidth=linewidth,
        color=Color.colors[color_idx],
        alpha = alpha
    )

def _add_styling_epoch(ax : Axes, ylabel : str,  title : str, color_idx : int = 0):
    color_idx = color_idx % len(Color.colors)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(
        title, fontsize=15, fontweight="bold", color=Color.colors[color_idx]
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

def plot_loss_convergence(experiments_data: List[Dict]):
    """Plot validation loss convergence for all experiments."""
    _, ax = plt.subplots(figsize=(14, 7))
    all_losses = []
    for idx, exp in enumerate(experiments_data):
        df = exp["results"]
        val_total = total_loss(df,"val")
        all_losses.extend(val_total)
        _plot_epoch(df, val_total, exp["label"], idx)
    
    set_percentile_ylim(ax, all_losses, floor=0)
    
    _add_styling_epoch(
        ax,
        ylabel="Validation Total Loss",
        title=f"Phase {PhaseName.name}: Validation Loss Convergence Comparison",
    )
    save_plot("1_loss_convergence.png")


def plot_train_val_comparison_per_experiment(experiments_data: List[Dict]):
    """Plot train vs validation loss per experiment in subplots."""
    n_exp = len(experiments_data)
    cols = n_exp if n_exp <= 3 else (n_exp + 1) // 2

    _, axes = make_grid(n_exp, max_cols=cols, base_width=6, base_height=5)

    for idx, (ax, exp) in enumerate(zip(axes, experiments_data)):
        df = exp["results"]
        train_total = total_loss(df, "train")
        val_total = total_loss(df, "val")

        _plot_epoch(df, train_total, "Train Loss", 0)
        _plot_epoch(df, val_total, "Val Loss", 1)

        _add_styling_epoch(ax, "Total Loss", exp["label"], idx)

        all_losses = list(train_total.values) + list(val_total.values)
        set_percentile_ylim(ax, all_losses, floor=0)

    hide_unused_axes(axes, n_exp)

    plt.suptitle(
        f"Phase {PhaseName.name}: Train vs Validation Loss per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    save_plot("2b_train_val_per_experiment.png", dpi=200)


def plot_loss_components_per_experiment(experiments_data: List[Dict]):
    """Plot individual validation loss components per experiment."""
    n_exp = len(experiments_data)
    cols = n_exp if n_exp <= 2 else (n_exp + 1) // 2

    _, axes = make_grid(n_exp, max_cols=cols, base_width=8, base_height=6)

    for idx, (ax, exp) in enumerate(zip(axes, experiments_data)):
        df = exp["results"]

        _plot_epoch(df, df["val/box_loss"], "Box Loss", 1, linewidth=2)
        _plot_epoch(df, df["val/cls_loss"], "Class Loss", 0, linewidth=2)
        _plot_epoch(df, df["val/dfl_loss"], "DFL Loss", 2, linewidth=2)
        _plot_epoch(df, df["val/spatial_loss"], "Spatial Loss", 3, linewidth=2)

        _add_styling_epoch(ax, "Validation Loss", exp["label"], idx)

        all_losses = np.concatenate([
            df["val/box_loss"],
            df["val/cls_loss"],
            df["val/dfl_loss"],
            df["val/spatial_loss"]
        ])
        set_percentile_ylim(ax, all_losses, floor=0, scale=1.15)

    hide_unused_axes(axes, n_exp)

    plt.suptitle(
        f"Phase {PhaseName.name}: Validation Loss Components per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    save_plot("2c_loss_components_per_experiment.png")


def plot_train_val_gap(experiments_data: List[Dict]):
    """Plot train-validation loss gap for overfitting analysis."""
    _, ax = plt.subplots(figsize=(14, 7))
    all_gaps = []

    for idx, exp in enumerate(experiments_data):
        df = exp["results"]
        train_total = total_loss(df, "train")
        val_total = total_loss(df, "val")
        gap_percent = ((train_total - val_total) / train_total * 100).abs()
        all_gaps.extend(gap_percent)

        _plot_epoch(df, gap_percent, exp["label"], idx)

    set_percentile_ylim(ax, all_gaps, floor=0)

    _add_styling_epoch(
        ax,
        ylabel="Train-Val Gap (%)",
        title=f"Phase {PhaseName.name}: Overfitting Analysis (Train-Val Loss Gap)",
    )
    save_plot("2_train_val_gap.png")


