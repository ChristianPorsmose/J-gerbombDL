from typing import Dict, List
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from analysis.experiment_list import COLORS, OUTPUT_DIR, PHASE_NAME
from analysis.utils import calculate_convergence_epoch, calculate_stability, extract_hyperparameters_from_name
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils.echo import log_success
from analysis.utils import total_loss
from analysis.plotting.utils import (
    plot_epoch, save_plot, add_bars, add_bar_labels, add_styling_bars, add_styling_epoch
)

def plot_loss_convergence(experiments_data: List[Dict]):
    """Plot validation loss convergence for all experiments."""
    _, ax = plt.subplots(figsize=(14, 7))

    # Collect all loss values to compute 95th percentile
    all_losses = []

    for exp_data in experiments_data:
        df = exp_data["results"]
        val_total = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        all_losses.extend(val_total.values)

        plot_epoch(df, val_total, exp_data["label"],COLORS[exp_data["name"]])

    # Set y-axis limits to 95th percentile to avoid outlier scaling
    # Filter out NaN and Inf
    all_losses_clean = [x for x in all_losses if np.isfinite(x)]

    if len(all_losses_clean) > 0:
        y_max = np.percentile(all_losses_clean, 95)
        y_min = np.percentile(all_losses_clean, 5)
        plt.ylim(bottom=max(0, y_min * 0.9), top=y_max * 1.1)

    add_styling_epoch(
        ax,
        "Validation Total Loss", 
        f"Phase {PHASE_NAME}: Validation Loss Convergence Comparison",
        epoch_size=13,
        title_size=15
        )

    save_plot("1_loss_convergence.png", 300)


def plot_train_val_comparison_per_experiment(experiments_data: List[Dict]):
    """Plot train vs validation loss for each experiment in separate subplots."""
    n_experiments = len(experiments_data)

    # Create subplots: 2 rows if more than 3 experiments, else 1 row
    if n_experiments <= 3:
        rows, cols = 1, n_experiments
        figsize = (6 * n_experiments, 5)
    else:
        rows = 2
        cols = (n_experiments + 1) // 2
        figsize = (6 * cols, 10)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (ax, exp_data) in enumerate(zip(axes, experiments_data)):
        df = exp_data["results"]

        train_total = total_loss(df, "train")
        val_total = total_loss(df, "val")

        plot_epoch(df, train_total, "Train Loss","#3498db")
        plot_epoch(df, val_total, "Val Loss", "#e74c3c")
 
        add_styling_epoch(ax, "Total Loss",exp_data["label"], color=COLORS[exp_data["name"]])

        # Set y-axis limits based on percentiles to avoid outliers
        all_losses = list(train_total.values) + list(val_total.values)
        # Filter out NaN and Inf
        all_losses_clean = [x for x in all_losses if np.isfinite(x)]

        if len(all_losses_clean) > 0:
            y_min = np.percentile(all_losses_clean, 5)
            y_max = np.percentile(all_losses_clean, 95)
            ax.set_ylim(bottom=max(0, y_min * 0.9), top=y_max * 1.1)
        ax.set_ylim(1, 5)

    # Hide extra subplots if any
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Phase {PHASE_NAME}: Train vs Validation Loss per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    save_plot("2b_train_val_per_experiment.png",200)

def plot_loss_components_per_experiment(experiments_data: List[Dict]):
    """Plot individual loss components for each experiment in separate subplots."""
    n_experiments = len(experiments_data)

    # Create subplots: 2 rows if more than 2 experiments, else 1 row
    if n_experiments <= 2:
        rows, cols = 1, n_experiments
        figsize = (8 * n_experiments, 6)
    else:
        rows = 2
        cols = (n_experiments + 1) // 2
        figsize = (8 * cols, 12)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    losses = ["box_loss", "cls_loss", "dfl_loss", "spatial_loss"]

    for idx, (ax, exp_data) in enumerate(zip(axes, experiments_data)):
        df = exp_data["results"]

        plot_epoch(df, df["val/box_loss"], "Box Loss","#e74c3c", linewidth=2)
        plot_epoch(df,df["val/cls_loss"], "Class Loss","#3498db", linewidth=2)
        plot_epoch(df,df["val/dfl_loss"],"DFL Loss", "#2ecc71", linewidth=2 )

        if "val/spatial_loss" in df.columns:
            plot_epoch(df, df["val/spatial_loss"],"Spatial Loss", "#9b59b6",linewidth=2 )

        add_styling_epoch(ax, "Validation Loss", exp_data["label"], color=COLORS[exp_data["name"]], title_size=11)

        # Set y-axis limits based on percentiles
        all_losses = []
        for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
            all_losses.extend(df[col].values)
        if "val/spatial_loss" in df.columns:
            all_losses.extend(df["val/spatial_loss"].values)

        # Filter out NaN and Inf
        all_losses_clean = [x for x in all_losses if np.isfinite(x)]

        if len(all_losses_clean) > 0:
            y_min = np.percentile(all_losses_clean, 5)
            y_max = np.percentile(all_losses_clean, 95)
            ax.set_ylim(bottom=max(0, y_min * 0.2), top=y_max * 0.8)

    # Hide extra subplots if any
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Phase {PHASE_NAME}: Validation Loss Components per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    save_plot("2c_loss_components_per_experiment.png")

def plot_train_val_gap(experiments_data: List[Dict]):
    """Plot train-val gap over time for overfitting analysis."""
    _, ax = plt.subplots(figsize=(14, 7))

    # Collect all gap values for percentile-based scaling
    all_gaps = []
    for exp_data in experiments_data:
        df = exp_data["results"]
        train_total = total_loss(df, "train")
        val_total = total_loss(df, "val")
        gap_percent = ((train_total - val_total) / train_total * 100).abs()
        all_gaps.extend(gap_percent.values)

        plot_epoch(df, gap_percent, exp_data["label"],COLORS[exp_data["name"]])

    # Set y-axis limits based on 5th-95th percentile to handle outliers
    # Filter out NaN and Inf
    all_gaps_clean = [x for x in all_gaps if np.isfinite(x)]

    if len(all_gaps_clean) > 0:
        p5 = np.percentile(all_gaps_clean, 5)
        p95 = np.percentile(all_gaps_clean, 95)
        plt.ylim(bottom=max(0, p5 * 0.9), top=p95 * 1.1)

    plt.ylim(0, 10)

    add_styling_epoch(
        ax,"Train-Val Gap (%)",
        f"Phase {PHASE_NAME}: Overfitting Analysis (Train-Val Loss Gap)",
        title_size=15,
        epoch_size=13
        )

    save_plot("2_train_val_gap.png")


def plot_convergence_speed(experiments_data: List[Dict]):
    """Plot convergence speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = [exp["label"] for exp in experiments_data]
    convergence_epochs = [
        calculate_convergence_epoch(exp["results"]) for exp in experiments_data
    ]
    colors_list = [COLORS[exp["name"]] for exp in experiments_data]

    bars = ax.barh(
        labels, convergence_epochs, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax.set_xlabel("Epochs to Reach 95% of Final mAP", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Phase {PHASE_NAME}: Convergence Speed Comparison",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, convergence_epochs)):
        ax.text(
            val + 0.5,
            i,
            f"{val}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    save_plot("5_convergence_speed.png")
