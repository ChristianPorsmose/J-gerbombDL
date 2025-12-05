from typing import Dict, List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from analysis.experiment_list import COLORS, PHASE_NAME
from analysis.plotting.utils import normalize_metrics, save_plot
from analysis.utils import calculate_convergence_epoch

def _add_bars(ax, x, width, data_series, series_labels):
    colors = plt.cm.tab10.colors

    num_series = len(data_series)
    bars = []
    
    group_span = num_series * width
    
    initial_offset = (group_span / 2) - (width / 2) 
    
    for i in range(num_series):
        bar_center_position = x - initial_offset + (i * width)
        bar_container = ax.bar(
            bar_center_position,
            data_series[i],  
            width,
            label=series_labels[i],
            color=colors[i % len(colors)], 
            edgecolor="black",
            linewidth=1.5,
        )
        bars.append(bar_container)
    return bars

def _add_bar_labels(ax: Axes, bar_containers: List, format_str: str, offset: float, threshold: float = None, size: int = 8):
    for bar_container in bar_containers:
        for bar in bar_container:
            height = bar.get_height()
    
            is_valid = not np.isnan(height)
            if threshold is not None:
                is_valid = is_valid and (height >= threshold)
            
            if is_valid:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + offset,
                    format_str.format(height),
                    ha="center",
                    va="bottom",
                    fontsize=size,
                    fontweight="bold",
                )


def _add_styling_bars(ax : Axes, ylabel : str,  title : str,labels,x = None, rotation:int = 15):
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(
        title, fontsize=15, fontweight="bold"
    )
    if x is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

def plot_bar_groups(
    ax: Axes,
    x_labels: List[str],
    data_groups: List[List[float]],
    group_labels: List[str],
    title: str,
    metric: str
):
    x = np.arange(len(x_labels))
    width = 0.8 / len(data_groups)

    bars = _add_bars(ax, x, width, data_groups, group_labels)
    _add_styling_bars(ax, f"mAP@{metric}", title, x_labels, x)
    _add_bar_labels(ax, bars, "{:.4f}", 0.002, size=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")


def plot_map_comparison_bar_plot(experiments_data, base, run):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels = [exp["label"] for exp in experiments_data]
    colors = [COLORS[exp["name"]] for exp in experiments_data]

    metrics = ["mAP50-95(B)", "mAP50(B)"]

    for ax, metric in zip((ax1, ax2), metrics):
        metric_clean = metric.split("(")[0]
        if base == "results":
            values = [exp[base][f"metrics/{metric}"].iloc[-1] for exp in experiments_data]
        else:
            values = [exp[base]["metrics"][metric_clean] for exp in experiments_data]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)

        _add_styling_bars(ax, f"mAP@{metric}", f"{run} mAP@{metric}", labels)
        _add_bar_labels(ax, [bars], "{:.4f}", 0.002)

    save_plot(f"{run}.png")


def plot_val_test_gap_bar_plot(experiments_data):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels = [exp["label"] for exp in experiments_data]
    val50 = [exp["results"]["metrics/mAP50(B)"].iloc[-1] for exp in experiments_data]
    val95 = [exp["results"]["metrics/mAP50-95(B)"].iloc[-1] for exp in experiments_data]
    test50 = [exp["test_results"]["metrics"]["mAP50"] for exp in experiments_data]
    test95 = [exp["test_results"]["metrics"]["mAP50-95"] for exp in experiments_data]

    plot_bar_groups(
        ax1, labels,
        data_groups=[val50, test50],
        group_labels=["Validation mAP@0.5", "Test mAP@0.5"],
        title="Validation vs Test Performance (mAP@0.5)",
        metric="0.5"
    )

    plot_bar_groups(
        ax2, labels,
        data_groups=[val95, test95],
        group_labels=["Validation mAP@0.5:0.95", "Test mAP@0.5:0.95"],
        title="Validation vs Test Performance (mAP@0.5:0.95)",
        metric="0.5:0.95"
    )

    save_plot("4_val_test_gap.png")


def plot_test_loss_comparison_bar_plot(experiments_data: List[Dict]):
    _, ax = plt.subplots(figsize=(14, 7))

    labels = [exp["label"] for exp in experiments_data]

    # Extract test loss values
    test_box = []
    test_cls = []
    test_dfl = []
    test_spatial = []

    for exp in experiments_data:
        test_box.append(exp["test_results"]["losses"]["box"])
        test_cls.append(exp["test_results"]["losses"]["cls"])
        test_dfl.append(exp["test_results"]["losses"]["dfl"])
        test_spatial.append(exp["test_results"]["losses"]["spatial"])

    # Collect all loss values for 95th percentile calculation
    all_losses = test_box + test_cls + test_dfl + test_spatial

    x = np.arange(len(labels))
    width = 0.2

    labels = ["Box Loss", "Class Loss", "DFL Loss", "Spatial Loss"]
    data_series = [test_box, test_cls, test_dfl, test_spatial]
    all_bar_containers = _add_bars(ax, x,width,data_series, labels)

    # Set y-axis limits to 95th percentile to avoid outlier scaling
    if len(all_losses) > 0:
        HEADROOM = 1.15
        y_max = np.percentile(all_losses, 95)
        ax.set_ylim(top=y_max * HEADROOM)

    _add_styling_bars(ax,"Loss Value",f"Phase {PHASE_NAME}: Test Loss Components Comparison", labels, x )
    _add_bar_labels(ax, all_bar_containers,"{:.3f}", 0.005, 0.01)
    save_plot("3c_test_loss_comparison.png")


def plot_convergence_speed_bar_plot(experiments_data: List[Dict]):
    """Plot convergence speed comparison."""
    _, ax = plt.subplots(figsize=(12, 7))

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


def create_top_n_comparison_bar_plot(df_grid: pd.DataFrame, n: int = 10):
    """Create comparison plot of top N configurations using _add_bars."""

    actual_n = min(n, len(df_grid))
    df_top = df_grid.nlargest(actual_n, "val_map50_95")

    if actual_n < n:
        print(f"Top N comparison: Only {actual_n} valid experiments (requested {n})")

    labels = [name[:40] + "..." if len(name) > 40 else name for name in df_top["label"]]
    x = np.arange(actual_n)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_top["test_map50_95"] = df_top["test_map50_95"].replace([np.inf, -np.inf], np.nan).fillna(0)
    val_heights = df_top["val_map50_95"].values
    test_heights = df_top["test_map50_95"].values

    width = 0.35
    data_series = [val_heights, test_heights]
    series_labels = ["Val mAP@0.5:0.95", "Test mAP@0.5:0.95"]

    bars = _add_bars(ax1, x, width, data_series, series_labels)
    _add_bar_labels(ax1, bars, "{:.4f}", 0.002)

    _add_styling_bars(
        ax1,
        "mAP Score",
        f"Top {actual_n} Configurations by Validation mAP",
        labels,
        x,
        rotation=45
    )
    ax1.legend(series_labels, fontsize=10)

    metrics = ["val_map50_95", "convergence_epoch", "stability"]
    df_norm = normalize_metrics(df_top, metrics)

    x_metric = np.arange(len(metrics))
    width = 0.8 / actual_n

    data_series = []
    for i in range(actual_n):
        values = [df_norm[m + "_norm"].iloc[i] for m in metrics]
        data_series.append(values)

    bars = _add_bars(ax2, x_metric, width, data_series, labels)
    _add_bar_labels(ax2, bars, "{:.3f}", 0.002)

    metric_labels = ["Val mAP\n(↑)", "Convergence\nEpoch (↓)", "Stability\nStd (↓)"]
    _add_styling_bars(
        ax2,
        "Normalized Score (0-1)",
        "Multi-Metric Comparison (Normalized)",
        metric_labels,
        x_metric
    )
    ax2.legend(labels, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save_plot(f"top_{actual_n}_comparison.png")
