from typing import Dict, List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from analysis.experiment_list import COLORS, PHASE_NAME
from analysis.plotting.utils import add_bar_labels, add_bars, add_styling_bars, save_plot

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

    bars = add_bars(ax, x, width, data_groups, group_labels)
    add_styling_bars(ax, f"mAP@{metric}", title, x_labels, x)
    add_bar_labels(ax, bars, "{:.4f}", 0.002, size=9)
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

        add_styling_bars(ax, f"mAP@{metric}", f"{run} mAP@{metric}", labels)
        add_bar_labels(ax, [bars], "{:.4f}", 0.002)

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
    all_bar_containers = add_bars(ax, x,width,data_series, labels)

    # Set y-axis limits to 95th percentile to avoid outlier scaling
    if len(all_losses) > 0:
        HEADROOM = 1.15
        y_max = np.percentile(all_losses, 95)
        ax.set_ylim(top=y_max * HEADROOM)

    add_styling_bars(ax,"Loss Value",f"Phase {PHASE_NAME}: Test Loss Components Comparison", labels, x )
    add_bar_labels(ax, all_bar_containers,"{:.3f}", 0.005, 0.01)
    save_plot("3c_test_loss_comparison.png")