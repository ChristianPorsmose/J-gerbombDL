

from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from analysis.experiment_list import OUTPUT_DIR
from utils.echo import log_success


def save_plot(path : Path, dpi: int = 300):
    plt.tight_layout()
    save_path = OUTPUT_DIR / path
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log_success(f"Saved: {save_path}")

def plot_epoch(df : dict, val_total : float, label : str, color, alpha : int = 0.8, linewidth : float = 2.5):
    plt.plot(
        df["epoch"],
        val_total,
        label=label,
        linewidth=linewidth,
        color=color,
        alpha = alpha
    )

def add_bars(ax, x, width, data_series, series_labels):
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

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

def add_bar_labels(ax: Axes, bar_containers: List, format_str: str, offset: float, threshold: float = None, size: int = 8):
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


def add_styling_bars(ax : Axes, ylabel : str,  title : str,labels,x = None
                    ):
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(
        title, fontsize=15, fontweight="bold"
    )
    if x:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

def add_styling_epoch(ax : Axes, ylabel : str,  title : str, color:str = "black", epoch_size:int=11, title_size :int = 9 ):
    ax.set_xlabel("Epoch", fontsize=epoch_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(
        title, fontsize=title_size, fontweight="bold", color=color
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

def set_percentile_ylim(ax, values, pmin=5, pmax=95, floor=0, scale=1.1):
    clean = [v for v in values if np.isfinite(v)]
    if not clean:
        return
    y_min = np.percentile(clean, pmin)
    y_max = np.percentile(clean, pmax)
    ax.set_ylim(bottom=max(floor, y_min * 0.9), top=y_max * scale)

def make_grid(n, max_cols=3, base_width=6, base_height=5):
    if n <= max_cols:
        rows, cols = 1, n
    else:
        rows = (n + max_cols - 1) // max_cols
        cols = max_cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * base_width, rows * base_height))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    return fig, axes

def hide_unused_axes(axes, used):
    for ax in axes[used:]:
        ax.set_visible(False)
