

from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from analysis.experiment_list import OUTPUT_DIR
from utils.echo import log_success


def save_plot(path : Path, dpi: int = 300):
    plt.tight_layout()
    save_path = OUTPUT_DIR / path
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log_success(f"Saved: {save_path}")

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


def plot_line(ax,df,param,label, y, title, color, marker="o"):
    ax.plot(
        df[param],
        df[y],
        marker + "-",
        linewidth=2.5,
        markersize=8,
        color=color,
        label=y,
    )
    ax.set_xlabel(label, fontsize=11, fontweight="bold")
    ax.set_title(f"{title} vs {label}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def normalize_column(df, col, out_col):
    unique_vals = sorted(df[col].unique())
    if len(unique_vals) > 1:
        mapping = {v: i / (len(unique_vals) - 1) for i, v in enumerate(unique_vals)}
        df[out_col] = df[col].map(mapping)
    else:
        df[out_col] = 0.5


def pretty_label(name: str) -> str:
    if name == "val_map50_95_norm":
        return "Val mAP@0.5:0.95"
    base = name.replace("_norm", "").replace("_", " ").title()
    return {"Pretrained Flag": "Model Variant"}.get(base, base)


def compute_best_by_variant(df):
    result = {}
    if "pretrained_flag" not in df.columns:
        return result
    for variant in sorted(df["pretrained_flag"].unique()):
        sub = df[df["pretrained_flag"] == variant]
        if len(sub) == 0:
            continue
        idx = sub["val_map50_95"].idxmax()
        try:
            result[int(idx)] = int(variant)
        except Exception:
            result[idx] = int(variant)
    return result


def draw_glow_highlight(ax, x_positions, values, color, marker, label=None, z=3):
    for w in [6, 4, 2]:
        ax.plot(x_positions, values, color=color, alpha=0.18, linewidth=w, zorder=1)
    ax.plot(
        x_positions,
        values,
        color=color,
        linewidth=2.8,
        marker=marker,
        markersize=6,
        linestyle="-",
        markerfacecolor=color,
        markeredgecolor="white",
        label=label,
        zorder=z,
    )


def annotate_axis(ax, axis_x, orig_param, df, is_map=False):
    unique_vals = sorted(df[orig_param].unique())
    for j, val in enumerate(unique_vals):
        pos = j / (len(unique_vals) - 1) if len(unique_vals) > 1 else 0.5

        if is_map:
            text = f"{val:.4f}"
        else:
            if orig_param == "pretrained_flag":
                text = "Small" if int(val) == 0 else "Pretrained"
            elif isinstance(val, (int, np.integer)):
                text = f"{int(val)}"
            elif isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)

        ax.text(
            axis_x + 0.15,
            pos,
            text,
            fontsize=8,
            va="center",
            ha="left",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.7,
                edgecolor="gray",
            ),
        )


def normalize_map_values(df):
    non_zero = df[df["val_map50_95"] > 0]["val_map50_95"]
    m_min = non_zero.min() if len(non_zero) else df["val_map50_95"].min()
    m_max = df["val_map50_95"].max()
    return m_min, m_max


def normalize_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Normalize metrics to 0-1, inverting for 'lower is better'."""
    df_norm = df.copy()
    for metric in metrics:
        vals = df_norm[metric].values.astype(float)
        if metric == "val_map50_95":  # higher is better
            df_norm[metric + "_norm"] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        else:  # lower is better
            df_norm[metric + "_norm"] = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return df_norm