
from typing import Dict, List

from matplotlib import cm, pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from analysis.experiment_list import OUTPUT_DIR, PHASE_NAME
from analysis.plotting.bar_plots import create_top_n_comparison_bar_plot
from analysis.plotting.utils import annotate_axis, compute_best_by_variant, draw_glow_highlight, make_grid, normalize_column, normalize_map_values, plot_line, pretty_label, save_plot
from analysis.utils import calculate_convergence_epoch, calculate_stability, extract_hyperparameters_from_name
from utils.echo import log_success, log

def build_grid_dataframe(experiments_data: List[Dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments_data:
        params = extract_hyperparameters_from_name(exp["name"])
        res = exp["results"]
        metrics = {
            "name": exp["name"],
            "label": exp["label"],
            "val_map50_95": res["metrics/mAP50-95(B)"].iloc[-1],
            "val_map50": res["metrics/mAP50(B)"].iloc[-1],
            "final_val_loss": (res["val/box_loss"].iloc[-1] + res["val/cls_loss"].iloc[-1] + res["val/dfl_loss"].iloc[-1]),
            "convergence_epoch": calculate_convergence_epoch(res),
            "stability": calculate_stability(res),
            "test_map50": exp["test_results"]["metrics"]["mAP50"],
            "test_map50_95": exp["test_results"]["metrics"]["mAP50-95"],
        }
        rows.append({**params, **metrics})
    return pd.DataFrame(rows)

def _fmt_param(p: str) -> str:
    return p.replace("_", " ").title()

def create_grid_search_plots(experiments_data: List[Dict]):
    """Create comprehensive grid search visualization plots."""
    df_grid = build_grid_dataframe(experiments_data)

    # Identify which hyperparameters vary
    varying_params = []
    # Include standard params plus special-case params used in jagerloss grid
    for col in [
        "lr",
        "momentum",
        "weight_decay",
        "batch_size",
        "optimizer",
        "pretrained_flag",
        "lambda_rate",
        ]:
        if col in df_grid.columns and df_grid[col].nunique() > 1:
            varying_params.append(col)

    log(f"   Varying hyperparameters: {', '.join(varying_params)}")

    if len(varying_params) >= 2:
        create_heatmap_plots(df_grid, varying_params)

    for param in varying_params:
        create_parameter_sweep_plot(df_grid, param)

    plot_df, best_idx, best_mAP, best_by_variant = prepare_parallel_coordinates_data(df_grid, varying_params)
    plot_parallel_coordinates(plot_df, varying_params, best_idx, best_mAP, best_by_variant)

    create_top_n_comparison_bar_plot(df_grid,n=min(10, len(df_grid)))

    if len(varying_params) >= 2:
        create_parameter_importance_plot(df_grid, varying_params)

    csv_path = OUTPUT_DIR / "grid_search_results.csv"
    df_grid.to_csv(csv_path, index=False, float_format="%.6f")
    log_success(f"Saved: {csv_path}")
    log_success(f"Grid search visualizations complete!")



def create_heatmap_plots(df_grid: pd.DataFrame, varying_params: List[str]):
    metrics_to_plot = [
        ("val_map50_95", "Validation mAP@0.5:0.95", "RdYlGn"),
        ("final_val_loss", "Final Validation Loss", "RdYlGn_r"),
        ("convergence_epoch", "Convergence Epoch", "RdYlGn_r"),
    ]

    for i, p1 in enumerate(varying_params):
        for p2 in varying_params[i + 1:]:
            if df_grid[p1].nunique() < 2 or df_grid[p2].nunique() < 2:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (metric, title, cmap ) in zip(axes, metrics_to_plot):
                pivot = df_grid.pivot_table(values=metric, index=p2, columns=p1, aggfunc="mean")
                im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_xticklabels([f"{v:.4f}" for v in pivot.columns], rotation=45, ha="right")
                ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
                ax.set_xlabel(_fmt_param(p1))
                ax.set_ylabel(_fmt_param(p2))
                ax.set_title(title)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=9)

                # Annotate
                rows, cols = pivot.shape
                for r in range(rows):
                    for c in range(cols):
                        val = pivot.values[r, c]
                        text_color = "white" if (im.norm(val) > 0.5) else "black"
                        ax.text(c, r, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

            fig.suptitle(f"Grid Search Heatmap: {p1} vs {p2}", fontsize=14, fontweight="bold")
            save_plot(f"heatmap_{p1}_vs_{p2}.png")

def create_parameter_sweep_plot(df_grid: pd.DataFrame, param: str):
    """Create line plots showing effect of single parameter."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    df_sorted = df_grid.sort_values(param)
    param_label = _fmt_param(param)

    ax = axes[0]
    plot_line(ax,df_sorted,param,param_label, "val_map50_95", "Performance", "#3498db", "o")
    plot_line(ax,df_sorted,param,param_label, "val_map50", "Performance", "#2ecc71", "s")
    if "test_map50_95" in df_sorted.columns:
        plot_line(ax,df_sorted,param,param_label, "test_map50_95", "Performance", "#e74c3c", "^")

    ax.set_ylabel("mAP Score", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    plot_line(ax,df_sorted,param,param_label,"final_val_loss", "Loss", "#e74c3c", "o")
    ax.set_ylabel("Final Validation Loss", fontsize=11, fontweight="bold")

    ax = axes[2]
    plot_line(ax,df_sorted,param,param_label, "convergence_epoch", "Convergence Speed", "#9b59b6")
    ax.set_ylabel("Convergence Epoch", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    ax = axes[3]
    plot_line(ax,df_sorted,param,param_label, "stability", "Stability", "#f39c12")
    ax.set_ylabel("Stability (Std Dev)", fontsize=11, fontweight="bold")

    save_plot(f"sweep_{param}.png")


def prepare_parallel_coordinates_data(df_grid: pd.DataFrame, varying_params: List[str]):
    """Precompute all information needed for parallel coordinates plot."""
    plot_df = df_grid.copy()

    # Best configs
    best_idx = plot_df["val_map50_95"].idxmax()
    best_mAP = plot_df.loc[best_idx, "val_map50_95"]
    best_by_variant = compute_best_by_variant(plot_df)

    # Normalize parameters
    for p in varying_params:
        if p in plot_df.columns and plot_df[p].dtype in [np.float64, np.int64, int]:
            normalize_column(plot_df, p, f"{p}_norm")
    normalize_column(plot_df, "val_map50_95", "val_map50_95_norm")

    # Performance categories
    plot_df["Performance"] = pd.cut(
        plot_df["val_map50_95"], bins=3, labels=["Low", "Medium", "High"]
    )

    return plot_df, best_idx, best_mAP, best_by_variant


def plot_parallel_coordinates(
    plot_df: pd.DataFrame,
    varying_params: List[str],
    best_idx: int,
    best_mAP: float,
    best_by_variant: dict
):
    """Draw parallel coordinates plot given precomputed data."""
    
    cols_to_plot = [f"{p}_norm" for p in varying_params if f"{p}_norm" in plot_df]
    cols_to_plot += ["val_map50_95_norm", "Performance"]
    if len(cols_to_plot) < 3:
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("RdYlGn")
    m_min, m_max = normalize_map_values(plot_df)
    used_labels = set()

    for idx, row in plot_df.iterrows():
        values = [row[c] for c in cols_to_plot[:-1]]
        x_positions = list(range(len(values)))
        m_norm = ((row["val_map50_95"] - m_min) / (m_max - m_min)) if m_max > m_min else 0.5
        color = cmap(norm(m_norm))

        # Highlight logic
        if idx in best_by_variant:
            variant = best_by_variant[idx]
            key = "small" if variant == 0 else "pretrained"
            label = None if key in used_labels else ("Best Small" if variant==0 else "Best Pretrained") + f": {row['val_map50_95']:.4f}"
            hl_color = "#2c3e50" if variant==0 else "#8e44ad"
            hl_marker = "o" if variant==0 else "s"
            draw_glow_highlight(ax, x_positions, values, hl_color, hl_marker, label)
            used_labels.add(key)
        elif idx == best_idx:
            draw_glow_highlight(ax, x_positions, values, "black", "o", f"Best: {best_mAP:.4f}")
        else:
            ax.plot(x_positions, values, color=color, alpha=0.6, linewidth=1.5)

    for i in range(len(cols_to_plot) - 1):
        ax.axvline(i, color="gray", linewidth=1.5, alpha=0.5)
    ax.set_xticks(range(len(cols_to_plot) - 1))
    ax.set_xticklabels([pretty_label(c) for c in cols_to_plot[:-1]], rotation=45, ha="right")

    for i, col in enumerate(cols_to_plot[:-1]):
        orig_param = col.replace("_norm", "")
        annotate_axis(ax, i, orig_param, plot_df, is_map=(col=="val_map50_95_norm"))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Val mAP@0.5:0.95", fontsize=11, fontweight="bold")
    ticks = np.linspace(0, 1, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{m_min + (m_max - m_min) * t:.4f}" for t in ticks])

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, len(cols_to_plot) - 1.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Parallel Coordinates: Hyperparameter Configuration Overview", fontsize=14, fontweight="bold")

    save_plot("parallel_coordinates.png", dpi=250)

def get_numeric_params(df: pd.DataFrame, params: List[str]) -> List[str]:
    """Return subset of params that are numeric in df."""
    return [p for p in params if p in df.columns and df[p].dtype in [np.float64, np.int64]]


def compute_metric(df: pd.DataFrame, params: List[str], func) -> dict:
    """Compute metric (e.g., correlation or variance) for numeric params."""
    numeric_params = get_numeric_params(df, params)
    metrics = {}
    for p in numeric_params:
        try:
            val = func(df, p)
            if not np.isnan(val):
                metrics[p] = val
        except Exception:
            continue
    return metrics


def plot_horizontal_bars(ax, data_dict: dict, xlabel: str, title: str, color: str, fmt="{:.3f}"):
    """Plot horizontal bar chart with values annotated."""
    if not data_dict:
        return
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    param_names = [_fmt_param(p) for p, _ in sorted_items]
    values = [v for _, v in sorted_items]

    bars = ax.barh(param_names, values, color=color, edgecolor="black")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 1e-8, i, fmt.format(val), va="center", ha="left", fontsize=10, fontweight="bold")


def create_parameter_importance_plot(df_grid: pd.DataFrame, varying_params: List[str]):
    """Analyze and plot parameter importance using correlation and variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    corr_func = lambda df, p: abs(df[[p, "val_map50_95"]].corr().iloc[0, 1])
    correlations = compute_metric(df_grid, varying_params, corr_func)
    plot_horizontal_bars(ax1, correlations, "Absolute Correlation with Val mAP", "Hyperparameter Importance", "#3498db")

    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(subset=["val_map50_95"])
    variance_func = lambda df, p: (df.groupby(p)["val_map50_95"].max() - df.groupby(p)["val_map50_95"].min()).mean()
    variance_explained = compute_metric(df_clean, varying_params, variance_func)
    plot_horizontal_bars(ax2, variance_explained, "mAP Range Induced", "Parameter Impact on Performance", "#e74c3c", fmt="{:.4f}")

    save_plot("parameter_importance.png")


def style_row(table, col_labels, row_idx: int, facecolor: str, text_color: str = "black", bold: bool = True):
    for i in range(len(col_labels)):
        table[(row_idx, i)].set_facecolor(facecolor)
        table[(row_idx, i)].set_text_props(weight="bold" if bold else "normal", color=text_color)

def save_decision_matrix(df_matrix: pd.DataFrame):
    """Save decision matrix to CSV and create visualization."""

    # Save to CSV
    csv_path = OUTPUT_DIR / "decision_matrix.csv"
    df_matrix.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✅ Saved: {csv_path}")

    # Create visual table
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("tight")
    ax.axis("off")

    # Columns to display and their formatted labels
    display_cols = get_display_cols()
    col_labels = get_col_lables()
    col_widths = [0.15, 0.11, 0.11, 0.1, 0.11, 0.1, 0.1, 0.1]

    table = ax.table(
        cellText=df_matrix[display_cols].values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )

    # Table styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)


    style_row(table, col_labels, 0, "#3498db", text_color="white")

    style_row(table, col_labels, "#2ecc71")

    plt.title(
        f"Phase {PHASE_NAME} Decision Matrix (Sorted by Total Score)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    save_plot("6_decision_matrix.png")



def get_col_lables():
    col_labels = [
        "Experiment",
        "Val mAP\n@0.5:0.95 ↑",
        "Test mAP\n@0.5 ↑",
        "Val-Test\nGap % ↓",
        "Train-Val\nGap % ↓",
        "Converge\nEpoch ↓",
        "Stability\nStd ↓",
        "Score ↓",
    ]
    
    return col_labels

def get_display_cols():
    display_cols = [
        "Experiment",
        "Val_mAP@0.5:0.95",
        "Test_mAP@0.5",
        "Val-Test_Gap_%",
        "Train-Val_Gap_%",
        "Convergence_Epoch",
        "Stability_Std",
        "Total_Score",
    ]
    
    return display_cols
