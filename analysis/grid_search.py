
from typing import Dict, List

from matplotlib import cm, pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from analysis.experiment_list import OUTPUT_DIR, PHASE_NAME
from analysis.plotting.bar_plots import create_parameter_importance_plot, create_top_n_comparison_bar_plot
from analysis.plotting.heat_maps import create_heatmap_plots
from analysis.plotting.sweep_plots import create_parameter_sweep_plot
from analysis.plotting.utils import annotate_axis, compute_best_by_variant, draw_glow_highlight, fmt_param, make_grid, normalize_column, normalize_map_values, plot_line, pretty_label, save_plot
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