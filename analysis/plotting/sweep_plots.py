from matplotlib import pyplot as plt
import pandas as pd

from analysis.plotting.utils import fmt_param, plot_line, save_plot


def create_parameter_sweep_plot(df_grid: pd.DataFrame, param: str):
    """Create line plots showing effect of single parameter."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    df_sorted = df_grid.sort_values(param)
    param_label = fmt_param(param)

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
