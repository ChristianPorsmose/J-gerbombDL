from typing import List
from matplotlib import pyplot as plt
import pandas as pd

from analysis.plotting.utils import fmt_param, save_plot


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
                ax.set_xlabel(fmt_param(p1))
                ax.set_ylabel(fmt_param(p2))
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