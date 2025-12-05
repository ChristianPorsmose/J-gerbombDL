"""
Phase 1A Analysis Script
Comprehensive analysis of optimizer experiments to select the best configuration.

Generates:
- Loss convergence comparison plots
- Train-val gap analysis (overfitting)
- mAP@0.5:0.95 bar chart comparison
- Validation-test gap comparison
- Convergence speed analysis
- Stability metrics
- Decision matrix table
"""

from pathlib import Path
from typing import Optional
import click
import matplotlib.cm as cm
from analysis.experiment_list import COLORS, EXPERIMENT_LABELS, EXPERIMENTS_DIR
from analysis.plotting.bar_plots import plot_map_comparison_bar_plot, plot_test_loss_comparison_bar_plot, plot_val_test_gap_bar_plot
from analysis.utils import load_experiment_data


def apply_filter(filter:str):
    print(f"🔍 Filtering experiments with pattern: '{filter}'")
    # Search in experiments directory for matching folders
    available_experiments = [
        d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()
    ]
    experiments_to_analyze = [
        exp for exp in available_experiments if filter in exp
    ]
    if not experiments_to_analyze:
        print(f"❌ No experiments found matching filter '{filter}'")
        print(
            f"Available experiments: {', '.join(sorted(available_experiments)[:10])}..."
        )
        return
    print(f"✅ Found {len(experiments_to_analyze)} matching experiments:")
    for exp in sorted(experiments_to_analyze):
        print(f"   • {exp}")
    print()
    # Create dynamic labels and colors for filtered experiments
    EXPERIMENT_LABELS.clear()
    COLORS.clear()
    # Generate colors using a colormap
    cmap = cm.get_cmap("tab20")
    for idx, exp_name in enumerate(sorted(experiments_to_analyze)):
        EXPERIMENT_LABELS[exp_name] = exp_name.replace("_", " ").title()
        COLORS[exp_name] = cm.colors.to_hex(cmap(idx % 20))
    return experiments_to_analyze


@click.command(help=f"Phase Analysis Pipeline.")
@click.option(
    "--mode",
    type=click.Choice(["best", "average", "latest"]),
    default="best",
    help="Analysis mode: 'best' (best run by val loss), 'average' (average all runs), 'latest' (most recent)",
)
@click.option(
    "--filter",
    type=str,
    default=None,
    help="Filter experiments by substring match (e.g., 'adamW_lr0_001' to match all experiments with that pattern)",
)
@click.option(
    "--grid-search",
    type=str,
    default=None,
    help="Create grid search visualization plots. Specify name for output directory (e.g., 'adamw_lr_search')",
)
@click.option(
    "--top",
    type=int,
    default=None,
    help="Only analyze top N experiments by validation mAP (e.g., --top 10)",
)
def main(mode: str, filter: Optional[str], grid_search: Optional[str], top: Optional[int]):
    experiments_to_analyze = apply_filter(filter)
    
    experiments_data = []
    for exp_name in experiments_to_analyze:
        exp_data = load_experiment_data(exp_name, mode=mode)
        experiments_data.append(exp_data)

    #plot_map_comparison_bar_plot(experiments_data,"results", "3 Final Validation")
    #plot_map_comparison_bar_plot(experiments_data,"test_results", "4 Test")
    #plot_val_test_gap_bar_plot(experiments_data)
    #plot_test_loss_comparison_bar_plot(experiments_data)

    


if __name__ == "__main__":
    main()
