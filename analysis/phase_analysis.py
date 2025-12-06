import os
from pathlib import Path
import click
import matplotlib.cm as cm
from analysis.decision_matrix.decision_matrix import save_decision_matrix
from analysis.decision_matrix.generate import generate_decision_matrix
from analysis.globals import Output, PhaseName, Experiments
from analysis.plotting.line_plots import plot_loss_components_per_experiment, plot_loss_convergence, plot_train_val_comparison_per_experiment, plot_train_val_gap
from analysis.plotting.bar_plots import plot_convergence_speed_bar_plot, plot_map_comparison_bar_plot, plot_test_loss_comparison_bar_plot, plot_val_test_gap_bar_plot
from analysis.utils import load_experiment_data
from analysis.grid_search import create_grid_search_plots
from utils.echo import log_error, log_info, log_success, log

def apply_filter(filter:str):
    log_info(f"Filtering experiments with pattern: '{filter}'")
    
    available_experiments = [
        d.name for d in Experiments.path.iterdir() if d.is_dir()
    ]
    experiments_to_analyze = [
        exp for exp in available_experiments if filter in exp
    ]
    if not experiments_to_analyze:
        log_error(f"No experiments found matching filter '{filter}'")
        log(
            f"Available experiments: {', '.join(sorted(available_experiments)[:10])}..."
        )
        return
    log_success(f"Found {len(experiments_to_analyze)} matching experiments:")
    for exp in sorted(experiments_to_analyze):
        log(f"   • {exp}")
    log("")

    return experiments_to_analyze


@click.command(help=f"Phase Analysis Pipeline.")
@click.option(
    "--filter",
    type=str,
    default=None,
    help="Filter experiments by substring match (e.g., 'adamW_lr0_001' to match all experiments with that pattern)",
)
@click.option(
    "--grid",
    is_flag=True,          
    default=False,    
    help="Toggle generation of detailed grid-search diagnostic plots."
)
@click.option(
    "-o",
    "--output",
    default="Analysis_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Directory where analysis results will be saved. Created if it does not exist.",
)
def main(filter: str, grid: bool, output):
    os.makedirs(output, exist_ok=True)
    
    Output.path = Path(output)
    PhaseName.name = Path(filter)

    experiments_to_analyze = apply_filter(filter)

    experiments_data = []
    for exp_name in experiments_to_analyze:
        exp_data = load_experiment_data(exp_name)
        experiments_data.append(exp_data)

    plot_map_comparison_bar_plot(experiments_data,"results", "3 Final Validation")
    plot_map_comparison_bar_plot(experiments_data,"test_results", "4 Test")
    plot_val_test_gap_bar_plot(experiments_data)
    plot_test_loss_comparison_bar_plot(experiments_data) # DEN HER SKAL TJEKKES EFTER
    plot_convergence_speed_bar_plot(experiments_data)
    plot_loss_convergence(experiments_data)
    plot_train_val_comparison_per_experiment(experiments_data)
    plot_loss_components_per_experiment(experiments_data)
    plot_train_val_gap(experiments_data)

    if grid:
        create_grid_search_plots(experiments_data)

    df_matrix = generate_decision_matrix(experiments_data)
    save_decision_matrix(df_matrix)


if __name__ == "__main__":
    main()
