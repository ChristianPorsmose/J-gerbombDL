"""
Grid Search Experiment Runner

This script performs a grid search over hyperparameter combinations.
It takes a base configuration dictionary and multiple parameter lists,
then runs all combinations of those parameters.

Usage:
    python run_grid_search.py
"""

import click
import yaml
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from experiments_scripts.experiment_configs import (
    OPTIMIZER_PRETRAINED,
    OPTIMIZER_SMALL,
    PRETRAINED_DATASET_BASE,
    SMALL_CUSTOM_LOSS_BASE,
    PRETRAINED_CUSTOM_LOSS_BASE,
    LAMBDA_RATES,
    LEARNING_RATES,
    MOMENTUMS,
    SMALL_DATASET_BASE,
    WEIGHT_DECAYS,
    PRETRAINED_OPTIONS,
)
from utils.echo import log, log_error, log_warning, log_success, log_info


def set_nested(config: dict, key: str, value: any):
    """
    Set a value in a nested dict using dot notation for keys.
    """
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def generate_experiment_configs(
    base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from the base config and parameter grid.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    combinations = list(itertools.product(*param_values))

    configs = []
    for combo_idx, combo in enumerate(combinations, 1):

        config = base_config.copy()

        base_exp_name = base_config.get("experiment_name", "grid_search")

        name_parts = [base_exp_name]

        for param_name, param_value in zip(param_names, combo):
            set_nested(config, param_name, param_value)

            if isinstance(param_value, float):
                name_parts.append(f"{param_name}{param_value:.4f}".replace(".", "_"))
            else:
                name_parts.append(f"{param_name}{param_value}")

        config["experiment_name"] = "_".join(name_parts)

        configs.append(config)

    return configs


def log_experiment_header(experiment_idx, total_experiments, experiment_name):
    log(f"\n{'='*80}")
    log_info(f"Experiment [{experiment_idx}/{total_experiments}]: {experiment_name}")
    log(f"{'='*80}")


def log_key_params(config):
    log("Parameters:")
    for key in [
        "lr",
        "momentum",
        "weight_decay",
        "optimizer",
        "augmentation",
        "dataset_size",
        "freeze_backbone_layers",
    ]:
        if key in config:
            log(f"  {key}: {config[key]}")
    log("")


def run_experiment_from_config(
    config: Dict[str, Any], experiment_idx: int, total_experiments: int
) -> bool:
    """
    Run a single experiment from a configuration dictionary.
    """
    experiment_name = config.get("experiment_name", f"experiment_{experiment_idx}")

    log_experiment_header(experiment_idx, total_experiments, experiment_name)

    log_key_params(config)

    temp_config_path = Path(f"temp_config_{experiment_idx}.yaml")

    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        subprocess.run(
            ["python", "main.py", "--config", str(temp_config_path)],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
        )

        log_success(f"Experiment '{experiment_name}' completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        log_error(
            f"Experiment '{experiment_name}' failed with error code {e.returncode}"
        )
        return False

    except KeyboardInterrupt:
        log_warning(f"Experiment '{experiment_name}' interrupted by user")
        raise

    finally:
        temp_config_path.unlink()


def run_grid_search(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Run grid search over all parameter combinations.
    """
    start_time = datetime.now()

    configs = generate_experiment_configs(base_config, param_grid)

    results = {"total": len(configs), "succeeded": 0, "failed": 0, "experiments": []}

    log_experiment_options(
        base_config, param_grid, continue_on_error, start_time, configs
    )

    for i, config in enumerate(configs, 1):
        try:
            success = run_experiment_from_config(config, i, len(configs))

            results["experiments"].append(
                {
                    "name": config["experiment_name"],
                    "config": config,
                    "success": success,
                }
            )

            if success:
                results["succeeded"] += 1
            else:
                results["failed"] += 1
                if not continue_on_error:
                    log_warning(
                        f"Stopping grid search due to failure (continue_on_error=False)"
                    )
                    break

        except KeyboardInterrupt:
            log_warning(f"Grid search interrupted by user")
            break

    end_time = datetime.now()
    duration = end_time - start_time

    log_summary(results, end_time, duration)

    return results


def log_summary(results, end_time, duration):
    log(f"\n{'#'*80}")
    log_info(f"GRID SEARCH SUMMARY")
    log(f"{'#'*80}")
    log_info(f"Total experiments: {results['total']}")
    log_success(f"Succeeded: {results['succeeded']}")
    log_error(f"Failed: {results['failed']}")
    log_info(f"Duration: {duration}")
    log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Results by experiment:")
    for exp in results["experiments"]:
        status_fn = log_success if exp["success"] else log_error
        status_fn(f"{exp['name']}")
    log(f"{'#'*80}\n")


def log_experiment_options(
    base_config, param_grid, continue_on_error, start_time, configs
):
    log(f"\n{'#'*80}")
    log_info(f"GRID SEARCH")
    log(f"{'#'*80}")
    log_info(f"Base configuration:")
    for key, value in base_config.items():
        if key not in param_grid:  # Only show fixed parameters
            log(f"  {key}: {value}")
    log(f"Parameter grid:")
    for param_name, param_values in param_grid.items():
        log(f"  {param_name}: {param_values} ({len(param_values)} values)")
    log_info(f"\nTotal combinations: {len(configs)}")
    log(f"Continue on error: {continue_on_error}")
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'#'*80}\n")


LEARNING_RATE_GRID = {
    "lr": LEARNING_RATES,
    "momentum": MOMENTUMS,
    "weight_decay": WEIGHT_DECAYS,
}

CUSTOM_LOSS_GRID = {
    "lamda_rate": LAMBDA_RATES,
}

BASE_CONFIGS = {
    "small": SMALL_CUSTOM_LOSS_BASE,
    "pretrained": PRETRAINED_CUSTOM_LOSS_BASE,
    "optimizer_small": OPTIMIZER_SMALL,
    "optimizer_pretrained": OPTIMIZER_PRETRAINED,
    "small_dataset": SMALL_DATASET_BASE,
    "pretrained_dataset": PRETRAINED_DATASET_BASE,
}

PHASES = {
    "phase1": ["optimizer_small", "optimizer_pretrained"],
    "phase2": ["small", "pretrained"],
    "phase3": ["small_dataset", "pretrained_dataset"],
}


def total_combinations(grid):
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def log_grid_info(name, grid):
    total = total_combinations(grid)
    log_warning(f"Running grid: {name} -> {total} combinations")
    log(f"Parameters: {list(grid.keys())}")
    log(f"Estimated time: {total * 20 / 60:.1f} hours\n")


def validate_input(phase, base, grid):
    if phase == "single":
        if base is None:
            raise click.UsageError("--base is required when --phase=single")
        if grid is None:
            raise click.UsageError("--grid is required when --phase=single")

    if phase != "single" and grid is not None:
        raise click.UsageError(
            "--grid should NOT be used with phase1 or phase2. "
            "Grid is automatically selected by phase."
        )


@click.command("grid-search")
@click.option(
    "--phase",
    type=click.Choice(["phase1", "phase2", "phase3", "single"]),
    default="single",
    help="Choose experiment phase",
)
@click.option(
    "--base",
    type=click.Choice(list(BASE_CONFIGS.keys())),
    default=None,
    help="Base config (required for --phase=single)",
)
@click.option(
    "--grid",
    type=click.Choice(["learning", "custom-loss"]),
    default=None,
    help="Grid to run (only used with --phase=single)",
)
@click.option("--continue-on-error", is_flag=True, default=True)
def run_grid_cmd(phase, base, grid, continue_on_error):
    """
    Run a grid search over selected configuration(s).
    """
    validate_input(phase, base, grid)

    if grid == "learning":
        param_grid = LEARNING_RATE_GRID
        grid_name = "Learning-rate grid search"
    else:
        param_grid = CUSTOM_LOSS_GRID
        grid_name = "Custom-loss λ-rate grid search"

    log_grid_info(grid_name, param_grid)

    if phase == "single":
        bases_to_run = [base]
    else:
        bases_to_run = PHASES[phase]

    results = {}

    for base_key in bases_to_run:
        base_cfg = BASE_CONFIGS[base_key]

        log(f"\n=== Running grid search for base: {base_key} ===")
        result = run_grid_search(
            base_config=base_cfg,
            param_grid=param_grid,
            continue_on_error=continue_on_error,
        )
        results[base_key] = result

    log_success("\nGrid search completed.")


if __name__ == "__main__":
    run_grid_cmd()
