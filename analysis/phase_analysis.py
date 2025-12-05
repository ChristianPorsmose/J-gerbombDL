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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
EXPERIMENTS_DIR = Path("../experiments")
PHASE1A_EXPERIMENTS = [
    "phase1a_sgd_standard",
    "phase1a_sgd_conservative",
    "phase1a_adamw_standard",
    "phase1a_adamw_conservative",
    "phase1a_adamw_aggressive",
]

EXPERIMENT_1A_LABELS = {
    "phase1a_sgd_standard": "SGD Standard",
    "phase1a_sgd_conservative": "SGD Conservative",
    "phase1a_adamw_standard": "AdamW Standard",
    "phase1a_adamw_conservative": "AdamW Conservative",
    "phase1a_adamw_aggressive": "AdamW Aggressive",
}

COLORS_1A = {
    "phase1a_sgd_standard": "#e74c3c",
    "phase1a_sgd_conservative": "#c0392b",
    "phase1a_adamw_standard": "#3498db",
    "phase1a_adamw_conservative": "#2980b9",
    "phase1a_adamw_aggressive": "#9b59b6",
}

PHASE1B_EXPERIMENTS = [
    # "phase1b_not_frozen",
    # "phase1b_fully_frozen",
    # "phase1b_frozen_half",
    # "phase2_no_augmentation_20_images",
    # "phase2_no_augmentation_40_images",
    # "phase2_no_augmentation_80_images",
    # "phase2_no_augmentation_all_images",
    # "phase2_light_augmentation_20_images",
    # "phase2_light_augmentation_40_images",
    # "phase2_light_augmentation_80_images",
    # "phase2_light_augmentation_all_images",
    # "phase2_geo_augmentation_20_images",
    # "phase2_geo_augmentation_40_images",
    # "phase2_geo_augmentation_80_images",
    # "phase2_geo_augmentation_all_images",
    # "phase2_full_augmentation_20_images",
    # "phase2_full_augmentation_40_images",
    # "phase2_full_augmentation_80_images",
    # "phase2_full_augmentation_all_images",
    # "full_aug_dfl_freeze",
    # "dfl_frozen",
    # "dfl_unfrozen",
    # "phase1b_not_frozen_timed",
    # "phase1b_frozen_half_timed",
    # "phase1b_fully_frozen_timed",
    # "full_augmentation_all_images",
    # "geo_augmentation_all_images",
    # "light_augmentation_all_images",
    # "no_augmentation_all_images",
    # "phase1b_not_frozen_timed_mini",
    # "phase1b_frozen_half_timed_mini",
    # "phase1b_fully_frozen_timed_mini",
    # "phase1b_frozen_all_timed_mini"
    # "new_dataset",
    # "old_dataset",
    # "new_dataset_new_gains",
    # "old_dataset_new_gains"
    # "normal_loss",
    # "spatial_loss",
    # "adamW_lr0_0001_momentum0_9000_weight_decay0_0010",
    # "adamW_lr0_0001_momentum0_9000_weight_decay0_0010",
    "dataset_size_small_dataset_size1_0000",
    "dataset_size_small_dataset_size0_5000",
    "dataset_size_small_dataset_size0_2500",
    "dataset_size_small_dataset_size0_1250",
    "dataset_size_pretrained_dataset_size1_0000",
    "dataset_size_pretrained_dataset_size0_5000",
    "dataset_size_pretrained_dataset_size0_2500",
    "dataset_size_pretrained_dataset_size0_1250",
]


EXPERIMENT_1B_LABELS = {
    # "phase1b_not_frozen": "Not Frozen",
    # "phase1b_fully_frozen": "Fully Frozen",
    # "phase1b_frozen_half": "Frozen Half",
    # "phase2_no_augmentation_20_images": "Phase 2 No Augmentation (20 Images)",
    # "phase2_no_augmentation_40_images": "Phase 2 No Augmentation (40 Images)",
    # "phase2_no_augmentation_80_images": "Phase 2 No Augmentation (80 Images)",
    # "phase2_no_augmentation_all_images": "Phase 2 No Augmentation (All Images)",
    # "phase2_light_augmentation_20_images": "Phase 2 Light Augmentation (20 Images)",
    # "phase2_light_augmentation_40_images": "Phase 2 Light Augmentation (40 Images)",
    # "phase2_light_augmentation_80_images": "Phase 2 Light Augmentation (80 Images)",
    # "phase2_light_augmentation_all_images": "Phase 2 Light Augmentation (All Images)",
    # "phase2_geo_augmentation_20_images": "Phase 2 Geo Augmentation (20 Images)",
    # "phase2_geo_augmentation_40_images": "Phase 2 Geo Augmentation (40 Images)",
    # "phase2_geo_augmentation_80_images": "Phase 2 Geo Augmentation (80 Images)",
    # "phase2_geo_augmentation_all_images": "Phase 2 Geo Augmentation (All Images)",
    # "phase2_full_augmentation_20_images": "Phase 2 Full Augmentation (20 Images)",
    # "phase2_full_augmentation_40_images": "Phase 2 Full Augmentation (40 Images)",
    # "phase2_full_augmentation_80_images": "Phase 2 Full Augmentation (80 Images)",
    # "phase2_full_augmentation_all_images": "Phase 2 Full Augmentation (All Images)",
    # "full_aug_dfl_freeze": "Full Augmentation + FreezeDFL",
    # "dfl_frozen": "DFL Frozen",
    # "dfl_unfrozen": "DFL Unfrozen",
    # "phase1b_not_frozen_timed": "Not Frozen",
    # "phase1b_frozen_half_timed": "Frozen Half",
    # "phase1b_fully_frozen_timed": "Fully Frozen",
    # "full_augmentation_all_images": "Full Augmentation (All Images) 300 epochs",
    # "geo_augmentation_all_images": "Geo Augmentation (All Images) 300 epochs",
    # "light_augmentation_all_images": "Light Augmentation (All Images) 300 epochs",
    # "no_augmentation_all_images": "No Augmentation (All Images) 300 epochs",
    # "phase1b_not_frozen_timed_mini": "Not Frozen Mini",
    # "phase1b_frozen_half_timed_mini": "Frozen Half Mini",
    # "phase1b_fully_frozen_timed_mini": "Fully Frozen Mini",
    # "phase1b_frozen_all_timed_mini": "Frozen All Mini",
    # "new_dataset": "New Dataset",
    # "old_dataset": "Old Dataset",
    # "new_dataset_new_gains": "New Dataset New Gains",
    # "old_dataset_new_gains": "Old Dataset New Gains",
    # "normal_loss": "Normal Loss",
    # "spatial_loss": "Spatial Consistency Loss",
    "dataset_size_small_dataset_size1_0000": "Small Dataset Size 1,000",
    "dataset_size_small_dataset_size0_5000": "Small Dataset Size 0.5",
    "dataset_size_small_dataset_size0_2500": "Small Dataset Size 0.25",
    "dataset_size_small_dataset_size0_1250": "Small Dataset Size 0.125",
    "dataset_size_pretrained_dataset_size1_0000": "Pretrained Dataset Size 1,000",
    "dataset_size_pretrained_dataset_size0_5000": "Pretrained Dataset Size 0.5",
    "dataset_size_pretrained_dataset_size0_2500": "Pretrained Dataset Size 0.25",
    "dataset_size_pretrained_dataset_size0_1250": "Pretrained Dataset Size 0.125",
}

COLORS_1B = {
    # "phase1b_not_frozen": "#d61c3b",
    # "phase1b_fully_frozen": "#11d663",
    # "phase1b_frozen_half": "#1440d1",
    # "phase2_no_augmentation_20_images": "#ff9d00ff",
    # "phase2_no_augmentation_40_images": "#c27800ff",
    # "phase2_no_augmentation_80_images": "#925a00ff",
    # "phase2_no_augmentation_all_images": "#553400ff",
    # "phase2_light_augmentation_20_images": "#3cff00",
    # "phase2_light_augmentation_40_images": "#29b100",
    # "phase2_light_augmentation_80_images": "#1b7200",
    # "phase2_light_augmentation_all_images": "#114900",
    # "phase2_geo_augmentation_20_images": "#8e44ad",
    # "phase2_geo_augmentation_40_images": "#2c3e50",
    # "phase2_geo_augmentation_80_images": "#d35400",
    # "phase2_geo_augmentation_all_images": "#7f8c8d",
    # "phase2_full_augmentation_20_images": "#3498db",
    # "phase2_full_augmentation_40_images": "#2980b9",
    # "phase2_full_augmentation_80_images": "#1abc9c",
    # "phase2_full_augmentation_all_images": "#0b20db",
    # "full_aug_dfl_freeze": "#e42d15",
    # "dfl_frozen": "#e42d15",
    # "dfl_unfrozen": "#27ae60",
    # "phase1b_not_frozen_timed": "#d61c3b",
    # "phase1b_frozen_half_timed": "#1440d1",
    # "phase1b_fully_frozen_timed": "#11d663",
    # "full_augmentation_all_images": "#FF0000",
    # "geo_augmentation_all_images": "#ff00aa",
    # "light_augmentation_all_images": "#260ff1",
    # "no_augmentation_all_images": "#ffee00",
    # "phase1b_not_frozen_timed_mini": "#d61c3b",
    # "phase1b_frozen_half_timed_mini": "#1440d1",
    # "phase1b_fully_frozen_timed_mini": "#11d663",
    # "phase1b_frozen_all_timed_mini": "#ff8800",
    # "new_dataset": "#22aa22",
    # "old_dataset": "#aa2222",
    # "new_dataset_new_gains": "#2288ff",
    # "old_dataset_new_gains": "#ff8822",
    # "normal_loss": "#3498db",
    # "spatial_loss": "#e74c3c"
    "dataset_size_small_dataset_size1_0000": "#d61c3b",
    "dataset_size_small_dataset_size0_5000": "#ffae00",
    "dataset_size_small_dataset_size0_2500": "#ffe600",
    "dataset_size_small_dataset_size0_1250": "#00ff22",
    "dataset_size_pretrained_dataset_size1_0000": "#00ffff",
    "dataset_size_pretrained_dataset_size0_5000": "#0044ff",
    "dataset_size_pretrained_dataset_size0_2500": "#9900ff",
    "dataset_size_pretrained_dataset_size0_1250": "#ff00d4",
}

PHASE_NAME = "1B"
PHASE_EXPERIMENTS = PHASE1B_EXPERIMENTS
EXPERIMENT_LABELS = EXPERIMENT_1B_LABELS
COLORS = COLORS_1B

OUTPUT_DIR = Path("phase1b_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_all_runs(experiment_name: str) -> List[Path]:
    """Find all run directories for an experiment."""
    exp_dir = EXPERIMENTS_DIR / experiment_name / "runs"
    if not exp_dir.exists():
        raise FileNotFoundError(f"No runs found for {experiment_name}")

    # Find all train_* directories
    runs = sorted(exp_dir.glob("train_*"))
    if not runs:
        raise FileNotFoundError(f"No training runs found in {exp_dir}")

    return runs


def find_best_run(experiment_name: str) -> Path:
    """Find the best run (lowest final validation loss) for an experiment."""
    runs = find_all_runs(experiment_name)

    best_run = None
    best_val_loss = float("inf")

    for run_dir in runs:
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            final_val_loss = (
                df["val/box_loss"].iloc[-1]
                + df["val/cls_loss"].iloc[-1]
                + df["val/dfl_loss"].iloc[-1]
            )

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_run = run_dir

    if best_run is None:
        raise FileNotFoundError(f"No valid runs found for {experiment_name}")

    return best_run


def load_single_run_data(run_dir: Path) -> Dict:
    """Load results.csv and test_results.json for a single run."""
    # Load results CSV
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv found at {csv_path}")
    results_df = pd.read_csv(csv_path)

    # Load test results JSON
    test_json_path = run_dir / "test_results.json"
    test_results = None
    if test_json_path.exists():
        with open(test_json_path, "r") as f:
            test_results = json.load(f)

    return {"results": results_df, "test_results": test_results, "run_dir": run_dir}


def average_runs(experiment_name: str, runs: List[Path]) -> Dict:
    """Average metrics across multiple runs."""
    all_results = []
    all_test_results = []

    for run_dir in runs:
        try:
            run_data = load_single_run_data(run_dir)
            all_results.append(run_data["results"])
            if run_data["test_results"]:
                all_test_results.append(run_data["test_results"])
        except Exception as e:
            print(f"    ⚠️  Skipping {run_dir.name}: {e}")
            continue

    if len(all_results) == 0:
        raise ValueError(f"No valid runs found for {experiment_name}")

    # Average the results DataFrames (align by epoch)
    # Find common epochs across all runs
    min_epochs = min(len(df) for df in all_results)

    # Truncate all DataFrames to common length and average
    truncated_dfs = [df.iloc[:min_epochs].copy() for df in all_results]

    # Average numeric columns
    avg_df = truncated_dfs[0].copy()
    numeric_cols = avg_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != "epoch":
            values = np.array([df[col].values for df in truncated_dfs])
            avg_df[col] = np.mean(values, axis=0)

    # Average test results
    avg_test_results = None
    if all_test_results:
        avg_test_results = {
            "metrics": {
                "precision": np.mean(
                    [r["metrics"]["precision"] for r in all_test_results]
                ),
                "recall": np.mean([r["metrics"]["recall"] for r in all_test_results]),
                "mAP50": np.mean([r["metrics"]["mAP50"] for r in all_test_results]),
                "mAP50-95": np.mean(
                    [r["metrics"]["mAP50-95"] for r in all_test_results]
                ),
            },
            "losses": {
                "box": np.mean([r["losses"]["box"] for r in all_test_results]),
                "cls": np.mean([r["losses"]["cls"] for r in all_test_results]),
                "dfl": np.mean([r["losses"]["dfl"] for r in all_test_results]),
                "spatial": np.mean([r["losses"]["spatial"] for r in all_test_results]),
                "total": np.mean([r["losses"]["total"] for r in all_test_results]),
            },
        }

    return {
        "results": avg_df,
        "test_results": avg_test_results,
        "num_runs": len(all_results),
    }


def load_experiment_data(experiment_name: str, mode: str = "best") -> Dict:
    """
    Load experiment data.

    Args:
        experiment_name: Name of the experiment
        mode: "best" (best run by val loss), "average" (average all runs), or "latest" (most recent)
    """
    runs = find_all_runs(experiment_name)

    # Get label, fallback to experiment name if not in predefined labels
    label = EXPERIMENT_LABELS.get(
        experiment_name, experiment_name.replace("_", " ").title()
    )

    if mode == "best":
        run_dir = find_best_run(experiment_name)
        run_data = load_single_run_data(run_dir)
        return {
            "name": experiment_name,
            "label": label,
            "results": run_data["results"],
            "test_results": run_data["test_results"],
            "run_dir": run_dir,
            "mode": "best",
            "num_runs": 1,
        }

    elif mode == "average":
        avg_data = average_runs(experiment_name, runs)
        return {
            "name": experiment_name,
            "label": label + f" (avg n={avg_data['num_runs']})",
            "results": avg_data["results"],
            "test_results": avg_data["test_results"],
            "run_dir": None,
            "mode": "average",
            "num_runs": avg_data["num_runs"],
        }

    elif mode == "latest":
        run_dir = runs[-1]
        run_data = load_single_run_data(run_dir)
        return {
            "name": experiment_name,
            "label": label,
            "results": run_data["results"],
            "test_results": run_data["test_results"],
            "run_dir": run_dir,
            "mode": "latest",
            "num_runs": 1,
        }

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'best', 'average', or 'latest'")


def calculate_convergence_epoch(
    results_df: pd.DataFrame, threshold: float = 0.95
) -> int:
    """Calculate epoch where model reaches 95% of final mAP."""
    final_map = results_df["metrics/mAP50(B)"].iloc[-1]
    target = threshold * final_map

    converged = results_df[results_df["metrics/mAP50(B)"] >= target]
    if len(converged) == 0:
        return len(results_df)  # Never converged

    return int(converged.iloc[0]["epoch"])


def calculate_train_val_gap(results_df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate mean and std of train-val gap percentage."""
    train_total = (
        results_df["train/box_loss"]
        + results_df["train/cls_loss"]
        + results_df["train/dfl_loss"]
    )
    val_total = (
        results_df["val/box_loss"]
        + results_df["val/cls_loss"]
        + results_df["val/dfl_loss"]
    )

    # Calculate gap as percentage
    gap_percent = ((train_total - val_total) / train_total * 100).abs()

    return gap_percent.mean(), gap_percent.std()


def calculate_stability(results_df: pd.DataFrame, last_n: int = 20) -> float:
    """Calculate validation loss stability (std dev) in last N epochs."""
    val_total = (
        results_df["val/box_loss"]
        + results_df["val/cls_loss"]
        + results_df["val/dfl_loss"]
    )

    last_n_losses = val_total.iloc[-last_n:]
    return last_n_losses.std()


def plot_loss_convergence(experiments_data: List[Dict]):
    """Plot validation loss convergence for all experiments."""
    plt.figure(figsize=(14, 7))

    # Collect all loss values to compute 95th percentile
    all_losses = []

    for exp_data in experiments_data:
        df = exp_data["results"]
        val_total = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        all_losses.extend(val_total.values)

        plt.plot(
            df["epoch"],
            val_total,
            label=exp_data["label"],
            linewidth=2.5,
            color=COLORS[exp_data["name"]],
        )

    # Set y-axis limits to 95th percentile to avoid outlier scaling
    # Filter out NaN and Inf
    all_losses_clean = [x for x in all_losses if np.isfinite(x)]

    if len(all_losses_clean) > 0:
        y_max = np.percentile(all_losses_clean, 95)
        y_min = np.percentile(all_losses_clean, 5)
        plt.ylim(bottom=max(0, y_min * 0.9), top=y_max * 1.1)

    plt.xlabel("Epoch", fontsize=13, fontweight="bold")
    plt.ylabel("Validation Total Loss", fontsize=13, fontweight="bold")
    plt.title(
        f"Phase {PHASE_NAME}: Validation Loss Convergence Comparison",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "1_loss_convergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_train_val_comparison_per_experiment(experiments_data: List[Dict]):
    """Plot train vs validation loss for each experiment in separate subplots."""
    n_experiments = len(experiments_data)

    # Create subplots: 2 rows if more than 3 experiments, else 1 row
    if n_experiments <= 3:
        rows, cols = 1, n_experiments
        figsize = (6 * n_experiments, 5)
    else:
        rows = 2
        cols = (n_experiments + 1) // 2
        figsize = (6 * cols, 10)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (ax, exp_data) in enumerate(zip(axes, experiments_data)):
        df = exp_data["results"]

        # Calculate total losses
        train_total = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
        val_total = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

        # Add spatial loss if present
        if "train/spatial_loss" in df.columns:
            train_total += df["train/spatial_loss"]
        if "val/spatial_loss" in df.columns:
            val_total += df["val/spatial_loss"]

        # Plot
        ax.plot(
            df["epoch"],
            train_total,
            label="Train Loss",
            linewidth=2.5,
            color="#3498db",
            alpha=0.8,
        )
        ax.plot(
            df["epoch"],
            val_total,
            label="Val Loss",
            linewidth=2.5,
            color="#e74c3c",
            alpha=0.8,
        )

        # Styling
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total Loss", fontsize=11, fontweight="bold")
        ax.set_title(
            exp_data["label"],
            fontsize=9,
            fontweight="bold",
            color=COLORS[exp_data["name"]],
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on percentiles to avoid outliers
        all_losses = list(train_total.values) + list(val_total.values)
        # Filter out NaN and Inf
        all_losses_clean = [x for x in all_losses if np.isfinite(x)]

        if len(all_losses_clean) > 0:
            y_min = np.percentile(all_losses_clean, 5)
            y_max = np.percentile(all_losses_clean, 95)
            ax.set_ylim(bottom=max(0, y_min * 0.9), top=y_max * 1.1)
        ax.set_ylim(1, 5)

    # Hide extra subplots if any
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Phase {PHASE_NAME}: Train vs Validation Loss per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    save_path = OUTPUT_DIR / "2b_train_val_per_experiment.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_loss_components_per_experiment(experiments_data: List[Dict]):
    """Plot individual loss components for each experiment in separate subplots."""
    n_experiments = len(experiments_data)

    # Create subplots: 2 rows if more than 2 experiments, else 1 row
    if n_experiments <= 2:
        rows, cols = 1, n_experiments
        figsize = (8 * n_experiments, 6)
    else:
        rows = 2
        cols = (n_experiments + 1) // 2
        figsize = (8 * cols, 12)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for easier iteration
    if n_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (ax, exp_data) in enumerate(zip(axes, experiments_data)):
        df = exp_data["results"]

        # Plot validation loss components
        ax.plot(
            df["epoch"],
            df["val/box_loss"],
            label="Box Loss",
            linewidth=2,
            color="#e74c3c",
            alpha=0.8,
        )
        ax.plot(
            df["epoch"],
            df["val/cls_loss"],
            label="Class Loss",
            linewidth=2,
            color="#3498db",
            alpha=0.8,
        )
        ax.plot(
            df["epoch"],
            df["val/dfl_loss"],
            label="DFL Loss",
            linewidth=2,
            color="#2ecc71",
            alpha=0.8,
        )

        # Add spatial loss if present
        if "val/spatial_loss" in df.columns:
            ax.plot(
                df["epoch"],
                df["val/spatial_loss"],
                label="Spatial Loss",
                linewidth=2,
                color="#9b59b6",
                alpha=0.8,
            )

        # Styling
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation Loss", fontsize=11, fontweight="bold")
        ax.set_title(
            exp_data["label"],
            fontsize=12,
            fontweight="bold",
            color=COLORS[exp_data["name"]],
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on percentiles
        all_losses = []
        for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
            all_losses.extend(df[col].values)
        if "val/spatial_loss" in df.columns:
            all_losses.extend(df["val/spatial_loss"].values)

        # Filter out NaN and Inf
        all_losses_clean = [x for x in all_losses if np.isfinite(x)]

        if len(all_losses_clean) > 0:
            y_min = np.percentile(all_losses_clean, 5)
            y_max = np.percentile(all_losses_clean, 95)
            ax.set_ylim(bottom=max(0, y_min * 0.2), top=y_max * 0.8)

    # Hide extra subplots if any
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Phase {PHASE_NAME}: Validation Loss Components per Experiment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    save_path = OUTPUT_DIR / "2c_loss_components_per_experiment.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_train_val_gap(experiments_data: List[Dict]):
    """Plot train-val gap over time for overfitting analysis."""
    plt.figure(figsize=(14, 7))

    # Collect all gap values for percentile-based scaling
    all_gaps = []
    for exp_data in experiments_data:
        df = exp_data["results"]
        train_total = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
        val_total = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
        gap_percent = ((train_total - val_total) / train_total * 100).abs()
        all_gaps.extend(gap_percent.values)

        plt.plot(
            df["epoch"],
            gap_percent,
            label=exp_data["label"],
            linewidth=2.5,
            color=COLORS[exp_data["name"]],
        )

    # plt.axhline(y=15, color='red', linestyle='--', linewidth=2,
    #             label='Overfitting Threshold (15%)', alpha=0.7)

    # Set y-axis limits based on 5th-95th percentile to handle outliers
    # Filter out NaN and Inf
    all_gaps_clean = [x for x in all_gaps if np.isfinite(x)]

    if len(all_gaps_clean) > 0:
        p5 = np.percentile(all_gaps_clean, 5)
        p95 = np.percentile(all_gaps_clean, 95)
        plt.ylim(bottom=max(0, p5 * 0.9), top=p95 * 1.1)

    plt.ylim(0, 10)

    plt.xlabel("Epoch", fontsize=13, fontweight="bold")
    plt.ylabel("Train-Val Gap (%)", fontsize=13, fontweight="bold")
    plt.title(
        f"Phase {PHASE_NAME}: Overfitting Analysis (Train-Val Loss Gap)",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "2_train_val_gap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_map_comparison(experiments_data: List[Dict]):
    """Plot final mAP@0.5:0.95 comparison as bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels = [exp["label"] for exp in experiments_data]
    map50_95 = [
        exp["results"]["metrics/mAP50-95(B)"].iloc[-1] for exp in experiments_data
    ]
    map50 = [exp["results"]["metrics/mAP50(B)"].iloc[-1] for exp in experiments_data]
    colors_list = [COLORS[exp["name"]] for exp in experiments_data]

    # Plot 1: mAP@0.5:0.95
    bars1 = ax1.bar(
        labels, map50_95, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("mAP@0.5:0.95", fontsize=13, fontweight="bold")
    ax1.set_title("Final Validation mAP@0.5:0.95", fontsize=14, fontweight="bold")
    ax1.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax1.set_ylim([0.9, 1.0])
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: mAP@0.5
    bars2 = ax2.bar(labels, map50, color=colors_list, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("mAP@0.5", fontsize=13, fontweight="bold")
    ax2.set_title("Final Validation mAP@0.5", fontsize=14, fontweight="bold")
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "3_map_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_test_map_comparison(experiments_data: List[Dict]):
    """Plot final TEST mAP comparison as bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels = [exp["label"] for exp in experiments_data]
    colors_list = [COLORS[exp["name"]] for exp in experiments_data]

    # Extract test mAP values
    test_map50_95 = []
    test_map50 = []

    for exp in experiments_data:
        if exp["test_results"] and "metrics" in exp["test_results"]:
            test_map50_95.append(exp["test_results"]["metrics"]["mAP50-95"])
            test_map50.append(exp["test_results"]["metrics"]["mAP50"])
        else:
            test_map50_95.append(np.nan)
            test_map50.append(np.nan)

    # Plot 1: Test mAP@0.5:0.95
    bars1 = ax1.bar(
        labels, test_map50_95, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax1.set_ylabel("mAP@0.5:0.95", fontsize=13, fontweight="bold")
    ax1.set_title("Final Test mAP@0.5:0.95", fontsize=14, fontweight="bold")
    ax1.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax1.set_ylim([0.9, 1.0])
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Add value labels
    for bar, val in zip(bars1, test_map50_95):
        if not np.isnan(val):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 2: Test mAP@0.5
    bars2 = ax2.bar(
        labels, test_map50, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("mAP@0.5", fontsize=13, fontweight="bold")
    ax2.set_title("Final Test mAP@0.5", fontsize=14, fontweight="bold")
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # Add value labels
    for bar, val in zip(bars2, test_map50):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "3b_test_map_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_test_loss_comparison(experiments_data: List[Dict]):
    """Plot final TEST loss components comparison as grouped bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [exp["label"] for exp in experiments_data]

    # Extract test loss values
    test_box = []
    test_cls = []
    test_dfl = []
    test_spatial = []

    for exp in experiments_data:
        if exp["test_results"] and "losses" in exp["test_results"]:
            test_box.append(exp["test_results"]["losses"]["box"])
            test_cls.append(exp["test_results"]["losses"]["cls"])
            test_dfl.append(exp["test_results"]["losses"]["dfl"])
            test_spatial.append(exp["test_results"]["losses"]["spatial"])
        else:
            test_box.append(0)
            test_cls.append(0)
            test_dfl.append(0)
            test_spatial.append(0)

    # Collect all loss values for 95th percentile calculation
    all_losses = test_box + test_cls + test_dfl + test_spatial
    all_losses = [v for v in all_losses if v > 0]  # Exclude zeros
    # Filter out NaN and Inf
    all_losses_clean = [v for v in all_losses if np.isfinite(v)]

    x = np.arange(len(labels))
    width = 0.2

    # Create grouped bars
    bars1 = ax.bar(
        x - 1.5 * width,
        test_box,
        width,
        label="Box Loss",
        color="#e74c3c",
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        test_cls,
        width,
        label="Class Loss",
        color="#3498db",
        edgecolor="black",
        linewidth=1,
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        test_dfl,
        width,
        label="DFL Loss",
        color="#2ecc71",
        edgecolor="black",
        linewidth=1,
    )
    bars4 = ax.bar(
        x + 1.5 * width,
        test_spatial,
        width,
        label="Spatial Loss",
        color="#9b59b6",
        edgecolor="black",
        linewidth=1,
    )

    # Set y-axis limits to 95th percentile to avoid outlier scaling
    if len(all_losses_clean) > 0:
        y_max = np.percentile(all_losses_clean, 95)
        ax.set_ylim(top=y_max * 1.15)  # Add 15% headroom for labels

    ax.set_ylabel("Loss Value", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Phase {PHASE_NAME}: Test Loss Components Comparison",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars (only for non-zero values)
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only show label if value is significant
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "3c_test_loss_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_val_test_gap(experiments_data: List[Dict]):
    """Plot validation vs test performance comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels = [exp["label"] for exp in experiments_data]

    # Extract mAP@0.5 values
    val_map50 = [
        exp["results"]["metrics/mAP50(B)"].iloc[-1] for exp in experiments_data
    ]
    test_map50 = []

    # Extract mAP@0.5:0.95 values
    val_map50_95 = [
        exp["results"]["metrics/mAP50-95(B)"].iloc[-1] for exp in experiments_data
    ]
    test_map50_95 = []

    for exp in experiments_data:
        if exp["test_results"] and "metrics" in exp["test_results"]:
            test_map50.append(exp["test_results"]["metrics"]["mAP50"])
            test_map50_95.append(exp["test_results"]["metrics"]["mAP50-95"])
        else:
            test_map50.append(np.nan)
            test_map50_95.append(np.nan)

    x = np.arange(len(labels))
    width = 0.35

    # Plot 1: mAP@0.5
    bars1 = ax1.bar(
        x - width / 2,
        val_map50,
        width,
        label="Validation mAP@0.5",
        color="#3498db",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        test_map50,
        width,
        label="Test mAP@0.5",
        color="#e74c3c",
        edgecolor="black",
        linewidth=1.5,
    )

    ax1.set_ylabel("mAP@0.5", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Validation vs Test Performance (mAP@0.5)", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0.9, 1.0])

    # Add value labels for mAP@0.5
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.002,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # Plot 2: mAP@0.5:0.95
    bars3 = ax2.bar(
        x - width / 2,
        val_map50_95,
        width,
        label="Validation mAP@0.5:0.95",
        color="#3498db",
        edgecolor="black",
        linewidth=1.5,
    )
    bars4 = ax2.bar(
        x + width / 2,
        test_map50_95,
        width,
        label="Test mAP@0.5:0.95",
        color="#e74c3c",
        edgecolor="black",
        linewidth=1.5,
    )

    ax2.set_ylabel("mAP@0.5:0.95", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Validation vs Test Performance (mAP@0.5:0.95)", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim([0.9, 1.0])

    # Add value labels for mAP@0.5:0.95
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.002,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "4_val_test_gap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_convergence_speed(experiments_data: List[Dict]):
    """Plot convergence speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = [exp["label"] for exp in experiments_data]
    convergence_epochs = [
        calculate_convergence_epoch(exp["results"]) for exp in experiments_data
    ]
    colors_list = [COLORS[exp["name"]] for exp in experiments_data]

    bars = ax.barh(
        labels, convergence_epochs, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax.set_xlabel("Epochs to Reach 95% of Final mAP", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Phase {PHASE_NAME}: Convergence Speed Comparison",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, convergence_epochs)):
        ax.text(
            val + 0.5,
            i,
            f"{val}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "5_convergence_speed.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def generate_decision_matrix(experiments_data: List[Dict]) -> pd.DataFrame:
    """Generate comprehensive decision matrix with all metrics."""

    matrix_data = []

    for exp in experiments_data:
        df = exp["results"]

        # Extract metrics
        final_map50_95 = df["metrics/mAP50-95(B)"].iloc[-1]
        final_map50 = df["metrics/mAP50(B)"].iloc[-1]
        final_precision = df["metrics/precision(B)"].iloc[-1]
        final_recall = df["metrics/recall(B)"].iloc[-1]

        train_val_gap_mean, train_val_gap_std = calculate_train_val_gap(df)
        convergence_epoch = calculate_convergence_epoch(df)
        stability = calculate_stability(df)

        # Test metrics
        test_map50 = np.nan
        val_test_gap = np.nan
        if exp["test_results"] and "metrics" in exp["test_results"]:
            test_map50 = exp["test_results"]["metrics"]["mAP50"]
            val_test_gap = abs(final_map50 - test_map50) / final_map50 * 100

        matrix_data.append(
            {
                "Experiment": exp["label"],
                "Val_mAP@0.5": final_map50,
                "Val_mAP@0.5:0.95": final_map50_95,
                "Precision": final_precision,
                "Recall": final_recall,
                "Test_mAP@0.5": test_map50,
                "Val-Test_Gap_%": val_test_gap,
                "Train-Val_Gap_%": train_val_gap_mean,
                "Gap_Std_%": train_val_gap_std,
                "Convergence_Epoch": convergence_epoch,
                "Stability_Std": stability,
            }
        )

    df_matrix = pd.DataFrame(matrix_data)

    # Calculate rankings (lower rank = better)
    # For metrics where higher is better, we invert
    df_matrix["Rank_mAP50-95"] = df_matrix["Val_mAP@0.5:0.95"].rank(ascending=False)
    df_matrix["Rank_Test_mAP"] = df_matrix["Test_mAP@0.5"].rank(ascending=False)
    df_matrix["Rank_Train-Val_Gap"] = df_matrix["Train-Val_Gap_%"].rank(ascending=True)
    df_matrix["Rank_Val-Test_Gap"] = df_matrix["Val-Test_Gap_%"].rank(ascending=True)
    df_matrix["Rank_Convergence"] = df_matrix["Convergence_Epoch"].rank(ascending=True)
    df_matrix["Rank_Stability"] = df_matrix["Stability_Std"].rank(ascending=True)

    # Weighted scoring
    df_matrix["Total_Score"] = (
        df_matrix["Rank_mAP50-95"] * 3  # mAP most important
        + df_matrix["Rank_Test_mAP"] * 2  # Test performance
        + df_matrix["Rank_Train-Val_Gap"] * 2  # Generalization
        + df_matrix["Rank_Val-Test_Gap"] * 1.5
        + df_matrix["Rank_Convergence"] * 1  # Efficiency
        + df_matrix["Rank_Stability"] * 1  # Stability
    )

    # Sort by total score (lower is better)
    df_matrix = df_matrix.sort_values("Total_Score")

    return df_matrix


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

    # Select columns to display
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

    table_data = df_matrix[display_cols].values
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

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.11, 0.11, 0.1, 0.11, 0.1, 0.1, 0.1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color best performer (first row)
    for i in range(len(col_labels)):
        table[(1, i)].set_facecolor("#2ecc71")
        table[(1, i)].set_text_props(weight="bold")

    plt.title(
        f"Phase {PHASE_NAME} Decision Matrix (Sorted by Total Score)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    save_path = OUTPUT_DIR / "6_decision_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def extract_hyperparameters_from_name(exp_name: str) -> Dict:
    """Extract hyperparameter values from experiment name."""
    params = {}

    # Common patterns
    import re

    # Learning rate: lr0_0001 -> 0.0001
    lr_match = re.search(r"lr(0_\d+)", exp_name)
    if lr_match:
        params["lr"] = float(lr_match.group(1).replace("_", "."))

    # Momentum: momentum0_9000 -> 0.9000
    momentum_match = re.search(r"momentum(0_\d+)", exp_name)
    if momentum_match:
        params["momentum"] = float(momentum_match.group(1).replace("_", "."))

    # Weight decay: weight_decay0_0010 -> 0.0010
    wd_match = re.search(r"weight_decay(0_\d+)", exp_name)
    if wd_match:
        params["weight_decay"] = float(wd_match.group(1).replace("_", "."))

    # Batch size
    bs_match = re.search(r"batch_?size_?(\d+)", exp_name, re.IGNORECASE)
    if bs_match:
        params["batch_size"] = int(bs_match.group(1))

    # Optimizer type
    if "adamw" in exp_name.lower():
        params["optimizer"] = "AdamW"
    elif "adam" in exp_name.lower():
        params["optimizer"] = "Adam"
    elif "sgd" in exp_name.lower():
        params["optimizer"] = "SGD"

    # Special-case handling for jagerloss grid where two variables vary:
    # - model variant: small vs pretrained
    # - lambda rate: 160 vs 80
    # Example names:
    #   "jagerloss_no_prop_config_small_lamda_rate160"
    #   "jagerloss_no_prop_config_pretrained_lamda_rate80"
    if "jagerloss_no_prop_config" in exp_name:
        # Model variant flag (numeric for plotting)
        if "small" in exp_name:
            params["pretrained_flag"] = 0
            params["model_variant"] = "small"
        elif "pretrained" in exp_name:
            params["pretrained_flag"] = 1
            params["model_variant"] = "pretrained"

        # Lambda rate (numeric)
        lam_match = re.search(r"lamda[_-]?rate(\d+)", exp_name, re.IGNORECASE)
        if lam_match:
            try:
                params["lambda_rate"] = int(lam_match.group(1))
            except ValueError:
                pass

    return params


def create_grid_search_plots(experiments_data: List[Dict], output_name: str):
    """Create comprehensive grid search visualization plots."""

    # Create output directory
    grid_output_dir = Path("./analysis") / output_name
    grid_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔬 Creating grid search visualizations in: {grid_output_dir}")

    # Track problematic experiments
    nan_experiments = []
    inf_experiments = []

    # Extract hyperparameters and metrics
    grid_data = []
    for exp in experiments_data:
        params = extract_hyperparameters_from_name(exp["name"])

        metrics = {
            "name": exp["name"],
            "label": exp["label"],
            "val_map50_95": exp["results"]["metrics/mAP50-95(B)"].iloc[-1],
            "val_map50": exp["results"]["metrics/mAP50(B)"].iloc[-1],
            "final_val_loss": (
                exp["results"]["val/box_loss"].iloc[-1]
                + exp["results"]["val/cls_loss"].iloc[-1]
                + exp["results"]["val/dfl_loss"].iloc[-1]
            ),
            "convergence_epoch": calculate_convergence_epoch(exp["results"]),
            "stability": calculate_stability(exp["results"]),
        }

        # Check for NaN or Inf values
        has_nan = False
        has_inf = False
        for key, value in metrics.items():
            if key not in ["name", "label"]:
                if np.isnan(value):
                    has_nan = True
                elif np.isinf(value):
                    has_inf = True

        if has_nan:
            nan_experiments.append(exp["name"])
        if has_inf:
            inf_experiments.append(exp["name"])

        # Add test metrics if available
        if exp["test_results"] and "metrics" in exp["test_results"]:
            metrics["test_map50"] = exp["test_results"]["metrics"]["mAP50"]
            metrics["test_map50_95"] = exp["test_results"]["metrics"]["mAP50-95"]
        else:
            metrics["test_map50"] = np.nan
            metrics["test_map50_95"] = np.nan

        grid_data.append({**params, **metrics})

    df_grid = pd.DataFrame(grid_data)

    # Report problematic experiments
    if nan_experiments:
        print(
            f"\n   ⚠️  WARNING: {len(nan_experiments)} experiment(s) contain NaN values (training likely failed):"
        )
        for exp_name in nan_experiments[:5]:  # Show first 5
            print(f"      • {exp_name}")
        if len(nan_experiments) > 5:
            print(f"      ... and {len(nan_experiments) - 5} more")

    if inf_experiments:
        print(
            f"\n   ⚠️  WARNING: {len(inf_experiments)} experiment(s) contain Inf values (gradient explosion):"
        )
        for exp_name in inf_experiments[:5]:  # Show first 5
            print(f"      • {exp_name}")
        if len(inf_experiments) > 5:
            print(f"      ... and {len(inf_experiments) - 5} more")

    if nan_experiments or inf_experiments:
        print(
            f"\n   ℹ️  These experiments will be excluded from visualizations but included in CSV.\n"
        )

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

    print(f"   Varying hyperparameters: {', '.join(varying_params)}")

    # 1. Heatmap plots for 2D grids
    if len(varying_params) >= 2:
        create_heatmap_plots(df_grid, varying_params, grid_output_dir)

    # 2. Line plots for single parameter sweeps
    for param in varying_params:
        create_parameter_sweep_plot(df_grid, param, grid_output_dir)

    # 3. Parallel coordinates plot
    create_parallel_coordinates_plot(df_grid, varying_params, grid_output_dir)

    # 4. Top N comparison
    create_top_n_comparison(df_grid, grid_output_dir, n=min(10, len(df_grid)))

    # 5. Parameter importance plot
    if len(varying_params) >= 2:
        create_parameter_importance_plot(df_grid, varying_params, grid_output_dir)

    # Save grid search results CSV
    csv_path = grid_output_dir / "grid_search_results.csv"
    df_grid.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"   ✅ Saved: {csv_path}")

    # Save problematic experiments report
    if nan_experiments or inf_experiments:
        report_path = grid_output_dir / "problematic_experiments.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PROBLEMATIC EXPERIMENTS REPORT\n")
            f.write("=" * 80 + "\n\n")

            if nan_experiments:
                f.write(f"NaN VALUES DETECTED ({len(nan_experiments)} experiments):\n")
                f.write("These experiments likely failed during training.\n\n")
                for exp in nan_experiments:
                    f.write(f"  • {exp}\n")
                f.write("\n")

            if inf_experiments:
                f.write(f"INF VALUES DETECTED ({len(inf_experiments)} experiments):\n")
                f.write("These experiments likely experienced gradient explosion.\n")
                f.write(
                    "Consider: lower learning rate, gradient clipping, or different optimizer.\n\n"
                )
                for exp in inf_experiments:
                    f.write(f"  • {exp}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("NOTE: These experiments are excluded from visualizations but\n")
            f.write("      included in grid_search_results.csv for reference.\n")
            f.write("=" * 80 + "\n")

        print(f"   ✅ Saved: {report_path.name}")

    print(f"✅ Grid search visualizations complete!\n")


def create_heatmap_plots(
    df_grid: pd.DataFrame, varying_params: List[str], output_dir: Path
):
    """Create heatmap visualizations for 2D parameter grids."""

    # Filter out rows with NaN or Inf in key metrics
    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["val_map50_95", "final_val_loss", "convergence_epoch"]
    )

    if len(df_clean) == 0:
        print(f"   ⚠️  Skipping heatmaps: No valid data after removing NaN/Inf values")
        return

    if len(df_clean) < len(df_grid):
        print(
            f"   ℹ️  Heatmaps: Using {len(df_clean)}/{len(df_grid)} experiments (excluded NaN/Inf)"
        )

    metrics_to_plot = [
        ("val_map50_95", "Validation mAP@0.5:0.95", "RdYlGn", True),
        ("final_val_loss", "Final Validation Loss", "RdYlGn_r", False),
        ("convergence_epoch", "Convergence Epoch", "RdYlGn_r", False),
    ]

    # Try all pairs of varying parameters
    for i, param1 in enumerate(varying_params):
        for param2 in varying_params[i + 1 :]:

            # Skip if not enough variation
            if df_clean[param1].nunique() < 2 or df_clean[param2].nunique() < 2:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for ax, (metric, title, cmap, higher_better) in zip(axes, metrics_to_plot):
                # Create pivot table
                pivot = df_clean.pivot_table(
                    values=metric, index=param2, columns=param1, aggfunc="mean"
                )

                # Create heatmap
                im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

                # Set ticks and labels
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_xticklabels(
                    [
                        f"{v:.4f}" if isinstance(v, float) else str(v)
                        for v in pivot.columns
                    ],
                    rotation=45,
                    ha="right",
                )
                ax.set_yticklabels(
                    [
                        f"{v:.4f}" if isinstance(v, float) else str(v)
                        for v in pivot.index
                    ]
                )

                ax.set_xlabel(
                    param1.replace("_", " ").title(), fontsize=11, fontweight="bold"
                )
                ax.set_ylabel(
                    param2.replace("_", " ").title(), fontsize=11, fontweight="bold"
                )
                ax.set_title(title, fontsize=12, fontweight="bold")

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=9)

                # Annotate cells with values
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            text_color = "white" if (im.norm(val) > 0.5) else "black"
                            ax.text(
                                j,
                                i,
                                f"{val:.3f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=9,
                                fontweight="bold",
                            )

            plt.suptitle(
                f"Grid Search Heatmap: {param1} vs {param2}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            save_path = output_dir / f"heatmap_{param1}_vs_{param2}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"   ✅ Saved: {save_path.name}")


def create_parameter_sweep_plot(df_grid: pd.DataFrame, param: str, output_dir: Path):
    """Create line plots showing effect of single parameter."""

    if param not in df_grid.columns or df_grid[param].nunique() < 2:
        return

    # Filter out rows with NaN or Inf in key metrics
    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["val_map50_95", "final_val_loss", "convergence_epoch", "stability"]
    )

    if len(df_clean) == 0:
        print(
            f"   ⚠️  Skipping sweep plot for {param}: No valid data after removing NaN/Inf values"
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Sort by parameter value
    df_sorted = df_clean.sort_values(param)

    # Plot 1: mAP metrics
    ax = axes[0]
    ax.plot(
        df_sorted[param],
        df_sorted["val_map50_95"],
        "o-",
        linewidth=2.5,
        markersize=8,
        label="Val mAP@0.5:0.95",
        color="#3498db",
    )
    ax.plot(
        df_sorted[param],
        df_sorted["val_map50"],
        "s-",
        linewidth=2.5,
        markersize=8,
        label="Val mAP@0.5",
        color="#2ecc71",
    )
    if "test_map50_95" in df_sorted.columns:
        ax.plot(
            df_sorted[param],
            df_sorted["test_map50_95"],
            "^-",
            linewidth=2.5,
            markersize=8,
            label="Test mAP@0.5:0.95",
            color="#e74c3c",
        )
    ax.set_xlabel(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("mAP Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Performance vs " + param.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation loss
    ax = axes[1]
    ax.plot(
        df_sorted[param],
        df_sorted["final_val_loss"],
        "o-",
        linewidth=2.5,
        markersize=8,
        color="#e74c3c",
    )
    ax.set_xlabel(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Final Validation Loss", fontsize=11, fontweight="bold")
    ax.set_title(
        "Loss vs " + param.replace("_", " ").title(), fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Plot 3: Convergence speed
    ax = axes[2]
    ax.plot(
        df_sorted[param],
        df_sorted["convergence_epoch"],
        "o-",
        linewidth=2.5,
        markersize=8,
        color="#9b59b6",
    )
    ax.set_xlabel(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Convergence Epoch", fontsize=11, fontweight="bold")
    ax.set_title(
        "Convergence Speed vs " + param.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower is better

    # Plot 4: Stability
    ax = axes[3]
    ax.plot(
        df_sorted[param],
        df_sorted["stability"],
        "o-",
        linewidth=2.5,
        markersize=8,
        color="#f39c12",
    )
    ax.set_xlabel(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Stability (Std Dev)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Stability vs " + param.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"sweep_{param}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path.name}")


def create_parallel_coordinates_plot(
    df_grid: pd.DataFrame, varying_params: List[str], output_dir: Path
):
    """Create parallel coordinates plot for multi-dimensional visualization."""

    from pandas.plotting import parallel_coordinates

    # Filter out rows with NaN or Inf
    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["val_map50_95"]
    )

    if len(df_clean) == 0:
        print(
            f"   ⚠️  Skipping parallel coordinates: No valid data after removing NaN/Inf values"
        )
        return

    # Prepare data
    plot_df = df_clean.copy()

    # Find the best performing configuration
    best_idx = plot_df["val_map50_95"].idxmax()
    best_mAP = plot_df.loc[best_idx, "val_map50_95"]
    # If model variant axis is present, compute best per variant (0=Small, 1=Pretrained)
    has_variant_axis = "pretrained_flag" in plot_df.columns
    best_by_variant = {}
    if has_variant_axis:
        for variant in sorted(plot_df["pretrained_flag"].unique()):
            sub = plot_df[plot_df["pretrained_flag"] == variant]
            if len(sub) > 0:
                idx_best_var = sub["val_map50_95"].idxmax()
                try:
                    best_by_variant[int(idx_best_var)] = int(variant)
                except Exception:
                    # Fallback without casting index
                    best_by_variant[idx_best_var] = int(variant)

    # Normalize parameters to 0-1 range for better visualization
    # Using rank normalization for evenly spaced discrete values
    for param in varying_params:
        if param in plot_df.columns and plot_df[param].dtype in [
            np.float64,
            np.int64,
            int,
        ]:
            unique_vals = sorted(plot_df[param].unique())
            if len(unique_vals) > 1:
                # Map each unique value to evenly spaced positions
                val_to_norm = {
                    val: i / (len(unique_vals) - 1) for i, val in enumerate(unique_vals)
                }
                plot_df[f"{param}_norm"] = plot_df[param].map(val_to_norm)
            else:
                plot_df[f"{param}_norm"] = 0.5

    # Normalize val_map50_95 with evenly spaced positions
    unique_map_vals = sorted(plot_df["val_map50_95"].unique())
    if len(unique_map_vals) > 1:
        map_to_norm = {
            val: i / (len(unique_map_vals) - 1) for i, val in enumerate(unique_map_vals)
        }
        plot_df["val_map50_95_norm"] = plot_df["val_map50_95"].map(map_to_norm)
    else:
        plot_df["val_map50_95_norm"] = 0.5

    # Create performance categories
    plot_df["Performance"] = pd.cut(
        plot_df["val_map50_95"], bins=3, labels=["Low", "Medium", "High"]
    )

    # Select columns for parallel coordinates
    cols_to_plot = [
        f"{p}_norm" for p in varying_params if f"{p}_norm" in plot_df.columns
    ]
    cols_to_plot.append("val_map50_95_norm")
    cols_to_plot.append("Performance")

    if len(cols_to_plot) < 3:
        return

    fig, ax = plt.subplots(figsize=(15, 7))

    # Create colormap for continuous mAP values (normalized to 0-1 like other axes)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    # Normalize mAP to 0-1 range for consistent scaling
    mAP_min = plot_df["val_map50_95"].min()
    mAP_max = plot_df["val_map50_95"].max()
    norm = Normalize(vmin=0, vmax=1)  # Colorbar range
    cmap = cm.get_cmap("RdYlGn")

    # Plot each line individually with color based on normalized mAP
    used_labels = set()
    for idx, row in plot_df.iterrows():
        # Normalize mAP to 0-1
        mAP_min = plot_df["val_map50_95"].min()
        mAP_max = plot_df["val_map50_95"].max()
        # Exclude zero values from min calculation
        non_zero_map = plot_df[plot_df["val_map50_95"] > 0]["val_map50_95"]
        if len(non_zero_map) > 0:
            mAP_min = non_zero_map.min()

        mAP_norm = (
            (row["val_map50_95"] - mAP_min) / (mAP_max - mAP_min)
            if mAP_max > mAP_min
            else 0.5
        )
        color = cmap(norm(mAP_norm))
        values = [row[col] for col in cols_to_plot[:-1]]  # Exclude 'Performance'

        # Highlight logic
        if has_variant_axis and idx in best_by_variant:
            variant = best_by_variant[idx]
            label_key = "small" if variant == 0 else "pretrained"
            pretty_label = "Best Small" if variant == 0 else "Best Pretrained"
            label_text = (
                None
                if label_key in used_labels
                else f"{pretty_label}: {row['val_map50_95']:.4f}"
            )

            # Distinct styling for each variant
            if variant == 0:
                hl_color = "#2c3e50"  # deep navy
                hl_ls = "-"  # solid
                hl_marker = "o"  # circle
            else:
                hl_color = "#8e44ad"  # purple
                hl_ls = "-"  # dashed
                hl_marker = "s"  # square

            x_positions = list(range(len(values)))
            # Glow effect with variant color
            for glow_width in [6, 4, 2]:
                ax.plot(
                    x_positions,
                    values,
                    color=hl_color,
                    alpha=0.18,
                    linewidth=glow_width,
                    zorder=1,
                )
            # Main highlight line with markers
            ax.plot(
                x_positions,
                values,
                color=hl_color,
                linestyle=hl_ls,
                linewidth=2.8,
                marker=hl_marker,
                markersize=6,
                markerfacecolor=hl_color,
                markeredgecolor="white",
                label=label_text,
                zorder=3,
            )
            used_labels.add(label_key)
        elif not has_variant_axis and idx == best_idx:
            # Highlight global best only when variant axis isn't present
            x_positions = list(range(len(values)))
            for glow_width in [6, 4, 2]:
                ax.plot(
                    x_positions,
                    values,
                    color="black",
                    alpha=0.15,
                    linewidth=glow_width,
                    zorder=1,
                )
            ax.plot(
                x_positions,
                values,
                color="black",
                linewidth=2.5,
                label=f"Best: {best_mAP:.4f}",
                zorder=3,
            )
        else:
            ax.plot(
                range(len(values)),
                values,
                color=color,
                alpha=0.6,
                linewidth=1.5,
                zorder=2,
            )

    # Draw vertical lines for each axis
    for i in range(len(cols_to_plot[:-1])):
        ax.axvline(x=i, color="gray", linestyle="-", linewidth=1.5, alpha=0.5, zorder=0)

    # Update x-tick labels to show original parameter names
    new_labels = []
    custom_loss = False
    for col in cols_to_plot[:-1]:  # Exclude 'Performance' column
        if col == "val_map50_95_norm":
            new_labels.append("Val mAP@0.5:0.95")
        else:
            label = col.replace("_norm", "")
            label = label.replace("_", " ").title()
            # Custom friendly names
            if label == "Pretrained Flag":
                label = "Model Variant"
            elif label == "Lambda Rate":
                label = "Lambda Rate"
            new_labels.append(label)

    ax.set_xticklabels(new_labels, rotation=45, ha="right")

    # Add actual values on each axis (evenly spaced)
    for i, col in enumerate(cols_to_plot[:-1]):  # Exclude 'Performance'
        axis_x = i

        if col == "val_map50_95_norm":
            # Show actual mAP values
            unique_vals = sorted(plot_df["val_map50_95"].unique())
            for j, val in enumerate(unique_vals):
                norm_pos = j / (len(unique_vals) - 1) if len(unique_vals) > 1 else 0.5
                ax.text(
                    axis_x + 0.15,
                    norm_pos,
                    f"{val:.4f}",
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
        elif col.endswith("_norm"):
            # Get original parameter name
            orig_param = col.replace("_norm", "")
            if orig_param in plot_df.columns:
                # Get unique values and their evenly spaced normalized positions
                unique_vals = sorted(plot_df[orig_param].unique())
                for j, val in enumerate(unique_vals):
                    norm_pos = (
                        j / (len(unique_vals) - 1) if len(unique_vals) > 1 else 0.5
                    )
                    # Add text annotation
                    # Special-case: map pretrained_flag numeric values to labels
                    if orig_param == "pretrained_flag":
                        if int(val) == 0:
                            val_text = "Small"
                        else:
                            val_text = "Pretrained"
                    else:
                        # Show integers without decimals for clarity
                        if isinstance(val, (int, np.integer)):
                            val_text = f"{int(val)}"
                        elif isinstance(val, float):
                            val_text = f"{val:.4f}"
                        else:
                            val_text = str(val)
                    ax.text(
                        axis_x + 0.15,
                        norm_pos,
                        val_text,
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

    # Add colorbar with actual mAP range
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Val mAP@0.5:0.95", fontsize=11, fontweight="bold")

    # Update colorbar ticks to show actual mAP values
    n_ticks = 5
    tick_positions = np.linspace(0, 1, n_ticks)
    tick_labels = [
        f"{mAP_min + (mAP_max - mAP_min) * pos:.4f}" for pos in tick_positions
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=9)

    # Remove y axis label for clarity
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Set x-axis limits and ticks
    ax.set_xlim(-0.5, len(cols_to_plot[:-1]) - 0.5)
    ax.set_xticks(range(len(cols_to_plot[:-1])))

    ax.set_title(
        "Parallel Coordinates: Hyperparameter Configuration Overview",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = output_dir / "parallel_coordinates.png"
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path.name}")


def create_top_n_comparison(df_grid: pd.DataFrame, output_dir: Path, n: int = 10):
    """Create comparison plot of top N configurations."""

    # Filter out rows with NaN or Inf
    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["val_map50_95"]
    )

    if len(df_clean) == 0:
        print(
            f"   ⚠️  Skipping top N comparison: No valid data after removing NaN/Inf values"
        )
        return

    # Sort by validation mAP
    actual_n = min(n, len(df_clean))
    df_top = df_clean.nlargest(actual_n, "val_map50_95")

    if actual_n < n:
        print(
            f"   ℹ️  Top N comparison: Only {actual_n} valid experiments (requested {n})"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Truncate labels for readability
    labels = [name[:40] + "..." if len(name) > 40 else name for name in df_top["label"]]

    # Replace remaining NaN/Inf in test metrics with 0 for visualization
    df_top = df_top.copy()
    df_top["test_map50_95"] = (
        df_top["test_map50_95"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    x = np.arange(len(labels))

    # Plot 1: mAP comparison
    width = 0.35
    bars1 = ax1.bar(
        x - width / 2,
        df_top["val_map50_95"],
        width,
        label="Val mAP@0.5:0.95",
        color="#3498db",
        edgecolor="black",
    )
    bars2 = ax1.bar(
        x + width / 2,
        df_top["test_map50_95"],
        width,
        label="Test mAP@0.5:0.95",
        color="#e74c3c",
        edgecolor="black",
    )

    ax1.set_ylabel("mAP Score", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Top {n} Configurations by Validation mAP", fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    # Plot 2: Multi-metric comparison
    metrics = ["val_map50_95", "convergence_epoch", "stability"]
    metric_labels = ["Val mAP\n(↑)", "Convergence\nEpoch (↓)", "Stability\nStd (↓)"]

    # Normalize metrics to 0-1 for comparison
    normalized_data = []
    for metric in metrics:
        vals = df_top[metric].values
        if metric == "val_map50_95":
            # Higher is better - normalize to 0-1
            normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        else:
            # Lower is better - invert and normalize
            normalized = 1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        normalized_data.append(normalized)

    x_metric = np.arange(len(metrics))
    width = 0.8 / len(labels)

    for i, (label, color) in enumerate(zip(labels, plt.cm.tab10.colors)):
        offset = (i - len(labels) / 2) * width
        values = [normalized_data[j][i] for j in range(len(metrics))]
        ax2.bar(x_metric + offset, values, width, label=label, color=color, alpha=0.8)

    ax2.set_ylabel("Normalized Score (0-1)", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Multi-Metric Comparison (Normalized)", fontsize=13, fontweight="bold"
    )
    ax2.set_xticks(x_metric)
    ax2.set_xticklabels(metric_labels, fontsize=10)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = output_dir / f"top_{n}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path.name}")


def create_parameter_importance_plot(
    df_grid: pd.DataFrame, varying_params: List[str], output_dir: Path
):
    """Analyze and plot parameter importance using correlation."""

    # Filter out rows with NaN or Inf
    df_clean = df_grid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["val_map50_95"]
    )

    if len(df_clean) == 0:
        print(
            f"   ⚠️  Skipping parameter importance: No valid data after removing NaN/Inf values"
        )
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate correlations with validation mAP
    correlations = {}
    for param in varying_params:
        if param in df_clean.columns and df_clean[param].dtype in [
            np.float64,
            np.int64,
        ]:
            corr = df_clean[[param, "val_map50_95"]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations[param] = abs(corr)  # Absolute correlation

    if not correlations:
        plt.close()
        return

    # Sort by importance
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    param_names = [p.replace("_", " ").title() for p, _ in sorted_params]
    importances = [imp for _, imp in sorted_params]

    # Plot 1: Parameter importance (correlation)
    bars = ax1.barh(param_names, importances, color="#3498db", edgecolor="black")
    ax1.set_xlabel("Absolute Correlation with Val mAP", fontsize=11, fontweight="bold")
    ax1.set_title("Hyperparameter Importance", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax1.text(
            val + 0.01,
            i,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Variance explained
    # Calculate range of mAP for each parameter's variations
    variance_explained = {}
    for param in varying_params:
        if param in df_clean.columns and df_clean[param].dtype in [
            np.float64,
            np.int64,
        ]:
            # Group by parameter and calculate mAP range
            grouped = df_clean.groupby(param)["val_map50_95"]
            mAP_range = grouped.max() - grouped.min()
            mean_range = mAP_range.mean()
            if not np.isnan(mean_range):
                variance_explained[param] = mean_range

    if variance_explained:
        sorted_variance = sorted(
            variance_explained.items(), key=lambda x: x[1], reverse=True
        )
        param_names_var = [p.replace("_", " ").title() for p, _ in sorted_variance]
        variances = [v for _, v in sorted_variance]

        bars2 = ax2.barh(param_names_var, variances, color="#e74c3c", edgecolor="black")
        ax2.set_xlabel("mAP Range Induced", fontsize=11, fontweight="bold")
        ax2.set_title("Parameter Impact on Performance", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, variances)):
            ax2.text(
                val + 0.0001,
                i,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    save_path = output_dir / "parameter_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path.name}")


def generate_summary_report(experiments_data: List[Dict], df_matrix: pd.DataFrame):
    """Generate text summary report."""

    report = []
    report.append("=" * 80)
    report.append(f"PHASE {PHASE_NAME}: OPTIMIZER SELECTION - ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Best performer
    best = df_matrix.iloc[0]
    report.append(f"🏆 RECOMMENDED CONFIGURATION: {best['Experiment']}")
    report.append("")
    report.append("JUSTIFICATION:")
    report.append(
        f"  • Validation mAP@0.5:0.95: {best['Val_mAP@0.5:0.95']:.4f} (Rank #{int(best['Rank_mAP50-95'])})"
    )
    report.append(
        f"  • Test mAP@0.5: {best['Test_mAP@0.5']:.4f} (Rank #{int(best['Rank_Test_mAP'])})"
    )
    report.append(
        f"  • Train-Val Gap: {best['Train-Val_Gap_%']:.2f}% (Rank #{int(best['Rank_Train-Val_Gap'])})"
    )
    report.append(
        f"  • Val-Test Gap: {best['Val-Test_Gap_%']:.2f}% (Rank #{int(best['Rank_Val-Test_Gap'])})"
    )
    report.append(
        f"  • Convergence: Epoch {int(best['Convergence_Epoch'])} (Rank #{int(best['Rank_Convergence'])})"
    )
    report.append(
        f"  • Stability: {best['Stability_Std']:.4f} std (Rank #{int(best['Rank_Stability'])})"
    )
    report.append(f"  • Total Score: {best['Total_Score']:.2f} (Lower is better)")
    report.append("")

    # Full rankings
    report.append("=" * 80)
    report.append("COMPLETE RANKINGS")
    report.append("=" * 80)
    report.append("")

    for idx, row in df_matrix.iterrows():
        rank = df_matrix.index.get_loc(idx) + 1
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        report.append(f"{medal} {row['Experiment']}")
        report.append(f"   Val mAP@0.5:0.95: {row['Val_mAP@0.5:0.95']:.4f}")
        report.append(f"   Test mAP@0.5: {row['Test_mAP@0.5']:.4f}")
        report.append(f"   Train-Val Gap: {row['Train-Val_Gap_%']:.2f}%")
        report.append(f"   Convergence: Epoch {int(row['Convergence_Epoch'])}")
        report.append(f"   Total Score: {row['Total_Score']:.2f}")
        report.append("")

    # Key insights
    report.append("=" * 80)
    report.append("KEY INSIGHTS")
    report.append("=" * 80)
    report.append("")

    # Check for overfitting
    high_gap = df_matrix[df_matrix["Train-Val_Gap_%"] > 15]
    if len(high_gap) > 0:
        report.append(
            f"⚠️  {len(high_gap)} experiment(s) show high train-val gap (>15%):"
        )
        for _, row in high_gap.iterrows():
            report.append(f"   • {row['Experiment']}: {row['Train-Val_Gap_%']:.2f}%")
        report.append("")
    else:
        report.append("✅ All experiments show acceptable train-val gap (<15%)")
        report.append("")

    # Check convergence
    avg_convergence = df_matrix["Convergence_Epoch"].mean()
    report.append(f"📊 Average convergence: Epoch {avg_convergence:.1f}")
    fastest = df_matrix.loc[df_matrix["Convergence_Epoch"].idxmin()]
    report.append(
        f"⚡ Fastest convergence: {fastest['Experiment']} (Epoch {int(fastest['Convergence_Epoch'])})"
    )
    report.append("")

    # Check generalization
    avg_gap = df_matrix["Val-Test_Gap_%"].mean()
    report.append(f"🎯 Average val-test gap: {avg_gap:.2f}%")
    best_gen = df_matrix.loc[df_matrix["Val-Test_Gap_%"].idxmin()]
    report.append(
        f"🏅 Best generalization: {best_gen['Experiment']} ({best_gen['Val-Test_Gap_%']:.2f}% gap)"
    )
    report.append("")

    report.append("=" * 80)
    report.append("RECOMMENDATION FOR SUBSEQUENT PHASES")
    report.append("=" * 80)
    report.append("")
    report.append(f"Use '{best['Experiment']}' configuration for:")
    report.append("  • Phase 1B: Transfer Learning Strategy")
    report.append("  • Phase 2: Augmentation Validation")
    report.append("  • Phase 3: Data Efficiency")
    report.append("  • Phase 4: Spatial Consistency Loss")
    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Saved: {report_path}")


def main():
    """Main analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Phase {PHASE_NAME} Optimizer Analysis"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="best",
        choices=["best", "average", "latest"],
        help="Analysis mode: 'best' (best run by val loss), 'average' (average all runs), 'latest' (most recent)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter experiments by substring match (e.g., 'adamW_lr0_001' to match all experiments with that pattern)",
    )
    parser.add_argument(
        "--grid-search",
        type=str,
        default=None,
        help="Create grid search visualization plots. Specify name for output directory (e.g., 'adamw_lr_search')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only analyze top N experiments by validation mAP (e.g., --top 10)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(
        f"PHASE {PHASE_NAME}: OPTIMIZER SELECTION - COMPREHENSIVE ANALYSIS (Mode: {args.mode})"
    )
    print("=" * 80 + "\n")

    # Determine which experiments to analyze
    if args.filter:
        print(f"🔍 Filtering experiments with pattern: '{args.filter}'")
        # Search in experiments directory for matching folders
        available_experiments = [
            d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()
        ]
        experiments_to_analyze = [
            exp for exp in available_experiments if args.filter in exp
        ]

        if not experiments_to_analyze:
            print(f"❌ No experiments found matching filter '{args.filter}'")
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
        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab20")
        for idx, exp_name in enumerate(sorted(experiments_to_analyze)):
            EXPERIMENT_LABELS[exp_name] = exp_name.replace("_", " ").title()
            COLORS[exp_name] = cm.colors.to_hex(cmap(idx % 20))
    else:
        experiments_to_analyze = PHASE_EXPERIMENTS

    # Load all experiment data
    print(f"📂 Loading experiment data (mode={args.mode})...")
    experiments_data = []
    for exp_name in experiments_to_analyze:
        try:
            exp_data = load_experiment_data(exp_name, mode=args.mode)
            experiments_data.append(exp_data)
            if exp_data["mode"] == "average":
                label = EXPERIMENT_LABELS.get(exp_name, exp_name)
                print(f"  ✅ Loaded: {label} (averaged {exp_data['num_runs']} runs)")
            else:
                print(f"  ✅ Loaded: {exp_data['label']}")
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipped {exp_name}: {e}")
        except Exception as e:
            print(f"  ⚠️  Error loading {exp_name}: {e}")

    if len(experiments_data) == 0:
        print(
            f"\n❌ No experiment data found. Please run Phase {PHASE_NAME} experiments first."
        )
        return

    # Filter to top N experiments if requested
    if args.top and args.top < len(experiments_data):
        print(f"\n🔝 Filtering to top {args.top} experiments by validation mAP...")
        # Build a safe list excluding NaN/Inf metrics to align with grid search behavior
        safe_experiments = []
        for exp in experiments_data:
            try:
                series = exp["results"]["metrics/mAP50-95(B)"]
            except KeyError:
                # Missing metric: deprioritize
                continue
            final_map = series.iloc[-1]
            if isinstance(final_map, (float, int)) and not (
                np.isnan(final_map) or np.isinf(final_map)
            ):
                safe_experiments.append((final_map, exp))

        if len(safe_experiments) == 0:
            print("   ⚠️ No valid mAP50-95 metrics found; skipping top filter.")
        else:
            # Sort by final validation mAP (descending)
            safe_experiments.sort(key=lambda t: t[0], reverse=True)
            experiments_data = [exp for _, exp in safe_experiments[: args.top]]
            print(f"   Selected experiments:")
            for i, exp in enumerate(experiments_data, 1):
                final_map = exp["results"]["metrics/mAP50-95(B)"].iloc[-1]
                print(f"   {i}. {exp['label']} (mAP50-95: {final_map:.4f})")
            print()

    print(f"\n📊 Analyzing {len(experiments_data)} experiments...\n")

    # Grid search mode - create specialized visualizations
    if args.grid_search:
        create_grid_search_plots(experiments_data, args.grid_search)

    # Generate standard plots
    print("🎨 Generating plots...")
    plot_loss_convergence(experiments_data)
    plot_train_val_gap(experiments_data)
    plot_train_val_comparison_per_experiment(experiments_data)  # NEW
    plot_loss_components_per_experiment(experiments_data)  # NEW
    plot_map_comparison(experiments_data)
    plot_test_map_comparison(experiments_data)
    plot_test_loss_comparison(experiments_data)
    plot_val_test_gap(experiments_data)
    plot_convergence_speed(experiments_data)

    # Generate decision matrix
    print("\n📋 Generating decision matrix...")
    df_matrix = generate_decision_matrix(experiments_data)
    save_decision_matrix(df_matrix)

    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_summary_report(experiments_data, df_matrix)

    print("\n" + "=" * 80)
    print(f"✅ ANALYSIS COMPLETE! All results saved to: {OUTPUT_DIR.absolute()}")
    print(f"📊 Analysis mode: {args.mode}")
    if args.filter:
        print(
            f"🔍 Filter applied: '{args.filter}' ({len(experiments_data)} experiments)"
        )
    if args.top:
        print(f"🔝 Top N filter: {args.top} best experiments")
    if args.grid_search:
        print(f"🔬 Grid search visualizations saved to: ./analysis/{args.grid_search}")
    if args.mode == "average":
        print("   Note: Metrics represent averages across all runs for each experiment")
    elif args.mode == "best":
        print("   Note: Using best run (lowest validation loss) for each experiment")
    elif args.mode == "latest":
        print("   Note: Using most recent run for each experiment")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
