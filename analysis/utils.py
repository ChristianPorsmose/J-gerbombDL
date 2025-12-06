
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analysis.globals import Output, PhaseName, Experiments


def total_loss(df : dict, base_key : str):
    return df.get(f"{base_key}/box_loss", 0) + df.get(f"{base_key}/cls_loss", 0) \
        + df.get("{base_key}/dfl_loss", 0) + df.get("{base_key}/spatial_loss", 0)


def generate_summary_report(experiments_data: List[Dict], df_matrix: pd.DataFrame):
    """Generate text summary report."""

    report = []
    report.append("=" * 80)
    report.append(f"PHASE {PhaseName.name}: OPTIMIZER SELECTION - ANALYSIS REPORT")
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
    report_path = Output.path / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Saved: {report_path}")

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


def load_experiment_data(experiment_name: str) -> Dict:
    """
    Load experiment data.

    Args:
        experiment_name: Name of the experiment
        mode: "best" (best run by val loss), "average" (average all runs), or "latest" (most recent)
    """
    label = experiment_name.replace("_", " ").title()
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

def find_all_runs(experiment_name: str) -> List[Path]:
    """Find all run directories for an experiment."""
    exp_dir = Experiments.path / experiment_name / "runs"
    if not exp_dir.exists():
        raise FileNotFoundError(f"No runs found for {experiment_name}")

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