from typing import Dict, List
import numpy as np
import pandas as pd
from analysis.utils import calculate_convergence_epoch, calculate_stability, calculate_train_val_gap


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