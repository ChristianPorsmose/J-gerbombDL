from matplotlib import pyplot as plt
import pandas as pd
from analysis.experiment_list import OUTPUT_DIR, PHASE_NAME
from analysis.plotting.utils import save_plot


def _style_row(table, col_labels, row_idx: int, facecolor: str, text_color: str = "black", bold: bool = True):
    for i in range(len(col_labels)):
        table[(row_idx, i)].set_facecolor(facecolor)
        table[(row_idx, i)].set_text_props(weight="bold" if bold else "normal", color=text_color)

def _get_col_lables():
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

def _get_display_cols():
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
    display_cols = _get_display_cols()
    col_labels = _get_col_lables()
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


    _style_row(table, col_labels, 0, "#3498db", text_color="white")

    _style_row(table, col_labels,1, "#2ecc71")

    plt.title(
        f"Phase {PHASE_NAME} Decision Matrix (Sorted by Total Score)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    save_plot("6_decision_matrix.png")

