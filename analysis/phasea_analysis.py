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
    "phase1b_not_frozen",
    "phase1b_fully_frozen",
    "phase1b_frozen_half",
]

EXPERIMENT_1B_LABELS = {
    "phase1b_not_frozen": "Not Frozen",
    "phase1b_fully_frozen": "Fully Frozen",
    "phase1b_frozen_half": "Frozen Half",
}

COLORS_1B = {
    "phase1b_not_frozen": "#d61c3b",
    "phase1b_fully_frozen": "#11d663",
    "phase1b_frozen_half": "#1440d1",
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
    best_val_loss = float('inf')
    
    for run_dir in runs:
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            final_val_loss = (df['val/box_loss'].iloc[-1] + 
                            df['val/cls_loss'].iloc[-1] + 
                            df['val/dfl_loss'].iloc[-1])
            
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
        with open(test_json_path, 'r') as f:
            test_results = json.load(f)
    
    return {
        "results": results_df,
        "test_results": test_results,
        "run_dir": run_dir
    }


def average_runs(experiment_name: str, runs: List[Path]) -> Dict:
    """Average metrics across multiple runs."""
    all_results = []
    all_test_results = []
    
    for run_dir in runs:
        try:
            run_data = load_single_run_data(run_dir)
            all_results.append(run_data['results'])
            if run_data['test_results']:
                all_test_results.append(run_data['test_results'])
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
        if col != 'epoch':
            values = np.array([df[col].values for df in truncated_dfs])
            avg_df[col] = np.mean(values, axis=0)
    
    # Average test results
    avg_test_results = None
    if all_test_results:
        avg_test_results = {
            'metrics': {
                'precision': np.mean([r['metrics']['precision'] for r in all_test_results]),
                'recall': np.mean([r['metrics']['recall'] for r in all_test_results]),
                'mAP50': np.mean([r['metrics']['mAP50'] for r in all_test_results]),
                'mAP50-95': np.mean([r['metrics']['mAP50-95'] for r in all_test_results]),
            },
            'losses': {
                'box': np.mean([r['losses']['box'] for r in all_test_results]),
                'cls': np.mean([r['losses']['cls'] for r in all_test_results]),
                'dfl': np.mean([r['losses']['dfl'] for r in all_test_results]),
                'spatial': np.mean([r['losses']['spatial'] for r in all_test_results]),
                'total': np.mean([r['losses']['total'] for r in all_test_results]),
            }
        }
    
    return {
        "results": avg_df,
        "test_results": avg_test_results,
        "num_runs": len(all_results)
    }


def load_experiment_data(experiment_name: str, mode: str = "best") -> Dict:
    """
    Load experiment data.
    
    Args:
        experiment_name: Name of the experiment
        mode: "best" (best run by val loss), "average" (average all runs), or "latest" (most recent)
    """
    runs = find_all_runs(experiment_name)
    
    if mode == "best":
        run_dir = find_best_run(experiment_name)
        run_data = load_single_run_data(run_dir)
        return {
            "name": experiment_name,
            "label": EXPERIMENT_LABELS[experiment_name],
            "results": run_data['results'],
            "test_results": run_data['test_results'],
            "run_dir": run_dir,
            "mode": "best",
            "num_runs": 1
        }
    
    elif mode == "average":
        avg_data = average_runs(experiment_name, runs)
        return {
            "name": experiment_name,
            "label": EXPERIMENT_LABELS[experiment_name] + f" (avg n={avg_data['num_runs']})",
            "results": avg_data['results'],
            "test_results": avg_data['test_results'],
            "run_dir": None,
            "mode": "average",
            "num_runs": avg_data['num_runs']
        }
    
    elif mode == "latest":
        run_dir = runs[-1]
        run_data = load_single_run_data(run_dir)
        return {
            "name": experiment_name,
            "label": EXPERIMENT_LABELS[experiment_name],
            "results": run_data['results'],
            "test_results": run_data['test_results'],
            "run_dir": run_dir,
            "mode": "latest",
            "num_runs": 1
        }
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'best', 'average', or 'latest'")


def calculate_convergence_epoch(results_df: pd.DataFrame, threshold: float = 0.95) -> int:
    """Calculate epoch where model reaches 95% of final mAP."""
    final_map = results_df['metrics/mAP50(B)'].iloc[-1]
    target = threshold * final_map
    
    converged = results_df[results_df['metrics/mAP50(B)'] >= target]
    if len(converged) == 0:
        return len(results_df)  # Never converged
    
    return int(converged.iloc[0]['epoch'])


def calculate_train_val_gap(results_df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate mean and std of train-val gap percentage."""
    train_total = (results_df['train/box_loss'] + 
                   results_df['train/cls_loss'] + 
                   results_df['train/dfl_loss'])
    val_total = (results_df['val/box_loss'] + 
                 results_df['val/cls_loss'] + 
                 results_df['val/dfl_loss'])
    
    # Calculate gap as percentage
    gap_percent = ((train_total - val_total) / train_total * 100).abs()
    
    return gap_percent.mean(), gap_percent.std()


def calculate_stability(results_df: pd.DataFrame, last_n: int = 20) -> float:
    """Calculate validation loss stability (std dev) in last N epochs."""
    val_total = (results_df['val/box_loss'] + 
                 results_df['val/cls_loss'] + 
                 results_df['val/dfl_loss'])
    
    last_n_losses = val_total.iloc[-last_n:]
    return last_n_losses.std()


def plot_loss_convergence(experiments_data: List[Dict]):
    """Plot validation loss convergence for all experiments."""
    plt.figure(figsize=(14, 7))
    
    for exp_data in experiments_data:
        df = exp_data['results']
        val_total = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        
        plt.plot(df['epoch'], val_total, 
                label=exp_data['label'],
                linewidth=2.5,
                color=COLORS[exp_data['name']])
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Total Loss', fontsize=13, fontweight='bold')
    plt.title(f'Phase {PHASE_NAME}: Validation Loss Convergence Comparison', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "1_loss_convergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_train_val_gap(experiments_data: List[Dict]):
    """Plot train-val gap over time for overfitting analysis."""
    plt.figure(figsize=(14, 7))
    
    for exp_data in experiments_data:
        df = exp_data['results']
        train_total = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
        val_total = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
        gap_percent = ((train_total - val_total) / train_total * 100).abs()
        
        plt.plot(df['epoch'], gap_percent,
                label=exp_data['label'],
                linewidth=2.5,
                color=COLORS[exp_data['name']])
    
    plt.axhline(y=15, color='red', linestyle='--', linewidth=2, 
                label='Overfitting Threshold (15%)', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Train-Val Gap (%)', fontsize=13, fontweight='bold')
    plt.title(f'Phase {PHASE_NAME}: Overfitting Analysis (Train-Val Loss Gap)', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "2_train_val_gap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_map_comparison(experiments_data: List[Dict]):
    """Plot final mAP@0.5:0.95 comparison as bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    labels = [exp['label'] for exp in experiments_data]
    map50_95 = [exp['results']['metrics/mAP50-95(B)'].iloc[-1] for exp in experiments_data]
    map50 = [exp['results']['metrics/mAP50(B)'].iloc[-1] for exp in experiments_data]
    colors_list = [COLORS[exp['name']] for exp in experiments_data]
    
    # Plot 1: mAP@0.5:0.95
    bars1 = ax1.bar(labels, map50_95, color=colors_list, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('mAP@0.5:0.95', fontsize=13, fontweight='bold')
    ax1.set_title('Final Validation mAP@0.5:0.95', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='0.95 threshold')
    ax1.set_ylim([0.9, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: mAP@0.5
    bars2 = ax2.bar(labels, map50, color=colors_list, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('mAP@0.5', fontsize=13, fontweight='bold')
    ax2.set_title('Final Validation mAP@0.5', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='0.95 threshold')
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "3_map_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_test_map_comparison(experiments_data: List[Dict]):
    """Plot final TEST mAP comparison as bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    labels = [exp['label'] for exp in experiments_data]
    colors_list = [COLORS[exp['name']] for exp in experiments_data]
    
    # Extract test mAP values
    test_map50_95 = []
    test_map50 = []
    
    for exp in experiments_data:
        if exp['test_results'] and 'metrics' in exp['test_results']:
            test_map50_95.append(exp['test_results']['metrics']['mAP50-95'])
            test_map50.append(exp['test_results']['metrics']['mAP50'])
        else:
            test_map50_95.append(np.nan)
            test_map50.append(np.nan)
    
    # Plot 1: Test mAP@0.5:0.95
    bars1 = ax1.bar(labels, test_map50_95, color=colors_list, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('mAP@0.5:0.95', fontsize=13, fontweight='bold')
    ax1.set_title('Final Test mAP@0.5:0.95', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='0.95 threshold')
    ax1.set_ylim([0.9, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, test_map50_95):
        if not np.isnan(val):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Test mAP@0.5
    bars2 = ax2.bar(labels, test_map50, color=colors_list, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('mAP@0.5', fontsize=13, fontweight='bold')
    ax2.set_title('Final Test mAP@0.5', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='0.95 threshold')
    ax2.set_ylim([0.9, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels
    for bar, val in zip(bars2, test_map50):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "3b_test_map_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_test_loss_comparison(experiments_data: List[Dict]):
    """Plot final TEST loss components comparison as grouped bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    labels = [exp['label'] for exp in experiments_data]
    
    # Extract test loss values
    test_box = []
    test_cls = []
    test_dfl = []
    test_spatial = []
    
    for exp in experiments_data:
        if exp['test_results'] and 'losses' in exp['test_results']:
            test_box.append(exp['test_results']['losses']['box'])
            test_cls.append(exp['test_results']['losses']['cls'])
            test_dfl.append(exp['test_results']['losses']['dfl'])
            test_spatial.append(exp['test_results']['losses']['spatial'])
        else:
            test_box.append(0)
            test_cls.append(0)
            test_dfl.append(0)
            test_spatial.append(0)
    
    x = np.arange(len(labels))
    width = 0.2
    
    # Create grouped bars
    bars1 = ax.bar(x - 1.5*width, test_box, width, label='Box Loss', 
                   color='#e74c3c', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x - 0.5*width, test_cls, width, label='Class Loss', 
                   color='#3498db', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + 0.5*width, test_dfl, width, label='DFL Loss', 
                   color='#2ecc71', edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + 1.5*width, test_spatial, width, label='Spatial Loss', 
                   color='#9b59b6', edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Loss Value', fontsize=13, fontweight='bold')
    ax.set_title(f'Phase {PHASE_NAME}: Test Loss Components Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for non-zero values)
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only show label if value is significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "3c_test_loss_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_val_test_gap(experiments_data: List[Dict]):
    """Plot validation vs test performance comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    labels = [exp['label'] for exp in experiments_data]
    
    # Extract mAP@0.5 values
    val_map50 = [exp['results']['metrics/mAP50(B)'].iloc[-1] for exp in experiments_data]
    test_map50 = []
    
    # Extract mAP@0.5:0.95 values
    val_map50_95 = [exp['results']['metrics/mAP50-95(B)'].iloc[-1] for exp in experiments_data]
    test_map50_95 = []
    
    for exp in experiments_data:
        if exp['test_results'] and 'metrics' in exp['test_results']:
            test_map50.append(exp['test_results']['metrics']['mAP50'])
            test_map50_95.append(exp['test_results']['metrics']['mAP50-95'])
        else:
            test_map50.append(np.nan)
            test_map50_95.append(np.nan)
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot 1: mAP@0.5
    bars1 = ax1.bar(x - width/2, val_map50, width, label='Validation mAP@0.5',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, test_map50, width, label='Test mAP@0.5',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('mAP@0.5', fontsize=13, fontweight='bold')
    ax1.set_title('Validation vs Test Performance (mAP@0.5)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.9, 1.0])
    
    # Add value labels for mAP@0.5
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: mAP@0.5:0.95
    bars3 = ax2.bar(x - width/2, val_map50_95, width, label='Validation mAP@0.5:0.95',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, test_map50_95, width, label='Test mAP@0.5:0.95',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('mAP@0.5:0.95', fontsize=13, fontweight='bold')
    ax2.set_title('Validation vs Test Performance (mAP@0.5:0.95)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.9, 1.0])
    
    # Add value labels for mAP@0.5:0.95
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "4_val_test_gap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_convergence_speed(experiments_data: List[Dict]):
    """Plot convergence speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = [exp['label'] for exp in experiments_data]
    convergence_epochs = [calculate_convergence_epoch(exp['results']) for exp in experiments_data]
    colors_list = [COLORS[exp['name']] for exp in experiments_data]
    
    bars = ax.barh(labels, convergence_epochs, color=colors_list, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Epochs to Reach 95% of Final mAP', fontsize=13, fontweight='bold')
    ax.set_title(f'Phase {PHASE_NAME}: Convergence Speed Comparison', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, convergence_epochs)):
        ax.text(val + 0.5, i, f'{val}',
               va='center', ha='left', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "5_convergence_speed.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def generate_decision_matrix(experiments_data: List[Dict]) -> pd.DataFrame:
    """Generate comprehensive decision matrix with all metrics."""
    
    matrix_data = []
    
    for exp in experiments_data:
        df = exp['results']
        
        # Extract metrics
        final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
        final_map50 = df['metrics/mAP50(B)'].iloc[-1]
        final_precision = df['metrics/precision(B)'].iloc[-1]
        final_recall = df['metrics/recall(B)'].iloc[-1]
        
        train_val_gap_mean, train_val_gap_std = calculate_train_val_gap(df)
        convergence_epoch = calculate_convergence_epoch(df)
        stability = calculate_stability(df)
        
        # Test metrics
        test_map50 = np.nan
        val_test_gap = np.nan
        if exp['test_results'] and 'metrics' in exp['test_results']:
            test_map50 = exp['test_results']['metrics']['mAP50']
            val_test_gap = abs(final_map50 - test_map50) / final_map50 * 100
        
        matrix_data.append({
            'Experiment': exp['label'],
            'Val_mAP@0.5': final_map50,
            'Val_mAP@0.5:0.95': final_map50_95,
            'Precision': final_precision,
            'Recall': final_recall,
            'Test_mAP@0.5': test_map50,
            'Val-Test_Gap_%': val_test_gap,
            'Train-Val_Gap_%': train_val_gap_mean,
            'Gap_Std_%': train_val_gap_std,
            'Convergence_Epoch': convergence_epoch,
            'Stability_Std': stability
        })
    
    df_matrix = pd.DataFrame(matrix_data)
    
    # Calculate rankings (lower rank = better)
    # For metrics where higher is better, we invert
    df_matrix['Rank_mAP50-95'] = df_matrix['Val_mAP@0.5:0.95'].rank(ascending=False)
    df_matrix['Rank_Test_mAP'] = df_matrix['Test_mAP@0.5'].rank(ascending=False)
    df_matrix['Rank_Train-Val_Gap'] = df_matrix['Train-Val_Gap_%'].rank(ascending=True)
    df_matrix['Rank_Val-Test_Gap'] = df_matrix['Val-Test_Gap_%'].rank(ascending=True)
    df_matrix['Rank_Convergence'] = df_matrix['Convergence_Epoch'].rank(ascending=True)
    df_matrix['Rank_Stability'] = df_matrix['Stability_Std'].rank(ascending=True)
    
    # Weighted scoring
    df_matrix['Total_Score'] = (
        df_matrix['Rank_mAP50-95'] * 3 +  # mAP most important
        df_matrix['Rank_Test_mAP'] * 2 +   # Test performance
        df_matrix['Rank_Train-Val_Gap'] * 2 +  # Generalization
        df_matrix['Rank_Val-Test_Gap'] * 1.5 +
        df_matrix['Rank_Convergence'] * 1 +  # Efficiency
        df_matrix['Rank_Stability'] * 1      # Stability
    )
    
    # Sort by total score (lower is better)
    df_matrix = df_matrix.sort_values('Total_Score')
    
    return df_matrix


def save_decision_matrix(df_matrix: pd.DataFrame):
    """Save decision matrix to CSV and create visualization."""
    
    # Save to CSV
    csv_path = OUTPUT_DIR / "decision_matrix.csv"
    df_matrix.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Saved: {csv_path}")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns to display
    display_cols = [
        'Experiment', 'Val_mAP@0.5:0.95', 'Test_mAP@0.5', 'Val-Test_Gap_%',
        'Train-Val_Gap_%', 'Convergence_Epoch', 'Stability_Std', 'Total_Score'
    ]
    
    table_data = df_matrix[display_cols].values
    col_labels = [
        'Experiment', 'Val mAP\n@0.5:0.95 ↑', 'Test mAP\n@0.5 ↑', 'Val-Test\nGap % ↓',
        'Train-Val\nGap % ↓', 'Converge\nEpoch ↓', 'Stability\nStd ↓', 'Score ↓'
    ]
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.11, 0.11, 0.1, 0.11, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performer (first row)
    for i in range(len(col_labels)):
        table[(1, i)].set_facecolor('#2ecc71')
        table[(1, i)].set_text_props(weight='bold')
    
    plt.title(f'Phase {PHASE_NAME} Decision Matrix (Sorted by Total Score)', 
             fontsize=16, fontweight='bold', pad=20)
    
    save_path = OUTPUT_DIR / "6_decision_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def generate_summary_report(experiments_data: List[Dict], df_matrix: pd.DataFrame):
    """Generate text summary report."""
    
    report = []
    report.append("="*80)
    report.append(f"PHASE {PHASE_NAME}: OPTIMIZER SELECTION - ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Best performer
    best = df_matrix.iloc[0]
    report.append(f"🏆 RECOMMENDED CONFIGURATION: {best['Experiment']}")
    report.append("")
    report.append("JUSTIFICATION:")
    report.append(f"  • Validation mAP@0.5:0.95: {best['Val_mAP@0.5:0.95']:.4f} (Rank #{int(best['Rank_mAP50-95'])})")
    report.append(f"  • Test mAP@0.5: {best['Test_mAP@0.5']:.4f} (Rank #{int(best['Rank_Test_mAP'])})")
    report.append(f"  • Train-Val Gap: {best['Train-Val_Gap_%']:.2f}% (Rank #{int(best['Rank_Train-Val_Gap'])})")
    report.append(f"  • Val-Test Gap: {best['Val-Test_Gap_%']:.2f}% (Rank #{int(best['Rank_Val-Test_Gap'])})")
    report.append(f"  • Convergence: Epoch {int(best['Convergence_Epoch'])} (Rank #{int(best['Rank_Convergence'])})")
    report.append(f"  • Stability: {best['Stability_Std']:.4f} std (Rank #{int(best['Rank_Stability'])})")
    report.append(f"  • Total Score: {best['Total_Score']:.2f} (Lower is better)")
    report.append("")
    
    # Full rankings
    report.append("="*80)
    report.append("COMPLETE RANKINGS")
    report.append("="*80)
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
    report.append("="*80)
    report.append("KEY INSIGHTS")
    report.append("="*80)
    report.append("")
    
    # Check for overfitting
    high_gap = df_matrix[df_matrix['Train-Val_Gap_%'] > 15]
    if len(high_gap) > 0:
        report.append(f"⚠️  {len(high_gap)} experiment(s) show high train-val gap (>15%):")
        for _, row in high_gap.iterrows():
            report.append(f"   • {row['Experiment']}: {row['Train-Val_Gap_%']:.2f}%")
        report.append("")
    else:
        report.append("✅ All experiments show acceptable train-val gap (<15%)")
        report.append("")
    
    # Check convergence
    avg_convergence = df_matrix['Convergence_Epoch'].mean()
    report.append(f"📊 Average convergence: Epoch {avg_convergence:.1f}")
    fastest = df_matrix.loc[df_matrix['Convergence_Epoch'].idxmin()]
    report.append(f"⚡ Fastest convergence: {fastest['Experiment']} (Epoch {int(fastest['Convergence_Epoch'])})")
    report.append("")
    
    # Check generalization
    avg_gap = df_matrix['Val-Test_Gap_%'].mean()
    report.append(f"🎯 Average val-test gap: {avg_gap:.2f}%")
    best_gen = df_matrix.loc[df_matrix['Val-Test_Gap_%'].idxmin()]
    report.append(f"🏅 Best generalization: {best_gen['Experiment']} ({best_gen['Val-Test_Gap_%']:.2f}% gap)")
    report.append("")
    
    report.append("="*80)
    report.append("RECOMMENDATION FOR SUBSEQUENT PHASES")
    report.append("="*80)
    report.append("")
    report.append(f"Use '{best['Experiment']}' configuration for:")
    report.append("  • Phase 1B: Transfer Learning Strategy")
    report.append("  • Phase 2: Augmentation Validation")
    report.append("  • Phase 3: Data Efficiency")
    report.append("  • Phase 4: Spatial Consistency Loss")
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✅ Saved: {report_path}")


def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description=f"Phase {PHASE_NAME} Optimizer Analysis")
    parser.add_argument("--mode", type=str, default="best", 
                       choices=["best", "average", "latest"],
                       help="Analysis mode: 'best' (best run by val loss), 'average' (average all runs), 'latest' (most recent)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"PHASE {PHASE_NAME}: OPTIMIZER SELECTION - COMPREHENSIVE ANALYSIS (Mode: {args.mode.upper()})")
    print("="*80 + "\n")
    
    # Load all experiment data
    print(f"📂 Loading experiment data (mode={args.mode})...")
    experiments_data = []
    for exp_name in PHASE_EXPERIMENTS:
        try:
            exp_data = load_experiment_data(exp_name, mode=args.mode)
            experiments_data.append(exp_data)
            if exp_data['mode'] == 'average':
                print(f"  ✅ Loaded: {EXPERIMENT_LABELS[exp_name]} (averaged {exp_data['num_runs']} runs)")
            else:
                print(f"  ✅ Loaded: {exp_data['label']}")
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipped {exp_name}: {e}")
        except Exception as e:
            print(f"  ⚠️  Error loading {exp_name}: {e}")
    
    if len(experiments_data) == 0:
        print(f"\n❌ No experiment data found. Please run Phase {PHASE_NAME} experiments first.")
        return
    
    print(f"\n📊 Analyzing {len(experiments_data)} experiments...\n")
    
    # Generate plots
    print("🎨 Generating plots...")
    plot_loss_convergence(experiments_data)
    plot_train_val_gap(experiments_data)
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
    
    print("\n" + "="*80)
    print(f"✅ ANALYSIS COMPLETE! All results saved to: {OUTPUT_DIR.absolute()}")
    print(f"📊 Analysis mode: {args.mode.upper()}")
    if args.mode == "average":
        print("   Note: Metrics represent averages across all runs for each experiment")
    elif args.mode == "best":
        print("   Note: Using best run (lowest validation loss) for each experiment")
    elif args.mode == "latest":
        print("   Note: Using most recent run for each experiment")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
