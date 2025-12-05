#!/usr/bin/env python3
"""
Plot train vs validation loss per experiment in subplots.
Optionally overlay a paired experiment (e.g., custom_loss) with dashed lines using --combine.

Usage examples:
  # Basic: plot all matching experiments individually
  python analysis/plot_train_val_compare_combined.py --filter "dataset_size_normal_loss"

  # Overlay custom vs normal for each matching base
  python analysis/plot_train_val_compare_combined.py --filter "dataset_size" \
      --combine "dataset_size_normal_loss,dataset_size_custom_loss"

Expected experiment directory layout:
  experiments/<experiment_name>/runs/<run_id>/results.csv
  experiments/<experiment_name>/runs/<run_id>/config.json

This script will use the latest run folder under each experiment.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = WORKSPACE_ROOT / "experiments"
OUTPUT_DIR = WORKSPACE_ROOT / "analysis" / "combined_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_run(exp_dir: Path) -> Optional[Path]:
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    # Latest by name or mtime
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def load_results_for_experiment(exp_name: str) -> Optional[Dict]:
    exp_dir = EXPERIMENTS_DIR / exp_name
    if not exp_dir.exists() or not exp_dir.is_dir():
        return None
    latest_run = find_latest_run(exp_dir)
    if latest_run is None:
        return None
    results_path = latest_run / "results.csv"
    if not results_path.exists():
        return None
    df = pd.read_csv(results_path)
    # Build a label
    label = exp_name.replace("_", " ").title()
    return {
        "name": exp_name,
        "label": label,
        "results": df,
        "run_path": latest_run,
    }


def discover_experiments(filter_substring: Optional[str]) -> List[str]:
    names = []
    for d in EXPERIMENTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            if filter_substring is None or filter_substring in d.name:
                names.append(d.name)
    return sorted(names)


def group_pairs(
    experiment_names: List[str], normal_token: str, custom_token: str
) -> List[Tuple[str, Optional[str]]]:
    """Group experiments into (normal, custom) pairs by replacing token.
    If a matching custom experiment doesn't exist, return None for that side.
    """
    name_set = set(experiment_names)
    pairs = []
    for name in experiment_names:
        if normal_token in name:
            base_custom = name.replace(normal_token, custom_token)
            custom_match = base_custom if base_custom in name_set else None
            pairs.append((name, custom_match))
    # Also catch cases where only custom exists
    for name in experiment_names:
        if custom_token in name:
            base_normal = name.replace(custom_token, normal_token)
            if base_normal not in name_set:
                pairs.append((base_normal, name))
    # Deduplicate by set
    unique = []
    seen = set()
    for n, c in pairs:
        key = (n or "", c or "")
        if key not in seen:
            unique.append((n, c))
            seen.add(key)
    return unique


def plot_train_val_subplot(
    ax,
    exp: Dict,
    color1: str = "blue",
    color2: str = "red",
    style: str = "solid",
    style2: str = "dashed",
    label_suffix: str = "",
):
    df = exp["results"]
    # Compute totals (box + cls + dfl)
    has_train = all(
        col in df.columns
        for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    )
    has_val = all(
        col in df.columns for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    )
    if not (has_train and has_val):
        return
    train_total = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
    val_total = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
    epochs = df["epoch"] if "epoch" in df.columns else np.arange(len(train_total))

    # Get 90 percent of data and set that as y_lim max
    ax.set_ylim(1, 5)

    # Training line (solid or dashed depending on style)
    ax.plot(
        epochs,
        train_total,
        label=f"Train{label_suffix}",
        color=color1,
        linestyle=style,
        linewidth=2.0,
    )
    # Validation line (dashed variant for clarity if style is solid, else use dotted)
    ax.plot(
        epochs,
        val_total,
        label=f"Val{label_suffix}",
        color=color2,
        linestyle=style2,
        linewidth=2.0,
    )


def make_color(idx: int) -> str:
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    return palette[idx % 20]


def plot_combined(
    experiment_pairs: List[Tuple[Optional[str], Optional[str]]],
    normal_token: str,
    custom_token: str,
):
    n = len(experiment_pairs)
    if n == 0:
        print("No experiment pairs to plot.")
        return

    # Layout similar to phase analysis function
    rows = 1 if n <= 3 else 2
    cols = n if n <= 3 else (n + 1) // 2
    figsize = (6 * cols, 5 if rows == 1 else 10)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (normal_name, custom_name) in enumerate(experiment_pairs):
        ax = axes[idx]
        color = make_color(idx)
        title = normal_name or custom_name or f"Pair {idx+1}"

        title_pretty = title.split("_")[-2:]
        model = title.split("_")[-4]
        # Make first letter uppercase
        model = model.capitalize()
        if model == "Small":
            model = "Custom "
        title_pretty = (
            model
            + " with training data proportion "
            + title_pretty[0][-1]
            + "."
            + title_pretty[1]
        )

        ax.set_title(title_pretty, fontsize=12, fontweight="bold")

        # Load normal
        normal_data = load_results_for_experiment(normal_name) if normal_name else None
        custom_data = load_results_for_experiment(custom_name) if custom_name else None

        if normal_data:
            plot_train_val_subplot(
                ax,
                normal_data,
                color1="blue",
                color2="red",
                style="solid",
                style2="solid",
                label_suffix=f" ({normal_token})",
            )
        if custom_data:
            plot_train_val_subplot(
                ax,
                custom_data,
                color1="green",
                color2="orange",
                style="dotted",
                style2="dotted",
                label_suffix=f" ({custom_token})",
            )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Total Loss", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"train_val_combined_{normal_token}_vs_{custom_token}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def plot_individual(experiment_names: List[str]):
    n = len(experiment_names)
    if n == 0:
        print("No experiments to plot.")
        return

    rows = 1 if n <= 3 else 2
    cols = n if n <= 3 else (n + 1) // 2
    figsize = (6 * cols, 5 if rows == 1 else 10)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, name in enumerate(experiment_names):
        ax = axes[idx]
        color = make_color(idx)
        data = load_results_for_experiment(name)
        if not data:
            ax.set_title(f"{name} (no data)", fontsize=12)
            ax.axis("off")
            continue
        ax.set_title(data["label"], fontsize=12, fontweight="bold")
        plot_train_val_subplot(ax, data, color=color, style="solid", label_suffix="")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Total Loss", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "train_val_individual.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val loss per experiment, with optional overlay combine pairs."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring to filter experiments by name",
    )
    parser.add_argument(
        "--combine",
        type=str,
        default=None,
        help='Comma-separated tokens: "normal_token,custom_token" to pair and overlay',
    )
    args = parser.parse_args()

    print(f"🔎 Experiments dir: {EXPERIMENTS_DIR}")
    exp_names = discover_experiments(args.filter)
    print(f"Found {len(exp_names)} experiments matching filter: {args.filter}")

    if args.combine:
        try:
            normal_token, custom_token = [s.strip() for s in args.combine.split(",")]
        except ValueError:
            print(
                '❌ --combine must be two comma-separated tokens, e.g., "dataset_size_normal_loss,dataset_size_custom_loss"'
            )
            return
        pairs = group_pairs(exp_names, normal_token, custom_token)
        print(f"Pairing results: {len(pairs)} pairs")
        plot_combined(pairs, normal_token, custom_token)
    else:
        plot_individual(exp_names)


if __name__ == "__main__":
    main()
