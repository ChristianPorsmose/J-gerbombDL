
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
            _save_plot(f"heatmap_{param1}_vs_{param2}.png")

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

    _save_plot(f"sweep_{param}.png")


def create_parallel_coordinates_plot(
    df_grid: pd.DataFrame, varying_params: List[str], output_dir: Path
):
    """Create parallel coordinates plot for multi-dimensional visualization."""
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

    _save_plot("parallel_coordinates.png", dpi=250)

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
    _save_plot("parameter_importance.png")


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

    _save_plot(f"top_{n}_comparison.png")


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

    _save_plot("6_decision_matrix.png")