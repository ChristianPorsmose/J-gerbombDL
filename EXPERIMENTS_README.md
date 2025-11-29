# Jäger Bomb Detection - Experiment Framework

This framework conducts systematic ablation studies to validate your model design choices.

## Quick Start

### 1. Generate Experiment Configurations

```bash
# Generate all Experiment 1 configs (Configuration Search)
python run_experiments.py --generate --experiment exp1
```

This creates configs in `experiments/exp1_configuration/`:
- `run1_baseline/` - Unfrozen + Standard LR + Standard Decay
- `run2_conservative/` - Unfrozen + Low LR + High Decay
- `run3_frozen_standard/` - Frozen Backbone + Standard LR
- `run4_frozen_low/` - Frozen Backbone + Low LR

### 2. Run Each Configuration

```bash
# Run baseline
python main.py --config experiments/exp1_configuration/run1_baseline/config.yaml

# Run conservative
python main.py --config experiments/exp1_configuration/run2_conservative/config.yaml

# Continue for all runs...
```

### 3. Analyze Results

```bash
# Analyze Experiment 1
python run_experiments.py --analyze --experiment exp1
```

### 4. Set Best Configuration

After analyzing Experiment 1, programmatically set the best config:

```python
from experiment_orchestrator import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator()

# Set the best run (example: run1_baseline performed best)
orchestrator.set_best_config("run1_baseline", {
    "mAP50": 0.85,
    "mAP50-95": 0.72,
    "precision": 0.88,
    "recall": 0.82
})

# Now generate Experiment 2 with the best config as baseline
orchestrator.generate_exp2_all()
```

## Experiment Structure

### Experiment 1: Configuration Search (Hyperparams & Freezing)

**Goal:** Find stable training configuration before testing complex features.

**Fixed:** Full Dataset (231 images), Standard Loss, Standard Augmentation

**Variables:**
- Backbone Freezing (Frozen/Unfrozen)
- Learning Rate (High/Low)
- Weight Decay (High/Low)

**Runs:**
- Run 1: Baseline (Unfrozen + Standard LR + Standard Decay)
- Run 2: Conservative (Unfrozen + Low LR + High Decay)
- Run 3: Frozen + Standard LR
- Run 4: Frozen + Low LR

**Expected Outcome:** Identify which configuration works best for your small dataset.

---

### Experiment 2: Data Scarcity Study

**Goal:** Determine minimum dataset size needed for good performance.

**Fixed:** Best config from Exp 1

**Variables:** Dataset Size

**Runs:**
- Run 5: 50 images
- Run 6: 100 images
- Run 7: 150 images
- Run 8: 200 images
- (Compare against Run X from Exp 1: 231 images)

**Expected Outcome:** Line chart showing mAP vs Dataset Size, demonstrating diminishing returns.

---

### Experiment 3: Robustness Study (Augmentation)

**Goal:** Prove domain-specific augmentations (Disco lights, Occlusion) help performance.

**Fixed:** Best config from Exp 1, Full dataset

**Variables:** Augmentation Pipeline

**Runs:**
- Run 9: No Augmentation (Resize only)
- Run 10: Geometric Only (Flip/Rotate)
- (Compare against Run X from Exp 1: Full Custom Pipeline)

**Expected Outcome:** Table showing your custom augmentations improve metrics.

---

### Experiment 4: Novelty Study (Custom Loss)

**Goal:** Validate your custom loss improves spatial coupling of cups/shots.

**Fixed:** Best config from Exp 1, Full dataset, Full augmentations

**Variables:** Loss Function

**Runs:**
- Run 11: Custom Loss (Spatial Consistency)
- (Compare against Run X from Exp 1: Standard YOLO Loss)

**Expected Outcome:** 
- Reduced "floating shots" (shots without cups)
- Better spatial coupling in confusion matrix
- This is your 10 ECTS contribution!

---

## File Structure

```
custom_training_loop/
├── experiment_orchestrator.py    # Main experiment manager
├── run_experiments.py            # CLI for running/analyzing experiments
├── main.py                       # Training script (updated for experiments)
├── experiments/                  # Generated experiment configs
│   ├── experiment_log.json      # Tracks all experiments
│   ├── exp1_configuration/
│   │   ├── run1_baseline/
│   │   │   └── config.yaml
│   │   ├── run2_conservative/
│   │   └── ...
│   ├── exp2_data_scarcity/
│   ├── exp3_robustness/
│   └── exp4_novelty/
└── runs/                        # Training results
    ├── train_20251129_214412/
    │   ├── results.csv
    │   ├── results.png
    │   ├── BoxF1_curve.png
    │   └── ...
    └── ...
```

## Configuration Options

### Available Settings

```yaml
# Training
lr: 0.0003                    # Learning rate
weight_decay: 0.0005          # Weight decay
epochs: 100                   # Training epochs
batch_size: 8                 # Batch size

# Model
freeze_backbone: false        # Freeze backbone layers
freeze_dfl: false            # Freeze DFL layer
use_ema: false               # Use exponential moving average

# Data
dataset_size: 231            # Number of images to use
augmentation: "full"         # "none", "geometric", or "full"

# Loss
loss_type: "standard"        # "standard" or "spatial_consistency"
```

## Analysis Tools

### View Experiment Summary

```bash
# Analyze all experiments
python run_experiments.py --analyze --experiment all
```

### Manual Analysis

```python
from experiment_orchestrator import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator()

# Access experiment results
exp1_results = orchestrator.experiments["exp1_configuration"]
for run in exp1_results:
    print(f"{run['run_name']}: mAP50 = {run['results']['mAP50']}")
```

## Report Generation Tips

### For Experiment 1
- Create a table comparing all 4 configurations
- Highlight the winning configuration
- Explain WHY it works (e.g., "Freezing necessary for small dataset")

### For Experiment 2
- Generate line plot: Dataset Size vs mAP
- Show diminishing returns curve
- Identify optimal dataset size

### For Experiment 3
- Create comparison table: No Aug | Geometric | Custom
- Show precision, recall, mAP for each
- Prove domain-specific augmentations help

### For Experiment 4
- Compare confusion matrices side-by-side
- Analyze specific failure cases
- Show reduction in "floating shot" errors
- Demonstrate spatial consistency improvements

## Tips

1. **Run experiments sequentially** - Complete Exp 1 before moving to Exp 2
2. **Keep good notes** - Document why each configuration worked/failed
3. **Save all results** - Each run creates a unique `runs/train_*` directory
4. **Use best config** - Always base subsequent experiments on your best Exp 1 config
5. **Visualize everything** - Use the generated plots in your report

## Troubleshooting

**Problem:** Config not found
```bash
# Make sure you generated configs first
python run_experiments.py --generate --experiment exp1
```

**Problem:** Can't set best_config for Exp 2
```python
# You must run and log Exp 1 first
orchestrator.set_best_config("run1_baseline", results_dict)
```

**Problem:** Results not saving
```bash
# Check that runs/ directory exists
mkdir -p runs
```
