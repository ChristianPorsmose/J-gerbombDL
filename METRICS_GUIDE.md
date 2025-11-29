# Metrics Tracking System

This system automatically tracks and visualizes your YOLO training metrics.

## Features

✅ **Metrics Tracked:**
- Training losses (box, cls, dfl)
- Validation losses (box, cls, dfl)
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Learning rate schedule for all parameter groups

✅ **Generated Plots:**
- `losses.png` - Training and validation losses over time
- `metrics.png` - Precision, Recall, mAP metrics
- `learning_rate.png` - LR schedule visualization
- `BoxPR_curve.png` - Precision-Recall curve
- `BoxF1_curve.png` - F1 score over training
- `BoxP_curve.png` - Precision over training
- `BoxR_curve.png` - Recall over training
- `confusion_matrix.png` - Confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix

✅ **CSV Export:**
All metrics are saved to `results.csv` in the same format as Ultralytics.

## How to Use

### Option 1: Replace the trainer import

In `main.py`, change:
```python
from jäger_bomb_trainer import JägerBombTrainer
```

To:
```python
from jäger_bomb_trainer_with_metrics import JägerBombTrainer
```

Then add an experiment name when creating the trainer:
```python
trainer = JägerBombTrainer(cfg, save_cfg, experiment_name="my_experiment")
```

### Option 2: Keep both versions

If you want to keep both versions, use:
```python
from jäger_bomb_trainer_with_metrics import JägerBombTrainer as MetricsTrainer

# Use the metrics-enabled version
trainer = MetricsTrainer(cfg, save_cfg, experiment_name="experiment_1")
```

### Option 3: Standalone metrics tracker

You can also use the `MetricsTracker` class directly in your existing trainer:

```python
from metrics_tracker import MetricsTracker

# In your trainer __init__:
self.metrics = MetricsTracker(save_dir="runs", experiment_name="exp1")

# During training:
self.metrics.update(
    epoch=epoch,
    elapsed_time=time_elapsed,
    train_losses={'box': box_loss, 'cls': cls_loss, 'dfl': dfl_loss},
    val_losses={'box': val_box, 'cls': val_cls, 'dfl': val_dfl},
    val_metrics={'precision': p, 'recall': r, 'mAP50': map50, 'mAP50-95': map50_95},
    learning_rates=[lr0, lr1, lr2]
)

# Generate plots
self.metrics.plot_metrics()
```

## Output Structure

```
runs/
└── experiment_name/
    ├── results.csv              # All metrics in CSV format
    ├── best_metrics.json        # Summary of best achieved metrics
    └── plots/
        ├── losses.png
        ├── metrics.png
        ├── learning_rate.png
        ├── BoxF1_curve.png
        ├── BoxP_curve.png
        ├── BoxR_curve.png
        ├── BoxPR_curve.png
        ├── confusion_matrix.png
        └── confusion_matrix_normalized.png
```

## Notes

- Plots are generated every 10 epochs and at the end of training
- The CSV file is updated after each epoch
- Best metrics summary is saved at the end of training
- All plots use high DPI (150) for publication quality

## Customization

You can customize class names for confusion matrix:
```python
# In your trainer, after training:
self.metrics_tracker._plot_confusion_matrix(class_names=["cup", "shot"])
```

## Requirements

Make sure you have these packages installed:
```bash
pip install matplotlib seaborn scikit-learn
```
