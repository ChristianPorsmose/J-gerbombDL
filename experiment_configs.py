"""
Strategic Experiment Configurations for Jäger Bomb Detection
Each phase tests specific hypotheses with isolated variables.

USAGE:
------
Option 1 (Programmatic):
    from experiment_configs import PHASE1A_SGD_STANDARD
    import yaml
    with open('setup.yaml', 'w') as f:
        yaml.dump(PHASE1A_SGD_STANDARD, f)
    # Then run: python main.py

Option 2 (Direct import in main.py):
    from experiment_configs import PHASE1A_SGD_STANDARD
    # Use PHASE1A_SGD_STANDARD directly instead of loading setup.yaml

Option 3 (Manual copy):
    Copy the config dict into setup.yaml and run: python main.py
"""

# ========== PHASE 1A: Optimizer Selection (5 runs) ==========
# Goal: Determine which optimizer + parameters work best for small dataset
# Variable: Optimizer type, learning rate, weight decay
# Fixed: Unfrozen backbone, full augmentation, all data
# Analysis: Compare convergence, stability, final mAP, overfitting

PHASE1A_SGD_STANDARD = {
    "experiment_name": "phase1a_sgd_standard",
    "model": "custom_yolo.yaml",
    "train_data_path": "/home/dl01e25/dataset_final_boxes_yolo/train.txt",
    "val_data_path": "/home/dl01e25/dataset_final_boxes_yolo/val.txt",
    "batch_size": 8,
    "epochs": 100,
    "log_interval": 10,
    "save_interval": 30,
    "save_path": "weights",
    "optimizer": "SGD",
    "lr": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "freeze_backbone": False,
    "freeze_dfl": False,
    "use_ema": False,
    "augmentation": "full",
}

PHASE1A_SGD_CONSERVATIVE = {
    "experiment_name": "phase1a_sgd_conservative",
    "model": "custom_yolo.yaml",
    "train_data_path": "/home/dl01e25/dataset_final_boxes_yolo/train.txt",
    "val_data_path": "/home/dl01e25/dataset_final_boxes_yolo/val.txt",
    "batch_size": 8,
    "epochs": 100,
    "log_interval": 10,
    "save_interval": 30,
    "save_path": "weights",
    "optimizer": "SGD",
    "lr": 0.001,  # 10x lower
    "momentum": 0.937,
    "weight_decay": 0.001,  # 2x higher for regularization
    "freeze_backbone": False,
    "freeze_dfl": False,
    "use_ema": False,
    "augmentation": "full",
}

PHASE1A_ADAMW_STANDARD = {
    "experiment_name": "phase1a_adamw_standard",
    "model": "custom_yolo.yaml",
    "train_data_path": "/home/dl01e25/dataset_final_boxes_yolo/train.txt",
    "val_data_path": "/home/dl01e25/dataset_final_boxes_yolo/val.txt",
    "batch_size": 8,
    "epochs": 100,
    "log_interval": 10,
    "save_interval": 30,
    "save_path": "weights",
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0005,
    "freeze_backbone": False,
    "freeze_dfl": False,
    "use_ema": False,
    "augmentation": "full",
}

PHASE1A_ADAMW_CONSERVATIVE = {
    "experiment_name": "phase1a_adamw_conservative",
    "model": "custom_yolo.yaml",
    "train_data_path": "/home/dl01e25/dataset_final_boxes_yolo/train.txt",
    "val_data_path": "/home/dl01e25/dataset_final_boxes_yolo/val.txt",
    "batch_size": 8,
    "epochs": 100,
    "log_interval": 10,
    "save_interval": 30,
    "save_path": "weights",
    "optimizer": "AdamW",
    "lr": 0.0001,  # 10x lower
    "weight_decay": 0.001,  # 2x higher for regularization
    "freeze_backbone": False,
    "freeze_dfl": False,
    "use_ema": False,
    "augmentation": "full",
}

PHASE1A_ADAMW_AGGRESSIVE = {
    "experiment_name": "phase1a_adamw_aggressive",
    "model": "custom_yolo.yaml",
    "train_data_path": "/home/dl01e25/dataset_final_boxes_yolo/train.txt",
    "val_data_path": "/home/dl01e25/dataset_final_boxes_yolo/val.txt",
    "batch_size": 8,
    "epochs": 100,
    "log_interval": 10,
    "save_interval": 30,
    "save_path": "weights",
    "optimizer": "AdamW",
    "lr": 0.003,  # 3x higher
    "weight_decay": 0.0003,  # Lower decay for faster learning
    "freeze_backbone": False,
    "freeze_dfl": False,
    "use_ema": False,
    "augmentation": "full",
}

# ========== PHASE 1B: Transfer Learning Strategy (2-3 runs) ==========
# Goal: Test if freezing backbone helps with small dataset
# Variable: Backbone freezing, learning rate
# Fixed: BEST optimizer from Phase 1A, full augmentation, all data
# Analysis: Compare mAP, train-val gap (overfitting), convergence speed

PHASE1B_FROZEN_STANDARD = {
    # TODO: Copy the BEST optimizer config from Phase 1A here
    # Then modify:
    "freeze_backbone": True,
    # Keep the winning LR from Phase 1A
}

PHASE1B_FROZEN_LOWER_LR = {
    # TODO: Copy the BEST optimizer config from Phase 1A here
    # Then modify:
    "freeze_backbone": True,
    # "lr": <divide Phase 1A winner LR by 3>
}

# ========== PHASE 2: Data Efficiency (5 runs) ==========
# Goal: Determine minimum viable dataset size
# Variable: Dataset size
# Fixed: BEST config from Phase 1A+1B
# Analysis: Plot mAP vs dataset size, find diminishing returns point

PHASE2_50_IMAGES = {
    # TODO: Copy BEST config from Phase 1
    # Then add:
    "dataset_size": 50,
}

PHASE2_100_IMAGES = {
    # TODO: Copy BEST config from Phase 1
    "dataset_size": 100,
}

PHASE2_150_IMAGES = {
    # TODO: Copy BEST config from Phase 1
    "dataset_size": 150,
}

PHASE2_200_IMAGES = {
    # TODO: Copy BEST config from Phase 1
    "dataset_size": 200,
}

PHASE2_231_IMAGES = {
    # TODO: Copy BEST config from Phase 1
    "dataset_size": 231,  # Full dataset
}

# ========== PHASE 3: Augmentation Validation (3 runs) ==========
# Goal: Prove domain-specific augmentations help
# Variable: Augmentation strategy
# Fixed: BEST config from Phase 1, full dataset
# Analysis: Compare mAP, per-class metrics, failure cases

PHASE3_NO_AUGMENTATION = {
    # TODO: Copy BEST config from Phase 1
    "augmentation": "none",  # Resize only
}

PHASE3_GENERIC_AUGMENTATION = {
    # TODO: Copy BEST config from Phase 1
    "augmentation": "geometric",  # Flip + Rotate (standard CV)
}

PHASE3_DOMAIN_AUGMENTATION = {
    # TODO: Copy BEST config from Phase 1
    "augmentation": "full",  # Your custom disco/occlusion pipeline
}

# ========== PHASE 4: Novel Contribution - Spatial Consistency Loss (2 runs) ==========
# Goal: Validate your 10 ECTS contribution
# Variable: Loss function
# Fixed: BEST config from Phases 1-3
# Analysis: Confusion matrix, "floating shots" count, spatial coupling metrics

PHASE4_STANDARD_LOSS = {
    # TODO: Copy BEST config from Phase 1-3
    "loss_type": "standard",  # YOLO loss (baseline)
}

PHASE4_SPATIAL_LOSS = {
    # TODO: Copy BEST config from Phase 1-3
    "loss_type": "spatial_consistency",  # Your novel loss
}

# ========== USAGE GUIDE ==========

"""
PHASE 1A: Optimizer Selection
------------------------------
Run all 5 configs:
1. PHASE1A_SGD_STANDARD
2. PHASE1A_SGD_CONSERVATIVE
3. PHASE1A_ADAMW_STANDARD
4. PHASE1A_ADAMW_CONSERVATIVE
5. PHASE1A_ADAMW_AGGRESSIVE

Analysis:
- Compare final mAP@0.5 and mAP@0.5:0.95
- Check training curves (runs/train_*/results.png)
- Look at train-val loss gap (overfitting indicator)
- Note convergence speed (which reaches best mAP first?)

Example insights:
- "SGD requires higher LR but is more stable"
- "AdamW converges faster but may overfit with high LR"
- "Conservative settings reduce overfitting but slower convergence"

Pick the winner (e.g., PHASE1A_ADAMW_STANDARD)


PHASE 1B: Transfer Learning
----------------------------
Copy the winning optimizer config, then test freezing:

Run 2-3 configs:
1. PHASE1B_FROZEN_STANDARD (with winning LR)
2. PHASE1B_FROZEN_LOWER_LR (if #1 unstable)

Analysis:
- Does freezing reduce overfitting? (check train-val gap)
- Does freezing hurt or help mAP?
- Per-class performance (does it help small objects?)

Example insights:
- "Freezing improved mAP by 3% (0.72 → 0.75)"
- "Train-val gap reduced from 15% to 8%, indicating less overfitting"
- "With only 231 images, transfer learning is essential"

Lock in your BEST overall training config.


PHASE 2: Data Efficiency
-------------------------
Use BEST config from Phase 1, vary dataset size.

Run 5 configs with different dataset_size values.

Analysis:
- Create line plot: X-axis = dataset size, Y-axis = mAP
- Identify the "knee" of the curve
- Calculate marginal gains: mAP improvement per 50 images

Example insights:
- "Performance plateaus at ~150 images (mAP 0.73)"
- "50→100 images: +15% mAP, but 150→200: only +2%"
- "Diminishing returns suggest 150 images sufficient for production"


PHASE 3: Augmentation
----------------------
Use BEST config from Phase 1, full dataset.

Run 3 configs with different augmentation modes.

Analysis:
- Compare precision/recall/mAP
- Check per-class improvements
- Count failure cases (false positives in extreme lighting)

Example insights:
- "Generic augmentation: mAP 0.70"
- "Domain-specific augmentation: mAP 0.75 (+7%)"
- "ColorJitter reduced false positives in disco lighting by 23%"
- "Domain knowledge matters: simulating real-world conditions helps"


PHASE 4: Novel Loss
--------------------
Use BEST config from Phases 1-3.

Run 2 configs: standard vs spatial_consistency loss.

CRITICAL: Don't just compare mAP!

Deep Analysis:
1. Confusion matrices side-by-side
2. Count "floating shots" (shots without nearby cups)
3. Calculate average shot-cup distance in predictions
4. Show qualitative examples on hard cases

Example insights:
- "Standard loss: 15 floating shot errors"
- "Spatial loss: 3 floating shot errors (-80%)"
- "Average shot-cup distance: 45px → 12px"
- "Confusion matrix shows 18% fewer shot→background errors"
- "Physical constraint enforcement improves logical consistency"


TOTAL RUNS: 5 + 2 + 5 + 3 + 2 = 17 experiments
Each experiment generates:
- results.csv (all metrics)
- results.png (training curves)
- Confusion matrix
- PR curves, F1 curves
"""

# ========== QUICK REFERENCE ==========

OPTIMIZER_NOTES = {
    "SGD": {
        "pros": "Stable, generalizes well, proven for YOLO",
        "cons": "Requires careful LR tuning, slower convergence",
        "lr_range": "0.001 - 0.01",
    },
    "AdamW": {
        "pros": "Fast convergence, adaptive per-parameter LR, good for small data",
        "cons": "Can overfit, requires weight decay tuning",
        "lr_range": "0.0001 - 0.003",
    }
}

AUGMENTATION_MODES = {
    "none": "Resize only - baseline to show augmentation impact",
    "geometric": "Flip + Rotate - standard CV augmentations",
    "full": "ColorJitter + Blur + Geometric - domain-specific disco/club simulation"
}

KEY_METRICS = {
    "mAP@0.5": "Primary metric - detection accuracy at 50% IoU",
    "mAP@0.5:0.95": "Strict metric - average across IoU thresholds",
    "Precision": "False positive rate - important for deployment",
    "Recall": "False negative rate - are we missing detections?",
    "Train-val gap": "Overfitting indicator - should be <10-15%"
}
