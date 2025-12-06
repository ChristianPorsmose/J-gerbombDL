from pathlib import Path

from analysis.colors_and_labels import COLORS_1B, EXPERIMENT_1B_LABELS

EXPERIMENTS_DIR = Path("experiments_results_temp")

PHASE1A_EXPERIMENTS = [
    "phase1a_sgd_standard",
    "phase1a_sgd_conservative",
    "phase1a_adamw_standard",
    "phase1a_adamw_conservative",
    "phase1a_adamw_aggressive",
]


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


PHASE_NAME = "1B"
PHASE_EXPERIMENTS = PHASE1B_EXPERIMENTS
EXPERIMENT_LABELS = EXPERIMENT_1B_LABELS
COLORS = COLORS_1B

OUTPUT_DIR = Path("TEMP_DIR")
OUTPUT_DIR.mkdir(exist_ok=True)