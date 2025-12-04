from dataclasses import fields, is_dataclass
import numpy as np
from pathlib import Path
import torch

import yaml

from configs import (
    ExperimentConfig, 
    FreezeConfig, 
    OptimizerConfig,
    PathsConfig, 
    TrainingConfig,
    ModelConfig,
    LossConfig
)
import random
import tempfile
import os
import click

def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_name = cfg_dict["experiment_name"],
        model = ModelConfig(**cfg_dict["model"]),
        paths = PathsConfig(**cfg_dict["paths"]),
        training = TrainingConfig(**cfg_dict["training"]),
        optimizer = OptimizerConfig(**cfg_dict["optimizer"]),
        freeze = FreezeConfig(**cfg_dict["freeze"]),
        augmentation = cfg_dict["augmentation"],
        dataset_size = cfg_dict["dataset_size"],
        loss_type = LossConfig(**cfg_dict["loss"])
    )

def convert_to_python_types(obj):
    """Recursively convert numpy/torch types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    else:
        return obj

def xywh_to_xyxy(bboxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Convert bboxes from normalized xywh to pixel xyxy format."""
    if bboxes.numel() == 0:
        return bboxes
    
    # bboxes: [N, 4] in format [x_center, y_center, width, height] normalized
    x_center = bboxes[:, 0] * img_w
    y_center = bboxes[:, 1] * img_h
    width = bboxes[:, 2] * img_w
    height = bboxes[:, 3] * img_h
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def create_experiment_train_path(params, train_data_path):
    with open(train_data_path, 'r') as f:
        all_train_paths = [line.strip() for line in f.readlines()]
        
    desired_size = int(params["dataset_size"]*len(all_train_paths))
        
    if desired_size < len(all_train_paths):
            # Create NESTED subset: shuffle once with fixed seed, then take first N
            # This ensures dataset_size=20 ⊂ dataset_size=40 ⊂ dataset_size=60, etc.
        random.seed(42)  # Fixed seed for reproducibility across all experiments
        all_train_paths_shuffled = all_train_paths.copy()
        random.shuffle(all_train_paths_shuffled)
            
            # Take first N paths (ensures nesting property)
        sampled_paths = all_train_paths_shuffled[:desired_size]
            
            # Create temporary file in the SAME DIRECTORY as original file
            # This is critical because the dataset prepends "../" and looks for labels/ relative to the file location
        orig_dir = os.path.dirname(train_data_path)
        temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                delete=False, 
                suffix='.txt',
                dir=orig_dir  # Create temp file in same directory as original
            )
        for path in sampled_paths:
            temp_file.write(path + '\n')
        temp_file.close()
        train_data_path = temp_file.name
            
        click.secho(
            f"[INFO] Dataset size limiting: Using {desired_size}/{len(all_train_paths)} training images (NESTED subset)", 
            fg="blue"
        )
        click.secho(f"[INFO] Temporary file created: {train_data_path}", fg="blue")
    else:
        click.secho(f"[INFO] Dataset size: Using all {len(all_train_paths)} training images", fg="blue")
    return train_data_path