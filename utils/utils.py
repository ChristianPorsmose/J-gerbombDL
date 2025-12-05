import torch
from contextlib import contextmanager

import yaml

from configs import (
    ExperimentConfig,
    FreezeConfig,
    OptimizerConfig,
    PathsConfig,
    TrainingConfig,
    ModelConfig,
    LossConfig,
)
import random
import tempfile
import os


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return ExperimentConfig(
        experiment_name=cfg_dict["experiment_name"],
        model=ModelConfig(**cfg_dict["model"]),
        paths=PathsConfig(**cfg_dict["paths"]),
        training=TrainingConfig(**cfg_dict["training"]),
        optimizer=OptimizerConfig(**cfg_dict["optimizer"]),
        freeze=FreezeConfig(**cfg_dict["freeze"]),
        augmentation=cfg_dict["augmentation"],
        dataset_size=cfg_dict["dataset_size"],
        loss_type=LossConfig(**cfg_dict["loss"]),
    )


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


@contextmanager
def create_experiment_train_path(cfg: ExperimentConfig):
    with open(cfg.paths.train, "r") as f:
        all_train_paths = [line.strip() for line in f.readlines()]

    desired_size = int(cfg.dataset_size * len(all_train_paths))
    temp_file_path = None

    if desired_size < len(all_train_paths):
        random.seed(42)
        all_train_paths_shuffled = all_train_paths.copy()
        random.shuffle(all_train_paths_shuffled)
        sampled_paths = all_train_paths_shuffled[:desired_size]

        orig_dir = os.path.dirname(cfg.paths.train)
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", dir=orig_dir
        )
        for path in sampled_paths:
            temp_file.write(path + "\n")
        temp_file.close()
        temp_file_path = temp_file.name
        train_data_path_to_use = temp_file_path
    else:
        train_data_path_to_use = cfg.paths.train

    try:
        yield train_data_path_to_use
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
