from torch.utils.data import DataLoader
from typing import Any
from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Any

from ultralytics.models.yolo.model import YOLO

@dataclass
class PathsConfig:
    train: str
    val: str
    test: str

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    log_interval: int
    save_interval: int
    save_path: str

@dataclass
class OptimizerConfig:
    type: str
    lr: float
    lr_scheduler: str
    weight_decay: float
    momentum: float

@dataclass
class FreezeConfig:
    backbone_layers: int
    dfl: bool

@dataclass
class ExperimentConfig:
    experiment_name: str
    model: str
    paths: PathsConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    freeze: FreezeConfig
    augmentation: str
    dataset_size: float
    use_ema: bool
    loss_type: str

@dataclass
class TrainerConfig:
    epochs: int
    log_interval: int
    use_ema: bool
    experiment_name: str
    loss_type: str
    device: str

@dataclass
class TrainerState:
    model: YOLO
    optimizer: optim.Optimizer
    scheduler: Any
    loss_fn: Any

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
