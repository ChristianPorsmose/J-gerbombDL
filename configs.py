from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    type: str
    pretrained: bool


@dataclass
class LossConfig:
    type: str
    lambda_rate: Optional[float] = None


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
    early_stop_count: int


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
    model: ModelConfig
    paths: PathsConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    freeze: FreezeConfig
    augmentation: str
    dataset_size: float
    loss_type: LossConfig
