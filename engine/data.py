from torch.utils.data import DataLoader
from typing import Any, Callable, Optional
from dataclasses import dataclass
import torch.optim as optim
from typing import Any
import torch
from ultralytics.models.yolo.model import YOLO
from metrics.metrics import Metrics

@dataclass
class TrainerConfig:
    epochs: int
    log_interval: int
    experiment_name: str

type LossFunc = Callable[..., tuple[torch.Tensor, torch.Tensor]]

@dataclass
class TrainerState:
    model: YOLO
    optimizer: optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    loss_fn: LossFunc
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

@dataclass
class LossComponent:
    box: float
    cls: float
    dfl: float
    spatial: Optional[float] = None


@dataclass
class BatchResult:
    train_loss: LossComponent
    val_loss: LossComponent
    metrics: Metrics