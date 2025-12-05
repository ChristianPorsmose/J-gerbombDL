from torch.utils.data import DataLoader
from typing import Callable, Union, TypeAlias, Tuple
from dataclasses import dataclass
import torch.optim as optim
import torch
from ultralytics.models.yolo.model import YOLO
from metrics.metrics import Metrics


LossFunc: TypeAlias = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainerState:
    experiment_name: str
    model: YOLO
    optimizer: optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    loss_fn: LossFunc
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class LossComponent:
    box: float = 0.0
    cls: float = 0.0
    dfl: float = 0.0
    spatial: float = 0.0

    def total(self) -> float:
        return self.box + self.cls + self.dfl + self.spatial

    def __truediv__(self, value: Union[int, float]) -> "LossComponent":
        return LossComponent(
            box=self.box / value,
            cls=self.cls / value,
            dfl=self.dfl / value,
            spatial=self.spatial / value,
        )

    def __add__(self, other: "LossComponent") -> "LossComponent":
        return LossComponent(
            box=self.box + other.box,
            cls=self.cls + other.cls,
            dfl=self.dfl + other.dfl,
            spatial=self.spatial + other.spatial,
        )

    def __iadd__(self, other: "LossComponent") -> "LossComponent":
        self.box += other.box
        self.cls += other.cls
        self.dfl += other.dfl
        self.spatial += other.spatial
        return self


@dataclass
class BatchResult:
    train_loss: LossComponent
    val_loss: LossComponent
    metrics: Metrics
