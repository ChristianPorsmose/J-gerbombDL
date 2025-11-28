from torch.utils.data import DataLoader
from typing import Callable, Any
from torch.optim import Optimizer
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class TrainingConfig:
    model: nn.Module
    optimizer: Optimizer
    scheduler: Any
    train_dataloader : DataLoader
    val_dataloader : DataLoader
    loss_fn : Callable
    device : str
    epochs : int
    log_interval : int
    yolo_model : Any

@dataclass
class SaveConfig:
    save_epoch_interval : int
    save_path : str