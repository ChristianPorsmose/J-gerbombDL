from configs import ExperimentConfig
from dataset.jäger_bomb_dataset import JägerBombDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Tuple
import click
from engine.jäger_bomb_loss import JägerBombLoss
from ultralytics.models import YOLO
import math
from torchvision.transforms import v2 as T
from ultralytics.utils.loss import v8DetectionLoss
from dataset.letter_box_transform import LetterBoxTransform
from ultralytics.utils.loss import v8DetectionLoss
from engine.data import TrainerConfig, TrainerState
from dataset.yolo_compose import YOLOCompose
from types import SimpleNamespace

def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets

class TrainingFactory:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def _create_yolo_model(self) -> YOLO:
        model = YOLO( self.cfg.model, task="detect").load('yolo11n.pt')
        model.model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)
        return model

    def create_transform_list(self, augmentation_mode):
        if augmentation_mode == "none":
            return [
                LetterBoxTransform(new_shape=(640, 640)),
                T.ToDtype(torch.float32, scale=True)
            ]
        if augmentation_mode == "geometric":
            return [
                LetterBoxTransform(new_shape=(640, 640)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
                T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
                T.ToDtype(torch.float32, scale=True)
            ]
        if augmentation_mode == "light":
            return [
                LetterBoxTransform(new_shape=(640, 640)),
                T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
                T.ToDtype(torch.float32, scale=True)
            ]
        return [
                LetterBoxTransform(new_shape=(640, 640)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomPerspective(distortion_scale=0.2, p=0.5,fill=114),
                T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
                T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
                T.ToDtype(torch.float32, scale=True)
            ]

    def _create_datasets(self, train_transforms, val_transforms) -> Tuple[JägerBombDataset, JägerBombDataset, JägerBombDataset]:
        train_ds = JägerBombDataset(self.cfg.paths.train, transforms=YOLOCompose(train_transforms))
        val_ds   = JägerBombDataset(self.cfg.paths.val, transforms=YOLOCompose(val_transforms))
        test_ds  = JägerBombDataset(self.cfg.paths.test, transforms=YOLOCompose(val_transforms))
        return train_ds, val_ds, test_ds

    def _create_dataloaders(self, train_ds, val_ds, test_ds) -> Tuple[DataLoader, DataLoader, DataLoader]:
        generator = torch.Generator(torch.get_default_device().type)
        train_dl = DataLoader(train_ds, batch_size=self.cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn, generator=generator)
        val_dl = DataLoader(val_ds, batch_size=self.cfg.training.batch_size,
                            shuffle=False, collate_fn=collate_fn, generator=generator)
        test_dl = DataLoader(test_ds, batch_size=self.cfg.training.batch_size,
                             shuffle=False, collate_fn=collate_fn, generator=generator)
        return train_dl, val_dl, test_dl

    def _create_optimizer_param_groups(self, torch_model) -> list:
        g = [], [], []  # parameter groups: [weights with decay, weights without decay, biases]
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers
        
        for module_name, module in torch_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # batch norm weights (no decay)
                    g[1].append(param)
                else:  # regular weights (with decay)
                    g[0].append(param)
        return g

    def _create_optimizer(self, torch_model) -> torch.optim.Optimizer:
        weights, bn_no_decay, biases = self._create_optimizer_param_groups(torch_model)
        optimizer_type = self.cfg.optimizer.type.upper()
        momentum = self.cfg.optimizer.momentum
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.weight_decay

        if optimizer_type == "SGD":
            use_nesterov = momentum > 0
            optimizer = torch.optim.SGD(biases, lr=lr, momentum=momentum, nesterov=use_nesterov, weight_decay=weight_decay)
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(biases, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        optimizer.add_param_group({"params": weights, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": bn_no_decay, "weight_decay": 0.0})

        click.secho(f"[SUCCESS] Optimizer created: {optimizer_type} | lr={lr}, weight_decay={weight_decay}, momentum={momentum}", fg="green")
        click.echo(f"Parameter groups: {len(weights)} weights(decay), {len(bn_no_decay)} batchnorm(no decay), {len(biases)} biases(no decay)")

        return optimizer

    def create_scheduler(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        epochs = self.cfg.training.epochs
        lr = self.cfg.optimizer.lr
        if self.cfg.optimizer.lr_scheduler == "cosine":
            def one_cycle_lr(epoch):
                """Cosine learning rate schedule from 1.0 to lrf over epochs."""
                lrf = 0.01  # final learning rate factor (1% of initial)
                # Cosine annealing: starts at 1.0, ends at lrf
                return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / epochs))

            click.echo(f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {epochs} epochs")
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)

        click.echo(f"Learning rate: fixed at {lr:.6f}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    def create_loss_func(self, torch_model, loss_type):
        if loss_type == "spatial_consistency":
            click.secho("[INFO] Using spatial consistency loss (JägerBombLoss)", fg="blue")
            return JägerBombLoss(torch_model, lamda_rate=1)
        click.secho("[INFO] Using standard YOLO loss (v8DetectionLoss)", fg="blue")
        return v8DetectionLoss(torch_model)

    def create(self) -> Tuple[TrainerConfig, TrainerState]:
        model = self._create_yolo_model()
        torch_model = model.model
        train_transforms = self.create_transform_list(self.cfg.augmentation)
        val_transforms = self.create_transform_list("none")
        train_ds, val_ds, test_ds = self._create_datasets(train_transforms, val_transforms)
        train_dl, val_dl, test_dl = self._create_dataloaders(train_ds, val_ds, test_ds)
        optimizer = self._create_optimizer(torch_model)
        scheduler = self.create_scheduler(optimizer)
        loss_fn = self.create_loss_func(torch_model, self.cfg.loss_type)

        trainer_cfg = TrainerConfig(
            epochs=self.cfg.training.epochs,
            log_interval=self.cfg.training.log_interval,
            use_ema=self.cfg.use_ema,
            experiment_name=self.cfg.experiment_name,
        )

        trainer_state = TrainerState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_loader=train_dl,
            val_loader=val_dl,
            test_loader=test_dl
        )

        return trainer_cfg, trainer_state