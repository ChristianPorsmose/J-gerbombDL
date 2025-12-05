from configs import ExperimentConfig, LossConfig
from dataset.jäger_bomb_dataset import JägerBombDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import Tuple
from engine.jäger_bomb_loss import JägerBombLoss
from ultralytics.models import YOLO
import math
from torchvision.transforms import v2 as T
from ultralytics.utils.loss import v8DetectionLoss
from dataset.letter_box_transform import LetterBoxTransform
from ultralytics.utils.loss import v8DetectionLoss
from engine.data import LossFunc, TrainerConfig, TrainerState
from dataset.yolo_compose import YOLOCompose
from types import SimpleNamespace
from utils.echo import log_success, log_info, log


def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets


class TrainingFactory:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def create_yolo_model(self) -> YOLO:
        if self.cfg.model.pretrained:
            log_info("Loading pretrained YOLOv11n model weights")
            return YOLO(self.cfg.model.type, task="detect").load("yolo11n.pt")
        return YOLO(self.cfg.model.type, task="detect")

    def _light_augmentation(self) -> list:
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True),
        ]

    def _geometric_augmentation(self) -> list:
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ToDtype(torch.float32, scale=True),
        ]

    def _no_augmentation(self) -> list:
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ToDtype(torch.float32, scale=True),
        ]

    def _full_augmentation(self) -> list:
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True),
        ]

    def _final_augmentation(self) -> list:
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True),
        ]

    def _create_transform_list(self, augmentation_mode: str) -> list:
        augmentation_dict = {
            "light": self._light_augmentation,
            "geometric": self._geometric_augmentation,
            "none": self._no_augmentation,
            "full": self._full_augmentation,
            "final": self._final_augmentation,
        }
        if augmentation_mode not in augmentation_dict:
            raise ValueError(f"Unsupported augmentation mode: {augmentation_mode}")
        return augmentation_dict[augmentation_mode]()

    def _create_datasets(
        self, train_transforms: list, val_transforms: list
    ) -> Tuple[JägerBombDataset, JägerBombDataset, JägerBombDataset]:
        train_ds = JägerBombDataset(
            self.cfg.paths.train, transforms=YOLOCompose(train_transforms)
        )
        val_ds = JägerBombDataset(
            self.cfg.paths.val, transforms=YOLOCompose(val_transforms)
        )
        test_ds = JägerBombDataset(
            self.cfg.paths.test, transforms=YOLOCompose(val_transforms)
        )
        return train_ds, val_ds, test_ds

    def _create_dataloaders(
        self,
        train_ds: JägerBombDataset,
        val_ds: JägerBombDataset,
        test_ds: JägerBombDataset,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        generator = torch.Generator(torch.get_default_device().type)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            generator=generator,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            generator=generator,
        )
        return train_dl, val_dl, test_dl

    def _create_optimizer(self, params: list) -> torch.optim.Optimizer:
        optimizer_type = self.cfg.optimizer.type.upper()
        momentum = self.cfg.optimizer.momentum
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.weight_decay

        if optimizer_type == "SGD":
            use_nesterov = momentum > 0
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                nesterov=use_nesterov,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(
                params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        log_success(
            f"Optimizer created: {optimizer_type} | lr={lr}, weight_decay={weight_decay}, momentum={momentum}"
        )
        return optimizer

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LambdaLR:
        epochs = self.cfg.training.epochs
        lr = self.cfg.optimizer.lr
        if self.cfg.optimizer.lr_scheduler == "cosine":

            def one_cycle_lr(epoch):
                """Cosine learning rate schedule from 1.0 to lrf over epochs."""
                lrf = 0.01  # final learning rate factor (1% of initial)
                # Cosine annealing: starts at 1.0, ends at lrf
                return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / epochs))

            log(
                f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {epochs} epochs"
            )
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)

        log(f"Learning rate: fixed at {lr:.6f}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    def _create_loss_func(
        self, torch_model: torch.nn.Module, loss_type: LossConfig
    ) -> LossFunc:
        torch_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        if loss_type.type == "spatial":
            log_info("Using spatial consistency loss (JägerBombLoss)")
            return JägerBombLoss(torch_model, lamda_rate=loss_type.lambda_rate)
        log_info("Using standard YOLO loss (v8DetectionLoss)")
        return v8DetectionLoss(torch_model)

    def create(self, model: YOLO, params: list) -> Tuple[TrainerConfig, TrainerState]:
        torch_model = model.model
        train_transforms = self._create_transform_list(self.cfg.augmentation)
        val_transforms = self._create_transform_list("none")
        train_ds, val_ds, test_ds = self._create_datasets(
            train_transforms, val_transforms
        )
        train_dl, val_dl, test_dl = self._create_dataloaders(train_ds, val_ds, test_ds)
        optimizer = self._create_optimizer(params)
        scheduler = self._create_scheduler(optimizer)
        loss_fn = self._create_loss_func(torch_model, self.cfg.loss_type)

        trainer_cfg = TrainerConfig(
            epochs=self.cfg.training.epochs,
            log_interval=self.cfg.training.log_interval,
            experiment_name=self.cfg.experiment_name,
            early_stop_count=self.cfg.training.early_stop_count,
        )

        trainer_state = TrainerState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_loader=train_dl,
            val_loader=val_dl,
            test_loader=test_dl,
        )

        return trainer_cfg, trainer_state
