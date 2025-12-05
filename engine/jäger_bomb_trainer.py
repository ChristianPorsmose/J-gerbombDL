from dataclasses import asdict
import json
import time
import torch
import numpy as np
from pathlib import Path
from torch import nn
from datetime import datetime
from metrics.jäger_bomb_metric_tracker import JägerBombMetricTracker
from metrics.jäger_bomb_metric_logger import JägerBombMetricLogger
from metrics.metric_visualization import plot_all_metrics
from engine.bomb_visualize import visualize_batch, visualize_predictions
from engine.data import BatchResult, LossComponent, TrainerState
from configs import TrainingConfig
from engine.log_helpers import log_loss
from ultralytics.models.yolo.model import YOLO
from utils.echo import log_warning, log_info, log_success, log
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader


class JägerBombTrainer:
    def __init__(self, cfg: TrainingConfig, state: TrainerState):
        self.cfg = cfg
        self.torch_model: nn.Module = state.model.model
        self.state = state
        self.device = torch.get_default_device().type
        self._init_grad_scaler()
        self._init_metrics()

    def _init_grad_scaler(self):
        enable = self.device.startswith("cuda")
        self.scaler = GradScaler(self.device, enabled=enable)

    def _init_metrics(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.state.experiment_name
        save_dir = (
            Path("experiments_results")
            / experiment_name
            / "runs"
            / f"train_{timestamp}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_tracker = JägerBombMetricTracker(
            self.state.model.names, save_dir=Path(save_dir)
        )
        self.metrics_logger = JägerBombMetricLogger(file_path=save_dir / "metrics.csv")

    def test_best_model(self):
        best_model_path = self.metric_tracker.save_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            log_warning(
                f"best.pt not found at {best_model_path}, skipping test evaluation"
            )
            return
        log_info(f"Loading best model from {best_model_path}...")
        test_model = YOLO(str(best_model_path))
        test_model.model.eval()
        test_model.model.to(self.device)

        loss = self._evaluate(None, self.state.test_loader, "TEST RESULTS")
        results_path = self.metric_tracker.save_dir / "test_results.json"

        # FIX ME: THIS IS TOO HACKY, LOSSCOMPONENT SHOULD ALWAYS BE FLOATS
        loss_dict = {k: float(v) for k, v in asdict(loss).items()}
        with open(results_path, "w") as f:
            json.dump(loss_dict, f, indent=2)

    @torch.no_grad()
    def _evaluate(
        self, epoch: int, loader: DataLoader, header: str = "VALIDATION"
    ) -> LossComponent:
        """evaluate model and compute metrics."""
        self.torch_model.eval()

        val_loss = LossComponent()

        self.metric_tracker.reset()

        for X_val, y_val in loader:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

            batch = self._prepare_batch_dict(X_val, y_val)

            pred_val = self.torch_model(X_val)

            _, last_loss = self.state.loss_fn(pred_val, batch)
            loss_values = last_loss.cpu().numpy().round(3)

            new_loss = self._extract_loss_component(loss_values)

            val_loss += new_loss

            img_h, img_w = X_val.shape[2], X_val.shape[3]

            metrics_batch = self.metric_tracker.prepare_batch(y_val, img_h, img_w)

            self.metric_tracker.update(pred_val, metrics_batch)

        average_loss = val_loss / len(loader)

        log_loss(epoch, average_loss, header)

        return average_loss

    def _save_model(self, epoch: int, is_best=False):
        """Save model checkpoint."""
        save_dir = Path(self.metric_tracker.save_dir) / "weights"
        save_dir.mkdir(parents=True, exist_ok=True)

        last_path = save_dir / "last.pt"
        self.state.model.save(str(last_path))

        # Save best.pt (only when validation improves)
        if is_best:
            best_path = save_dir / "best.pt"
            self.state.model.save(str(best_path))
            log_success(f"Model saved at epoch {epoch} → {best_path}")
        else:
            log_info(f"Checkpoint saved → {last_path}")

    def _save_training_time(self, start: float, end: float):
        total_training_time = end - start
        time_per_epoch = total_training_time / self.cfg.epochs
        timing_info = {
            "total_training_time_seconds": total_training_time,
            "total_training_time_minutes": total_training_time / 60,
            "total_training_time_hours": total_training_time / 3600,
            "time_per_epoch_seconds": time_per_epoch,
            "time_per_epoch_minutes": time_per_epoch / 60,
            "num_epochs": self.cfg.epochs,
        }

        timing_path = Path(self.metric_tracker.save_dir) / "timing.json"
        with open(timing_path, "w") as f:
            json.dump(timing_info, f, indent=2)

        log_info(f"Training Time Summary:")
        log(
            f"  Total: {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes)"
        )
        log(
            f"  Per Epoch: {time_per_epoch/60:.2f} minutes ({time_per_epoch:.2f} seconds)"
        )
        log(f"  Saved to: {timing_path}")

    def train(self):
        self.torch_model.to(self.device)
        best_loss = np.inf
        early_stop_counter = 0
        early_stop_limit = (
            999999 if self.cfg.early_stop_count == -1 else self.cfg.early_stop_count
        )
        WARMUP_EPOCHS = 3.0

        train_loader_len = len(self.state.train_loader)
        nr_warmup_iterations = max(round(WARMUP_EPOCHS * train_loader_len), 100)

        epoch_train_losses = LossComponent()

        training_start_time = time.time()

        for epoch in range(self.cfg.epochs):
            self.torch_model.train(True)

            epoch_train_losses = LossComponent()
            batch_count = 0

            # Track first batch for visualization
            first_batch_saved = False

            batch_count += self.train_one_epoch(
                train_loader_len,
                nr_warmup_iterations,
                epoch_train_losses,
                epoch,
                first_batch_saved,
            )

            average_train_loss = epoch_train_losses / batch_count

            val_losses = self._evaluate(epoch, self.state.val_loader)
            # Compute detection metrics and generate plots every N epochs or at end
            generate_plots = epoch == self.cfg.epochs - 1

            if generate_plots:
                visualize_predictions(
                    epoch,
                    self.torch_model,
                    self.state.val_loader,
                    self.metric_tracker.save_dir,
                )

            det_metrics = self.metric_tracker.compute(plot=generate_plots)

            if epoch >= WARMUP_EPOCHS:
                self.state.scheduler.step()

            current_lr = self.state.optimizer.param_groups[0]["lr"]
            log(f"Learning rate: {current_lr:.6f}")

            self.metrics_logger.log_batch_result(
                batchResult=BatchResult(
                    train_loss=average_train_loss,
                    val_loss=val_losses,
                    metrics=det_metrics,
                ),
                epoch=epoch,
                lr=current_lr,
            )

            self._save(best_loss, epoch, val_losses, early_stop_counter)

            if early_stop_counter >= early_stop_limit:
                log_info(f"Early stopping at epoch {epoch}")
                break
        training_end_time = time.time()

        self._save_training_time(training_start_time, training_end_time)

        log_info("[INFO] Generating final metrics and plots...")

        plot_all_metrics(
            confusion_matrix=self.metric_tracker.confusion_matrix,
            base_dir=Path(self.metric_tracker.save_dir),
        )

        log_success("Training complete!")

    def _save(
        self,
        best_loss: int,
        epoch: int,
        val_losses: LossComponent,
        early_stop_counter: int,
    ):
        curr_val_loss = val_losses.total()
        if curr_val_loss < best_loss:
            self._save_model(epoch, is_best=True)
            best_loss = curr_val_loss
            early_stop_counter = 0
        else:
            self._save_model(epoch, is_best=False)
            early_stop_counter += 1

    def train_one_epoch(
        self,
        train_loader_len: int,
        nr_warmup_iterations: int,
        epoch_train_losses: LossComponent,
        epoch: int,
        first_batch_saved: bool,
    ):
        batch_count = 0
        for batch_idx, (X, y) in enumerate(self.state.train_loader):

            # Warmup learning rate for first few epochs
            nr_integrated_batches = batch_idx + train_loader_len * epoch
            if nr_integrated_batches <= nr_warmup_iterations:
                warmup_iteration_range = [0, nr_warmup_iterations]

                # gradually increase LR from 0.1 to target
                for j, x in enumerate(self.state.optimizer.param_groups):
                    x["lr"] = np.interp(
                        nr_integrated_batches,
                        warmup_iteration_range,
                        [0.1 * x["initial_lr"], x["initial_lr"]],
                    )

            X, y = X.to(self.device), y.to(self.device)
            batch = self._prepare_batch_dict(X, y)

            if batch_idx == 0 and epoch == 0 and not first_batch_saved:
                visualize_batch(
                    X,
                    batch,
                    self.metric_tracker.save_dir,
                    predictions=None,
                    epoch=epoch,
                    is_train=True,
                    max_imgs=4,
                )
                first_batch_saved = True

            with torch.amp.autocast(
                device_type=self.device, enabled=self.scaler.is_enabled()
            ):
                pred = self.torch_model.forward(X)
                batch_loss, last_loss = self.state.loss_fn(pred, batch)

            loss_values = last_loss.detach().cpu().numpy().round(3)

            new_loss = self._extract_loss_component(loss_values)

            loss = batch_loss.sum()

            epoch_train_losses += new_loss

            batch_count += 1

            self.state.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (prevents exploding gradients)
            self.scaler.unscale_(self.state.optimizer)
            torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=10.0)

            self.scaler.step(self.state.optimizer)
            self.scaler.update()

            if batch_idx % self.cfg.log_interval == 0:
                log_loss(epoch, new_loss, header=f"TRAIN — Batch {batch_idx} ")

        return batch_count

    def _prepare_batch_dict(self, images: torch.Tensor, targets: torch.Tensor) -> dict:
        valid_mask = targets[:, :, 0] != -1
        samples_with_valid_targets = valid_mask.any(dim=1)
        if not samples_with_valid_targets.any():
            return {
                "img": images,
                "batch_idx": torch.empty(0, device=images.device),
                "cls": torch.empty(0, device=images.device),
                "bboxes": torch.empty((0, 4), device=images.device),
            }
        valid_targets = targets[valid_mask]
        target_lengths = [int(v.sum().item()) for v in valid_mask]
        batch_idx = torch.cat(
            [
                torch.full((length,), i, device=images.device)
                for i, length in enumerate(target_lengths)
                if length > 0
            ]
        )
        cls = valid_targets[:, 0]
        bboxes = valid_targets[:, 1:]
        return {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

    def _extract_loss_component(self, loss_values) -> LossComponent:
        return LossComponent(
            box=loss_values[0],
            cls=loss_values[1],
            dfl=loss_values[2],
            spatial=loss_values[3] if len(loss_values) > 3 else 0.0,
        )
