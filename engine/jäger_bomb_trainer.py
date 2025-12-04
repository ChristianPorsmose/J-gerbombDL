import torch
import numpy as np
from pathlib import Path
from torch import nn
from utils.utils import xywh_to_xyxy
from datetime import datetime
import click
import traceback
from metrics.jäger_bomb_metric_tracker import JägerBombMetricTracker
from metrics.jäger_bomb_metric_logger import JägerBombMetricLogger
from metrics.metric_visualization import plot_all_metrics
from engine.bomb_visualize import visualize_batch, visualize_predictions
from engine.data import BatchResult, LossComponent, TrainerConfig, TrainerState
from engine.log_helpers import log_loss

class JägerBombTrainer:
    def __init__(self, cfg: TrainerConfig, state : TrainerState):
        self.cfg = cfg
        self.torch_model : nn.Module = state.model.model
        self.state = state
        self.device = torch.get_default_device().type
        self._init_grad_scaler()
        self._init_metrics()

    def _init_grad_scaler(self):
        """Initialize gradient scaler for mixed precision training."""
        enable = self.device.startswith('cuda')
        self.scaler = torch.amp.GradScaler(self.device, enabled=enable)
    
    def _init_metrics(self):
        """Initialize metrics tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.cfg.experiment_name
        save_dir = Path("experiments_results") / experiment_name / "runs" / f"train_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_tracker = JägerBombMetricTracker(
            self.state.model.names,
            save_dir=str(save_dir)
        )
        self.metrics_logger = JägerBombMetricLogger(
            file_path=save_dir / "metrics.csv"
        )
    
    @torch.no_grad()
    def _validate(self, epoch) -> LossComponent:
        """Validate model and compute metrics."""
        self.torch_model.eval()
        
        val_loss = LossComponent()
        count = 0

        self.metric_tracker.reset()
        
        for X_val, y_val in self.state.val_loader:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)
            
            batch = self._prepare_batch_dict(X_val, y_val)
            
            pred_val =  self.torch_model(X_val)
            
            _, last_loss = self.state.loss_fn(pred_val, batch)
            loss_values = last_loss.cpu().numpy().round(3)
            
            new_loss = self._extract_loss_component(loss_values)
            
            val_loss += new_loss

            count += 1
            
            img_h, img_w = X_val.shape[2], X_val.shape[3]
            
            metrics_batch = self.metric_tracker.prepare_batch(y_val, img_h, img_w)
            
            self.metric_tracker.update(pred_val, metrics_batch)
        
        average_loss = val_loss / count
        
        log_loss(epoch, average_loss, header="VALIDATION")
        
        return average_loss



    def _prepare_batch_dict(self, images: torch.Tensor, targets: torch.Tensor) -> dict:
        valid_mask = targets[:, :, 0] != -1

        samples_with_valid_targets = valid_mask.any(dim=1)

        if not samples_with_valid_targets.any():
            return {"img": images, "batch_idx": torch.empty(0, device=images.device), "cls": torch.empty(0, device=images.device), "bboxes": torch.empty((0, 4), device=images.device)}

        valid_targets = targets[valid_mask]

        target_lengths = [int(v.sum().item()) for v in valid_mask]
        batch_idx = torch.cat([torch.full((length,), i, device=images.device) for i, length in enumerate(target_lengths) if length > 0])

        cls = valid_targets[:, 0]
        bboxes = valid_targets[:, 1:]

        return {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

    def _save_model(self, epoch, is_best=False):
        """Save model checkpoint."""
        save_dir = Path(self.metric_tracker.save_dir) / "weights"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        last_path = save_dir / "last.pt"
        self.state.model.save(str(last_path))
        
        # Save best.pt (only when validation improves)
        if is_best:
            best_path = save_dir / "best.pt"
            self.state.model.save(str(best_path))
            click.secho(f"[SUCCESS] Model saved at epoch {epoch} → {best_path}", fg="green")
        else:
            click.secho(f"[INFO] Checkpoint saved → {last_path}", fg="blue")  

    def train(self):
        self.torch_model.to(self.device)
        best_loss = np.inf
        count = 0
        EARLY_STOPPAGE_COUNT = 9999999999
        WARMUP_EPOCHS = 3.0
    
        train_loader_len = len(self.state.train_loader)
        nr_warmup_iterations = max(round(WARMUP_EPOCHS * train_loader_len), 100)
        
        epoch_train_losses = LossComponent()
        
        for epoch in range(self.cfg.epochs):
            self.torch_model.train(True)
            
            epoch_train_losses = LossComponent()
            batch_count = 0
            
            # Track first batch for visualization
            first_batch_saved = False
            
            batch_count += self.train_one_epoch(train_loader_len, nr_warmup_iterations, epoch_train_losses, epoch, first_batch_saved) # FIX ME
            
            average_train_loss = epoch_train_losses / batch_count
            
            val_losses = self._validate(epoch)
            # Compute detection metrics and generate plots every N epochs or at end
            generate_plots = (epoch == self.cfg.epochs - 1)
            
            if generate_plots:
                visualize_predictions(epoch, self.torch_model, self.state.val_loader, self.device)
            
            det_metrics = self.metric_tracker.compute(plot=generate_plots)

            if epoch >= WARMUP_EPOCHS:
                self.state.scheduler.step()
            
            current_lr = self.state.optimizer.param_groups[0]['lr']
            click.echo(f"Learning rate: {current_lr:.6f}")
            
            self.metrics_logger.log_batch_result(
                batchResult=BatchResult(
                    train_loss=average_train_loss,
                    val_loss=val_losses,
                    metrics=det_metrics
                ),
                epoch=epoch,
                lr=current_lr
                )
            
            count = self._save(best_loss, epoch, val_losses)
            
            if count >= EARLY_STOPPAGE_COUNT:
                click.secho(f"[INFO] Early stopping at epoch {epoch}", fg="blue")
                break
        
        click.secho("\n[INFO] Generating final metrics and plots...", fg="blue")
        
        plot_all_metrics(
            confusion_matrix=self.metric_tracker.confusion_matrix,
            base_dir=Path(self.metric_tracker.save_dir)
        )
        
        # Evaluate best model on test set
        # click.secho("\n[INFO] Evaluating best model on test set...", fg="blue")
        # self._evaluate_test_set()
        
        click.secho("[SUCCESS] Training complete!", fg="green")

    def _save(self, best_loss : int, epoch: int, val_losses : LossComponent) -> int:
        curr_val_loss = val_losses.total()
        if curr_val_loss < best_loss:
            self._save_model(epoch, is_best=True)
            best_loss = curr_val_loss
            count = 0
        else:
            self._save_model(epoch, is_best=False)
            count += 1
        return count


    def train_one_epoch(self, train_loader_len, nr_warmup_iterations, epoch_train_loss : LossComponent, epoch, first_batch_saved):
        batch_count = 0
        for batch_idx, (X, y) in enumerate(self.state.train_loader):
            
            # Warmup learning rate for first few epochs
            nr_integrated_batches = batch_idx + train_loader_len * epoch 
            if nr_integrated_batches <= nr_warmup_iterations:
                warmup_iteration_range = [0, nr_warmup_iterations] 
                
                # Warmup: gradually increase LR from 0.1 to target
                for j, x in enumerate(self.state.optimizer.param_groups):
                    x['lr'] = np.interp(nr_integrated_batches, warmup_iteration_range, [0.1 * x['initial_lr'], x['initial_lr']])
                
            X, y = X.to(self.device), y.to(self.device) 
            batch = self._prepare_batch_dict(X, y)
                
            if batch_idx == 0 and epoch == 0 and not first_batch_saved:
                visualize_batch(X, batch,self.metric_tracker.save_dir ,predictions=None, 
                                        epoch=epoch, is_train=True, max_imgs=4)
                first_batch_saved = True

            with torch.amp.autocast(device_type=self.device, enabled=self.scaler.is_enabled()):
                pred = self.torch_model.forward(X)
                batch_loss, last_loss = self.state.loss_fn(pred, batch)
                
            loss_values = last_loss.detach().cpu().numpy().round(3)
     
            new_loss = self._extract_loss_component(loss_values)
                
            loss = batch_loss.sum()
                
            epoch_train_loss += new_loss
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

    def _extract_loss_component(self, loss_values) -> LossComponent:
        return LossComponent(
            box=loss_values[0],
            cls=loss_values[1],
            dfl=loss_values[2],
            spatial=loss_values[3] if len(loss_values) > 3 else 0.0
        )