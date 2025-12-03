from configs import TrainerConfig, TrainerState
import torch
import numpy as np
from pathlib import Path
from torch import nn
from utils import xywh_to_xyxy
from jäger_bomb_visualizer import JägerBombVisualizer
from jäger_bomb_metrics import JägerBombMetrics
from ultralytics.models import YOLO
from jäger_bomb_metrics import JägerBombMetrics
from datetime import datetime
import click
import traceback

class JägerBombTrainer:
    def __init__(self, cfg: TrainerConfig, state : TrainerState):
        self.cfg = cfg
        self.torch_model : nn.Module = state.model.model
        self.state = state
        self._init_grad_scaler()
        self._init_metrics()
        self.visualizer = JägerBombVisualizer(self.metrics.save_dir)

    def _init_grad_scaler(self):
        """Initialize gradient scaler for mixed precision training."""
        enable = self.cfg.device.startswith('cuda')
        self.scaler = torch.amp.GradScaler(self.cfg.device, enabled=enable)
    
    def _init_metrics(self):
        """Initialize metrics tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.cfg.experiment_name
        save_dir = Path("experiments") / experiment_name / "runs" / f"train_{timestamp}"
        
        # FIX ME: DOES THIS NAME STUFF WORK AS EXPECTED?
        self.metrics = JägerBombMetrics(
            names=self.state.model.names,
            save_dir=str(save_dir),
            device=self.cfg.device
        )
        click.secho(f"[SUCCESS] Metrics tracking initialized → {save_dir}", fg="green")
    
    @torch.no_grad()
    def _validate(self, epoch):
        """Validate model and compute metrics."""
        self.cfg.model.eval()
        
        val_box = 0
        val_cls = 0
        val_dfl = 0
        val_spatial = 0
        count = 0

        self.metrics.reset()
        
        for X_val, y_val in self.state.val_loader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)
            
            batch = self._prepare_batch_dict(X_val, y_val)
            
            pred_val =  self.torch_model(X_val)
            
            _, last_loss = self.state.loss_fn(pred_val, batch)
            loss_values = last_loss.cpu().numpy().round(3)
            
            # Handle both standard loss (3 elements) and spatial loss (4 elements)
            if len(loss_values) == 4:
                box_loss, cls_loss, dfl_loss, spatial_loss = loss_values
                val_spatial += spatial_loss
            else:
                box_loss, cls_loss, dfl_loss = loss_values
            
            val_box += box_loss
            val_cls += cls_loss
            val_dfl += dfl_loss
            count += 1
            
            # Update metrics with predictions
            # Convert bboxes from xywh normalized to xyxy pixel format
            img_h, img_w = X_val.shape[2], X_val.shape[3]
            
            # batch dict from _prepare_batch_dict may have filtered targets
            # We need to use the original y_val to get all ground truths
            # y_val shape: [batch_size, max_objects, 5] where 5 = [cls, x, y, w, h]
            all_gt_cls = []
            all_gt_bboxes = []
            all_batch_idx = []
            
            for bi in range(y_val.shape[0]):
                # Get targets for this image (filter out padding)
                img_targets = y_val[bi][y_val[bi, :, 0] != -1]
                if img_targets.shape[0] > 0:
                    cls = img_targets[:, 0]
                    bboxes_xywh = img_targets[:, 1:]
                    # Convert to xyxy pixel coordinates
                    bboxes_xyxy = xywh_to_xyxy(bboxes_xywh, img_w, img_h)
                    
                    all_gt_cls.append(cls)
                    all_gt_bboxes.append(bboxes_xyxy)
                    all_batch_idx.extend([bi] * cls.shape[0])
            
            if len(all_gt_cls) > 0:
                metrics_batch = {
                    'cls': torch.cat(all_gt_cls),
                    'bboxes': torch.cat(all_gt_bboxes),
                    'batch_idx': torch.tensor(all_batch_idx, device=y_val.device)
                }
            else:
                metrics_batch = {
                    'cls': torch.empty(0, device=y_val.device),
                    'bboxes': torch.empty((0, 4), device=y_val.device),
                    'batch_idx': torch.empty(0, dtype=torch.long, device=y_val.device)
                }
            
            try:
                self.metrics.update(pred_val, metrics_batch)
            except Exception as e:
                click.secho(f"[ERROR] Metrics update failed: {e}", fg="red")
                traceback.print_exc()
        
        # Compute average losses
        avg_box = val_box / count
        avg_cls = val_cls / count
        avg_dfl = val_dfl / count
        avg_spatial = val_spatial / count
        avg_total = avg_box + avg_cls + avg_dfl + avg_spatial
        
        spatial_str = f", Spatial: {avg_spatial:.4f}" if avg_spatial > 0 else ""
        click.echo(
            f"VALIDATION — Epoch {epoch}: "
            f"Box: {avg_box:.4f}, "
            f"Cls: {avg_cls:.4f}, "
            f"DFL: {avg_dfl:.4f}"
            f"{spatial_str}, "
            f"Total: {avg_total:.4f}"
        )
        
        return {
            'box': avg_box,
            'cls': avg_cls,
            'dfl': avg_dfl,
            'spatial': avg_spatial,
            'total': avg_total
        }

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
        save_dir = Path(self.metrics.save_dir) / "weights"
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

    def _reset_train_losses(self):
        return {'box': 0, 'cls': 0, 'dfl': 0, 'spatial': 0}


    def train(self):
        self.torch_model.to(self.cfg.device)
        best_loss = np.inf
        count = 0
        early_stoppage_count = 9999999999
        
        warmup_epochs = 3.0
        train_loader_len = len(self.state.train_loader)
        nr_warmup_iterations = max(round(warmup_epochs * train_loader_len), 100)
        
        # Track training losses per epoch
        epoch_train_losses = self._reset_train_losses()
        
        for epoch in range(self.cfg.epochs):
            self.state.model.train(True)
            
            epoch_train_losses = self._reset_train_losses()
            batch_count = 0
            
            # Track first batch for visualization
            first_batch_saved = False
            
            for batch_idx, (X, y) in enumerate(self.state.train_loader):
                # Warmup learning rate for first few epochs
                ni = batch_idx + train_loader_len * epoch  # number integrated batches
                if ni <= nr_warmup_iterations:
                    xi = [0, nr_warmup_iterations]  # warmup iteration range
                    # Warmup: gradually increase LR from 0.1 to target
                    for j, x in enumerate(self.cfg.optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [0.1 * x['initial_lr'], x['initial_lr']])
                
                X, y = X.to(self.cfg.device), y.to(self.cfg.device) 
                batch = self._prepare_batch_dict(X, y)
                
                # Save first training batch for visualization (only at epoch 0)
                if batch_idx == 0 and epoch == 0 and not first_batch_saved:
                    self.visualizer.visualize_batch(X, batch, predictions=None, 
                                        epoch=epoch, is_train=True, max_imgs=4)
                    first_batch_saved = True
                
                # Forward pass with AMP
                with torch.amp.autocast(device_type=self.cfg.device, enabled=self.scaler.is_enabled()):
                    pred = self.torch_model.forward(X)
                    batch_loss, last_loss = self.state.loss_fn(pred, batch)
                
                loss_values = last_loss.detach().cpu().numpy().round(3)
                
                # Handle both standard loss (3 elements) and spatial loss (4 elements)
                if len(loss_values) == 4:
                    box_loss, cls_loss, dfl_loss, spatial_loss = loss_values
                    epoch_train_losses['spatial'] += spatial_loss
                else:
                    box_loss, cls_loss, dfl_loss = loss_values
                
                loss = batch_loss.sum()
                
                # Accumulate epoch losses
                epoch_train_losses['box'] += box_loss
                epoch_train_losses['cls'] += cls_loss
                epoch_train_losses['dfl'] += dfl_loss
                batch_count += 1
                
                # Backward pass with gradient scaling
                self.state.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (prevents exploding gradients)
                self.scaler.unscale_(self.state.optimizer)
                torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=10.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.state.optimizer)
                self.scaler.update()
                

                if batch_idx % self.cfg.log_interval == 0:
                    click.echo(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f}, "
                        f"Total: {(box_loss+cls_loss+dfl_loss):.4f}"
                    )
            
            # Compute average training losses for the epoch
            avg_train_losses = {
                'box': epoch_train_losses['box'] / batch_count,
                'cls': epoch_train_losses['cls'] / batch_count,
                'dfl': epoch_train_losses['dfl'] / batch_count,
                'spatial': epoch_train_losses['spatial'] / batch_count
            }
            
            # Validation
            val_losses = self._validate(epoch)
            
            # Compute detection metrics and generate plots every N epochs or at end
            generate_plots = (epoch == self.cfg.epochs - 1)
            
            # Generate prediction visualizations every 10 epochs
            if generate_plots:
                # Visualize validation predictions
                self.visualizer.visualize_predictions(epoch, self.cfg.model, self.cfg.val_dataloader)
            
            try:
                det_metrics = self.metrics.compute_metrics(plot=generate_plots)
                
                # Print metrics
                click.echo(
                    f"METRICS — Epoch {epoch}: "
                    f"P: {det_metrics['precision']:.4f}, "
                    f"R: {det_metrics['recall']:.4f}, "
                    f"mAP50: {det_metrics['mAP50']:.4f}, "
                    f"mAP50-95: {det_metrics['mAP50-95']:.4f}"
                )
            except Exception as e:
                det_metrics = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
                click.secho(f"Could not compute metrics: {e}", fg="red")
            
            # Step the learning rate scheduler (only after warmup)
            if epoch >= warmup_epochs:
                self.cfg.scheduler.step()
            
            current_lr = self.cfg.optimizer.param_groups[0]['lr']
            click.echo(f"Learning rate: {current_lr:.6f}")
            
            # Log epoch to CSV
            self.metrics.log_epoch(
                epoch=epoch,
                train_losses=avg_train_losses,
                val_losses=val_losses,
                metrics=det_metrics,
                lr=current_lr
            )
            
            # Model saving based on validation loss
            curr_val_loss = val_losses['total']
            if curr_val_loss < best_loss:
                self._save_model(epoch, is_best=True)
                best_loss = curr_val_loss
                count = 0
            else:
                self._save_model(epoch, is_best=False)
                count += 1
            
            if count >= early_stoppage_count:
                click.secho(f"[INFO] Early stopping at epoch {epoch}", fg="blue")
                break
        
        # Finalize metrics (generate final plots)
        click.secho("\n[INFO] Generating final metrics and plots...", fg="blue")
        self.metrics.finalize()
        
        # Evaluate best model on test set
        # click.secho("\n[INFO] Evaluating best model on test set...", fg="blue")
        # self._evaluate_test_set()
        
        click.secho("[SUCCESS] Training complete!", fg="green")