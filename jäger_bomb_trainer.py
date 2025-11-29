from configs import TrainingConfig, SaveConfig
import torch
import numpy as np
from pathlib import Path

class JägerBombTrainer:
    def __init__(self, cfg: TrainingConfig, save_cfg : SaveConfig):
        self.cfg = cfg
        self.save_cfg = save_cfg
        
        # Initialize AMP (Automatic Mixed Precision) scaler for better training
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            self.amp = True
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)
            self.amp = False
        
        # EMA configurable
        if getattr(self.cfg, 'use_ema', False):
            from ultralytics.utils.torch_utils import ModelEMA
            self.ema = ModelEMA(self.cfg.model)
            print("✅ EMA (Exponential Moving Average) enabled")
        else:
            self.ema = None
            print("⚠️ EMA (Exponential Moving Average) disabled")
        
        # Initialize metrics tracker
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics tracking."""
        from jäger_bomb_metrics import JägerBombMetrics
        from datetime import datetime
        
        # Get class names from model
        names = getattr(self.cfg.yolo_model, 'names', {0: "class0", 1: "class1"})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        
        # Create unique save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if running an experiment (config loaded from experiments/ folder)
        experiment_name = getattr(self.cfg, 'experiment_name', None)
        if experiment_name:
            # Save under the experiment folder
            save_dir = Path("experiments") / experiment_name / "runs" / f"train_{timestamp}"
        else:
            # Default runs folder
            save_dir = Path("runs") / f"train_{timestamp}"
        
        self.metrics = JägerBombMetrics(
            names=names,
            save_dir=str(save_dir),
            device=self.cfg.device
        )
        print(f"✅ Metrics tracking initialized → {save_dir}")
    
    @torch.no_grad()
    def _validate(self, epoch):
        """Validate model and compute metrics."""
        # Use EMA model for validation if enabled
        model_to_validate = self.ema.ema if self.ema else self.cfg.model
        model_to_validate.eval()
        
        val_box = 0
        val_cls = 0
        val_dfl = 0
        count = 0
        
        # Reset metrics for this validation run
        self.metrics.reset()
        
        for X_val, y_val in self.cfg.val_dataloader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)
            
            batch = self._prepare_batch_dict(X_val, y_val)
            
            # Get predictions
            pred_val = model_to_validate(X_val)
            
            # Compute loss
            _, last_loss = self.cfg.loss_fn(pred_val, batch)
            box_loss, cls_loss, dfl_loss = last_loss.cpu().numpy().round(3)
            
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
                    bboxes_xyxy = self._xywh_to_xyxy(bboxes_xywh, img_w, img_h)
                    
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
                print(f"⚠️ Metrics update failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute average losses
        avg_box = val_box / count
        avg_cls = val_cls / count
        avg_dfl = val_dfl / count
        avg_total = avg_box + avg_cls + avg_dfl
        
        print(
            f"VALIDATION — Epoch {epoch}: "
            f"Box: {avg_box:.4f}, "
            f"Cls: {avg_cls:.4f}, "
            f"DFL: {avg_dfl:.4f}, "
            f"Total: {avg_total:.4f}"
        )
        
        return {
            'box': avg_box,
            'cls': avg_cls,
            'dfl': avg_dfl,
            'total': avg_total
        }
    
    def _xywh_to_xyxy(self, bboxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        """Convert bboxes from normalized xywh to pixel xyxy format."""
        if bboxes.numel() == 0:
            return bboxes
        
        # bboxes: [N, 4] in format [x_center, y_center, width, height] normalized
        x_center = bboxes[:, 0] * img_w
        y_center = bboxes[:, 1] * img_h
        width = bboxes[:, 2] * img_w
        height = bboxes[:, 3] * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)

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

    def _save_model(self, epoch):
        save_path = f"{self.save_cfg.save_path}.pt"
        self.cfg.yolo_model.save(save_path)
        print(f"Model saved at epoch {epoch} → {save_path}")  

    def train(self):
        self.cfg.model.to(self.cfg.device)
        best_loss = np.inf
        count = 0
        early_stoppage_count = 15
        
        # Warmup settings (like Ultralytics)
        warmup_epochs = 3.0
        nb = len(self.cfg.train_dataloader)
        nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations
        
        # Track training losses per epoch
        epoch_train_losses = {'box': 0, 'cls': 0, 'dfl': 0}
        
        for epoch in range(self.cfg.epochs):
            self.cfg.model.train(True)
            
            # Reset epoch training losses
            epoch_train_losses = {'box': 0, 'cls': 0, 'dfl': 0}
            batch_count = 0
            
            for batch_idx, (X, y) in enumerate(self.cfg.train_dataloader):
                # Warmup learning rate for first few epochs
                ni = batch_idx + nb * epoch  # number integrated batches
                if ni <= nw:
                    xi = [0, nw]  # warmup iteration range
                    # Warmup: gradually increase LR from 0.1 to target
                    for j, x in enumerate(self.cfg.optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [0.1 * x['initial_lr'], x['initial_lr']])
                
                X, y = X.to(self.cfg.device), y.to(self.cfg.device) 
                batch = self._prepare_batch_dict(X, y)
                
                # Forward pass with AMP
                with torch.amp.autocast(device_type='cuda' if self.amp else 'cpu', enabled=self.amp):
                    pred = self.cfg.model.forward(X)
                    batch_loss, last_loss = self.cfg.loss_fn(pred, batch)
                
                box_loss, cls_loss, dfl_loss = last_loss.detach().cpu().numpy().round(3)
                loss = batch_loss.sum()
                
                # Accumulate epoch losses
                epoch_train_losses['box'] += box_loss
                epoch_train_losses['cls'] += cls_loss
                epoch_train_losses['dfl'] += dfl_loss
                batch_count += 1
                
                # Backward pass with gradient scaling
                self.cfg.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (prevents exploding gradients)
                self.scaler.unscale_(self.cfg.optimizer)
                torch.nn.utils.clip_grad_norm_(self.cfg.model.parameters(), max_norm=10.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.cfg.optimizer)
                self.scaler.update()
                
                # Update EMA after optimizer step
                if self.ema:
                    self.ema.update(self.cfg.model)

                if batch_idx % self.cfg.log_interval == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f}, "
                        f"Total: {(box_loss+cls_loss+dfl_loss):.4f}"
                    )
            
            # Calculate average training losses for epoch
            avg_train_losses = {
                'box': epoch_train_losses['box'] / batch_count,
                'cls': epoch_train_losses['cls'] / batch_count,
                'dfl': epoch_train_losses['dfl'] / batch_count
            }
            
            # Validation
            val_losses = self._validate(epoch)
            
            # Compute detection metrics and generate plots every N epochs or at end
            generate_plots = (epoch % 10 == 0) or (epoch == self.cfg.epochs - 1)
            try:
                det_metrics = self.metrics.compute_metrics(plot=generate_plots)
                
                # Print metrics
                print(
                    f"METRICS — Epoch {epoch}: "
                    f"P: {det_metrics['precision']:.4f}, "
                    f"R: {det_metrics['recall']:.4f}, "
                    f"mAP50: {det_metrics['mAP50']:.4f}, "
                    f"mAP50-95: {det_metrics['mAP50-95']:.4f}"
                )
            except Exception as e:
                det_metrics = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
                print(f"⚠️ Could not compute metrics: {e}")
            
            # Step the learning rate scheduler (only after warmup)
            if epoch >= warmup_epochs:
                self.cfg.scheduler.step()
            
            current_lr = self.cfg.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
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
                self._save_model(epoch)
                best_loss = curr_val_loss
                count = 0
            else:
                count += 1
            
            if count >= early_stoppage_count:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Finalize metrics (generate final plots)
        print("\n📊 Generating final metrics and plots...")
        self.metrics.finalize()
        print("✅ Training complete!")