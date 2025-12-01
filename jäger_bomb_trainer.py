from configs import TrainingConfig, SaveConfig
import torch
import numpy as np
from pathlib import Path
from utils import convert_to_python_types, xywh_to_xyxy
from jäger_bomb_visualizer import JägerBombVisualizer
from jäger_bomb_logger import JägerBombLogger
import json
from jäger_bomb_metrics import JägerBombMetrics
from ultralytics.models import YOLO
from jäger_bomb_metrics import JägerBombMetrics
from datetime import datetime

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
        
        # Initialize metrics tracker
        self._init_metrics()
        self.visualizer = JägerBombVisualizer(self.metrics.save_dir)
    
    def _init_metrics(self):
        """Initialize metrics tracking."""
        
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
        
        # Store config parameters for logging
        self.config_params = getattr(self.cfg, 'config_params', None)
    
    @torch.no_grad()
    def _evaluate_test_set(self):
        """Evaluate best.pt model on test dataset and save results."""
        
        # Load best model
        best_model_path = Path(self.metrics.save_dir) / "weights" / "best.pt"
        if not best_model_path.exists():
            print(f"⚠️ best.pt not found at {best_model_path}, skipping test evaluation")
            return
        
        print(f"Loading best model from {best_model_path}...")
        test_model = YOLO(str(best_model_path))
        test_model.model.eval()
        test_model.model.to(self.cfg.device)
        
        # Check if test dataloader exists
        if not hasattr(self.cfg, 'test_dataloader') or self.cfg.test_dataloader is None:
            print("⚠️ No test dataloader configured, skipping test evaluation")
            return
        
        # Initialize metrics for test set
        names = getattr(self.cfg.yolo_model, 'names', {0: "class0", 1: "class1"})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        
        test_metrics = JägerBombMetrics(
            names=names,
            save_dir=str(Path(self.metrics.save_dir) / "test_results"),
            device=self.cfg.device
        )
        
        print("Running inference on test set...")
        test_box = 0
        test_cls = 0
        test_dfl = 0
        test_spatial = 0
        count = 0
        
        for X_test, y_test in self.cfg.test_dataloader:
            X_test, y_test = X_test.to(self.cfg.device), y_test.to(self.cfg.device)
            
            batch = self._prepare_batch_dict(X_test, y_test)
            
            # Get predictions
            pred_test = test_model.model(X_test)
            
            # Compute loss (optional, for comparison)
            _, last_loss = self.cfg.loss_fn(pred_test, batch)
            loss_values = last_loss.cpu().numpy().round(3)
            
            if len(loss_values) == 4:
                box_loss, cls_loss, dfl_loss, spatial_loss = loss_values
                test_spatial += spatial_loss
            else:
                box_loss, cls_loss, dfl_loss = loss_values
            
            test_box += box_loss
            test_cls += cls_loss
            test_dfl += dfl_loss
            count += 1
            
            # Update metrics with predictions
            img_h, img_w = X_test.shape[2], X_test.shape[3]
            
            all_gt_cls = []
            all_gt_bboxes = []
            all_batch_idx = []
            
            for bi in range(y_test.shape[0]):
                img_targets = y_test[bi][y_test[bi, :, 0] != -1]
                if img_targets.shape[0] > 0:
                    cls = img_targets[:, 0]
                    bboxes_xywh = img_targets[:, 1:]
                    bboxes_xyxy = self._xywh_to_xyxy(bboxes_xywh, img_w, img_h)
                    
                    all_gt_cls.append(cls)
                    all_gt_bboxes.append(bboxes_xyxy)
                    all_batch_idx.extend([bi] * cls.shape[0])
            
            if len(all_gt_cls) > 0:
                metrics_batch = {
                    'cls': torch.cat(all_gt_cls),
                    'bboxes': torch.cat(all_gt_bboxes),
                    'batch_idx': torch.tensor(all_batch_idx, device=y_test.device)
                }
            else:
                metrics_batch = {
                    'cls': torch.empty(0, device=y_test.device),
                    'bboxes': torch.empty((0, 4), device=y_test.device),
                    'batch_idx': torch.empty(0, dtype=torch.long, device=y_test.device)
                }
            
            test_metrics.update(pred_test, metrics_batch)
        
        # Compute average losses
        avg_test_losses = {
            'box': test_box / count,
            'cls': test_cls / count,
            'dfl': test_dfl / count,
            'spatial': test_spatial / count,
            'total': (test_box + test_cls + test_dfl + test_spatial) / count
        }
        
        print(
            f"TEST SET LOSSES: "
            f"Box: {avg_test_losses['box']:.4f}, "
            f"Cls: {avg_test_losses['cls']:.4f}, "
            f"DFL: {avg_test_losses['dfl']:.4f}, "
            f"Spatial: {avg_test_losses['spatial']:.4f}, "
            f"Total: {avg_test_losses['total']:.4f}"
        )
        
        # Compute detection metrics
        try:
            det_metrics = test_metrics.compute_metrics(plot=True)
            print(
                f"TEST SET METRICS: "
                f"P: {det_metrics['precision']:.4f}, "
                f"R: {det_metrics['recall']:.4f}, "
                f"mAP50: {det_metrics['mAP50']:.4f}, "
                f"mAP50-95: {det_metrics['mAP50-95']:.4f}"
            )
            
            # Plot confusion matrices for test set
            try:
                test_metrics.confusion_matrix.plot(normalize=True, save_dir=str(Path(self.metrics.save_dir) / "test_results"))
                test_metrics.confusion_matrix.plot(normalize=False, save_dir=str(Path(self.metrics.save_dir) / "test_results"))
                print("✅ Test confusion matrices generated")
            except Exception as e:
                print(f"⚠️ Could not generate test confusion matrices: {e}")
            
        except Exception as e:
            det_metrics = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
            print(f"⚠️ Could not compute test metrics: {e}")
        
        # Save test results to JSON (convert numpy types to native Python)
        
        test_results = {
            'losses': convert_to_python_types(avg_test_losses),
            'metrics': convert_to_python_types(det_metrics),
            'model': str(best_model_path)
        }
        
        results_path = Path(self.metrics.save_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"✅ Test results saved → {results_path}")
    
    @torch.no_grad()
    def _validate(self, epoch):
        """Validate model and compute metrics."""
        self.cfg.model.eval()
        
        val_box = 0
        val_cls = 0
        val_dfl = 0
        val_spatial = 0
        count = 0
        
        # Reset metrics for this validation run
        self.metrics.reset()
        
        for X_val, y_val in self.cfg.val_dataloader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)
            
            batch = self._prepare_batch_dict(X_val, y_val)
            
            # Get predictions
            pred_val =  self.cfg.model(X_val)
            
            # Compute loss
            _, last_loss = self.cfg.loss_fn(pred_val, batch)
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
                print(f"⚠️ Metrics update failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute average losses
        avg_box = val_box / count
        avg_cls = val_cls / count
        avg_dfl = val_dfl / count
        avg_spatial = val_spatial / count
        avg_total = avg_box + avg_cls + avg_dfl + avg_spatial
        
        spatial_str = f", Spatial: {avg_spatial:.4f}" if avg_spatial > 0 else ""
        print(
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
        # Save to experiment-specific directory if available
        if hasattr(self.metrics, 'save_dir'):
            save_dir = Path(self.metrics.save_dir) / "weights"
        else:
            save_dir = Path(self.save_cfg.save_path).parent
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save last.pt (always)
        last_path = save_dir / "last.pt"
        self.cfg.yolo_model.save(str(last_path))
        
        # Save best.pt (only when validation improves)
        if is_best:
            best_path = save_dir / "best.pt"
            self.cfg.yolo_model.save(str(best_path))
            print(f"✅ Model saved at epoch {epoch} → {best_path}")
        else:
            print(f"💾 Checkpoint saved → {last_path}")  

    def train(self):
        self.cfg.model.to(self.cfg.device)
        best_loss = np.inf
        count = 0
        early_stoppage_count = 9999999999
        
        JägerBombLogger.log_config_params(self.metrics.save_dir, self.cfg, self.config_params)
        
        # Warmup settings (like Ultralytics)
        warmup_epochs = 3.0
        nb = len(self.cfg.train_dataloader)
        nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations
        
        # Track training losses per epoch
        epoch_train_losses = {'box': 0, 'cls': 0, 'dfl': 0, 'spatial': 0}
        
        for epoch in range(self.cfg.epochs):
            self.cfg.model.train(True)
            
            # Reset epoch training losses
            epoch_train_losses = {'box': 0, 'cls': 0, 'dfl': 0, 'spatial': 0}
            batch_count = 0
            
            # Track first batch for visualization
            first_batch_saved = False
            
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
                
                # Save first training batch for visualization (only at epoch 0)
                if batch_idx == 0 and epoch == 0 and not first_batch_saved:
                    self.visualizer.visualize_batch(X, batch, predictions=None, 
                                        epoch=epoch, is_train=True, max_imgs=4)
                    first_batch_saved = True
                
                # Forward pass with AMP
                with torch.amp.autocast(device_type='cuda' if self.amp else 'cpu', enabled=self.amp):
                    pred = self.cfg.model.forward(X)
                    batch_loss, last_loss = self.cfg.loss_fn(pred, batch)
                
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
                self.cfg.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (prevents exploding gradients)
                self.scaler.unscale_(self.cfg.optimizer)
                torch.nn.utils.clip_grad_norm_(self.cfg.model.parameters(), max_norm=10.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.cfg.optimizer)
                self.scaler.update()
                

                if batch_idx % self.cfg.log_interval == 0:
                    print(
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
                self._save_model(epoch, is_best=True)
                best_loss = curr_val_loss
                count = 0
            else:
                self._save_model(epoch, is_best=False)
                count += 1
            
            if count >= early_stoppage_count:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Finalize metrics (generate final plots)
        print("\n📊 Generating final metrics and plots...")
        self.metrics.finalize()
        
        # Evaluate best model on test set
        print("\n🧪 Evaluating best model on test set...")
        self._evaluate_test_set()
        
        print("✅ Training complete!")