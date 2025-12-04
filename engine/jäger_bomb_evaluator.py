

import json
from pathlib import Path

import click
import torch

from jäger_bomb_metrics_old import JägerBombMetrics
from ultralytics.models.yolo.model import YOLO
from utils.utils import convert_to_python_types


class JägerBombEvaluator:
    pass

    @torch.no_grad()
    def _evaluate_test_set(self):
        """Evaluate best.pt model on test dataset and save results."""
        
        # Load best model
        best_model_path = Path(self.metrics.save_dir) / "weights" / "best.pt"
        if not best_model_path.exists():
            click.secho(f"[WARNING] best.pt not found at {best_model_path}, skipping test evaluation", fg="yellow")
            return
        
        click.echo(f"[INFO] Loading best model from {best_model_path}...", fg="blue")
        test_model = YOLO(str(best_model_path))
        test_model.model.eval()
        test_model.model.to(self.cfg.device)
        
        names = getattr(self.cfg.yolo_model, 'names', {0: "class0", 1: "class1"})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        
        test_metrics = JägerBombMetrics(
            names=self.state.model.names,
            save_dir=str(Path(self.metrics.save_dir) / "test_results"),
            device=self.cfg.device
        )
        
        click.echo("Running inference on test set...")
        test_box = 0
        test_cls = 0
        test_dfl = 0
        test_spatial = 0
        count = 0
        
        for X_test, y_test in self.state.test_loader:
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
        
        click.echo(
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
            click.echo(
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
                click.secho("[SUCCESS] Test confusion matrices generated", fg="green")
            except Exception as e:
                click.secho(f"[ERROR] Could not generate test confusion matrices: {e}", fg="red")
            
        except Exception as e:
            det_metrics = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
            click.secho(f"[ERROR] Could not compute test metrics: {e}", fg="red")
        
        # Save test results to JSON (convert numpy types to native Python)
        
        test_results = {
            'losses': convert_to_python_types(avg_test_losses),
            'metrics': convert_to_python_types(det_metrics),
            'model': str(best_model_path)
        }
        
        results_path = Path(self.metrics.save_dir) / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        click.secho(f"Test results saved → {results_path}", fg="green")