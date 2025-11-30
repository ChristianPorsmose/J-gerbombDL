"""
Metrics tracking for JägerBomb custom YOLO training loop.
Uses Ultralytics' metrics utilities for proper detection evaluation.
"""

import csv
from pathlib import Path
import numpy as np
import torch

from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou


class JägerBombMetrics:
    """
    Metrics tracker for custom YOLO training.
    Computes and saves: BoxF1, BoxPR, BoxP, BoxR curves, Confusion Matrix, and CSV logs.
    """
    
    def __init__(self, names: dict, save_dir: str = "runs", device: str = "cuda"):
        """
        Initialize metrics tracker.
        
        Args:
            names: Dictionary mapping class indices to names, e.g., {0: "shot", 1: "cup"}
            save_dir: Directory to save metrics and plots
            device: Device for tensor operations
        """
        self.names = names
        self.nc = len(names)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # IoU thresholds for mAP calculation
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        
        # Initialize metrics
        self.det_metrics = DetMetrics(names=names)
        self.confusion_matrix = ConfusionMatrix(names=names, task="detect")
        
        # Image counter for target_img tracking
        self.seen = 0
        
        # CSV logging
        self.csv_path = self.save_dir / "results.csv"
        # Check if using spatial consistency loss
        self.has_spatial_loss = device is not None  # Will be set properly by trainer
        self.csv_headers = [
            "epoch", 
            "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/spatial_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/spatial_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "lr/pg0"
        ]
        self._init_csv()
        
        # Store epoch results
        self.epoch_results = []
        
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.det_metrics = DetMetrics(names=self.names)
        self.confusion_matrix = ConfusionMatrix(names=self.names, task="detect")
        self.seen = 0
    
    def update(self, preds: torch.Tensor, targets: dict, conf_thres: float = 0.001, iou_thres: float = 0.6):
        """
        Update metrics with predictions and ground truth.
        
        Args:
            preds: Model predictions (raw output from model)
            targets: Dictionary with 'cls', 'bboxes', 'batch_idx' keys (bboxes in xyxy pixel format)
            conf_thres: Confidence threshold for predictions
            iou_thres: IoU threshold for NMS
        """
        from ultralytics.utils.nms import non_max_suppression
        
        # Get raw predictions
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Apply NMS
        pred_list = non_max_suppression(
            preds,
            conf_thres,
            iou_thres,
            nc=self.nc,
            multi_label=True,
            agnostic=False,
            max_det=300,
        )
        
        # Process each image in batch
        for si, pred in enumerate(pred_list):
            self.seen += 1
            
            # Get ground truth for this image
            idx = targets['batch_idx'] == si
            gt_cls = targets['cls'][idx]
            gt_bboxes = targets['bboxes'][idx]
            nl = gt_cls.shape[0]  # number of labels
            
            # Target classes for this image
            tcls = gt_cls.cpu().numpy() if nl else np.array([])
            
            # No predictions case
            if pred.shape[0] == 0:
                if nl:
                    # Update stats with no predictions but we have labels
                    self.det_metrics.update_stats({
                        'tp': np.zeros((0, self.niou), dtype=bool),
                        'conf': np.array([]),
                        'pred_cls': np.array([]),
                        'target_cls': tcls,
                        'target_img': np.full(nl, self.seen - 1),
                    })
                continue
            
            # Get predictions
            pred_bboxes = pred[:, :4]
            pred_conf = pred[:, 4]
            pred_cls = pred[:, 5]
            
            # Update confusion matrix
            pred_dict = {'bboxes': pred_bboxes, 'conf': pred_conf, 'cls': pred_cls}
            gt_dict = {'bboxes': gt_bboxes, 'cls': gt_cls}
            self.confusion_matrix.process_batch(pred_dict, gt_dict, conf=0.25, iou_thres=0.45)
            
            # Compute correct predictions (true positives)
            correct = self._process_batch(pred_bboxes, pred_cls, gt_bboxes, gt_cls)
            
            # Update detection metrics
            stats_update = {
                'tp': correct,
                'conf': pred_conf.cpu().numpy(),
                'pred_cls': pred_cls.cpu().numpy(),
                'target_cls': tcls,
                'target_img': np.full(nl, self.seen - 1) if nl else np.array([]),
            }
            
            
            self.det_metrics.update_stats(stats_update)
    
    def _process_batch(self, pred_bboxes: torch.Tensor, pred_cls: torch.Tensor, 
                       gt_bboxes: torch.Tensor, gt_cls: torch.Tensor) -> np.ndarray:
        """
        Compute true positives for predictions vs ground truth.
        
        Returns:
            correct: Array of shape (num_preds, 10) for 10 IoU thresholds
        """
        # Move everything to CPU for processing
        pred_bboxes = pred_bboxes.cpu()
        pred_cls = pred_cls.cpu()
        gt_bboxes = gt_bboxes.cpu()
        gt_cls = gt_cls.cpu()
        iouv = self.iouv.cpu()
        
        npred = pred_cls.shape[0]
        ngt = gt_cls.shape[0]
        
        correct = np.zeros((npred, self.niou), dtype=bool)
        
        if ngt == 0 or npred == 0:
            return correct
        
        # Compute IoU between all predictions and ground truths
        iou = box_iou(gt_bboxes, pred_bboxes)  # [ngt, npred]
        
        # Class match matrix
        correct_class = gt_cls[:, None] == pred_cls  # [ngt, npred]
        
        # For each IoU threshold
        for i, threshold in enumerate(iouv):
            # Find matches above threshold with correct class
            matches = torch.where((iou >= threshold) & correct_class)
            
            if matches[0].numel():
                # Create match array: [gt_idx, pred_idx, iou]
                match_array = torch.cat([
                    matches[0].unsqueeze(1).float(),
                    matches[1].unsqueeze(1).float(), 
                    iou[matches].unsqueeze(1)
                ], dim=1).numpy()
                
                if match_array.shape[0] > 1:
                    # Sort by IoU descending
                    match_array = match_array[match_array[:, 2].argsort()[::-1]]
                    # Keep only best match per prediction
                    _, unique_pred_idx = np.unique(match_array[:, 1], return_index=True)
                    match_array = match_array[unique_pred_idx]
                    # Keep only best match per ground truth
                    _, unique_gt_idx = np.unique(match_array[:, 0], return_index=True)
                    match_array = match_array[unique_gt_idx]
                
                # Mark matched predictions as correct
                correct[match_array[:, 1].astype(int), i] = True
        
        return correct
    
    def compute_metrics(self, plot: bool = True) -> dict:
        """
        Compute final metrics and optionally generate plots.
        
        Args:
            plot: Whether to generate and save plots
            
        Returns:
            Dictionary with precision, recall, mAP50, mAP50-95
        """
        # Check if we have any stats
        has_stats = any(len(v) > 0 for v in self.det_metrics.stats.values())
        
        if not has_stats:
            print("⚠️ No detection stats collected")
            return {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
        
        # Debug: print stats summary
        stats_summary = {k: len(v) for k, v in self.det_metrics.stats.items()}
        print(f"📊 Stats summary: {stats_summary}")
        
        # Process detection metrics (this generates plots if plot=True)
        try:
            self.det_metrics.process(save_dir=self.save_dir, plot=plot)
        except Exception as e:
            print(f"⚠️ Error processing metrics: {e}")
            import traceback
            traceback.print_exc()
            return {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
        
        # Get results
        results = self.det_metrics.mean_results()
        if len(results) >= 4:
            precision, recall, map50, map50_95 = results[:4]
        else:
            precision, recall, map50, map50_95 = 0, 0, 0, 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'mAP50': float(map50),
            'mAP50-95': float(map50_95)
        }
    
    def log_epoch(self, epoch: int, train_losses: dict, val_losses: dict, metrics: dict, lr: float):
        """
        Log epoch results to CSV.
        
        Args:
            epoch: Current epoch number
            train_losses: Dict with 'box', 'cls', 'dfl', 'spatial' keys
            val_losses: Dict with 'box', 'cls', 'dfl', 'spatial' keys
            metrics: Dict with 'precision', 'recall', 'mAP50', 'mAP50-95' keys
            lr: Current learning rate
        """
        row = [
            epoch,
            train_losses.get('box', 0),
            train_losses.get('cls', 0),
            train_losses.get('dfl', 0),
            train_losses.get('spatial', 0),
            val_losses.get('box', 0),
            val_losses.get('cls', 0),
            val_losses.get('dfl', 0),
            val_losses.get('spatial', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('mAP50', 0),
            metrics.get('mAP50-95', 0),
            lr
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.epoch_results.append(row)
    
    def finalize(self):
        """Generate final plots and save metrics."""
        # Plot confusion matrices (normalized and non-normalized)
        try:
            self.confusion_matrix.plot(normalize=True, save_dir=str(self.save_dir))
            self.confusion_matrix.plot(normalize=False, save_dir=str(self.save_dir))
        except Exception as e:
            print(f"⚠️ Could not generate confusion matrix: {e}")
        
        # Generate collective results plot
        try:
            self._plot_results()
            print(f"✅ Results plot generated: {self.save_dir / 'results.png'}")
        except Exception as e:
            print(f"⚠️ Could not generate results plot: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Metrics saved to {self.save_dir}")
        print(f"   - Results CSV: {self.csv_path}")
        print(f"   - Plots: {self.save_dir}/*.png")
    
    def _plot_results(self):
        """Generate comprehensive results plot with all training metrics."""
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d
        
        # Read CSV data
        data = {}
        with open(self.csv_path, 'r') as f:
            import csv
            reader = csv.reader(f)
            headers = next(reader)
            for header in headers:
                data[header] = []
            
            for row in reader:
                for i, value in enumerate(row):
                    try:
                        data[headers[i]].append(float(value))
                    except:
                        data[headers[i]].append(0)
        
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
        
        epochs = data['epoch']
        
        # Define plot layout: losses on left, metrics on right
        plot_configs = [
            ('train/box_loss', 'Train Box Loss'),
            ('train/cls_loss', 'Train Class Loss'),
            ('train/dfl_loss', 'Train DFL Loss'),
            ('train/spatial_loss', 'Train Spatial Loss'),
            ('val/box_loss', 'Val Box Loss'),
            ('val/cls_loss', 'Val Class Loss'),
            ('val/dfl_loss', 'Val DFL Loss'),
            ('val/spatial_loss', 'Val Spatial Loss'),
            ('metrics/precision(B)', 'Precision'),
            ('metrics/recall(B)', 'Recall'),
            ('metrics/mAP50(B)', 'mAP@0.5'),
            ('metrics/mAP50-95(B)', 'mAP@0.5:0.95'),
            ('lr/pg0', 'Learning Rate'),
        ]
        
        # Create figure with subplots
        n_plots = len(plot_configs)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, (key, title) in enumerate(plot_configs):
            ax = axes[idx]
            
            if key in data and len(data[key]) > 0:
                y = data[key]
                
                # Plot actual values
                ax.plot(epochs, y, marker='o', markersize=3, linewidth=1.5, label='Actual')
                
                # Plot smoothed curve if enough data points
                if len(y) > 3:
                    y_smooth = gaussian_filter1d(y, sigma=2)
                    ax.plot(epochs, y_smooth, linestyle='--', linewidth=2, alpha=0.7, label='Smooth')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')
        
        # Hide extra subplots
        for idx in range(len(plot_configs), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'results.png', dpi=200, bbox_inches='tight')
        plt.close()
