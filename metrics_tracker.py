import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix
import json

class MetricsTracker:
    def __init__(self, save_dir="runs", experiment_name=None, nc=None, class_names=None):
        """
        Initialize metrics tracker for YOLO training.
        
        Args:
            save_dir: Directory to save metrics and plots
            experiment_name: Name of the experiment (if None, uses timestamp)
            nc: Number of classes (for confusion matrix background class handling)
            class_names: List of class names
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.nc = nc
        self.class_names = class_names
        
        self.csv_path = self.save_dir / "results.csv"
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize CSV
        self.csv_headers = [
            'epoch', 'time',
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
        
        # Storage for metrics history
        self.history = {header: [] for header in self.csv_headers}
        
        # Storage for confusion matrix (per-epoch, replaced each epoch)
        self.all_predictions = []
        self.all_targets = []
        self.all_confidences = []
        
        # Storage for proper PR curve (stores TP, confidence, class for each detection)
        self.pr_curve_data = {
            'tp': [],           # True positive flags (1 or 0)
            'conf': [],         # Confidence scores
            'pred_cls': [],     # Predicted classes
            'n_gt_per_class': {}  # Number of ground truth per class
        }
        
        print(f"📊 Metrics tracker initialized at: {self.save_dir}")
    
    def update(self, epoch, elapsed_time, train_losses, val_losses, val_metrics, learning_rates):
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            elapsed_time: Total training time in seconds
            train_losses: dict with 'box', 'cls', 'dfl' losses from training
            val_losses: dict with 'box', 'cls', 'dfl' losses from validation
            val_metrics: dict with 'precision', 'recall', 'mAP50', 'mAP50-95'
            learning_rates: list of learning rates for each param group [pg0, pg1, pg2]
        """
        row = {
            'epoch': epoch + 1,  # 1-indexed
            'time': round(elapsed_time, 4),
            'train/box_loss': round(train_losses['box'], 5),
            'train/cls_loss': round(train_losses['cls'], 5),
            'train/dfl_loss': round(train_losses['dfl'], 5),
            'metrics/precision(B)': round(val_metrics.get('precision', 0), 5),
            'metrics/recall(B)': round(val_metrics.get('recall', 0), 5),
            'metrics/mAP50(B)': round(val_metrics.get('mAP50', 0), 5),
            'metrics/mAP50-95(B)': round(val_metrics.get('mAP50-95', 0), 5),
            'val/box_loss': round(val_losses['box'], 5),
            'val/cls_loss': round(val_losses['cls'], 5),
            'val/dfl_loss': round(val_losses['dfl'], 5),
            'lr/pg0': learning_rates[0],
            'lr/pg1': learning_rates[1],
            'lr/pg2': learning_rates[2]
        }
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(row)
        
        # Store in history
        for key, value in row.items():
            self.history[key].append(value)
    
    def add_predictions(self, predictions, targets, confidences=None):
        """
        Set predictions and targets for confusion matrix calculation.
        This replaces any existing data (only latest epoch is used).
        
        Args:
            predictions: numpy array or list of predicted class indices
            targets: numpy array or list of ground truth class indices
            confidences: numpy array or list of confidence scores (optional)
        """
        # Replace instead of extend - we only want the latest epoch's data
        self.all_predictions = list(predictions)
        self.all_targets = list(targets)
        if confidences is not None:
            self.all_confidences = list(confidences)
        else:
            self.all_confidences = []
    
    def set_pr_curve_data(self, tp, conf, pred_cls, n_gt_per_class):
        """
        Set data for proper PR curve calculation.
        This replaces any existing data (only latest epoch is used).
        
        Args:
            tp: list of true positive flags (1=TP, 0=FP) for each detection
            conf: list of confidence scores for each detection
            pred_cls: list of predicted class indices for each detection
            n_gt_per_class: dict mapping class index to number of ground truth objects
        """
        self.pr_curve_data = {
            'tp': list(tp),
            'conf': list(conf),
            'pred_cls': list(pred_cls),
            'n_gt_per_class': dict(n_gt_per_class)
        }
    
    def plot_metrics(self):
        """Generate all metric plots."""
        if len(self.history['epoch']) < 2:
            print("⚠️ Not enough data to plot (need at least 2 epochs)")
            return
        
        self._plot_losses()
        self._plot_metrics()
        self._plot_learning_rate()
        self._plot_pr_curve()
        self._plot_training_pr_progress()
        self._plot_f1_curve()
        self._plot_p_curve()
        self._plot_r_curve()
        self._plot_confusion_matrix(class_names=self.class_names, nc=self.nc)
        
        print(f"✅ All plots saved to: {self.plots_dir}")
    
    def _plot_losses(self):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = self.history['epoch']
        
        # Box loss
        axes[0].plot(epochs, self.history['train/box_loss'], label='Train', marker='o', markersize=3)
        axes[0].plot(epochs, self.history['val/box_loss'], label='Val', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Box Loss')
        axes[0].set_title('Box Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Class loss
        axes[1].plot(epochs, self.history['train/cls_loss'], label='Train', marker='o', markersize=3)
        axes[1].plot(epochs, self.history['val/cls_loss'], label='Val', marker='s', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Class Loss')
        axes[1].set_title('Classification Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # DFL loss
        axes[2].plot(epochs, self.history['train/dfl_loss'], label='Train', marker='o', markersize=3)
        axes[2].plot(epochs, self.history['val/dfl_loss'], label='Val', marker='s', markersize=3)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('DFL Loss')
        axes[2].set_title('DFL Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'losses.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics(self):
        """Plot precision, recall, and mAP metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = self.history['epoch']
        
        # Precision
        axes[0, 0].plot(epochs, self.history['metrics/precision(B)'], marker='o', markersize=3, color='blue')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision (Box)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # Recall
        axes[0, 1].plot(epochs, self.history['metrics/recall(B)'], marker='s', markersize=3, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall (Box)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # mAP50
        axes[1, 0].plot(epochs, self.history['metrics/mAP50(B)'], marker='^', markersize=3, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP@0.5')
        axes[1, 0].set_title('mAP@0.5 (Box)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])
        
        # mAP50-95
        axes[1, 1].plot(epochs, self.history['metrics/mAP50-95(B)'], marker='d', markersize=3, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP@0.5:0.95')
        axes[1, 1].set_title('mAP@0.5:0.95 (Box)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_rate(self):
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = self.history['epoch']
        
        ax.plot(epochs, self.history['lr/pg0'], label='pg0 (weights with decay)', marker='o', markersize=3)
        ax.plot(epochs, self.history['lr/pg1'], label='pg1 (batch norm)', marker='s', markersize=3)
        ax.plot(epochs, self.history['lr/pg2'], label='pg2 (biases)', marker='^', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'learning_rate.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self):
        """Plot proper Precision-Recall curve by varying confidence threshold.
        
        This creates a true PR curve where each point represents the precision/recall
        at a specific confidence threshold.
        """
        tp = np.array(self.pr_curve_data['tp'])
        conf = np.array(self.pr_curve_data['conf'])
        pred_cls = np.array(self.pr_curve_data['pred_cls'])
        n_gt_per_class = self.pr_curve_data['n_gt_per_class']
        
        if len(tp) == 0:
            print("⚠️ No PR curve data available")
            return
        
        # Get unique classes
        classes = sorted(set(pred_cls.tolist()) | set(n_gt_per_class.keys()))
        nc = len(classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, nc))
        
        all_ap = []
        
        for c_idx, c in enumerate(classes):
            if c not in n_gt_per_class or n_gt_per_class[c] == 0:
                continue
            
            # Get predictions for this class
            class_mask = pred_cls == c
            if not class_mask.any():
                all_ap.append(0.0)
                continue
            
            tp_c = tp[class_mask]
            conf_c = conf[class_mask]
            n_gt = n_gt_per_class[c]
            
            # Sort by confidence (descending)
            sorted_idx = np.argsort(-conf_c)
            tp_c = tp_c[sorted_idx]
            conf_c = conf_c[sorted_idx]
            
            # Compute cumulative TP and FP
            tp_cumsum = np.cumsum(tp_c)
            fp_cumsum = np.cumsum(1 - tp_c)
            
            # Compute precision and recall at each threshold
            precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            recall_curve = tp_cumsum / (n_gt + 1e-16)
            
            # Prepend (0, 1) point for proper curve
            recall_curve = np.concatenate([[0], recall_curve])
            precision_curve = np.concatenate([[1], precision_curve])
            
            # Compute AP using 101-point interpolation
            x = np.linspace(0, 1, 101)
            # Make precision monotonically decreasing
            for i in range(len(precision_curve) - 2, -1, -1):
                precision_curve[i] = max(precision_curve[i], precision_curve[i + 1])
            ap = np.trapz(np.interp(x, recall_curve, precision_curve), x)
            all_ap.append(ap)
            
            # Plot curve for this class
            ax.plot(recall_curve, precision_curve, color=colors[c_idx], 
                   linewidth=2, label=f'Class {c} (AP={ap:.3f})')
        
        # Plot mean PR curve (average precision)
        mean_ap = np.mean(all_ap) if all_ap else 0.0
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve (mAP@0.5 = {mean_ap:.3f})', fontsize=14)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'BoxPR_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_training_pr_progress(self):
        """Plot how precision and recall evolve during training (not a true PR curve)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = self.history['epoch']
        
        ax.plot(epochs, self.history['metrics/precision(B)'], marker='o', markersize=3, 
                linewidth=2, label='Precision', color='blue')
        ax.plot(epochs, self.history['metrics/recall(B)'], marker='s', markersize=3, 
                linewidth=2, label='Recall', color='green')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Precision & Recall over Training')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'PR_training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_f1_curve(self):
        """Plot F1 score over epochs."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        precision = np.array(self.history['metrics/precision(B)'])
        recall = np.array(self.history['metrics/recall(B)'])
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        epochs = self.history['epoch']
        
        ax.plot(epochs, f1, marker='o', markersize=3, linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score over Training')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Mark best F1
        best_idx = np.argmax(f1)
        ax.axvline(x=epochs[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best F1 @ epoch {epochs[best_idx]}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'BoxF1_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_p_curve(self):
        """Plot Precision over epochs."""
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = self.history['epoch']
        precision = self.history['metrics/precision(B)']
        
        ax.plot(epochs, precision, marker='o', markersize=3, linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.set_title('Precision over Training (Box)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Mark best precision
        best_idx = np.argmax(precision)
        ax.axvline(x=epochs[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best @ epoch {epochs[best_idx]}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'BoxP_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_r_curve(self):
        """Plot Recall over epochs."""
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = self.history['epoch']
        recall = self.history['metrics/recall(B)']
        
        ax.plot(epochs, recall, marker='s', markersize=3, linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.set_title('Recall over Training (Box)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Mark best recall
        best_idx = np.argmax(recall)
        ax.axvline(x=epochs[best_idx], color='r', linestyle='--', alpha=0.5, label=f'Best @ epoch {epochs[best_idx]}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'BoxR_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, class_names=None, nc=None):
        """Plot confusion matrix (both raw and normalized).
        
        Args:
            class_names: List of class names. If None, uses generic names.
            nc: Number of actual classes (excludes background). If provided,
                class nc is treated as background class.
        """
        if len(self.all_predictions) == 0 or len(self.all_targets) == 0:
            print("⚠️ No predictions available for confusion matrix")
            return
        
        # Get unique classes
        all_classes = sorted(set(self.all_targets) | set(self.all_predictions))
        
        # Determine if we have background class (typically the highest class index)
        if nc is not None:
            # nc is the number of actual classes, so index nc is background
            if class_names is None:
                class_names = [f"Class {i}" for i in range(nc)] + ["Background"]
            else:
                class_names = list(class_names) + ["Background"]
        else:
            if class_names is None:
                class_names = [f"Class {i}" for i in all_classes]
        
        # Compute confusion matrix
        cm = confusion_matrix(self.all_targets, self.all_predictions, labels=all_classes)
        
        # Plot raw confusion matrix
        plt.figure()  # Create fresh figure
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                    yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close('all')  # Close all figures
        
        # Plot normalized confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure()  # Create fresh figure
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    vmin=0, vmax=1, cbar_kws={'label': 'Normalized Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
        plt.close('all')  # Close all figures
    
    def save_best_metrics(self):
        """Save summary of best metrics achieved."""
        if len(self.history['epoch']) == 0:
            return
        
        summary = {
            'best_mAP50': float(max(self.history['metrics/mAP50(B)'])),
            'best_mAP50_epoch': int(self.history['epoch'][np.argmax(self.history['metrics/mAP50(B)'])]),
            'best_mAP50-95': float(max(self.history['metrics/mAP50-95(B)'])),
            'best_mAP50-95_epoch': int(self.history['epoch'][np.argmax(self.history['metrics/mAP50-95(B)'])]),
            'best_precision': float(max(self.history['metrics/precision(B)'])),
            'best_precision_epoch': int(self.history['epoch'][np.argmax(self.history['metrics/precision(B)'])]),
            'best_recall': float(max(self.history['metrics/recall(B)'])),
            'best_recall_epoch': int(self.history['epoch'][np.argmax(self.history['metrics/recall(B)'])]),
            'final_train_loss': float(self.history['train/box_loss'][-1] + 
                                     self.history['train/cls_loss'][-1] + 
                                     self.history['train/dfl_loss'][-1]),
            'final_val_loss': float(self.history['val/box_loss'][-1] + 
                                   self.history['val/cls_loss'][-1] + 
                                   self.history['val/dfl_loss'][-1])
        }
        
        with open(self.save_dir / 'best_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print("📈 TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"Best mAP@0.5: {summary['best_mAP50']:.4f} (epoch {summary['best_mAP50_epoch']})")
        print(f"Best mAP@0.5:0.95: {summary['best_mAP50-95']:.4f} (epoch {summary['best_mAP50-95_epoch']})")
        print(f"Best Precision: {summary['best_precision']:.4f} (epoch {summary['best_precision_epoch']})")
        print(f"Best Recall: {summary['best_recall']:.4f} (epoch {summary['best_recall_epoch']})")
        print(f"{'='*50}\n")
