from configs import TrainingConfig, SaveConfig
import torch
import numpy as np
import time
from metrics_tracker import MetricsTracker
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using 101-point interpolation (COCO style).
    
    Args:
        recall: numpy array of recall values
        precision: numpy array of precision values
    
    Returns:
        ap: float, average precision
    """
    # Append sentinel values at the beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing (from right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 101-point interpolation (COCO style)
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    
    return ap


def match_predictions_to_gt(pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using IoU.
    
    Args:
        pred_boxes: tensor [N, 4] in xyxy format
        pred_classes: tensor [N]
        pred_confs: tensor [N]
        gt_boxes: tensor [M, 4] in xyxy format  
        gt_classes: tensor [M]
        iou_threshold: float, IoU threshold for matching
    
    Returns:
        tp: numpy array of true positives (1) and false positives (0)
        conf: numpy array of confidence scores
        pred_cls: numpy array of predicted classes
        matched_gt_indices: numpy array of matched GT indices (-1 if no match)
    """
    if len(pred_boxes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    if len(gt_boxes) == 0:
        # All predictions are false positives
        matched_gt_indices = np.full(len(pred_boxes), -1, dtype=np.int32)
        return np.zeros(len(pred_boxes)), pred_confs.cpu().numpy(), pred_classes.cpu().numpy(), matched_gt_indices
    
    # Compute IoU between all predictions and ground truth
    ious = box_iou(pred_boxes, gt_boxes)
    
    # For each prediction, find the best matching gt (if any)
    tp = np.zeros(len(pred_boxes))
    matched_gt_indices = np.full(len(pred_boxes), -1, dtype=np.int32)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    # Sort predictions by confidence (already sorted by NMS but ensure)
    sorted_idx = torch.argsort(pred_confs, descending=True)
    
    for i in sorted_idx:
        pred_cls = pred_classes[i].item()
        best_iou = 0
        best_gt_idx = -1
        
        for j in range(len(gt_boxes)):
            if gt_matched[j]:
                continue
            if gt_classes[j].item() != pred_cls:
                continue
            if ious[i, j] > best_iou:
                best_iou = ious[i, j].item()
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
            matched_gt_indices[i] = best_gt_idx
    
    return tp, pred_confs.cpu().numpy(), pred_classes.cpu().numpy(), matched_gt_indices


class JägerBombTrainer:
    def __init__(self, cfg: TrainingConfig, save_cfg : SaveConfig, experiment_name=None):
        self.cfg = cfg
        self.save_cfg = save_cfg
        
        # Get number of classes from model
        nc = getattr(self.cfg.model, 'nc', 2)
        self.nc = nc
        
        # Try to get class names from model
        class_names = getattr(self.cfg.model, 'names', None)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(nc)]
        elif isinstance(class_names, dict):
            class_names = [class_names.get(i, f"Class {i}") for i in range(nc)]
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            save_dir="runs", 
            experiment_name=experiment_name,
            nc=nc,
            class_names=class_names
        )
        
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
        
        # Storage for training losses per epoch
        self.train_box_losses = []
        self.train_cls_losses = []
        self.train_dfl_losses = []
        
    @torch.no_grad()
    def _validate(self, epoch):
        # Use EMA model for validation if enabled
        model_to_validate = self.ema.ema if self.ema else self.cfg.model
        model_to_validate.eval() 
        val_box = 0
        val_cls = 0
        val_dfl = 0
        count = 0
        
        # Number of classes
        nc = self.nc
        
        # Storage for metrics computation
        # For each class: lists of (tp, conf) tuples
        stats = {
            'tp': [],           # True positive flags per detection
            'conf': [],         # Confidence per detection  
            'pred_cls': [],     # Predicted class per detection
            'target_cls': [],   # All ground truth classes
        }
        
        # Storage for mAP50-95 computation (need to store boxes for re-matching)
        all_detections = []  # List of dicts: {'boxes': tensor, 'classes': tensor, 'confs': tensor}
        all_ground_truths = []  # List of dicts: {'boxes': tensor, 'classes': tensor}
        
        # Collect predictions for confusion matrix
        all_pred_classes = []
        all_target_classes = []
        
        # Image size (assuming square, get from first batch)
        img_size = 640
        
        for X_val, y_val in self.cfg.val_dataloader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)
            batch_size = X_val.shape[0]
            img_size = X_val.shape[2]  # Get actual image size

            batch = self._prepare_batch_dict(X_val, y_val)

            # Forward pass - model returns (inference_output, loss_output) tuple
            pred_val = model_to_validate(X_val)

            # Compute loss
            _, last_loss = self.cfg.loss_fn(pred_val, batch)
            box_loss, cls_loss, dfl_loss = last_loss.cpu().numpy().round(3)

            val_box += box_loss
            val_cls += cls_loss
            val_dfl += dfl_loss
            count += 1
            
            # Decode predictions using NMS
            # pred_val is either a tensor or tuple - NMS handles both
            detections = non_max_suppression(
                pred_val,
                conf_thres=0.001,  # Low threshold to get more predictions for mAP
                iou_thres=0.6,
                nc=nc,
                max_det=300
            )
            
            # Process each image in batch
            for batch_idx_i in range(batch_size):
                # Get detections for this image: [N, 6] = [x1, y1, x2, y2, conf, cls]
                det = detections[batch_idx_i]
                
                # Get ground truth for this image
                gt_mask = (batch['batch_idx'] == batch_idx_i) if len(batch['batch_idx']) > 0 else torch.zeros(0, dtype=torch.bool)
                gt_cls = batch['cls'][gt_mask] if len(batch['cls']) > 0 else torch.tensor([])
                gt_bboxes = batch['bboxes'][gt_mask] if len(batch['bboxes']) > 0 else torch.tensor([])
                
                # Convert gt bboxes from xywh normalized to xyxy pixel coords
                if len(gt_bboxes) > 0:
                    gt_bboxes_xyxy = xywh2xyxy(gt_bboxes.clone())
                    gt_bboxes_xyxy[:, [0, 2]] *= img_size  # x coordinates
                    gt_bboxes_xyxy[:, [1, 3]] *= img_size  # y coordinates
                else:
                    gt_bboxes_xyxy = torch.tensor([]).to(self.cfg.device)
                
                # Add target classes for stats (used in mAP calculation)
                if len(gt_cls) > 0:
                    stats['target_cls'].extend(gt_cls.cpu().numpy().astype(int).tolist())
                
                # Store detections and ground truths for mAP50-95 computation
                all_ground_truths.append({
                    'boxes': gt_bboxes_xyxy.cpu() if len(gt_bboxes_xyxy) > 0 else torch.tensor([]),
                    'classes': gt_cls.cpu().int() if len(gt_cls) > 0 else torch.tensor([])
                })
                
                if len(det) == 0:
                    # No predictions - store empty detection
                    all_detections.append({
                        'boxes': torch.tensor([]),
                        'classes': torch.tensor([]),
                        'confs': torch.tensor([])
                    })
                    # All GT are false negatives (missed detections)
                    if len(gt_cls) > 0:
                        gt_classes_np = gt_cls.cpu().numpy().astype(int)
                        for gt_c in gt_classes_np:
                            all_target_classes.append(int(gt_c))
                            all_pred_classes.append(nc)  # predicted as background
                    continue
                
                # Extract prediction components
                pred_boxes = det[:, :4]  # xyxy format already
                pred_confs = det[:, 4]
                pred_classes = det[:, 5].int()
                
                # Store for mAP50-95
                all_detections.append({
                    'boxes': pred_boxes.cpu(),
                    'classes': pred_classes.cpu(),
                    'confs': pred_confs.cpu()
                })
                
                # We'll collect confusion matrix pairs after matching
                
                # Match predictions to ground truth for mAP calculation
                if len(gt_bboxes_xyxy) > 0:
                    # Compute IoU between predictions and ground truth
                    ious = box_iou(pred_boxes, gt_bboxes_xyxy)
                    
                    # For mAP50, use IoU threshold 0.5
                    tp, conf, pred_cls, matched_gt_indices = match_predictions_to_gt(
                        pred_boxes, pred_classes, pred_confs, 
                        gt_bboxes_xyxy, gt_cls.int(), 
                        iou_threshold=0.5
                    )
                    
                    stats['tp'].extend(tp.tolist())
                    stats['conf'].extend(conf.tolist())
                    stats['pred_cls'].extend(pred_cls.tolist())
                    
                    # Collect confusion matrix pairs
                    # Use background_class = nc (number of classes) for unmatched
                    gt_classes_np = gt_cls.cpu().numpy().astype(int)
                    pred_classes_np = pred_classes.cpu().numpy().astype(int)
                    pred_confs_np = pred_confs.cpu().numpy()
                    
                    # Track which GT boxes are matched
                    gt_matched = set()
                    
                    # For each prediction with conf > 0.25
                    for i, (pred_c, p_conf, is_tp) in enumerate(zip(pred_classes_np, pred_confs_np, tp)):
                        if p_conf < 0.25:
                            continue
                        if is_tp == 1 and matched_gt_indices[i] >= 0:
                            # True positive: matched to GT
                            gt_idx = int(matched_gt_indices[i])
                            gt_matched.add(gt_idx)
                            all_target_classes.append(int(gt_classes_np[gt_idx]))
                            all_pred_classes.append(int(pred_c))
                        else:
                            # False positive: predicted but no matching GT
                            all_target_classes.append(nc)  # background class
                            all_pred_classes.append(int(pred_c))
                    
                    # False negatives: GT boxes not matched by any high-conf prediction
                    for gt_idx, gt_c in enumerate(gt_classes_np):
                        if gt_idx not in gt_matched:
                            all_target_classes.append(int(gt_c))
                            all_pred_classes.append(nc)  # predicted as background
                else:
                    # All predictions are false positives (no GT)
                    stats['tp'].extend([0] * len(det))
                    stats['conf'].extend(pred_confs.cpu().numpy().tolist())
                    stats['pred_cls'].extend(pred_classes.cpu().numpy().tolist())
                    
                    # Add FP to confusion matrix
                    pred_classes_np = pred_classes.cpu().numpy().astype(int)
                    pred_confs_np = pred_confs.cpu().numpy()
                    for pred_c, p_conf in zip(pred_classes_np, pred_confs_np):
                        if p_conf >= 0.25:
                            all_target_classes.append(nc)  # background
                            all_pred_classes.append(int(pred_c))

        avg_box = val_box/count if count > 0 else 0
        avg_cls = val_cls/count if count > 0 else 0
        avg_dfl = val_dfl/count if count > 0 else 0
        
        # Compute metrics (including proper mAP50-95)
        precision, recall, mAP50, mAP50_95 = self._compute_metrics(stats, nc, img_size, all_detections, all_ground_truths)
        
        print(
            f"VALIDATION — Epoch {epoch}: "
            f"Box: {avg_box:.4f}, "
            f"Cls: {avg_cls:.4f}, "
            f"DFL: {avg_dfl:.4f}, "
            f"Total: {(avg_box+avg_cls+avg_dfl):.4f} | "
            f"P: {precision:.4f}, R: {recall:.4f}, "
            f"mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}"
        )
        
        # Add predictions to metrics tracker for confusion matrix
        if len(all_pred_classes) > 0 and len(all_target_classes) > 0:
            self.metrics_tracker.add_predictions(all_pred_classes, all_target_classes)
        
        # Set PR curve data for proper PR curve plotting
        # Count number of ground truths per class
        n_gt_per_class = {}
        for c in range(nc):
            n_gt_per_class[c] = stats['target_cls'].count(c)
        
        self.metrics_tracker.set_pr_curve_data(
            tp=stats['tp'],
            conf=stats['conf'],
            pred_cls=stats['pred_cls'],
            n_gt_per_class=n_gt_per_class
        )
        
        return {
            'box': avg_box,
            'cls': avg_cls,
            'dfl': avg_dfl,
            'total': avg_box + avg_cls + avg_dfl
        }, {
            'precision': precision,
            'recall': recall,
            'mAP50': mAP50,
            'mAP50-95': mAP50_95
        }
    
    def _compute_metrics(self, stats, nc, img_size, all_detections, all_ground_truths):
        """
        Compute precision, recall, mAP50 and mAP50-95 from accumulated stats.
        
        Args:
            stats: dict with 'tp', 'conf', 'pred_cls', 'target_cls' lists
            nc: number of classes
            img_size: image size for scaling
            all_detections: list of detection dicts with boxes, classes, confs
            all_ground_truths: list of ground truth dicts with boxes, classes
        
        Returns:
            precision, recall, mAP50, mAP50-95
        """
        tp = np.array(stats['tp'])
        conf = np.array(stats['conf'])
        pred_cls = np.array(stats['pred_cls'])
        target_cls = np.array(stats['target_cls'])
        
        if len(tp) == 0 or len(target_cls) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Sort by confidence
        sorted_idx = np.argsort(-conf)
        tp = tp[sorted_idx]
        conf = conf[sorted_idx]
        pred_cls = pred_cls[sorted_idx]
        
        # Compute metrics per class
        ap50_per_class = []
        precision_per_class = []
        recall_per_class = []
        
        for c in range(nc):
            # Get predictions and targets for this class
            class_mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = class_mask.sum()
            
            if n_gt == 0 or n_pred == 0:
                ap50_per_class.append(0.0)
                continue
            
            # Compute cumulative TP and FP
            tp_class = tp[class_mask]
            conf_class = conf[class_mask]
            
            # Sort by confidence
            sorted_class_idx = np.argsort(-conf_class)
            tp_class = tp_class[sorted_class_idx]
            
            # Cumulative sum
            tp_cumsum = np.cumsum(tp_class)
            fp_cumsum = np.cumsum(1 - tp_class)
            
            # Precision and recall
            precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            recall_curve = tp_cumsum / (n_gt + 1e-16)
            
            # Compute AP using 101-point interpolation
            ap50 = compute_ap(recall_curve, precision_curve)
            ap50_per_class.append(ap50)
            
            # Store final precision and recall
            if len(precision_curve) > 0:
                precision_per_class.append(precision_curve[-1])
                recall_per_class.append(recall_curve[-1])
        
        mAP50 = np.mean(ap50_per_class) if ap50_per_class else 0.0
        
        # Compute proper mAP50-95 at all IoU thresholds
        mAP50_95 = self._compute_map50_95(all_detections, all_ground_truths, nc)
        
        # Average precision and recall across classes
        precision = np.mean(precision_per_class) if precision_per_class else 0.0
        recall = np.mean(recall_per_class) if recall_per_class else 0.0
        
        return precision, recall, mAP50, mAP50_95
    
    def _compute_map50_95(self, all_detections, all_ground_truths, nc):
        """
        Compute mAP at IoU thresholds from 0.5 to 0.95 with step 0.05.
        
        Args:
            all_detections: list of dicts with 'boxes', 'classes', 'confs'
            all_ground_truths: list of dicts with 'boxes', 'classes'
            nc: number of classes
        
        Returns:
            mAP averaged over IoU thresholds [0.5, 0.55, 0.6, ..., 0.95]
        """
        iou_thresholds = np.arange(0.5, 0.95 + 0.05, 0.05)  # [0.5, 0.55, ..., 0.95]
        ap_per_iou = []
        
        for iou_thresh in iou_thresholds:
            # Collect stats at this IoU threshold
            stats_at_iou = {
                'tp': [],
                'conf': [],
                'pred_cls': [],
                'target_cls': []
            }
            
            for det, gt in zip(all_detections, all_ground_truths):
                pred_boxes = det['boxes']
                pred_classes = det['classes']
                pred_confs = det['confs']
                gt_boxes = gt['boxes']
                gt_classes = gt['classes']
                
                # Add target classes
                if len(gt_classes) > 0:
                    stats_at_iou['target_cls'].extend(gt_classes.numpy().astype(int).tolist())
                
                if len(pred_boxes) == 0:
                    continue
                
                if len(gt_boxes) == 0:
                    # All predictions are FP
                    stats_at_iou['tp'].extend([0] * len(pred_boxes))
                    stats_at_iou['conf'].extend(pred_confs.numpy().tolist())
                    stats_at_iou['pred_cls'].extend(pred_classes.numpy().tolist())
                    continue
                
                # Match at this IoU threshold
                tp, conf, pred_cls, _ = match_predictions_to_gt(
                    pred_boxes, pred_classes, pred_confs,
                    gt_boxes, gt_classes,
                    iou_threshold=iou_thresh
                )
                
                stats_at_iou['tp'].extend(tp.tolist())
                stats_at_iou['conf'].extend(conf.tolist())
                stats_at_iou['pred_cls'].extend(pred_cls.tolist())
            
            # Compute AP at this IoU threshold
            tp = np.array(stats_at_iou['tp'])
            conf = np.array(stats_at_iou['conf'])
            pred_cls = np.array(stats_at_iou['pred_cls'])
            target_cls = np.array(stats_at_iou['target_cls'])
            
            if len(tp) == 0 or len(target_cls) == 0:
                ap_per_iou.append(0.0)
                continue
            
            # Sort by confidence
            sorted_idx = np.argsort(-conf)
            tp = tp[sorted_idx]
            pred_cls = pred_cls[sorted_idx]
            
            ap_per_class = []
            for c in range(nc):
                class_mask = pred_cls == c
                n_gt = (target_cls == c).sum()
                n_pred = class_mask.sum()
                
                if n_gt == 0 or n_pred == 0:
                    ap_per_class.append(0.0)
                    continue
                
                tp_class = tp[class_mask]
                tp_cumsum = np.cumsum(tp_class)
                fp_cumsum = np.cumsum(1 - tp_class)
                
                precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                recall_curve = tp_cumsum / (n_gt + 1e-16)
                
                ap = compute_ap(recall_curve, precision_curve)
                ap_per_class.append(ap)
            
            mAP_at_iou = np.mean(ap_per_class) if ap_per_class else 0.0
            ap_per_iou.append(mAP_at_iou)
        
        # Average over all IoU thresholds
        mAP50_95 = np.mean(ap_per_iou) if ap_per_iou else 0.0
        return mAP50_95

    def _prepare_batch_dict(self, images: torch.Tensor, targets: torch.Tensor) -> dict:
        valid_mask = targets[:, :, 0] != -1

        samples_with_valid_targets = valid_mask.any(dim=1)

        if not samples_with_valid_targets.any():
            return {"img": images, "batch_idx": torch.empty(0), "cls": torch.empty(0), "bboxes": torch.empty(0)}

        valid_targets = targets[valid_mask]

        target_lengths = [int(v.sum().item()) for v in valid_mask]
        batch_idx = torch.cat([torch.full((length,), i) for i, length in enumerate(target_lengths) if length > 0])

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
        start_time = time.time()
        
        # Warmup settings (like Ultralytics)
        warmup_epochs = 3.0
        nb = len(self.cfg.train_dataloader)
        nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations
        
        for epoch in range(self.cfg.epochs):
            self.cfg.model.train(True)
            
            # Reset epoch losses
            self.train_box_losses = []
            self.train_cls_losses = []
            self.train_dfl_losses = []
            
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
                
                box_loss, cls_loss, dfl_loss = last_loss.detach().cpu().numpy()
                loss = batch_loss.sum()
                
                # Store losses for this epoch
                self.train_box_losses.append(float(box_loss))
                self.train_cls_losses.append(float(cls_loss))
                self.train_dfl_losses.append(float(dfl_loss))
                
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

            val_losses, val_metrics = self._validate(epoch)
            
            # Step the learning rate scheduler (only after warmup)
            if epoch >= warmup_epochs:
                self.cfg.scheduler.step()
            
            # Get learning rates for all param groups
            lrs = [pg['lr'] for pg in self.cfg.optimizer.param_groups]
            print(f"Learning rate: {lrs[0]:.6f}")
            
            # Update metrics tracker
            elapsed_time = time.time() - start_time
            train_losses = {
                'box': float(np.mean(self.train_box_losses)) if self.train_box_losses else 0,
                'cls': float(np.mean(self.train_cls_losses)) if self.train_cls_losses else 0,
                'dfl': float(np.mean(self.train_dfl_losses)) if self.train_dfl_losses else 0
            }
            self.metrics_tracker.update(
                epoch=epoch,
                elapsed_time=elapsed_time,
                train_losses=train_losses,
                val_losses=val_losses,
                val_metrics=val_metrics,
                learning_rates=lrs
            )
            
            if(val_losses['total']<best_loss):
                self._save_model(epoch)
                best_loss = val_losses['total']
                count = 0
            else:
                count+=1
            
            # Generate plots every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.metrics_tracker.plot_metrics()
            
            if(count >= early_stoppage_count):
                print(f"Early stopping at epoch {epoch}")
                break
            #if (epoch + 1) % self.save_cfg.save_epoch_interval == 0:
            #    self._save_model(epoch)
        
        # Training complete - generate final plots and summary
        print("\n🎨 Generating final plots...")
        self.metrics_tracker.plot_metrics()
        self.metrics_tracker.save_best_metrics()
        print(f"✅ Results saved to: {self.metrics_tracker.save_dir}")