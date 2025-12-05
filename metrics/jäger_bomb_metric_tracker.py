from pathlib import Path
import traceback
import click
from metrics.metrics import Metrics
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
import torch 
from ultralytics.utils.nms import non_max_suppression
import numpy as np
from utils.echo import log_warning, log_info, log_error

from utils.utils import xywh_to_xyxy

class JägerBombMetricTracker:
    """
    Metric tracker for Jäger Bomb training.
    """
    def __init__(self, names : dict ,save_dir : Path):
        self.save_dir = save_dir
        self.names = names
        self.nc = len(names)
        # IoU thresholds for mAP calculation
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel() 

        self.reset()

    def reset(self):
        self.det_metrics = DetMetrics(names=self.names)
        self.confusion_matrix = ConfusionMatrix(names=self.names, task="detect")
        self.seen = 0

    def prepare_batch(self, y_val, img_w, img_h) -> dict:
        all_gt_cls = []
        all_gt_bboxes = []
        all_batch_idx = []
        
        for bi in range(y_val.shape[0]):
            # Get targets for this image (filter out padding)
            img_targets = y_val[bi][y_val[bi, :, 0] != -1]
            if img_targets.shape[0] > 0:
                cls = img_targets[:, 0]
                bboxes_xywh = img_targets[:, 1:]
                bboxes_xyxy = xywh_to_xyxy(bboxes_xywh, img_w, img_h)
                
                all_gt_cls.append(cls)
                all_gt_bboxes.append(bboxes_xyxy)
                all_batch_idx.extend([bi] * cls.shape[0])
        
        if len(all_gt_cls) > 0:
            return {
                'cls': torch.cat(all_gt_cls),
                'bboxes': torch.cat(all_gt_bboxes),
                'batch_idx': torch.tensor(all_batch_idx, device=y_val.device)
            }
        return {
                'cls': torch.empty(0, device=y_val.device),
                'bboxes': torch.empty((0, 4), device=y_val.device),
                'batch_idx': torch.empty(0, dtype=torch.long, device=y_val.device)
            }


    def _process_batch(self, pred_bboxes: torch.Tensor, pred_cls: torch.Tensor, 
                       gt_bboxes: torch.Tensor, gt_cls: torch.Tensor) -> np.ndarray:
        """
        Compute true positives for predictions vs ground truth.
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
        
        iou = box_iou(gt_bboxes, pred_bboxes) 
        
        # Class match matrix
        correct_class = gt_cls[:, None] == pred_cls  # [ngt, npred]
        
        for i, threshold in enumerate(iouv):
            matches = torch.where((iou >= threshold) & correct_class)
            
            if matches[0].numel():
                # match array: [gt_idx, pred_idx, iou]
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


    def update(self, preds: torch.Tensor, targets: dict, conf_thres: float = 0.001, iou_thres: float = 0.6):
        """
        Update metrics with predictions and ground truth.
        """
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
            
            pred_bboxes, pred_conf, pred_cls = self.get_predictions(pred)

            pred_dict = {'bboxes': pred_bboxes, 'conf': pred_conf, 'cls': pred_cls}
            gt_dict = {'bboxes': gt_bboxes, 'cls': gt_cls}

            self.confusion_matrix.process_batch(pred_dict, gt_dict)
            
            correct = self._process_batch(pred_bboxes, pred_cls, gt_bboxes, gt_cls)

            stats_update = {
                'tp': correct,
                'conf': pred_conf.cpu().numpy(),
                'pred_cls': pred_cls.cpu().numpy(),
                'target_cls': tcls,
                'target_img': np.full(nl, self.seen - 1) if nl else np.array([]),
            }
            
            self.det_metrics.update_stats(stats_update)

    def get_predictions(self, pred):
        pred_bboxes = pred[:, :4]
        pred_conf = pred[:, 4]
        pred_cls = pred[:, 5]
        return pred_bboxes,pred_conf,pred_cls

    def compute(self, plot: bool = True) -> Metrics:
        """
        Compute final metrics and optionally generate plots.
        """
        has_stats = any(len(v) > 0 for v in self.det_metrics.stats.values())
        
        if not has_stats:
            log_warning("No detection stats collected")
            return Metrics()
        
        stats_summary = {k: len(v) for k, v in self.det_metrics.stats.items()}
        log_info(f"Stats summary: {stats_summary}")
        
        # FOR SOME REASON ULTRALYTICS PUT PLOTTING INSIDE PROCESS METHOD??
        try:
            self.det_metrics.process(save_dir=self.save_dir, plot=plot)
        except Exception as e:
            log_error(f"Error processing metrics: {e}")
            traceback.print_exc()
            return Metrics()
        
        return Metrics(*self.det_metrics.mean_results())