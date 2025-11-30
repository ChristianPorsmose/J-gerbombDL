from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
import torch 

class JägerBombLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk = 10, lamda_rate = 0.001):
        super().__init__(model, tal_topk)
        self.lamda_rate = lamda_rate

    
    def __call__(self, preds, batch):
        """Calculate loss including spatial consistency on predicted boxes."""
        # Get standard detection loss from parent class
        loss, loss_item = super().__call__(preds, batch)
        
        # Now compute spatial loss on predictions
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Preprocess ground truth targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predictions to bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Run TAL assigner to match predictions to ground truth
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # ===== SPATIAL CONSISTENCY LOSS ON PREDICTIONS =====
        radial_loss = 0.0
        
        # Convert pred_bboxes to pixel coordinates once (multiply by stride per anchor)
        pred_bboxes_pixels = (pred_bboxes * stride_tensor).detach()
        
        for i_img in range(batch_size):
            # Get predictions for this image that were matched (foreground)
            img_fg_mask = fg_mask[i_img]  # [h*w] boolean mask
            
            if not img_fg_mask.any():
                continue
            
            # Get predicted boxes and their assigned GT classes
            pred_boxes_matched = pred_bboxes_pixels[i_img][img_fg_mask]  # [N, 4] xyxy in pixels
            gt_idx_matched = target_gt_idx[i_img][img_fg_mask]  # [N] indices into gt_labels/gt_bboxes
            
            # Get the ground truth classes for matched predictions
            pred_classes = gt_labels[i_img][gt_idx_matched].squeeze(-1)  # [N]
            
            # Separate predicted shots and cups
            shot_mask = pred_classes == 0
            cup_mask = pred_classes == 1
            
            pred_shots = pred_boxes_matched[shot_mask]
            pred_cups = pred_boxes_matched[cup_mask]
            
            radial_loss += self._radial_containment_loss(pred_shots, pred_cups)

        # Average spatial loss over batch
        radial_loss = radial_loss / batch_size
        
        # Extend loss tensor from [box, cls, dfl] to [box, cls, dfl, spatial]
        weighted_spatial = self.lamda_rate * radial_loss * batch_size
        loss = torch.cat([loss, weighted_spatial.unsqueeze(0)])
        
        # Extend loss_item for logging
        loss_item = torch.cat([loss_item, radial_loss.unsqueeze(0)])

        return loss, loss_item
    

    def _radial_containment_loss(self, gt_shots, gt_cups):
        """
        Bi-directional spatial consistency loss:
        1. Each shot must be inside a cup (shot → cup)
        2. Each cup must contain a shot (cup → shot)
        """
        # FIX ME: den her skal lige tænkes over ( det skal være et og )
        if len(gt_shots) == 0 or len(gt_cups) == 0:
            if(len(gt_shots) != len(gt_cups)):
                return torch.tensor(10, device=gt_cups.device)
            return torch.tensor(0.0, device=gt_cups.device)

        # Compute centers and radii
        def center_and_radius(box):
            x1, y1, x2, y2 = box.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            rw = (x2 - x1) / 2
            rh = (y2 - y1) / 2
            r = torch.sqrt(rw * rh)  # geometric mean
            return cx, cy, r
        
        cx_s, cy_s, r_s = center_and_radius(gt_shots)
        cx_c, cy_c, r_c = center_and_radius(gt_cups)

        # Pairwise distances: shape [num_shots, num_cups]
        dist = torch.sqrt((cx_s[:, None] - cx_c[None, :])**2 +
                        (cy_s[:, None] - cy_c[None, :])**2)

        # ===== LOSS 1: Shot → Cup containment =====
        # Each shot should be inside its nearest cup
        nearest_cup_per_shot = dist.argmin(dim=1)
        d_shot_to_cup = dist[torch.arange(len(dist)), nearest_cup_per_shot]
        r_c_matched = r_c[nearest_cup_per_shot]
        
        # Violation: shot center is outside the cup's radius
        violation_shot = d_shot_to_cup + r_s - r_c_matched
        loss_shot = torch.relu(violation_shot).mean()
        
        # ===== LOSS 2: Cup → Shot containment =====
        # Each cup should contain at least one shot inside it
        nearest_shot_per_cup = dist.argmin(dim=0)  # For each cup, find nearest shot
        d_cup_to_shot = dist.T[torch.arange(len(dist.T)), nearest_shot_per_cup]
        r_s_matched = r_s[nearest_shot_per_cup]
        
        # Violation: nearest shot is outside the cup's radius
        # Cup should contain the shot, so shot must be within cup radius
        violation_cup = d_cup_to_shot + r_s_matched - r_c
        loss_cup = torch.relu(violation_cup).mean()
        
        # Combined loss
        total_loss = (loss_shot + loss_cup) / 2
        #print(f"Spatial loss → shot→cup: {loss_shot:.4f}, cup→shot: {loss_cup:.4f}, total: {total_loss:.4f}")
        return total_loss