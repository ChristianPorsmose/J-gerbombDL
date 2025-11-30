from ultralytics.utils.loss import v8DetectionLoss
import torch 

class JägerBombLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk = 10, lamda = 0.001):
        super().__init__(model, tal_topk)
        self.lamda = lamda

    
    def __call__(self, preds, batch):
        loss, loss_item = super().__call__(preds, batch)

        # Extract features to compute image size (same as parent class)
        feats = preds[1] if isinstance(preds, tuple) else preds
        dtype = batch["img"].dtype
        batch_size = batch["img"].shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Preprocess targets with scale_tensor
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        radial_loss = 0.0

        for i_img in range(batch_size):
            labels_i = gt_labels[i_img].squeeze(-1)
            boxes_i  = gt_bboxes[i_img]             
            valid = boxes_i.sum(1) > 0
            labels_i = labels_i[valid]
            boxes_i  = boxes_i[valid]
            shots_i = boxes_i[labels_i == 0]
            cups_i  = boxes_i[labels_i == 1] 
            radial_loss += self._radial_containment_loss(shots_i, cups_i)

        # Average spatial loss over batch
        radial_loss = radial_loss / batch_size
        
        # Extend loss tensor from [box, cls, dfl] to [box, cls, dfl, spatial]
        # loss is already [box*bs, cls*bs, dfl*bs], so add spatial*bs
        weighted_spatial = self.lamda * radial_loss * batch_size
        loss = torch.cat([loss, weighted_spatial.unsqueeze(0)])
        
        # Extend loss_item from [box, cls, dfl] to [box, cls, dfl, spatial] for logging
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