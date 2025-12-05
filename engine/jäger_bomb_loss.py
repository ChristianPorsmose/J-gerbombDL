import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

SHOT_IDX = 0
CUP_IDX = 1

class JägerBombLoss(v8DetectionLoss):
    def __init__(self, model : torch.nn.Module, tal_topk=10, lamda_rate=0.001): 
        super().__init__(model, tal_topk)
        self.lamda_rate = lamda_rate

    def __call__(self, preds, batch):
        """
        Compute loss vector: [box, cls, dfl, containment].
        """
        loss, detached_losses = super().__call__(preds, batch)

        containment_loss = self._calculate_containment_loss_assigned(
            self.pred_bboxes * self.stride_tensor,
            self.target_bboxes * self.stride_tensor,
            self.target_scores,
            self.fg_mask
        ) * self.lamda_rate

        loss_vector = torch.zeros(4, device=self.device)
        loss_vector[:3] = loss
        loss_vector[3] = containment_loss
        detached_vector = torch.zeros_like(loss_vector)
        detached_vector[:3] = detached_losses
        detached_vector[3] = containment_loss.detach()
        return loss_vector, detached_vector

    def _box_intersection(self, boxes1, boxes2):
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.chunk(4, dim=1)
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.chunk(4, dim=1)

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)

        return (inter_w * inter_h).squeeze()
    
    def _calculate_containment_loss_assigned(self, pred_bboxes, target_bboxes, target_scores, fg_mask):
        device = pred_bboxes.device
        total = torch.tensor(0.0, device=device)

        # Select positives
        if not fg_mask.any():
            return total

        # Indices of positives per batch element
        # fg_mask: (B, N) -> per-image masks
        B = fg_mask.shape[0]
        for b in range(B):
            pos = fg_mask[b]
            if not pos.any():
                continue

            # Predicted boxes at positives
            pos_pred_boxes = pred_bboxes[b, pos]  # (M, 4)

            # Class scores at positives
            pos_scores = target_scores[b, pos]     # (M, C)
            if pos_scores.numel() == 0:
                continue

            cls_ids = pos_scores.argmax(dim=1)     # (M,)

            shot_mask = cls_ids == SHOT_IDX
            cup_mask = cls_ids == CUP_IDX

            shot_boxes = pos_pred_boxes[shot_mask]
            cup_boxes = pos_pred_boxes[cup_mask]

            shot_nearest_cup_loss = self._nearest_loss(shot_boxes, cup_boxes)
            cup_nearest_shot_loss = self._nearest_loss(cup_boxes, shot_boxes)

            total += (shot_nearest_cup_loss + cup_nearest_shot_loss) / 2.0

        return total / B

    def _nearest_loss(self, primary_boxes, candidate_boxes):
        if primary_boxes.shape[0] == 0 or candidate_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=primary_boxes.device)
        
        centers1 = (primary_boxes[:, :2] + primary_boxes[:, 2:]) / 2
        centers2 = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) / 2
        dists = torch.cdist(centers1, centers2, p=2)

        _, nearest_idx = dists.min(dim=1)
        nearest_boxes = candidate_boxes[nearest_idx]
        
        inter = self._box_intersection(primary_boxes, nearest_boxes)
        area = (primary_boxes[:, 2] - primary_boxes[:, 0]) * (primary_boxes[:, 3] - primary_boxes[:, 1])
        return (1.0 - (inter / (area + 1e-6))).clamp(min=0).mean()
