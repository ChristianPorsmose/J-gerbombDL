import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

SHOT_IDX = 0
CUP_IDX = 1

class JägerBombLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, lamda_rate=0.001): #proportion=0.5):
        super().__init__(model, tal_topk)
        self.lamda_rate = lamda_rate
        #self.proportion = proportion
        # Define your class indices here

    def __call__(self, preds, batch):
        """
        Compute loss vector: [box, cls, dfl, containment].
        Returns:
            loss_vector: (4,) tensor, per-component loss
            detached_losses: (4,) tensor, detached for logging
        """
        loss, detached_losses = super().__call__(preds, batch)

        containment_loss = self._calculate_containment_loss_assigned(
            self.pred_bboxes * self.stride_tensor,
            self.target_bboxes * self.stride_tensor,
            self.target_scores,
            self.fg_mask
        ) * self.lamda_rate

        loss_vector = torch.zeros(4, device=self.device)
        loss_vector[:3] = loss  # box, cls, dfl
        loss_vector[3] = containment_loss
        detached_vector = torch.zeros_like(loss_vector)
        detached_vector[:3] = detached_losses
        detached_vector[3] = containment_loss.detach()
        return loss_vector, detached_vector

    def _box_intersection(self, boxes1, boxes2):
        """
        Vectorized intersection calculation.
        boxes1: (N, 4) xyxy
        boxes2: (N, 4) xyxy
        Returns: (N,) intersection area
        """
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
        """
        Compute containment using assigned positives:
        - Use fg_mask anchors
        - Use target_scores to get class labels
        """
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
            # Target boxes at positives (closest GT assigned)
            pos_tgt_boxes = target_bboxes[b, pos]  # (M, 4)
            # Class scores at positives
            pos_scores = target_scores[b, pos]     # (M, C)
            if pos_scores.numel() == 0:
                continue

            # Class ids from target scores (argmax over classes with non-zero score)
            cls_ids = pos_scores.argmax(dim=1)     # (M,)

            # Split shots vs cups
            shot_mask = cls_ids == SHOT_IDX
            cup_mask = cls_ids == CUP_IDX

            shot_boxes = pos_pred_boxes[shot_mask]
            cup_boxes = pos_pred_boxes[cup_mask]

            if shot_boxes.shape[0] == 0 or cup_boxes.shape[0] == 0:
                continue

            # Centers
            shot_centers = (shot_boxes[:, :2] + shot_boxes[:, 2:]) / 2
            cup_centers = (cup_boxes[:, :2] + cup_boxes[:, 2:]) / 2

            dists = torch.cdist(shot_centers, cup_centers, p=2)

            # Shot -> nearest cup
            _, nearest_cup_idx = dists.min(dim=1)
            nearest_cups = cup_boxes[nearest_cup_idx]
            inter1 = self._box_intersection(shot_boxes, nearest_cups)
            shot_area = (shot_boxes[:, 2] - shot_boxes[:, 0]) * (shot_boxes[:, 3] - shot_boxes[:, 1])
            loss1 = (1.0 - (inter1 / (shot_area + 1e-6))).clamp(min=0).mean()

            # Cup -> nearest shot
            _, nearest_shot_idx = dists.min(dim=0)
            nearest_shots = shot_boxes[nearest_shot_idx]
            inter2 = self._box_intersection(nearest_shots, cup_boxes)
            shot_area2 = (nearest_shots[:, 2] - nearest_shots[:, 0]) * (nearest_shots[:, 3] - nearest_shots[:, 1])
            loss2 = (1.0 - (inter2 / (shot_area2 + 1e-6))).clamp(min=0).mean()

            total += (loss1 + loss2) / 2.0

        return total / B