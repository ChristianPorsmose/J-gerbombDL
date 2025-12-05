

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import torch
import numpy as np
from ultralytics.utils.nms import non_max_suppression
from utils.echo import log

CONFIDENCE_THRESHOLD = 0.25
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def visualize_batch(images, batch_dict,save_dir, predictions=None, epoch=0, is_train=True, max_imgs=4):
    """
    Visualize a batch with ground truth boxes and optionally predictions.
    """
    save_dir = Path(save_dir) / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    split = "train" if is_train else "val"
    batch_size = min(images.shape[0], max_imgs)
    
    fig, axes = plt.subplots(1, batch_size, figsize=(5*batch_size, 5))
    if batch_size == 1:
        axes = [axes]
    colors_dict = {0: 'red', 1: 'blue'}
    labels = {0: 'shot', 1: 'cup'}
    
    for idx in range(batch_size):
        ax = axes[idx]
        
        # Convert image tensor to numpy for visualization
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        ax.imshow(img)
        ax.axis('off')
        
        h, w = img.shape[:2]
        
        img_mask = batch_dict['batch_idx'] == idx
        if img_mask.any():
            _draw_ground_truth_boxes(batch_dict, colors_dict, labels, ax, h, w, img_mask)
        
        if predictions is not None:
            # Process predictions
            _draw_predictions(predictions, colors_dict, labels, idx, ax)
        
        ax.set_title(f'Image {idx}', fontsize=10)
    
    handles, labels_list = axes[0].get_legend_handles_labels()
    if handles:
        # Remove duplicate labels
        by_label = dict(zip(labels_list, handles))
        fig.legend(by_label.values(), by_label.keys(), 
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
    
    plt.tight_layout()
    
    save_path = save_dir / f"epoch{epoch:03d}_{split}_batch.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"Saved {split} visualization → {save_path}")

def _draw_predictions(predictions, colors_dict, labels, idx, ax):
    pred_boxes = predictions[0][idx]
            
    if len(pred_boxes) > 0:
        # pred_boxes in format: [x1, y1, x2, y2, conf, cls], I.e. bounding boxes, confidence, and class
        for pred in pred_boxes:
            if len(pred) < 6:
                continue
            x1, y1, x2, y2, conf, cls = pred[:6]
            cls = int(cls)
                    
            if conf < CONFIDENCE_THRESHOLD: 
                continue
                    
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=colors_dict.get(cls, 'green'),
                                   facecolor='none', linestyle='--',
                                   label=f'Pred {labels.get(cls, "?")} {conf:.2f}')
            ax.add_patch(rect)

def _draw_ground_truth_boxes(batch_dict, colors_dict, labels, ax, h, w, img_mask):
    gt_classes = batch_dict['cls'][img_mask].cpu().numpy()
    gt_bboxes = batch_dict['bboxes'][img_mask].cpu().numpy()  # normalized xywh
    for cls, bbox in zip(gt_classes, gt_bboxes):
        # Convert normalized xywh to pixel xyxy
        x_center, y_center, width, height = bbox
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        box_w = width * w
        box_h = height * h
        cls = int(cls)
        rect = Rectangle((x1, y1), box_w, box_h, 
                               linewidth=2, edgecolor=colors_dict[cls], 
                               facecolor='none', linestyle='-',
                               label=f'GT {labels[cls]}')
        ax.add_patch(rect)
    
    
def visualize_predictions(epoch, model, data_loader, save_dir : Path):
    device = torch.get_default_device().type
    """Run model on validation set and visualize predictions."""
    model.eval()
    
    save_dir = save_dir / "visualizations" / f"epoch{epoch:03d}_predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    colors_dict = {0: RED, 1: BLUE}
    labels = {0: 'shot', 1: 'cup'}
    
    log(f"Generating prediction visualizations for epoch {epoch}...")
    
    with torch.no_grad():
        for batch_idx, (X_val, y_val) in enumerate(data_loader):
            X_val = X_val.to(device)
            
            preds = model(X_val)
            
            # Post-process predictions with NMS
            # preds is typically a tuple (inference_out, loss_out) or just inference_out
            if isinstance(preds, tuple):
                preds = preds[0]
            
            # Apply NMS to filter predictions
            predictions = non_max_suppression(preds, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=0.45, max_det=300)
            
            # Visualize each image in batch
            for img_idx in range(X_val.shape[0]):
                # Convert tensor to numpy image for plotting
                img = X_val[img_idx].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8).copy()
                # Draw predictions for this image
                if predictions and len(predictions) > img_idx:
                    dets = predictions[img_idx]  # [N, 6] tensor: x1, y1, x2, y2, conf, cls, same as before
                    
                    if dets is not None and len(dets) > 0:
                        for det in dets:
                            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cls = int(cls)
                            
                            _draw_box(colors_dict, img, x1, y1, x2, y2, cls)
                            
                            _draw_label(colors_dict, labels, img, x1, y1, conf, cls)
                
                save_path = save_dir / f"batch{batch_idx:03d}_img{img_idx:02d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Limit to first few batches to avoid too many images
            if batch_idx >= 2: 
                break

    log(f"Saved prediction visualizations → {save_dir}")

def _draw_box(colors_dict, img, x1, y1, x2, y2, cls):
    cv2.rectangle(img, (x1, y1), (x2, y2), colors_dict.get(cls, GREEN), 2)

def _draw_label(colors_dict, labels, img, x1, y1, conf, cls):
    label_text = f'{labels.get(cls, "?")} {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1-th-4), (x1+tw, y1), colors_dict.get(cls,GREEN), -1)
    cv2.putText(img, label_text, (x1, y1-2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)