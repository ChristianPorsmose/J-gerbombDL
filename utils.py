import numpy as np
from pathlib import Path
import torch

def convert_to_python_types(obj):
    """Recursively convert numpy/torch types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    else:
        return obj
    
def convert_to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj
    

def xywh_to_xyxy(bboxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Convert bboxes from normalized xywh to pixel xyxy format."""
    if bboxes.numel() == 0:
        return bboxes
    
    # bboxes: [N, 4] in format [x_center, y_center, width, height] normalized
    x_center = bboxes[:, 0] * img_w
    y_center = bboxes[:, 1] * img_h
    width = bboxes[:, 2] * img_w
    height = bboxes[:, 3] * img_h
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)