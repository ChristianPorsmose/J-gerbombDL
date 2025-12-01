
from torchvision.transforms import v2 as T
import torch
from letter_box_transform import LetterBoxTransform


class YOLOCompose:
    """Custom compose that handles both images and bboxes."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LetterBoxTransform):
                if bboxes is not None and len(bboxes) > 0:
                    img, bboxes = t(img, bboxes)
                else:
                    img = t(img)
            elif isinstance(t, T.RandomHorizontalFlip) and bboxes is not None and len(bboxes) > 0:
                # Apply horizontal flip to both image and bboxes
                if torch.rand(1) < t.p:
                    img = T.functional.hflip(img)
                    # Flip bbox x-coordinates: x_center_new = 1 - x_center_old
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]  # flip x_center (column 1)
            elif isinstance(t, T.RandomVerticalFlip) and bboxes is not None and len(bboxes) > 0:
                # Apply vertical flip to both image and bboxes
                if torch.rand(1) < t.p:
                    img = T.functional.vflip(img)
                    # Flip bbox y-coordinates: y_center_new = 1 - y_center_old
                    bboxes[:, 2] = 1.0 - bboxes[:, 2]  # flip y_center (column 2)
            elif isinstance(t, T.RandomPerspective) and bboxes is not None and len(bboxes) > 0:
                # Apply perspective transform to both image and bboxes
                img, bboxes = self._apply_perspective_with_bboxes(img, bboxes, t)
            elif isinstance(t, T.RandomRotation) and bboxes is not None and len(bboxes) > 0:
                # Apply rotation to both image and bboxes
                img, bboxes = self._apply_rotation_with_bboxes(img, bboxes, t)
            else:
                # Regular transforms that only affect the image (color jitter, normalization, etc.)
                img = t(img)
        return img, bboxes
    
    def _apply_perspective_with_bboxes(self, img, bboxes, transform):
        """Apply perspective transform to image and transform bboxes accordingly."""
        import torchvision.transforms.functional as TF
        
        # Get image dimensions
        _, h, w = img.shape
        
        # Get perspective parameters
        if torch.rand(1) < transform.p:
            startpoints, endpoints = transform.get_params(w, h, transform.distortion_scale)
            
            # Apply perspective to image
            img = TF.perspective(img, startpoints, endpoints, transform.interpolation, transform.fill)
            
            # Convert normalized YOLO bboxes to pixel corners
            bboxes_corners = self._yolo_to_corners(bboxes, w, h)  # [N, 4] with [x1, y1, x2, y2]
            
            # Get all 4 corners of each box
            x1, y1, x2, y2 = bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 2], bboxes_corners[:, 3]
            corners = torch.stack([
                torch.stack([x1, y1], dim=1),  # top-left
                torch.stack([x2, y1], dim=1),  # top-right
                torch.stack([x1, y2], dim=1),  # bottom-left
                torch.stack([x2, y2], dim=1),  # bottom-right
            ], dim=1)  # [N, 4, 2]
            
            # Apply perspective transform to corners
            device = bboxes.device
            corners_transformed = self._transform_corners_perspective(corners, startpoints, endpoints, w, h)
            corners_transformed = corners_transformed.to(device)  # Ensure same device as input
            
            # Get new bounding boxes from transformed corners (axis-aligned)
            x_coords = corners_transformed[:, :, 0]  # [N, 4]
            y_coords = corners_transformed[:, :, 1]  # [N, 4]
            new_x1 = x_coords.min(dim=1)[0]
            new_y1 = y_coords.min(dim=1)[0]
            new_x2 = x_coords.max(dim=1)[0]
            new_y2 = y_coords.max(dim=1)[0]
            
            # Convert back to normalized YOLO format
            bboxes = self._corners_to_yolo(new_x1, new_y1, new_x2, new_y2, w, h, bboxes[:, 0])
        
        return img, bboxes
    
    def _apply_rotation_with_bboxes(self, img, bboxes, transform):
        """Apply rotation to image and transform bboxes accordingly."""
        import torchvision.transforms.functional as TF
        
        # Get image dimensions
        _, h, w = img.shape
        
        # Get rotation angle
        angle = transform.get_params(transform.degrees)
        
        # Determine rotation center
        if transform.center is None:
            center = [w / 2, h / 2]
        else:
            center = transform.center
        
        # Apply rotation to image
        img = TF.rotate(img, angle, transform.interpolation, transform.expand, center, transform.fill)
        
        # Convert normalized YOLO bboxes to pixel corners
        bboxes_corners = self._yolo_to_corners(bboxes, w, h)
        
        # Get all 4 corners of each box
        x1, y1, x2, y2 = bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 2], bboxes_corners[:, 3]
        corners = torch.stack([
            torch.stack([x1, y1], dim=1),  # top-left
            torch.stack([x2, y1], dim=1),  # top-right
            torch.stack([x1, y2], dim=1),  # bottom-left
            torch.stack([x2, y2], dim=1),  # bottom-right
        ], dim=1)  # [N, 4, 2]
        
        # Apply rotation to corners
        device = bboxes.device
        corners_transformed = self._transform_corners_rotation(corners, angle, center[0], center[1])
        corners_transformed = corners_transformed.to(device)  # Ensure same device as input
        
        # Get new bounding boxes from transformed corners (axis-aligned)
        x_coords = corners_transformed[:, :, 0]
        y_coords = corners_transformed[:, :, 1]
        new_x1 = x_coords.min(dim=1)[0]
        new_y1 = y_coords.min(dim=1)[0]
        new_x2 = x_coords.max(dim=1)[0]
        new_y2 = y_coords.max(dim=1)[0]
        
        # Convert back to normalized YOLO format
        bboxes = self._corners_to_yolo(new_x1, new_y1, new_x2, new_y2, w, h, bboxes[:, 0])
        
        return img, bboxes
    
    @staticmethod
    def _yolo_to_corners(bboxes, img_w, img_h):
        """Convert YOLO format [class, cx, cy, w, h] (normalized) to corners [x1, y1, x2, y2] (pixels)."""
        cx = bboxes[:, 1] * img_w
        cy = bboxes[:, 2] * img_h
        w = bboxes[:, 3] * img_w
        h = bboxes[:, 4] * img_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    @staticmethod
    def _corners_to_yolo(x1, y1, x2, y2, img_w, img_h, classes):
        """Convert corners [x1, y1, x2, y2] (pixels) back to YOLO format [class, cx, cy, w, h] (normalized)."""
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        
        # Clamp to valid range [0, 1]
        cx = torch.clamp(cx, 0, 1)
        cy = torch.clamp(cy, 0, 1)
        w = torch.clamp(w, 0, 1)
        h = torch.clamp(h, 0, 1)
        
        return torch.stack([classes, cx, cy, w, h], dim=1)
    
    @staticmethod
    def _transform_corners_perspective(corners, startpoints, endpoints, w, h):
        """Transform corners using perspective transformation matrix."""
        # Compute perspective transformation matrix
        import numpy as np
        import cv2
        
        # Convert to numpy for cv2
        startpoints_np = np.float32(startpoints)
        endpoints_np = np.float32(endpoints)
        matrix = cv2.getPerspectiveTransform(startpoints_np, endpoints_np)
        
        # Transform all corners
        N = corners.shape[0]
        corners_flat = corners.reshape(-1, 2).cpu().numpy()  # [N*4, 2]
        
        # Apply perspective transform
        corners_homogeneous = np.hstack([corners_flat, np.ones((corners_flat.shape[0], 1))])  # [N*4, 3]
        transformed = (matrix @ corners_homogeneous.T).T  # [N*4, 3]
        transformed = transformed[:, :2] / transformed[:, 2:3]  # Normalize by w coordinate
        
        # Convert back to torch and reshape
        corners_transformed = torch.from_numpy(transformed).float().reshape(N, 4, 2)
        
        return corners_transformed
    
    @staticmethod
    def _transform_corners_rotation(corners, angle, cx, cy):
        """Transform corners using rotation around specified center point."""
        import math
        
        # Convert angle to radians (NOTE: torchvision uses counter-clockwise, which is standard)
        angle_rad = math.radians(-angle)  # Negate because we need to match torchvision's convention
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Translate corners to origin (center of rotation), rotate, translate back
        corners_centered = corners.clone()
        corners_centered[:, :, 0] -= cx
        corners_centered[:, :, 1] -= cy
        
        # Apply rotation matrix
        # Standard 2D rotation: [cos -sin; sin cos]
        x_rot = corners_centered[:, :, 0] * cos_a - corners_centered[:, :, 1] * sin_a
        y_rot = corners_centered[:, :, 0] * sin_a + corners_centered[:, :, 1] * cos_a
        
        corners_rotated = torch.stack([x_rot + cx, y_rot + cy], dim=2)
        
        return corners_rotated