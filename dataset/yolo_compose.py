import math
import torch
import numpy as np
import cv2
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as TF
from dataset.letter_box_transform import LetterBoxTransform


class YOLOCompose:
    """Custom Compose handling images and YOLO bounding boxes."""

    def __init__(self, transforms):
        self.transforms = transforms
        self.handlers = {
            LetterBoxTransform: self._handle_letterbox,
            T.RandomHorizontalFlip: self._handle_hflip,
            T.RandomVerticalFlip: self._handle_vflip,
            T.RandomPerspective: self._handle_perspective,
            T.RandomRotation: self._handle_rotation,
        }

    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            handler = self.handlers.get(type(t), self._handle_image_only)
            img, bboxes = handler(t, img, bboxes)
        return img, bboxes

    def _handle_letterbox(self, t, img, bboxes):
        if bboxes is not None and len(bboxes) > 0:
            return t(img, bboxes)
        return t(img), bboxes

    def _handle_hflip(self, t, img, bboxes):
        if bboxes is not None and len(bboxes) > 0 and torch.rand(1) < t.p:
            img = T.functional.hflip(img)
            bboxes[:, 1] = 1.0 - bboxes[:, 1]
        return img, bboxes

    def _handle_vflip(self, t, img, bboxes):
        if bboxes is not None and len(bboxes) > 0 and torch.rand(1) < t.p:
            if torch.rand(1) < t.p:
                img = T.functional.vflip(img)
                bboxes[:, 2] = 1.0 - bboxes[:, 2]
        return img, bboxes

    def _handle_perspective(self, t, img, bboxes):
        if bboxes is not None and len(bboxes) > 0 and torch.rand(1) < t.p:
            _, h, w = img.shape
            startpoints, endpoints = t.get_params(w, h, t.distortion_scale)
            img = TF.perspective(img, startpoints, endpoints, t.interpolation, t.fill)
            bboxes = self._transform_bboxes_perspective(
                bboxes, startpoints, endpoints, w, h
            )
        return img, bboxes

    def _handle_rotation(self, t, img, bboxes):
        if bboxes is not None and len(bboxes) > 0:
            _, h, w = img.shape
            angle = t.get_params(t.degrees)
            center = t.center if t.center is not None else [w / 2, h / 2]
            img = TF.rotate(img, angle, t.interpolation, t.expand, center, t.fill)
            bboxes = self._transform_bboxes_rotation(
                bboxes, angle, center[0], center[1], w, h
            )
        return img, bboxes

    def _handle_image_only(self, t, img, bboxes):
        return t(img), bboxes

    def _yolo_to_corners(self, bboxes, w, h):
        cx, cy = bboxes[:, 1] * w, bboxes[:, 2] * h
        bw, bh = bboxes[:, 3] * w, bboxes[:, 4] * h
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _corners_to_yolo(self, x1, y1, x2, y2, w, h, classes):
        cx = torch.clamp((x1 + x2) / 2 / w, 0, 1)
        cy = torch.clamp((y1 + y2) / 2 / h, 0, 1)
        bw = torch.clamp((x2 - x1) / w, 0, 1)
        bh = torch.clamp((y2 - y1) / h, 0, 1)
        return torch.stack([classes, cx, cy, bw, bh], dim=1)

    def _transform_bboxes_perspective(self, bboxes, startpoints, endpoints, w, h):
        corners = self._yolo_to_corners(bboxes, w, h)
        N = corners.shape[0]

        x1, y1, x2, y2 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
        corners_pts = torch.stack(
            [
                torch.stack([x1, y1], dim=1),
                torch.stack([x2, y1], dim=1),
                torch.stack([x1, y2], dim=1),
                torch.stack([x2, y2], dim=1),
            ],
            dim=1,
        )

        matrix = cv2.getPerspectiveTransform(
            np.float32(startpoints), np.float32(endpoints)
        )
        pts_flat = corners_pts.reshape(-1, 2).cpu().numpy()
        pts_h = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])
        transformed = (matrix @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        transformed = torch.from_numpy(transformed).float().reshape(N, 4, 2)

        x_coords, y_coords = transformed[:, :, 0], transformed[:, :, 1]
        return self._corners_to_yolo(
            x_coords.min(1)[0],
            y_coords.min(1)[0],
            x_coords.max(1)[0],
            y_coords.max(1)[0],
            w,
            h,
            bboxes[:, 0],
        )

    def _transform_bboxes_rotation(self, bboxes, angle, cx, cy, w, h):
        corners = self._yolo_to_corners(bboxes, w, h)
        angle_rad = math.radians(-angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        corners_centered = corners.clone()
        corners_centered[:, :, 0] -= cx
        corners_centered[:, :, 1] -= cy

        x_rot = corners_centered[:, :, 0] * cos_a - corners_centered[:, :, 1] * sin_a
        y_rot = corners_centered[:, :, 0] * sin_a + corners_centered[:, :, 1] * cos_a

        rotated = torch.stack([x_rot + cx, y_rot + cy], dim=2)
        x_coords, y_coords = rotated[:, :, 0], rotated[:, :, 1]
        return self._corners_to_yolo(
            x_coords.min(1)[0],
            y_coords.min(1)[0],
            x_coords.max(1)[0],
            y_coords.max(1)[0],
            w,
            h,
            bboxes[:, 0],
        )
