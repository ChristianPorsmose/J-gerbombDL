from torchvision.transforms import v2 as T


class LetterBoxTransform:
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114)):
        self.new_shape = new_shape  # (H, W)
        self.color = color
        self.scale_ratio = None
        self.pad = None

    def __call__(self, img, bboxes=None):
        """
        Apply letterbox to image and optionally transform bboxes.
        """
        # img is a tensor [C, H, W]
        shape = img.shape[1:]  # current shape [H, W]
        new_shape = self.new_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        self.scale_ratio = r

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]

        dh /= 2
        dw /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        self.pad = (left, top, right, bottom)

        if shape != new_unpad:
            img = T.Resize(new_unpad)(img)

        img = T.Pad([left, top, right, bottom], fill=self.color[0])(img)

        if bboxes is not None and len(bboxes) > 0:
            # Bboxes are in normalized format [class, x_center, y_center, width, height]
            # We need to maintain them in normalized format relative to the NEW padded image

            orig_h, orig_w = shape

            new_h, new_w = new_shape

            # Convert from normalized to pixel coordinates in original image
            bboxes_pixel = bboxes.clone()
            bboxes_pixel[:, 1] *= orig_w  # x_center
            bboxes_pixel[:, 2] *= orig_h  # y_center
            bboxes_pixel[:, 3] *= orig_w  # width
            bboxes_pixel[:, 4] *= orig_h  # height

            bboxes_pixel[:, 1:5] *= r

            # Apply padding offset (only to centers)
            bboxes_pixel[:, 1] += left  # x_center
            bboxes_pixel[:, 2] += top  # y_center

            # Normalize to new image size
            bboxes_pixel[:, 1] /= new_w  # x_center
            bboxes_pixel[:, 2] /= new_h  # y_center
            bboxes_pixel[:, 3] /= new_w  # width
            bboxes_pixel[:, 4] /= new_h  # height

            return img, bboxes_pixel

        return img
