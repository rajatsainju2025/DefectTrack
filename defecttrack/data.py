"""Image-sequence loading for inference."""

import glob
import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def letterbox(img, height=512, width=1024, color=(127.5, 127.5, 127.5)):
    """Resize an image to fit (width, height), padding the rest with grey."""
    shape = img.shape[:2]  # (height, width)
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # (w, h)
    dw = (width - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


class LoadImages:
    """Iterate over the frames of an image sequence, in filename order.

    Yields ``(path, network_input, original_image)`` where ``network_input``
    is a normalised CHW float32 RGB array of size ``img_size`` (width, height).
    """

    def __init__(self, path, img_size=(1024, 512), median_blur=3):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.*")))
            self.files = [f for f in files
                          if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        elif os.path.isfile(path):
            self.files = [path]
        else:
            self.files = []

        if not self.files:
            raise FileNotFoundError(f"No images found in {path}")

        self.width, self.height = img_size
        self.median_blur = median_blur

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for path in self.files:
            yield (path, *self.load(path))

    def load(self, path):
        img0 = cv2.imread(path)  # BGR
        if img0 is None:
            raise IOError(f"Failed to load {path}")
        if self.median_blur:
            img0 = cv2.medianBlur(img0, self.median_blur)

        img, *_ = letterbox(img0, height=self.height, width=self.width)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, img0
