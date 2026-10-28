"""Draw tracked boxes and IDs on a frame."""

import cv2
import numpy as np


def get_color(idx):
    """Stable, distinct BGR colour for a track ID."""
    idx = idx * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255


def plot_tracking(image, tlwhs, obj_ids, frame_id=0):
    """Return a copy of ``image`` with one coloured, labelled box per track."""
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(1, image.shape[1] / 1600.0)
    text_thickness = 2 if text_scale > 1.1 else 1
    line_thickness = max(2, int(image.shape[1] / 500.0))

    for (x1, y1, w, h), obj_id in zip(tlwhs, obj_ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x1 + w, y1 + h))
        color = get_color(abs(int(obj_id)))
        cv2.rectangle(im, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
        cv2.putText(im, str(int(obj_id)), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, color, thickness=text_thickness)
    return im
