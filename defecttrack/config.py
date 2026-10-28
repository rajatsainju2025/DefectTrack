"""Settings for DefectTrack inference.

The defaults reproduce the settings used for the published model
(HRNet-18 backbone, 1024x512 input, confidence threshold 0.3).
"""

from dataclasses import dataclass, field

import torch


@dataclass
class TrackerConfig:
    # Model
    load_model: str = "weights/best.pth"
    arch: str = "hrnet_18"
    head_conv: int = 256
    reid_dim: int = 128
    num_classes: int = 1
    down_ratio: int = 4
    reg_offset: bool = True
    cat_spec_wh: bool = False

    # Input (width, height). The model was trained on 1024 x 512 frames.
    img_size: tuple = (1024, 512)
    median_blur: int = 3

    # Tracking
    conf_thres: float = 0.3      # detection confidence to start/keep a track
    track_buffer: int = 30       # frames a lost track is kept (at 30 fps)
    frame_rate: int = 100        # frame rate of the TEM video
    K: int = 128                 # max detections per frame

    # Output filtering
    min_box_area: float = 100    # drop boxes smaller than this (pixels^2)
    max_aspect_ratio: float = 1.6  # drop boxes wider than this (w / h)

    device: str = "auto"         # "auto", "cpu", "cuda" or "cuda:N"

    # Normalisation constants (kept for checkpoint compatibility)
    mean: list = field(default_factory=lambda: [0.408, 0.447, 0.470])
    std: list = field(default_factory=lambda: [0.289, 0.274, 0.278])

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

    @property
    def heads(self):
        heads = {"hm": self.num_classes,
                 "wh": 2 * self.num_classes if self.cat_spec_wh else 2,
                 "id": self.reid_dim}
        if self.reg_offset:
            heads["reg"] = 2
        return heads
