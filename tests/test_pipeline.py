"""Fast tests that need no pretrained weights (a random-init model is used)."""

import os

import numpy as np
import pytest
import torch

from defecttrack.config import TrackerConfig
from defecttrack.data import LoadImages, letterbox
from defecttrack.models.model import create_model
from defecttrack.tracker.matching import bbox_ious
from defecttrack.tracking import track_sequence

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "video_sample", "images")


def test_bbox_ious_matches_reference():
    a = np.array([[0, 0, 9, 9]], dtype=float)           # 10 x 10 box
    b = np.array([[0, 0, 9, 9], [5, 0, 14, 9], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(bbox_ious(a, b), [[1.0, 50 / 150, 0.0]])


def test_letterbox_shape():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    out, ratio, *_ = letterbox(img, height=512, width=1024)
    assert out.shape == (512, 1024, 3) and ratio == 0.5


def test_loader_reads_sample():
    loader = LoadImages(SAMPLE)
    path, img, img0 = next(iter(loader))
    assert len(loader) == 10
    assert img.shape == (3, 512, 1024) and img.dtype == np.float32


def test_model_output_heads():
    cfg = TrackerConfig(device="cpu")
    model = create_model(cfg.arch, cfg.heads).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 128, 256))[-1]
    assert {k: v.shape[1] for k, v in out.items()} == cfg.heads


@pytest.mark.slow
def test_track_sequence_runs(tmp_path):
    cfg = TrackerConfig(device="cpu")
    model = create_model(cfg.arch, cfg.heads)
    results = track_sequence(cfg, SAMPLE, str(tmp_path), save_frames=False, model=model)
    assert len(results) == 10
    assert (tmp_path / "results.txt").exists()
