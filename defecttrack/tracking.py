"""Run DefectTrack on an image sequence and save the results."""

import logging
import os

import cv2
import torch

from .data import LoadImages
from .tracker.basetrack import BaseTrack
from .tracker.multitracker import JDETracker
from .utils.visualization import plot_tracking

logger = logging.getLogger(__name__)

# MOTChallenge format: frame, id, x, y, w, h, score, -1, -1, -1
MOT_LINE = "{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{s:.4f},1,-1,-1,-1\n"


def write_results(path, results):
    """Write tracks to a MOTChallenge-style text file."""
    with open(path, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for (x1, y1, w, h), track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                f.write(MOT_LINE.format(frame=frame_id, id=track_id,
                                        x1=x1, y1=y1, w=w, h=h, s=score))
    logger.info("Saved tracks to %s", path)


def keep_track(tlwh, cfg):
    """Filter out tiny boxes and boxes that are much wider than tall."""
    w, h = tlwh[2], tlwh[3]
    return w * h > cfg.min_box_area and w / h <= cfg.max_aspect_ratio


def track_sequence(cfg, image_dir, output_dir, save_frames=True,
                   save_video=False, model=None):
    """Track defect clusters through one image sequence.

    Writes ``results.txt`` (and optionally annotated frames / a video)
    to ``output_dir``. Returns the per-frame results.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    loader = LoadImages(image_dir, cfg.img_size, median_blur=cfg.median_blur)
    BaseTrack._count = 0  # restart track IDs at 1 for every sequence
    tracker = JDETracker(cfg, frame_rate=cfg.frame_rate, model=model)

    results, writer = [], None
    for frame_id, (path, img, img0) in enumerate(loader, start=1):
        blob = torch.from_numpy(img).unsqueeze(0).to(cfg.device)
        targets = tracker.update(blob, img0)

        tracks = [t for t in targets if keep_track(t.tlwh, cfg)]
        tlwhs = [t.tlwh for t in tracks]
        ids = [t.track_id for t in tracks]
        scores = [t.score for t in tracks]
        results.append((frame_id, tlwhs, ids, scores))

        if save_frames or save_video:
            frame = plot_tracking(img0, tlwhs, ids, frame_id=frame_id)
            if save_frames:
                cv2.imwrite(os.path.join(frames_dir, f"{frame_id:06d}.png"), frame)
            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(output_dir, "tracking.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(frame)

        logger.info("Frame %d/%d: %d tracks", frame_id, len(loader), len(tracks))

    if writer is not None:
        writer.release()
    write_results(os.path.join(output_dir, "results.txt"), results)
    return results
