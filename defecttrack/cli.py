"""Command-line entry point: ``defecttrack <image_dir> --weights best.pth``."""

import argparse
import logging
import os

from .config import TrackerConfig
from .tracking import track_sequence


def parse_args(argv=None):
    d = TrackerConfig.__dataclass_fields__
    p = argparse.ArgumentParser(
        prog="defecttrack",
        description="Track defect clusters through an in-situ TEM image sequence.")
    p.add_argument("images", nargs="+",
                   help="folder(s) of frames, e.g. data/video_sample/images")
    p.add_argument("-w", "--weights", default=d["load_model"].default,
                   help="path to the pretrained checkpoint (default: %(default)s)")
    p.add_argument("-o", "--output", default="results",
                   help="output folder (default: %(default)s)")
    p.add_argument("--arch", default=d["arch"].default, choices=["hrnet_18", "hrnet_32"])
    p.add_argument("--conf-thres", type=float, default=d["conf_thres"].default,
                   help="detection confidence threshold (default: %(default)s)")
    p.add_argument("--min-box-area", type=float, default=d["min_box_area"].default)
    p.add_argument("--max-aspect-ratio", type=float, default=d["max_aspect_ratio"].default)
    p.add_argument("--frame-rate", type=int, default=d["frame_rate"].default,
                   help="video frame rate; sets how long lost tracks are kept")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--no-frames", action="store_true",
                   help="don't save annotated frames")
    p.add_argument("--video", action="store_true",
                   help="also save an annotated tracking.mp4")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if not os.path.isfile(args.weights):
        raise SystemExit(f"Checkpoint not found: {args.weights}\n"
                         "Download it with: python scripts/download_weights.py")

    cfg = TrackerConfig(load_model=args.weights, arch=args.arch,
                        conf_thres=args.conf_thres, min_box_area=args.min_box_area,
                        max_aspect_ratio=args.max_aspect_ratio,
                        frame_rate=args.frame_rate, device=args.device)

    for image_dir in args.images:
        name = os.path.basename(os.path.normpath(image_dir))
        if name == "images":  # data/<sequence>/images -> <sequence>
            name = os.path.basename(os.path.dirname(os.path.normpath(image_dir)))
        out_dir = os.path.join(args.output, name)
        results = track_sequence(cfg, image_dir, out_dir,
                                 save_frames=not args.no_frames, save_video=args.video)
        n_ids = len({i for _, _, ids, _ in results for i in ids})
        print(f"{name}: {len(results)} frames, {n_ids} tracked defects -> {out_dir}")


if __name__ == "__main__":
    main()
