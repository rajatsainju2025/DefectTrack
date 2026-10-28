# DefectTrack

Real-time multi-object tracking of defect clusters in in-situ TEM videos.

DefectTrack detects defect clusters in every frame and links them across
frames, giving each defect a stable ID. It uses an HRNet-18 backbone with a
FairMOT-style joint detection and re-identification head, followed by a
Kalman-filter tracker.

> **Paper:** R. Sainju *et al.*, "DefectTrack: a deep learning-based
> multi-object tracking algorithm for quantitative defect analysis of in-situ
> TEM videos in real-time," *Scientific Reports* **12**, 15705 (2022).
> [doi:10.1038/s41598-022-19697-1](https://doi.org/10.1038/s41598-022-19697-1)

## Quick start

```bash
git clone https://github.com/rajatsainju2025/DefectTrack.git
cd DefectTrack
pip install .                        # installs the `defecttrack` command
python scripts/download_weights.py   # pretrained model -> weights/best.pth

defecttrack data/video_sample/images --video
```

Results are written to `results/video_sample/`:

| File | What it is |
| --- | --- |
| `results.txt` | One line per box: `frame,id,x,y,w,h,score,1,-1,-1,-1` (MOTChallenge format) |
| `frames/*.png` | Frames with each tracked defect drawn and labelled by ID |
| `tracking.mp4` | The annotated frames as a video (with `--video`) |

It runs on CPU or GPU; a GPU is picked automatically when available.

## Using your own video

Export the video as numbered frames into a folder, then point DefectTrack at it:

```bash
defecttrack path/to/my_video/images -o results
```

Several folders can be passed at once. Frames are resized and padded to
1024 x 512, the size the model was trained on.

Useful options (see `defecttrack --help` for all):

| Option | Default | Meaning |
| --- | --- | --- |
| `--weights` | `weights/best.pth` | Checkpoint to load |
| `--conf-thres` | 0.3 | Detection confidence needed to start or keep a track |
| `--min-box-area` | 100 | Ignore boxes smaller than this (pixels²) |
| `--max-aspect-ratio` | 1.6 | Ignore boxes much wider than they are tall |
| `--frame-rate` | 100 | Video frame rate; controls how long a lost track is remembered |
| `--device` | auto | `cpu`, `cuda` or `cuda:N` |

### From Python

```python
from defecttrack.config import TrackerConfig
from defecttrack.tracking import track_sequence

cfg = TrackerConfig(load_model="weights/best.pth", conf_thres=0.3)
results = track_sequence(cfg, "data/video_sample/images", "results/video_sample")
```

## Docker

```bash
docker build -t defecttrack .
docker run --rm -v "$PWD/results:/results" defecttrack            # sample video
docker run --rm -v "$PWD/my_video:/data" -v "$PWD/results:/results" \
    defecttrack /data --output /results                              # your frames
```

## Project layout

```
defecttrack/
  cli.py          command-line interface
  config.py       all settings, with the published defaults
  tracking.py     runs the tracker over a sequence and saves results
  data.py         frame loading and letterbox resizing
  models/         HRNet backbone, output decoding, checkpoint loading
  tracker/        Kalman filter, association (IoU + embedding), track state
  utils/          box post-processing and drawing
data/video_sample/  10-frame sample sequence
scripts/            weight download helper
tests/              quick tests (no weights needed)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Training

This repository contains the inference code and the released model. Training
follows the [FairMOT](https://github.com/ifzhang/FairMOT) procedure with an
HRNet-18 backbone, as described in the paper.

## Acknowledgements

Built on [FairMOT](https://github.com/ifzhang/FairMOT) and
[HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation) (both MIT licensed).

## Citation

```bibtex
@article{sainju2022defecttrack,
  title   = {DefectTrack: a deep learning-based multi-object tracking algorithm
             for quantitative defect analysis of in-situ TEM videos in real-time},
  author  = {Sainju, Rajat and others},
  journal = {Scientific Reports},
  volume  = {12},
  pages   = {15705},
  year    = {2022},
  doi     = {10.1038/s41598-022-19697-1}
}
```
