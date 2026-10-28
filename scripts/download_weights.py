"""Download the pretrained DefectTrack checkpoint to weights/best.pth."""

import os
import sys
import urllib.request

URL = "https://github.com/rajatsainju2025/DefectTrack/releases/download/v1.0.0/best.pth"
DEST = os.path.join(os.path.dirname(__file__), "..", "weights", "best.pth")


def main(dest=DEST):
    dest = os.path.abspath(dest)
    if os.path.isfile(dest):
        print(f"Already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {URL}\n -> {dest}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, dest)
    print(f"Done ({os.path.getsize(dest) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
