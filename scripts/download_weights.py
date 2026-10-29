"""Download the pretrained DefectTrack checkpoint to weights/best.pth.

The model is hosted on figshare as DefectTrack_model.zip (~84 MB). This script
downloads the zip and extracts only the checkpoint from it.
"""

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

URL = ("https://figshare.com/ndownloader/files/35106460"
       "?private_link=9e7f6c0870e828dbc1a2")
DEST = os.path.join(os.path.dirname(__file__), "..", "weights", "best.pth")


def main(dest=DEST):
    dest = os.path.abspath(dest)
    if os.path.isfile(dest):
        print(f"Already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "DefectTrack_model.zip")
        print(f"Downloading DefectTrack_model.zip from figshare ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "defecttrack"})
        with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)

        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist()
                       if m.endswith(".pth") and "__MACOSX" not in m]
            if not members:
                raise SystemExit("No .pth checkpoint found in the downloaded zip.")
            with zf.open(members[0]) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    print(f"Saved {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
