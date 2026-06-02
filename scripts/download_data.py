"""Download the two open datasets into ./data.

Usage:
    python scripts/download_data.py            # both datasets
    python scripts/download_data.py bacteria   # just classification data
    python scripts/download_data.py polystyrene

Both datasets are openly licensed (see DATA_SOURCES.md).
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# bacteria-ID: a single Dropbox folder zip (~600 MB) of .npy arrays.
BACTERIA_ZIP = "https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=1"

# polystyrene LoD: Mendeley dataset 33wf5rtr4h, files fetched via the public API.
POLY_API = ("https://data.mendeley.com/public-api/datasets/"
            "33wf5rtr4h/files?folder_id=root&version=1")
POLY_WANT_PREFIXES = ("Fig. S", "Fig. 1", "Metadata")

_UA = {"User-Agent": "Mozilla/5.0 (raman-open-ml downloader)"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def download_bacteria():
    out = os.path.join(DATA, "bacteria_id")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "X_reference.npy")):
        print("[bacteria] already present, skipping")
        return
    print("[bacteria] downloading ~600 MB zip ...")
    blob = _get(BACTERIA_ZIP)
    print("[bacteria] extracting ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(out)
    print(f"[bacteria] done -> {out}")


def download_polystyrene():
    out = os.path.join(DATA, "polystyrene")
    os.makedirs(out, exist_ok=True)
    listing = json.loads(_get(POLY_API).decode("utf-8"))
    for f in listing:
        name = f["filename"]
        if not name.startswith(POLY_WANT_PREFIXES):
            continue
        local = os.path.join(out, name.replace(" ", "").replace("..", "."))
        if os.path.exists(local):
            continue
        url = f["content_details"]["download_url"]
        with open(local, "wb") as fh:
            fh.write(_get(url))
        print(f"[polystyrene] {os.path.basename(local)} "
              f"({os.path.getsize(local) // 1024} KB)")
    print(f"[polystyrene] done -> {out}")


def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if which in ("all", "bacteria"):
        download_bacteria()
    if which in ("all", "polystyrene", "poly"):
        download_polystyrene()


if __name__ == "__main__":
    main()
