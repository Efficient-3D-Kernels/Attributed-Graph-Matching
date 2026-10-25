"""Download the Oxford VGG Affine Covariant Regions benchmark (Mikolajczyk et al., IJCV 2005).

Eight real image sequences with ground-truth homographies H1toKp (K = 2..6):
  bark (zoom+rotation), bikes (blur), boat (zoom+rotation), graf (viewpoint), leuven (illumination),
  trees (blur), ubc (JPEG compression), wall (viewpoint).
Files are stored under benchmarks/data/oxford_affine/<sequence>/ ; SHA-256 of every archive is
written to benchmarks/data/oxford_affine/manifest.json.
"""
from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_common import ROOT, sha256  # noqa: E402

BASE = "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/{}.tar.gz"
SEQUENCES = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]
DATA = ROOT / "benchmarks" / "data" / "oxford_affine"


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for s in SEQUENCES:
        arc = DATA / f"{s}.tar.gz"
        if not arc.exists():
            print("downloading", s, flush=True)
            urllib.request.urlretrieve(BASE.format(s), arc)
        out = DATA / s
        if not out.exists():
            out.mkdir()
            with tarfile.open(arc) as t:
                t.extractall(out, filter="data")
        manifest[s] = {"url": BASE.format(s), "sha256": sha256(arc), "files": sorted(p.name for p in out.rglob("*") if p.is_file())}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    for k, v in fetch().items():
        print(k, v["files"])
