"""
Download the MediaPipe PoseLandmarker task bundles for offline operation.

The problem statement requires a system that "runs on offline standalone
system". The Tasks API needs its weights as an explicit file, so this fetches
them once into `models/mediapipe/` where `pipeline/pose_backend.py` looks for
them. After this runs, no network access is needed at inference time.

    python tools/fetch_pose_model.py            # 'full' only (9 MB)
    python tools/fetch_pose_model.py --all      # lite + full + heavy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

VARIANTS = {
    "lite": "fastest, lowest accuracy - matches MEDIAPIPE_MODEL_COMPLEXITY=0",
    "full": "balanced - complexity=1, the recommended default",
    "heavy": "most accurate, slowest - complexity=2",
}

DEST = Path(__file__).resolve().parent.parent / "models" / "mediapipe"


def fetch(variant: str, force: bool = False) -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / f"pose_landmarker_{variant}.task"
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"  {variant:6s} already present ({target.stat().st_size:,} bytes)")
        return target
    url = BASE.format(v=variant)
    print(f"  {variant:6s} downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()[:16]
    print(f"  {variant:6s} saved {len(data):,} bytes  sha256:{digest}...")
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Fetch every variant")
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    wanted = sorted(VARIANTS) if args.all else [args.variant]
    print(f"Fetching pose bundles into {DEST}")
    for variant in wanted:
        try:
            fetch(variant, force=args.force)
        except Exception as e:
            print(f"  {variant:6s} FAILED: {e}", file=sys.stderr)
            return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.pose_backend import describe
    info = describe()
    print("\nPose backend status:")
    print(f"  mediapipe   {info['mediapipe_version']}")
    print(f"  tasks API   {info['tasks_api']}")
    print(f"  usable      {info['usable']}")
    return 0 if info["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
