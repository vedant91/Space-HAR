"""
Build the training cache for train/hmr3d_model.py from dataset/blender_core.

For every frame of every take, in order (tracking is stateful - a shuffled
read would desync the box/angle from what the live pipeline will actually
see), this:
  1. Runs pipeline.hmr_backend.HMRBackend's OWN tracker (_track/_crop) over
     the frame - the exact same MediaPipe-driven box+angle-EMA code the live
     system and ROMP both used. This is the one thing that must never drift
     from the live path: a cache built by a different crop policy silently
     trains a model for a distribution it will not actually be served.
  2. Reads the matching row of the take's pose_2d.npy ground truth and maps
     its (x, y) into the SAME crop-local frame via the crop's own affine
     matrix - the crop-local coordinate system train/hmr3d_model.py's
     contract is defined in. z and visibility pass through unchanged (an
     in-plane crop rotation does not change true depth from the camera).
  3. Saves the resized crop as a small JPEG and appends one row to a single
     manifest covering every take, keyed by take index for a later
     GroupShuffleSplit-by-take (never by frame - see train/train_lstm.py's
     comment on why a window-level split leaks).

Run under D:/SIH-2026/.venv-hmr (the only environment with mediapipe AND
ultralytics installed together):
    D:/SIH-2026/.venv-hmr/Scripts/python.exe tools/prepare_hmr3d_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.hmr_backend import HMRBackend, square_crop  # noqa: E402
from train.hmr3d_model import CROP_SIZE  # noqa: E402


def take_camera_dir(take_dir: Path, meta: dict) -> Path:
    """Most takes render one camera named in meta['cameras']; use the first
    one that actually has a directory (take_0022 uses 'payload_wide', not
    the usual 'payload_a' - hardcoding the latter silently drops it)."""
    for cam in meta.get("cameras", []):
        d = take_dir / cam
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{take_dir}: none of meta['cameras']={meta.get('cameras')} exist on disk")


def crop_local_target(gt_row: np.ndarray, m: np.ndarray, w: int, h: int, side: float) -> np.ndarray:
    """gt_row: (33, 4) full-frame-normalized (x, y, z, vis) from pose_2d.npy.
    Returns (33, 4) in the crop's own local frame: x, y in [0,1] of the crop,
    z and vis unchanged (see module docstring on why z needs no adjustment)."""
    out = gt_row.copy()
    px = np.c_[gt_row[:, 0] * w, gt_row[:, 1] * h, np.ones(len(gt_row))]
    crop_px = px @ m.T
    out[:, 0] = crop_px[:, 0] / side
    out[:, 1] = crop_px[:, 1] / side
    return out


def process_take(take_dir: Path, cache_dir: Path, jpg_quality: int = 90):
    meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
    cam_dir = take_camera_dir(take_dir, meta)
    tag = meta.get("tag", "?")
    idx = meta["index"]

    gt = np.load(cam_dir / "pose_2d.npy").reshape(-1, 33, 4)
    frames = sorted((cam_dir / "frames").glob("*.png"))
    n = min(len(frames), len(gt))
    if n == 0:
        raise RuntimeError(f"take_{idx:04d} ({tag}): no frames to process")

    backend = HMRBackend(backend="none")   # __init__ returns early: no model, no tracker yet
    backend._load_tracker()                # sets up MediaPipe + stock YOLO only, no 3D model

    rows = []
    for i in range(n):
        rgb = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        backend._track(rgb)
        crop, m, side = backend._crop(rgb, backend._box, backend._angle)
        crop_small = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

        target = crop_local_target(gt[i], m, w, h, side)

        fname = f"take{idx:04d}_{i:05d}.jpg"
        cv2.imwrite(str(cache_dir / "crops" / fname),
                    cv2.cvtColor(crop_small, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        rows.append((fname, idx, tag, target))

    print(f"take_{idx:04d} ({tag}): {n} frames cached, "
          f"mp_hit_rate={backend.stats['mp_hits']/n:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "dataset" / "blender_core"))
    ap.add_argument("--out", default=str(REPO / "dataset" / "hmr3d_cache"))
    ap.add_argument("--takes", default=None, help="comma-separated take indices; default = all")
    args = ap.parse_args()

    cache_dir = Path(args.out)
    (cache_dir / "crops").mkdir(parents=True, exist_ok=True)

    take_dirs = sorted(Path(args.dataset).glob("take_*"))
    if args.takes:
        wanted = {int(t) for t in args.takes.split(",")}
        take_dirs = [d for d in take_dirs if int(d.name.split("_")[1]) in wanted]

    all_fnames, all_idx, all_tags, all_targets = [], [], [], []
    for d in take_dirs:
        for fname, idx, tag, target in process_take(d, cache_dir):
            all_fnames.append(fname)
            all_idx.append(idx)
            all_tags.append(tag)
            all_targets.append(target)

    np.savez(cache_dir / "manifest.npz",
             fnames=np.array(all_fnames),
             take_idx=np.array(all_idx, dtype=np.int32),
             tags=np.array(all_tags),
             targets=np.stack(all_targets).astype(np.float32))
    print(f"\nWrote {len(all_fnames)} rows across {len(take_dirs)} takes to {cache_dir}")


if __name__ == "__main__":
    main()
