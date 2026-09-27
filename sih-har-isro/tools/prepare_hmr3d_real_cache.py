"""
Build the REAL-footage training cache for train/train_hmr3d.py's
finetune_on_real(), from dataset/real_pseudo (data_generation/real_video_autolabel.py's
output) — same crop/target contract as tools/prepare_hmr3d_dataset.py, which
builds the synthetic cache, just sourced from real pseudo-labeled frames
instead of dataset/blender_core's exact ground truth.

Only the two wrist joints are ever labeled here (see
data_generation/real_video_autolabel.py's docstring for why — the same
"don't guess what you didn't measure" rule applies to a pose model as it does
to rack_frame.py's rotation math). Every other joint's target row stays
exactly zero (x, y, z, vis all 0), which is what finetune_on_real()'s masked
loss reads as "this joint contributes no gradient this frame" — not "this
joint is occluded" (that's a real, different, always-true synthetic label;
conflating the two would teach the model real joints are invisible when they
just weren't attempted).

Run once after data_generation/real_video_autolabel.py (needs mediapipe +
ultralytics, same env as tools/prepare_hmr3d_dataset.py):
    python tools/prepare_hmr3d_real_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config.experiment_config import REAL_PSEUDO_DIR
from pipeline.hmr_backend import HMRBackend
from train.hmr3d_model import CROP_SIZE
from tools.prepare_hmr3d_dataset import crop_local_target  # noqa: E402 (reuse, don't reimplement)


def process_video(video_dir: Path, cache_dir: Path, jpg_quality: int = 90):
    npz_path = video_dir / "poses.npz"
    frames_dir = video_dir / "frames"
    if not npz_path.exists() or not frames_dir.exists():
        return []

    data = np.load(str(npz_path))
    pose = data["pose"].reshape(-1, 33, 4)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    n = min(len(frame_files), len(pose))
    if n == 0:
        return []

    # Fresh tracker per video: HMRBackend._track is stateful (EMA'd box/angle),
    # and these are independent clips, not continuous frames of one take.
    backend = HMRBackend(backend="none")   # __init__ returns early: no model, no tracker yet
    backend._load_tracker()                # MediaPipe + stock YOLO only, no 3D model needed to cache

    rows = []
    n_usable = 0
    for i in range(n):
        if pose[i, :, 3].sum() < 1e-6:
            continue  # no wrist detected this frame at all — nothing to train on, skip rather than cache noise
        img = cv2.imread(str(frame_files[i]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        backend._track(rgb)
        crop, m, side = backend._crop(rgb, backend._box, backend._angle)
        crop_small = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

        target = crop_local_target(pose[i], m, w, h, side)

        fname = f"{video_dir.name}_{i:05d}.jpg"
        cv2.imwrite(str(cache_dir / "crops" / fname),
                    cv2.cvtColor(crop_small, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        rows.append((fname, video_dir.name, target))
        n_usable += 1

    print(f"{video_dir.name}: {n_usable}/{n} frames usable (had >=1 wrist detected), "
         f"mp_hit_rate={backend.stats['mp_hits']/max(n,1):.2f}")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", default=REAL_PSEUDO_DIR)
    ap.add_argument("--out", default=str(REPO / "dataset" / "hmr3d_real_cache"))
    args = ap.parse_args()

    pseudo_dir = Path(args.pseudo_dir)
    if not pseudo_dir.exists():
        raise FileNotFoundError(
            f"{pseudo_dir} doesn't exist — run data_generation/real_video_autolabel.py first.")

    cache_dir = Path(args.out)
    (cache_dir / "crops").mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(p for p in pseudo_dir.iterdir() if p.is_dir())
    all_fnames, all_video, all_targets = [], [], []
    for vd in video_dirs:
        for fname, video_name, target in process_video(vd, cache_dir):
            all_fnames.append(fname)
            all_video.append(video_name)
            all_targets.append(target)

    if not all_fnames:
        print(f"\nNo usable frames across {len(video_dirs)} video(s) — nothing to cache. "
             "finetune_on_real() will skip.")
        return

    np.savez(cache_dir / "manifest.npz",
             fnames=np.array(all_fnames),
             video=np.array(all_video),
             targets=np.stack(all_targets).astype(np.float32))
    print(f"\nWrote {len(all_fnames)} rows across {len(video_dirs)} videos to {cache_dir}")


if __name__ == "__main__":
    main()
