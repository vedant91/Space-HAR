"""Key-joint 2D error of saved raw MediaPipe poses vs Blender ground truth, in
native render pixels, same rule as tools/evaluate_real._pose_error (frames
with a detection, landmarks with gt visibility >= 0.9).

    python train/experiments/pose_error.py name=<rawpose_dir> ... --out x.json
A name=cache:<pose-cache digest-free> is not supported; pass directories of
take_XXXX.npy (132-dim per frame).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset" / "blender_core"
KEY = [11, 12, 13, 14, 15, 16, 23, 24]
WRISTS = [15, 16]
HOLD = {"orient_180", "orient_-60", "yaw_90", "pitch_60"}


def take_error(mp, take):
    meta = json.loads((DS / take / "meta.json").read_text())
    cam = DS / take / meta["cameras"][0]
    gt = np.load(cam / "pose_2d.npy")
    n = min(len(mp), len(gt))
    mp, gt = mp[:n].reshape(n, 33, 4), gt[:n].reshape(n, 33, 4)
    got = np.any(mp.reshape(n, -1) != 0, axis=1)
    use = (gt[..., 3] >= 0.9) & got[:, None]
    w, h = meta["resolution"]
    err = np.linalg.norm((mp[..., :2] - gt[..., :2]) * [w, h], axis=-1)
    key = np.zeros_like(use); key[:, KEY] = True
    wr = np.zeros_like(use); wr[:, WRISTS] = True
    return meta["tag"], err[use & key], err[use & wr], got.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists")
    rep = {}
    for s in args.sources:
        name, d = s.split("=", 1)
        rows, allk, allw, holdk, holdw = {}, [], [], [], []
        for f in sorted(Path(d).glob("take_*.npy")):
            tag, ek, ew, det = take_error(np.load(f), f.stem)
            rows[tag] = {"key_mean_px": round(float(ek.mean()), 1),
                         "wrist_mean_px": round(float(ew.mean()), 1),
                         "detect": round(float(det), 4)}
            allk.append(ek); allw.append(ew)
            if tag in HOLD:
                holdk.append(ek); holdw.append(ew)
        c = lambda xs: round(float(np.concatenate(xs).mean()), 1)
        rep[name] = {"all_key_mean_px": c(allk), "all_wrist_mean_px": c(allw),
                     "holdout_key_mean_px": c(holdk), "holdout_wrist_mean_px": c(holdw),
                     "takes": rows}
        print(name, {k: v for k, v in rep[name].items() if k != "takes"}, flush=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
