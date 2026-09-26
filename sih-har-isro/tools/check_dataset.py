"""
Gate a rendered dataset before anyone trains on it (or before paying for the
full render, when run on a --gt-only pass).

Fails loudly on the defects that cost this project a dataset before:
labels out of step with frames, a step missing from its sequence, holdout
tags that match nothing (empty test set), hands that teleport on a step
boundary, and a crew member rendered mostly out of frame.

    python tools/check_dataset.py dataset/blender_core
    python tools/check_dataset.py dataset/blender_core --holdout orient_180,yaw_90
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WRISTS = (15, 16)            # MediaPipe landmark indices
TORSO = (11, 12, 23, 24)     # shoulders + hips
MAX_WRIST_STEP_M = 0.04      # fastest real reach here is ~0.02 m/frame
MIN_TORSO_IN_FRAME = 0.5


def check(root: Path, holdout):
    errors, tags = [], {}
    takes = sorted(p for p in root.glob("take_*") if (p / "meta.json").exists())
    if not takes:
        return [f"no finished takes under {root}"], {}

    for take in takes:
        meta = json.loads((take / "meta.json").read_text(encoding="utf-8"))
        tags.setdefault(meta.get("tag", "nominal"), []).append(take.name)
        for cam in meta["cameras"]:
            d = take / cam
            name = f"{take.name}/{cam}"
            labels = np.load(d / "labels.npy")
            if meta.get("real"):   # recorded take: no pose ground truth to check
                n_png = len(list((d / "frames").glob("*.png")))
                if not (len(labels) == n_png == meta["n_frames"]):
                    errors.append(f"{name}: labels {len(labels)} / PNGs {n_png} / "
                                  f"n_frames {meta['n_frames']} disagree")
                continue
            pose2d = np.load(d / "pose_2d.npy")
            world = np.load(d / "pose_world.npy")

            if not (len(labels) == len(pose2d) == len(world) == meta["n_frames"]):
                errors.append(f"{name}: labels {len(labels)} / pose {len(pose2d)} / "
                              f"world {len(world)} / n_frames {meta['n_frames']} disagree")
            if meta.get("frames_written"):
                n_png = len(list((d / "frames").glob("*.png")))
                if n_png != meta["n_frames"]:
                    errors.append(f"{name}: {n_png} PNGs for {meta['n_frames']} frames")
            if set(labels.tolist()) != set(meta["sequence"]):
                errors.append(f"{name}: labels {sorted(set(labels.tolist()))} "
                              f"!= sequence {sorted(set(meta['sequence']))}")

            step = np.linalg.norm(np.diff(world[:, WRISTS, :], axis=0), axis=-1).max()
            if step > MAX_WRIST_STEP_M:
                f = int(np.argmax(np.linalg.norm(np.diff(world[:, WRISTS, :], axis=0), axis=-1).max(axis=1)))
                errors.append(f"{name}: wrist jumps {step:.3f} m in one frame "
                              f"at frame {f} (label {labels[f]} -> {labels[f + 1]})")

            xy = pose2d.reshape(len(pose2d), 33, 4)[:, TORSO, :2]
            inside = ((xy >= 0) & (xy <= 1)).all(axis=-1).mean()
            if inside < MIN_TORSO_IN_FRAME:
                errors.append(f"{name}: torso in frame only {inside:.0%} of the time")

    for tag in holdout:
        if tag not in tags:
            errors.append(f"holdout tag '{tag}' matches no take (have: {sorted(tags)})")
    return errors, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--holdout", default="orient_180,orient_-60,yaw_90,pitch_60")
    args = ap.parse_args()
    holdout = [t for t in args.holdout.split(",") if t]

    errors, tags = check(Path(args.dataset), holdout)
    print(f"{sum(len(v) for v in tags.values())} takes, tags: {sorted(tags)}")
    for e in errors:
        print("FAIL", e)
    print("DATASET_CHECK", "FAILED" if errors else "OK")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
