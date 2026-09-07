"""
Draw exported ground-truth landmarks onto the frames they came from.

This is the check that the ground truth is actually correct. Every other
number in this project - MediaPipe's landmark error, the LSTM's accuracy on
real renders, HSV recall against true boxes - is measured against
`pose_2d.npy`, so if that file is silently mirrored, offset, or off by a
frame then every downstream metric is wrong in a way no test would catch.

Overlaying it on the image makes that failure impossible to miss: a mirrored
skeleton lands on the wrong arm, a frame offset lags the motion visibly, and
a bad landmark-to-bone mapping puts a "knee" in an elbow.

Run (from the project venv, not from Blender):
    python tools/overlay_pose.py --take dataset/blender/take_0000 --frames 0,60,150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Pose topology, as (a, b) landmark index pairs.
POSE_CONNECTIONS = [
    (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

LEFT_IDS = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_IDS = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

COLOR_LEFT = (90, 220, 255)     # amber  (BGR)
COLOR_RIGHT = (255, 190, 90)    # blue
COLOR_MID = (200, 200, 200)
COLOR_BONE = (120, 255, 160)
COLOR_OCCLUDED = (90, 90, 200)


def _landmark_color(idx: int) -> tuple:
    if idx in LEFT_IDS:
        return COLOR_LEFT
    if idx in RIGHT_IDS:
        return COLOR_RIGHT
    return COLOR_MID


def draw_pose(img: np.ndarray, feats: np.ndarray,
              vis_threshold: float = 0.5,
              draw_payload: dict | None = None) -> np.ndarray:
    """feats: (132,) or (33, 4) in MediaPipe normalised convention."""
    h, w = img.shape[:2]
    lm = np.asarray(feats, dtype=np.float32).reshape(-1, 4)
    out = img.copy()

    def px(i):
        return int(round(lm[i, 0] * w)), int(round(lm[i, 1] * h))

    for a, b in POSE_CONNECTIONS:
        if lm[a, 3] < 0.05 or lm[b, 3] < 0.05:
            continue
        faded = lm[a, 3] < vis_threshold or lm[b, 3] < vis_threshold
        cv2.line(out, px(a), px(b),
                 COLOR_OCCLUDED if faded else COLOR_BONE,
                 1 if faded else 2, cv2.LINE_AA)

    for i in range(len(lm)):
        if lm[i, 3] < 0.05:
            continue
        radius = 4 if lm[i, 3] >= vis_threshold else 2
        cv2.circle(out, px(i), radius, _landmark_color(i), -1, cv2.LINE_AA)

    if draw_payload:
        for key, info in draw_payload.items():
            if not info.get("on_screen"):
                continue
            x1, y1, x2, y2 = (int(v) for v in info["bbox"])
            colour = {"red_box": (60, 60, 235),
                      "yellow_box": (60, 220, 235),
                      "main_box": (225, 225, 225)}.get(key, (255, 255, 255))
            solid = info.get("visible")
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2 if solid else 1)
            cv2.putText(out, f"{key} {info.get('visible_fraction', 0):.2f}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colour, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True, help="A take directory")
    ap.add_argument("--frames", default="0",
                    help="Comma-separated frame indices (0-based within the take)")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--camera", default=None,
                    help="Camera sub-directory; defaults to the first found")
    args = ap.parse_args()

    take = Path(args.take)
    meta = json.loads((take / "meta.json").read_text(encoding="utf-8"))
    camera = args.camera or meta["cameras"][0]

    poses = np.load(take / camera / "pose_2d.npy")
    labels = np.load(take / camera / "labels.npy")
    payload = json.loads((take / camera / "payload.json").read_text(encoding="utf-8"))
    frame_dir = take / camera / "frames"

    out_dir = Path(args.out) if args.out else take / camera / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    for token in args.frames.split(","):
        i = int(token)
        img_path = frame_dir / f"{i:05d}.png"
        if not img_path.exists():
            print(f"missing {img_path}")
            continue
        img = cv2.imread(str(img_path))
        vis = draw_pose(img, poses[i], draw_payload=payload[i])
        cv2.putText(vis, f"frame {i}  step {int(labels[i])}  [{camera}]",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        dest = out_dir / f"overlay_{i:05d}.png"
        cv2.imwrite(str(dest), vis)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
