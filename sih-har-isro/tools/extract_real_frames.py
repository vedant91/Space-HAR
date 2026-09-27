"""
Sample frames from a real (non-synthetic) video into dataset/yolo_detection/images/
for real-domain YOLO labeling. Consecutive video frames are near-duplicates, so this
samples on a time interval (default 1/sec), not every frame — see the real-data
scoping in the session that produced this: MVP target is ~150-250 real instances
per box class, which needs frame diversity (many short runs), not frame density
(one long clip sampled densely).

Usage:
    python tools/extract_real_frames.py --source "path/to/video.mp4" --interval-s 1.0
    python tools/extract_real_frames.py --source "path/to/video.mp4" --out dataset/yolo_detection/images/unlabeled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DEFAULT_OUT = "dataset/yolo_detection/images/unlabeled"


def extract(source: str, out_dir: str, interval_s: float) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps * interval_s))
    stem = Path(source).stem.replace(" ", "_")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    n_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                out_path = Path(out_dir) / f"{stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                n_written += 1
            frame_idx += 1
    finally:
        cap.release()

    return {"frames_read": frame_idx, "frames_written": n_written, "out_dir": out_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample frames from real video for labeling")
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output frame directory")
    parser.add_argument("--interval-s", type=float, default=1.0,
                        help="Seconds between sampled frames (default: 1.0)")
    args = parser.parse_args()

    result = extract(args.source, args.out, args.interval_s)
    print(f"Read {result['frames_read']} frames, wrote {result['frames_written']} "
         f"sampled frames to {result['out_dir']}")
