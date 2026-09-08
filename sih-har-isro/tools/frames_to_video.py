"""
Mux a directory of numbered PNG frames into an .mp4.

The Blender build on this machine has no FFMPEG muxer, so `render_video.py`
writes an image sequence and this encodes it with OpenCV (which carries its
own ffmpeg). Frames are taken in filename-sorted order.

    python tools/frames_to_video.py build/showcase/protocol_frames build/showcase/protocol.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def encode(frame_dir: Path, out: Path, fps: float) -> None:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        sys.exit(f"no PNG frames in {frame_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        sys.exit(f"cannot read {frames[0]}")
    h, w = first.shape[:2]

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        sys.exit("cv2.VideoWriter failed to open - codec unavailable")

    for f in frames:
        img = cv2.imread(str(f))
        if img is None or img.shape[:2] != (h, w):
            sys.exit(f"frame {f.name} missing or wrong size")
        writer.write(img)
    writer.release()

    size_mb = out.stat().st_size / 1e6
    print(f"VIDEO_OK {out} ({len(frames)} frames, {len(frames) / fps:.1f} s, "
          f"{size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frame_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    encode(args.frame_dir, args.out, args.fps)


if __name__ == "__main__":
    main()
