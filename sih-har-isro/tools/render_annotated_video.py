"""
Render an annotated video showing exactly what the real pipeline detects and
decides, frame by frame — the same HARPipeline.process_frame(..., annotate=True)
call the live GUI and camera loop use. Works on any video file (a real phone
recording, not just a rendered "take" — see tools/showcase_overlay.py for the
Blender-take-specific version this project also has).

Draws (via HARPipeline._annotate_frame, unmodified — this tool doesn't
reimplement drawing logic): detection boxes + labels, current/next step,
predicted step + confidence, elapsed time. What the box detector actually saw
is what you see; nothing here is a mock or a rendering of ground truth.

Usage:
    python tools/render_annotated_video.py --source "path/to/video.mp4" --out out.mp4
    python tools/render_annotated_video.py --source "path/to/video.mp4" --out out.mp4 \
        --detector-backend yolo --hmr-backend custom
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import FRAME_WIDTH, FRAME_HEIGHT, FPS


def render(source: str, out_path: str, detector_backend: str | None = None,
          hmr_backend: str | None = None, max_frames: int | None = None) -> dict:
    from pipeline.har_pipeline import HARPipeline

    pipeline = HARPipeline(
        source=source, headless=True, enable_voice=False, enable_recording=False,
        use_threaded=False,  # sequential: matches what annotate=True's timing reflects
        detector_backend=detector_backend, hmr_backend=hmr_backend,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        pipeline.close()
        raise RuntimeError(f"Cannot open video source: {source}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    t_start = time.perf_counter()
    n_written = 0
    try:
        while max_frames is None or n_written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, annotate=True)
            if result["vis"] is not None:
                writer.write(result["vis"])
                n_written += 1
    finally:
        writer.release()
        cap.release()
        pipeline.close()

    return {
        "frames_written": n_written,
        "out_path": out_path,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an annotated video from the real pipeline")
    parser.add_argument("--source", required=True, help="Input video file")
    parser.add_argument("--out", required=True, help="Output annotated .mp4 path")
    parser.add_argument("--detector-backend", type=str, default=None, choices=["hsv", "yolo"])
    parser.add_argument("--hmr-backend", type=str, default=None,
                        choices=["none", "mediapipe", "custom", "romp"])
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (quick smoke test)")
    args = parser.parse_args()

    result = render(args.source, args.out, detector_backend=args.detector_backend,
                    hmr_backend=args.hmr_backend, max_frames=args.max_frames)
    print(f"Wrote {result['frames_written']} annotated frames to {result['out_path']} "
         f"in {result['wall_s']}s")
