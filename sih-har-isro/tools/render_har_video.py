"""
Render a video file through the live HAR pipeline to an annotated MP4.

Every frame goes through `HARPipeline.process_frame()` - the same
perception, features, classifier and state machine the camera loop runs -
and the output is the pipeline's own monitor view (`_annotate_frame`):
camera frame with the perceived body and boxes, plus the side panel with the
3D pose, the current/next step, the 8-step checklist and a caption of every
line the voice system speaks.

    python tools/render_har_video.py --video ../testing-videos/test-har-3.mp4 \
        --out ../testing-videos/test-har3_monitor.mp4 [--show-3d-panel]

Frame timestamps come from the file (frame / fps), not the wall clock, so the
result does not depend on how fast this machine renders.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.project import load_project
from pipeline.har_pipeline import HARPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", type=Path, help="Project JSON for the monitor overlay")
    ap.add_argument("--lstm", default=None, help="checkpoint to serve (default: the promoted model)")
    ap.add_argument("--show-3d-panel", action="store_true",
                    help="load HMR3DNet as the 3D fallback when MediaPipe loses the body")
    ap.add_argument("--detector-weights", help="YOLO payload checkpoint for this render")
    ap.add_argument("--viz-3d", choices=["blazepose", "hmr", "lift", "refined"], default="blazepose",
                    help="3D view source: BlazePose world, skeleton-only lift, fused HMR depth, or refined BAS pose")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    if args.project and args.lstm:
        ap.error("Choose the classifier in the project file")
    try:
        project = load_project(args.project) if args.project else None
    except ValueError as exc:
        ap.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    p = HARPipeline(lstm_path=args.lstm, headless=True, enable_voice=False, enable_recording=False,
                    use_threaded=False, show_3d_panel=args.show_3d_panel, viz_3d=args.viz_3d,
                    detector_weights=args.detector_weights, project=project)
    out = Path(args.out)
    tmp = out.with_suffix(".raw.mp4")
    writer = None
    i, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        r = p.process_frame(frame, annotate=True, timestamp=i / fps)
        vis = r["vis"]
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        i += 1
        if i % 120 == 0:
            print(f"{i}/{total} frames, {time.time() - t0:.0f}s", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    p.close()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and tmp.exists():
        # H.264 so it plays in browsers / PowerPoint (mp4v often does not).
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-preset", "medium", str(out)], check=True)
        tmp.unlink()
    elif tmp.exists():
        tmp.replace(out)
    summary = p.state_machine.get_status_summary()
    print(f"wrote {out} ({i} frames)")
    for st in summary["steps"]:
        print(f"  step {st['id']} {st['name']:22s} {st['status']}")


if __name__ == "__main__":
    main()
