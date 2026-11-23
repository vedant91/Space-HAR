"""
Record a real performance of the protocol from a webcam, labelled live.

Writes the same layout as a rendered take, so tools/build_sequences.py (the
take-grouped, sanctioned path) consumes it unchanged:

    <out>/take_XXXX/meta.json
    <out>/take_XXXX/webcam/frames/00000.png ...
    <out>/take_XXXX/webcam/labels.npy    (T,) step id per frame
    <out>/take_XXXX/webcam/times.npy     (T,) capture time, seconds from start

A real person has no pose ground truth, so there is no pose_2d.npy; the take
is scored through MediaPipe like any live frame.

Keys:  SPACE  start / pause recording
       1-9    select a step by position (default BAS procedure has eight)
       [ / ]  select the previous / next step
       q      save the take and quit

    python tools/record_take.py --out dataset/real --tag real_nominal
    python tools/record_take.py --out dataset/real --tag real_skip       # skip a step on purpose
    python tools/record_take.py --out dataset/real --tag real_recover    # skip, then go back
"""

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import CAMERA_INDEX, EXPERIMENT_STEPS, FRAME_WIDTH, FRAME_HEIGHT, FPS  # noqa: E402
from config.project import load_project  # noqa: E402


def _writer(q: "queue.Queue"):
    while True:
        item = q.get()
        if item is None:
            return
        path, frame = item
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/real")
    ap.add_argument("--tag", default="real_nominal")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--project", type=Path, help="Record the steps defined in this project file")
    args = ap.parse_args()
    try:
        project = load_project(args.project) if args.project else None
    except ValueError as exc:
        ap.error(str(exc))
    steps = project.steps if project else EXPERIMENT_STEPS

    root = Path(args.out)
    index = len(list(root.glob("take_*")))
    cam_dir = root / f"take_{index:04d}" / "webcam"
    frame_dir = cam_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        sys.exit(f"camera {args.camera} did not open")

    # Blocking queue: if the disk falls behind, capture slows (and times.npy
    # records it) rather than frames being dropped silently.
    q = queue.Queue(maxsize=256)
    worker = threading.Thread(target=_writer, args=(q,), daemon=True)
    worker.start()

    labels, times = [], []
    position, recording, t_start = 0, False, None
    print(f"take_{index:04d} [{args.tag}] SPACE start  1-{min(len(steps), 9)} select  [/] previous/next  q save")
    for number, step in enumerate(steps, start=1):
        print(f"  {number}. {step['name']} (label {step['id']})")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed; saving what was recorded")
            break
        step = steps[position]
        if recording:
            if t_start is None:
                t_start = time.perf_counter()
            q.put((frame_dir / f"{len(labels):05d}.png", frame.copy()))
            labels.append(step["id"])
            times.append(time.perf_counter() - t_start)

        view = frame.copy()
        cv2.putText(view, f"{'REC' if recording else 'PAUSED'}  {position + 1}. {step['name']}  frames {len(labels)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.imshow("record_take", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif ord("1") <= key < ord("1") + min(len(steps), 9):
            position = key - ord("1")
        elif key == ord("["):
            position = max(0, position - 1)
        elif key == ord("]"):
            position = min(len(steps) - 1, position + 1)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    q.put(None)
    worker.join()

    if not labels:
        print("nothing recorded; no take written")
        return

    np.save(cam_dir / "labels.npy", np.array(labels, dtype=np.int64))
    np.save(cam_dir / "times.npy", np.array(times, dtype=np.float64))
    duration = times[-1] if times[-1] > 0 else 1.0
    meta = {
        "index": index,
        "tag": args.tag,
        "real": True,
        "sequence": [k for k, _ in groupby(labels)],   # steps in the order performed
        "cameras": ["webcam"],
        "n_frames": len(labels),
        "fps": round(len(labels) / duration, 2),       # measured, not assumed
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "frames_written": True,
        "per_camera": {"webcam": {"frames": len(labels), "labels": "labels.npy"}},
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if project:
        meta["project_id"] = project.id
    (cam_dir.parent / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved take_{index:04d}: {len(labels)} frames at {meta['fps']} fps, "
          f"sequence {meta['sequence']}")


if __name__ == "__main__":
    main()
