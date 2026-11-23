"""Extract labeled activity windows from a visually reviewed video interval (seconds)."""
import argparse
import hashlib
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.project import load_project
from pipeline.har_pipeline import HARPipeline


def extract(video, step, start, end, output, detector_weights=None, project=None):
    if output.exists() or not 0 <= start < end:
        raise ValueError("Choose a fresh output path and a positive interval")
    p = HARPipeline(headless=True, enable_voice=False, enable_recording=False,
                    detector_weights=detector_weights, project=project, use_threaded=False)
    cap = cv2.VideoCapture(str(video))
    windows = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0 or step not in p._lstm_label_map.values():
            raise ValueError("Invalid video or unknown classifier step ID")
        frame_id = 0
        while frame_id / fps < end:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_id / fps
            result = p.process_frame(frame, timestamp=t, label=step if t >= start else None)
            if result["sampled"] and p.skeleton_buffer.full and all(
                    label == step for label in p.skeleton_buffer.labels()):
                windows.append(p.skeleton_buffer.array().copy())
            frame_id += 1
        if not windows:
            raise ValueError("The reviewed interval contains no complete model window")
        with video.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            np.savez_compressed(stream, x=windows, step=step, source=digest,
                                project_id=project.id if project else "", layout=p.feature_layout,
                                sample_every=p.sample_every, rack_normalized=p.rack_normalizer is not None,
                                start=start, end=end)
        print(f"{len(windows)} reviewed windows saved to {output}")
    finally:
        cap.release()
        p.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("step", type=int)
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("output", type=Path)
    ap.add_argument("--detector-weights")
    ap.add_argument("--project", type=load_project)
    extract(**vars(ap.parse_args()))
