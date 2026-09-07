"""
Turn rendered Blender takes into LSTM training sequences.

Writes X_sequences.npy / y_labels.npy / groups.npy in exactly the layout
`train/train_lstm.py` already expects, so no training code changes.

Two things this does that the existing `data_generation/synthetic_pose.py`
path could not:

1.  `--pose-source mediapipe` runs the real pose estimator over the rendered
    frames, so the training distribution is *what the deployed pipeline will
    actually see* — MediaPipe's output, with its jitter, its dropouts and its
    confusion on an unusual body orientation. Training on ground-truth pose
    and deploying on MediaPipe output is a train/serve skew that no amount of
    held-out splitting will reveal.

2.  `groups.npy` is the TAKE index, not a synthetic trial id. train_lstm.py
    splits with GroupShuffleSplit, so a whole take — one crew orientation,
    one lighting condition, one complete performance — lands entirely in
    train or entirely in val. That makes the validation number mean
    "generalises to an unseen performance", which is the claim being made.
    Splitting by window instead lets two 50%-overlapping windows from the
    same second of the same take sit on opposite sides of the split.

Usage:
    python tools/build_sequences.py --dataset dataset/blender \
        --out dataset/sequences_mp --pose-source mediapipe \
        --holdout-tags orient_180,orient_-45
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import SEQUENCE_WINDOW, SKELETON_FEATURES  # noqa: E402


def _load_take(take_dir: Path, camera: Optional[str] = None) -> Optional[dict]:
    meta_path = take_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = camera or meta["cameras"][0]
    cam_dir = take_dir / cam
    if not (cam_dir / "pose_2d.npy").exists():
        return None
    return {"meta": meta, "camera": cam, "dir": cam_dir}


def _mediapipe_poses(cam_dir: Path, n_frames: int, complexity: int,
                     rack_normalize: bool) -> Tuple[np.ndarray, dict]:
    """Run the deployed pose backend over a take's rendered frames."""
    import cv2
    from pipeline.pose_backend import PoseBackend
    from pipeline.rack_frame import RackFrameNormalizer, pick_rack_rect
    from pipeline.hsv_detector import HSVBoxDetector

    frame_dir = cam_dir / "frames"
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no rendered frames under {frame_dir}")

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    backend = PoseBackend(complexity=complexity, min_det_conf=0.3,
                          min_trk_conf=0.3, downscale=1,
                          frame_width=w, frame_height=h,
                          static_image_mode=False)
    normalizer = RackFrameNormalizer() if rack_normalize else None
    detector = HSVBoxDetector(frame_width=w, frame_height=h) if rack_normalize else None

    out = np.zeros((len(frames), SKELETON_FEATURES), dtype=np.float32)
    detected = 0
    t0 = time.time()
    for i, path in enumerate(frames):
        img = cv2.imread(str(path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats = backend.process(rgb, timestamp_ms=int(i * 1000 / 30))
        if np.any(feats):
            detected += 1
        if normalizer is not None:
            feats = normalizer.normalize(
                feats, rack_rect=pick_rack_rect(detector.detect(img)))
        out[i] = feats
    backend.close()

    stats = {
        "frames": len(frames),
        "pose_detected": detected,
        "detect_rate": round(detected / max(len(frames), 1), 4),
        "seconds": round(time.time() - t0, 1),
        "backend": backend.backend,
    }
    return out, stats


def _windows(seq: np.ndarray, labels: np.ndarray, window: int, stride: int,
             purity: float = 0.75):
    """Sliding windows, labelled by the dominant step.

    `purity` is the fraction of frames that must share the majority label for
    the window to be kept. Requiring 1.0 (strictly pure windows) is the
    obvious choice and the wrong one here, for two reasons:

    * Data volume. A step lasting ~32 frames against a 30-frame window admits
      only 3 pure start positions, so a whole 8-step take yields ~24 windows
      even at stride 1. There is not enough signal there to train anything.

    * Realism. The deployed pipeline slides a 30-frame buffer continuously
      over a live camera; the majority of windows it ever sees near a
      transition ARE mixed. Training only on pure windows means every step
      boundary at inference is out of distribution — precisely the moments
      the state machine depends on. `STEP_CONFIRM_FRAMES` already debounces
      the noisy stretch, so a dominant-label window is exactly what the state
      machine is built to consume.

    The label is the majority step, not the last frame's: a window that is
    80% step 3 and ends on the first frame of step 4 is a picture of step 3.
    """
    xs, ys = [], []
    for start in range(0, len(seq) - window + 1, stride):
        lab = labels[start:start + window]
        values, counts = np.unique(lab, return_counts=True)
        dominant = int(values[int(np.argmax(counts))])
        if counts.max() / float(window) < purity:
            continue
        xs.append(seq[start:start + window])
        ys.append(dominant)
    if not xs:
        return (np.zeros((0, window, seq.shape[-1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def build(dataset_dir: str, out_dir: str, pose_source: str = "mediapipe",
          stride: int = 2, complexity: int = 1, camera: Optional[str] = None,
          purity: float = 0.75,
          holdout_tags: Sequence[str] = (), rack_normalize: bool = False,
          exclude_tags: Sequence[str] = ()) -> dict:
    root = Path(dataset_dir)
    takes = sorted(p for p in root.glob("take_*") if p.is_dir())
    if not takes:
        raise SystemExit(f"no takes under {root}")

    out_train = Path(out_dir)
    out_test = Path(f"{out_dir}_holdout")
    for d in (out_train, out_test):
        d.mkdir(parents=True, exist_ok=True)

    bucket = {"train": {"x": [], "y": [], "g": []},
              "holdout": {"x": [], "y": [], "g": []}}
    report: List[dict] = []

    for take_dir in takes:
        loaded = _load_take(take_dir, camera)
        if loaded is None:
            continue
        meta, cam_dir = loaded["meta"], loaded["dir"]
        tag = meta.get("tag", "nominal")
        if tag in exclude_tags:
            print(f"  skip {take_dir.name} (tag={tag}, excluded)")
            continue

        labels = np.load(cam_dir / "labels.npy")
        if pose_source == "groundtruth":
            poses = np.load(cam_dir / "pose_2d.npy")
            stats = {"frames": len(poses), "detect_rate": 1.0,
                     "backend": "blender_groundtruth"}
        else:
            if not meta.get("frames_written", True):
                print(f"  skip {take_dir.name} (no frames rendered)")
                continue
            poses, stats = _mediapipe_poses(cam_dir, len(labels), complexity,
                                            rack_normalize)

        n = min(len(poses), len(labels))
        poses, labels_n = poses[:n], labels[:n]

        xs, ys = _windows(poses, labels_n, SEQUENCE_WINDOW, stride, purity)
        split = "holdout" if tag in holdout_tags else "train"
        if len(xs):
            bucket[split]["x"].append(xs)
            bucket[split]["y"].append(ys)
            bucket[split]["g"].append(np.full(len(xs), meta["index"], dtype=np.int64))

        row = {"take": take_dir.name, "index": meta["index"], "tag": tag,
               "roll": meta.get("body_roll_deg"), "lighting": meta.get("lighting"),
               "distractors": meta.get("distractors"), "camera": loaded["camera"],
               "windows": int(len(xs)), "split": split, **stats}
        report.append(row)
        print(f"  {take_dir.name} tag={tag:16s} -> {len(xs):4d} windows "
              f"[{split}] detect={stats.get('detect_rate')}", flush=True)

    written = {}
    for split, target in (("train", out_train), ("holdout", out_test)):
        if not bucket[split]["x"]:
            continue
        X = np.concatenate(bucket[split]["x"])
        y = np.concatenate(bucket[split]["y"])
        g = np.concatenate(bucket[split]["g"])
        np.save(target / "X_sequences.npy", X)
        np.save(target / "y_labels.npy", y)
        np.save(target / "groups.npy", g)
        meta_out = {
            "total_windows": int(len(X)),
            "n_source_takes": int(len(np.unique(g))),
            "feature_dim": int(X.shape[-1]),
            "window": SEQUENCE_WINDOW,
            "stride": stride,
            "source": f"blender_render/{pose_source}",
            "rack_normalized": bool(rack_normalize),
            "class_counts": {int(k): int(v)
                             for k, v in zip(*np.unique(y, return_counts=True))},
        }
        (target / "metadata.json").write_text(json.dumps(meta_out, indent=2),
                                              encoding="utf-8")
        written[split] = {"dir": str(target), **meta_out}
        print(f"{split}: X={X.shape} y={y.shape} takes={meta_out['n_source_takes']}")

    summary = {"pose_source": pose_source, "stride": stride,
               "holdout_tags": list(holdout_tags), "takes": report,
               "written": written}
    (out_train / "build_report.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/blender")
    ap.add_argument("--out", default="dataset/sequences_mp")
    ap.add_argument("--pose-source", default="mediapipe",
                    choices=["mediapipe", "groundtruth"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--purity", type=float, default=0.75,
                    help="Fraction of a window that must share the majority "
                         "step label for it to be kept")
    ap.add_argument("--complexity", type=int, default=1,
                    help="0=lite 1=full 2=heavy pose bundle")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--rack-normalize", action="store_true")
    ap.add_argument("--holdout-tags", default="",
                    help="Comma-separated take tags to route into the holdout set")
    ap.add_argument("--exclude-tags", default="skip,recover",
                    help="Tags to leave out entirely (error sequences are for "
                         "state-machine evaluation, not step training)")
    args = ap.parse_args()

    holdout = [t for t in args.holdout_tags.split(",") if t]
    exclude = [t for t in args.exclude_tags.split(",") if t]
    build(args.dataset, args.out, args.pose_source, args.stride,
          args.complexity, args.camera, args.purity, holdout, args.rack_normalize, exclude)


if __name__ == "__main__":
    main()
