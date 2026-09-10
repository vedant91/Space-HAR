"""
Build LSTM/CNN training sequences from the real "gravitational-mimic" videos
==============================================================================
Bridges data_generation/real_video_autolabel.py's per-frame pseudo-labels
(dataset/real_pseudo/<video>/{frames/, poses.npz, step_labels.npy}) into the
exact same windowed-sequence contract data_generation/synthetic_pose.py's
generate_dataset() produces (X_sequences.npy, y_labels.npy, groups.npy,
metadata.json), using the (by then Stage-1+Stage-2 trained) custom PoseNet —
not the classical-CV wrist pseudo-labels — to featurize every frame, since
that's the exact feature distribution the deployed pipeline will produce.

Only contiguous runs of a single non-idle pseudo-label are windowed (an idle/
unknown-labeled frame, or a boundary between two different steps, never
contributes a window) — same "don't train on a mixed/ambiguous window"
principle simulation/space_sim.py's oracle scoring already applies. Short
runs (rare pick/examine actions caught for only a few frames in these short
clips) simply contribute no windows; synthetic data remains the reliable
backbone for full 8-class coverage. See merge_with_synthetic() for combining
the two into one training set with a leakage-safe group split still intact.

Usage:
    python data_generation/build_real_dataset.py                 # build + merge
    python data_generation/build_real_dataset.py --no-merge       # build only
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    REAL_PSEUDO_DIR, REAL_SEQUENCES_DIR, COMBINED_SEQUENCES_DIR, REAL_ANNOTATED_DIR,
    SEQUENCE_WINDOW, RACK_FRAME_NORMALIZE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_REAL_GROUP_OFFSET = 1_000_000  # keeps real group ids disjoint from synthetic ones when merged
_REAL_STRIDE = 10               # denser than synthetic's stride=15 — real runs are short


def _runs_of_equal_labels(labels: np.ndarray):
    """Yield (start, end_exclusive, label) for each maximal run of one value."""
    if len(labels) == 0:
        return
    start = 0
    cur = labels[0]
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != cur:
            yield start, i, int(cur)
            if i < len(labels):
                start, cur = i, labels[i]


def featurize_video(video_dir: Path, pose_wrapper) -> np.ndarray:
    """Run the (trained) PoseNetWrapper over every saved frame -> (N,132)."""
    frames_dir = video_dir / "frames"
    frame_files = sorted(frames_dir.glob("*.jpg"))
    feats = np.zeros((len(frame_files), 132), dtype=np.float32)
    for i, fp in enumerate(frame_files):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feats[i] = pose_wrapper.process(rgb)
    return feats


def build_real_sequences(pseudo_dir: str = REAL_PSEUDO_DIR,
                         out_dir: str = REAL_SEQUENCES_DIR,
                         window: int = SEQUENCE_WINDOW,
                         stride: int = _REAL_STRIDE,
                         rack_normalize: bool = RACK_FRAME_NORMALIZE,
                         min_run_frac_of_window: float = 1.0) -> Dict:
    """Featurize every dataset/real_pseudo/<video>/ with the trained PoseNet,
    window each contiguous non-idle pseudo-label run, and write the same
    (X,y,groups,metadata) contract synthetic_pose.generate_dataset() does."""
    from pipeline.pose_net import PoseNetWrapper
    pose_wrapper = PoseNetWrapper()
    if not pose_wrapper.available:
        logger.error("No trained PoseNet available — run train/train_posenet.py first.")
        return {"total_windows": 0, "error": "posenet unavailable"}

    normalizer = None
    if rack_normalize:
        from pipeline.rack_frame import RackFrameNormalizer, pick_rack_rect
        from pipeline.hsv_detector import HSVBoxDetector
        normalizer = RackFrameNormalizer()

    root = Path(pseudo_dir)
    video_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []

    all_x, all_y, all_groups = [], [], []
    group_id = 0
    per_step_windows: Dict[int, int] = {}
    per_video_report = []

    for vdir in video_dirs:
        labels_path = vdir / "step_labels.npy"
        if not labels_path.exists():
            continue
        labels = np.load(str(labels_path))
        feats = featurize_video(vdir, pose_wrapper)
        if rack_normalize:
            # Re-detect boxes per frame for the rack anchor (same HSV rect
            # extraction pipeline/rack_frame.py's docstring describes).
            det = HSVBoxDetector()
            normalizer.reset()
            frame_files = sorted((vdir / "frames").glob("*.jpg"))
            for i, fp in enumerate(frame_files):
                frame = cv2.imread(str(fp))
                rect = pick_rack_rect(det.detect(frame)) if frame is not None else None
                feats[i] = normalizer.normalize(feats[i], rack_rect=rect)

        n = min(len(labels), len(feats))
        video_windows = 0
        for start, end, label in _runs_of_equal_labels(labels[:n]):
            if label == 0:
                continue  # never window an idle/unknown-labeled run
            run_len = end - start
            if run_len < window * min_run_frac_of_window:
                continue
            for w_start in range(start, end - window + 1, stride):
                all_x.append(feats[w_start:w_start + window])
                all_y.append(label)
                all_groups.append(_REAL_GROUP_OFFSET + group_id)
                per_step_windows[label] = per_step_windows.get(label, 0) + 1
                video_windows += 1
            group_id += 1  # one group per contiguous run, not per window
        per_video_report.append({"video": vdir.name, "n_frames": n, "windows": video_windows})
        logger.info("%s: %d windows from pseudo-labeled runs", vdir.name, video_windows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not all_x:
        logger.warning("No windowable real sequences found (all runs shorter than "
                       "SEQUENCE_WINDOW=%d, or all pseudo-labels were idle/unknown).", window)
        meta = {"total_windows": 0, "per_video": per_video_report}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    X = np.stack(all_x, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    groups = np.array(all_groups, dtype=np.int64)
    np.save(str(out / "X_sequences.npy"), X)
    np.save(str(out / "y_labels.npy"), y)
    np.save(str(out / "groups.npy"), groups)

    meta = {
        "total_windows": int(len(X)),
        "n_source_trials": int(group_id),
        "feature_dim": int(X.shape[-1]),
        "window": int(window),
        "source": "real_gravitational_mimic",
        "posenet_backend": pose_wrapper.backend,
        "rack_normalized": bool(rack_normalize),
        "steps": [{"step_id": k, "windows": v} for k, v in sorted(per_step_windows.items())],
        "per_video": per_video_report,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s  X=%s y=%s groups=%s", out, X.shape, y.shape, groups.shape)
    return meta


def build_real_annotated_frames(pseudo_dir: str = REAL_PSEUDO_DIR,
                                out_dir: str = REAL_ANNOTATED_DIR) -> Dict:
    """Populate dataset/annotated_real/step_XX/ from the pseudo-labeled real
    frames, for train/train_cnn.py's FrameStackDataset (multi-root — see its
    docstring). Copies only frames inside a contiguous non-idle-label run,
    named `<video>_run{N}_frame{i}.jpg` so a sorted glob keeps each run's
    frames contiguous and never interleaves two different videos/runs."""
    root = Path(pseudo_dir)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)  # rebuild clean — stale frames from a re-run would double-count
    video_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []

    per_step_count: Dict[int, int] = {}
    for vdir in video_dirs:
        labels_path = vdir / "step_labels.npy"
        frames_dir = vdir / "frames"
        if not labels_path.exists() or not frames_dir.exists():
            continue
        labels = np.load(str(labels_path))
        frame_files = sorted(frames_dir.glob("*.jpg"))
        n = min(len(labels), len(frame_files))

        run_idx = 0
        for start, end, label in _runs_of_equal_labels(labels[:n]):
            if label == 0:
                continue
            step_dir = out / f"step_{label:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            for i in range(start, end):
                dest = step_dir / f"{vdir.name}_run{run_idx:02d}_frame{i - start:05d}.jpg"
                shutil.copyfile(frame_files[i], dest)
                per_step_count[label] = per_step_count.get(label, 0) + 1
            run_idx += 1

    meta = {"per_step_frame_count": per_step_count, "output_dir": str(out)}
    logger.info("Real annotated frames for CNN training: %s -> %s", per_step_count, out)
    return meta


def merge_with_synthetic(synthetic_dir: str = "dataset/skeleton_sequences",
                         real_dir: str = REAL_SEQUENCES_DIR,
                         out_dir: str = COMBINED_SEQUENCES_DIR) -> Dict:
    """Concatenate synthetic + real windowed sequences into one training set.
    Group ids stay disjoint (real ids were offset by _REAL_GROUP_OFFSET at
    build time) so train_lstm.py's GroupShuffleSplit still can't leak a
    window's near-duplicate across train/val."""
    syn_dir, real_p, out = Path(synthetic_dir), Path(real_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    syn_x = np.load(str(syn_dir / "X_sequences.npy"))
    syn_y = np.load(str(syn_dir / "y_labels.npy"))
    syn_groups = np.load(str(syn_dir / "groups.npy")) if (syn_dir / "groups.npy").exists() \
        else np.arange(len(syn_x))

    real_x_path = real_p / "X_sequences.npy"
    if not real_x_path.exists():
        logger.warning("No real sequences at %s — combined set == synthetic only.", real_p)
        real_x = np.zeros((0, *syn_x.shape[1:]), dtype=np.float32)
        real_y = np.zeros((0,), dtype=np.int64)
        real_groups = np.zeros((0,), dtype=np.int64)
    else:
        real_x = np.load(str(real_x_path))
        real_y = np.load(str(real_p / "y_labels.npy"))
        real_groups = np.load(str(real_p / "groups.npy"))

    X = np.concatenate([syn_x, real_x], axis=0)
    y = np.concatenate([syn_y, real_y], axis=0)
    groups = np.concatenate([syn_groups, real_groups], axis=0)

    rng = np.random.default_rng(23)
    idx = rng.permutation(len(X))
    X, y, groups = X[idx], y[idx], groups[idx]

    np.save(str(out / "X_sequences.npy"), X)
    np.save(str(out / "y_labels.npy"), y)
    np.save(str(out / "groups.npy"), groups)
    meta = {
        "total_windows": int(len(X)),
        "n_synthetic_windows": int(len(syn_x)),
        "n_real_windows": int(len(real_x)),
        "feature_dim": int(X.shape[-1]) if len(X) else None,
        "source": "synthetic+real_gravitational_mimic",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Combined dataset: %d synthetic + %d real = %d windows -> %s",
               len(syn_x), len(real_x), len(X), out)
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build real-video sequences and merge with synthetic")
    parser.add_argument("--pseudo-dir", default=REAL_PSEUDO_DIR)
    parser.add_argument("--out-dir", default=REAL_SEQUENCES_DIR)
    parser.add_argument("--synthetic-dir", default="dataset/skeleton_sequences")
    parser.add_argument("--combined-dir", default=COMBINED_SEQUENCES_DIR)
    parser.add_argument("--annotated-dir", default=REAL_ANNOTATED_DIR)
    parser.add_argument("--rack-normalize", action="store_true")
    parser.add_argument("--no-merge", action="store_true", help="Build real sequences only, skip merge")
    args = parser.parse_args()

    annotated_meta = build_real_annotated_frames(args.pseudo_dir, args.annotated_dir)
    print(json.dumps(annotated_meta, indent=2, default=str))
    real_meta = build_real_sequences(args.pseudo_dir, args.out_dir, rack_normalize=args.rack_normalize)
    print(json.dumps(real_meta, indent=2, default=str))
    if not args.no_merge:
        combined_meta = merge_with_synthetic(args.synthetic_dir, args.out_dir, args.combined_dir)
        print(json.dumps(combined_meta, indent=2, default=str))
