"""
Build LSTM training sequences from the TRAINED PoseNet's own inference
=========================================================================
Critical train/inference-consistency fix: data_generation/synthetic_pose.py's
generate_dataset() feeds the LSTM procedurally-generated GROUND-TRUTH pose
vectors — exact, noise-free x/y positions. That's what PoseNet is trained
*against* (see train/train_posenet.py), but it is NOT what the LSTM sees at
actual deployment: at inference, HARPoseNet's own (imperfect — see README's
"Current results", elbows/wrists score only 0.22-0.40 PCK) predictions feed
the LSTM. Training the LSTM almost entirely on clean ground truth and only a
sliver of real-video PoseNet output (see build_real_dataset.py — a few dozen
windows) left it unable to cope with PoseNet's actual output distribution:
measured end-to-end accuracy on synthetic video WITHOUT oracle pose injection
was 0.125 (near-random), vs. 1.000 with oracle injection.

The fix mirrors build_real_dataset.py's approach but for the SYNTHETIC domain:
re-render the same procedural sequences (simulation/renderer.py) and run the
already-trained PoseNet over every frame, using ITS predictions (not the
ground truth used only to pose the renderer's figure) as the LSTM's training
input. The step_id label is still exactly known (it drove the rendering), so
this needs no manual annotation — same "free labels" property as Stage 1
PoseNet training, just pointed at the LSTM instead.

Usage:
    python data_generation/build_posenet_realistic_dataset.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_WINDOW

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_dataset(output_dir: str, n_sequences_per_step: int = 24, frames_per_seq: int = 60,
                  window: int = SEQUENCE_WINDOW, stride: int = 15, seed: int = 17,
                  include_orientations: bool = True) -> Dict:
    from data_generation.synthetic_pose import generate_sequence, sequences_to_windows
    from simulation.renderer import render_frame, MicrogravityWorld
    from pipeline.pose_net import PoseNetWrapper

    pose_wrapper = PoseNetWrapper()
    if not pose_wrapper.available:
        raise RuntimeError("No trained PoseNet available — run train/train_posenet.py first.")
    logger.info("Using PoseNet backend: %s", pose_wrapper.backend)

    rng = np.random.default_rng(seed)
    orientations = (0, 1, 2, 3) if include_orientations else (0,)
    steps = list(range(1, 9))

    all_x, all_y, all_groups = [], [], []
    per_step = {}
    trial_id = 0
    t0 = time.time()
    n_ori = len(orientations)
    per_ori = max(1, n_sequences_per_step // n_ori)

    for step_id in steps:
        step_windows = 0
        for ori in orientations:
            for _ in range(per_ori):
                gt_seq = generate_sequence(step_id, n_frames=frames_per_seq, rng=rng,
                                           orientation=ori, drift=True)
                # A per-trial persistent MicrogravityWorld — matching
                # simulation/renderer.py:render_step_clip's own convention
                # (used for this project's existing CNN training frames) —
                # matters here: without one, render_frame falls back to
                # _box_layout's simpler step-keyed placement, which looks
                # different from what simulation/space_sim.py's full-protocol
                # sim (a single world shared across all 8 steps) renders.
                # Training on the wrong box-rendering style measurably hurt
                # live end-to-end accuracy (0.395 vs 0.953 on a matched-style
                # held-out set) even though PoseNet/LSTM were otherwise fine.
                world = MicrogravityWorld(FRAME_WIDTH, FRAME_HEIGHT, rng)

                # Render each frame from the ground-truth pose, then re-featurize
                # with the ACTUAL trained PoseNet — its output (not gt_seq) is
                # what the LSTM trains on, closing the train/inference gap.
                posenet_seq = np.zeros((frames_per_seq, gt_seq.shape[-1]), dtype=np.float32)
                for i in range(frames_per_seq):
                    t = i / max(frames_per_seq - 1, 1)
                    frame_bgr, _meta = render_frame(gt_seq[i], step_id, t,
                                                    width=FRAME_WIDTH, height=FRAME_HEIGHT,
                                                    rng=rng, world=world)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    posenet_seq[i] = pose_wrapper.process(frame_rgb)

                xw, yw = sequences_to_windows(posenet_seq, step_id, window, stride)
                if len(xw) == 0:
                    continue
                all_x.append(xw)
                all_y.append(yw)
                all_groups.append(np.full(len(xw), trial_id, dtype=np.int64))
                step_windows += len(xw)
                trial_id += 1
        per_step[int(step_id)] = int(step_windows)
        logger.info("Step %d: %d PoseNet-realistic windows (%.0fs elapsed)",
                   step_id, step_windows, time.time() - t0)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    idx = rng.permutation(len(X))
    X, y, groups = X[idx], y[idx], groups[idx]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / "X_sequences.npy"), X)
    np.save(str(out / "y_labels.npy"), y)
    np.save(str(out / "groups.npy"), groups)
    meta = {
        "total_windows": int(len(X)),
        "n_source_trials": int(trial_id),
        "feature_dim": int(X.shape[-1]),
        "window": int(window),
        "source": "synthetic_via_trained_posenet",
        "posenet_backend": pose_wrapper.backend,
        "seed": int(seed),
        "steps": [{"step_id": k, "windows": v} for k, v in per_step.items()],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s  X=%s y=%s groups=%s (%.0fs)", out, X.shape, y.shape, groups.shape,
               meta["elapsed_sec"])
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build LSTM training data from the trained "
                                                 "PoseNet's own predictions on synthetic renders")
    parser.add_argument("--output", default="dataset/skeleton_sequences_posenet")
    parser.add_argument("--n-sequences-per-step", type=int, default=24)
    parser.add_argument("--frames-per-seq", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    meta = build_dataset(args.output, n_sequences_per_step=args.n_sequences_per_step,
                         frames_per_seq=args.frames_per_seq, seed=args.seed)
    print(json.dumps(meta, indent=2))
