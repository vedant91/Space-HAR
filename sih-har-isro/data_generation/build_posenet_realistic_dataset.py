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


def build_protocol_dataset(output_dir: str, n_protocols: int = 32, frames_per_step: int = 72,
                           window: int = SEQUENCE_WINDOW, stride: int = 5, seed: int = 101) -> Dict:
    """Same idea as build_dataset, but each trial is a FULL 8-step protocol run in
    ONE persistent MicrogravityWorld (like simulation/renderer.py:render_protocol,
    what the deployed pipeline actually sees), not 60-frame clips that each start a
    fresh world. Object state (where the boxes are, who holds them) carries over
    between steps, which changes what PoseNet sees around the wrists.

    Orientation is balanced per step but independent ACROSS steps — deliberately NOT
    render_protocol's fixed step->orientation schedule, so the LSTM can't learn
    "orientation implies step" from training data that mirrors one evaluation recipe.
    Groups are per-protocol so a train/val split never shares a world across sides."""
    from data_generation.synthetic_pose import generate_sequence, sequences_to_windows
    from simulation.renderer import render_frame, MicrogravityWorld
    from pipeline.pose_net import PoseNetWrapper

    pose_wrapper = PoseNetWrapper()
    if not pose_wrapper.available:
        raise RuntimeError("No trained PoseNet available — run train/train_posenet.py first.")
    rng = np.random.default_rng(seed)
    all_x, all_y, all_g = [], [], []
    t0 = time.time()
    # Balanced orientation coverage: every step sees each of the 4 orientations equally
    # often (shuffled independently per step). Plain independent draws left some
    # (step, orientation) combos with ~0 samples at 12 protocols, and the LSTM then
    # misread those (e.g. step 8 read as step 2 at unseen orientations).
    ori_plan = [rng.permutation(np.resize(np.arange(4), n_protocols)) for _ in range(8)]
    for p in range(n_protocols):
        world = MicrogravityWorld(FRAME_WIDTH, FRAME_HEIGHT, rng)
        for step_id in range(1, 9):
            gt_seq = generate_sequence(step_id, n_frames=frames_per_step, rng=rng,
                                       orientation=int(ori_plan[step_id - 1][p]), drift=True)
            feats = np.zeros((frames_per_step, gt_seq.shape[-1]), dtype=np.float32)
            for i in range(frames_per_step):
                frame_bgr, _ = render_frame(gt_seq[i], step_id, i / max(frames_per_step - 1, 1),
                                            width=FRAME_WIDTH, height=FRAME_HEIGHT, rng=rng, world=world)
                feats[i] = pose_wrapper.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            xw, yw = sequences_to_windows(feats, step_id, window, stride)
            all_x.append(xw); all_y.append(yw)
            all_g.append(np.full(len(xw), 100000 + p, dtype=np.int64))
        logger.info("Protocol %d/%d done (%.0fs elapsed)", p + 1, n_protocols, time.time() - t0)

    X, y, groups = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_g)
    idx = rng.permutation(len(X))
    X, y, groups = X[idx], y[idx], groups[idx]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / "X_sequences.npy"), X)
    np.save(str(out / "y_labels.npy"), y)
    np.save(str(out / "groups.npy"), groups)
    meta = {"total_windows": int(len(X)), "n_protocols": n_protocols, "feature_dim": int(X.shape[-1]),
            "window": int(window), "stride": int(stride), "source": "protocol_via_trained_posenet",
            "posenet_backend": pose_wrapper.backend, "seed": int(seed),
            "elapsed_sec": round(time.time() - t0, 1)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s  X=%s (%.0fs)", out, X.shape, meta["elapsed_sec"])
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build LSTM training data from the trained "
                                                 "PoseNet's own predictions on synthetic renders")
    parser.add_argument("--output", default="dataset/skeleton_sequences_posenet")
    parser.add_argument("--n-sequences-per-step", type=int, default=24)
    parser.add_argument("--frames-per-seq", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--protocol", action="store_true",
                        help="build from full persistent-world protocol runs instead of per-step clips")
    parser.add_argument("--n-protocols", type=int, default=32)
    args = parser.parse_args()
    if args.protocol:
        meta = build_protocol_dataset(args.output, n_protocols=args.n_protocols)
    else:
        meta = build_dataset(args.output, n_sequences_per_step=args.n_sequences_per_step,
                             frames_per_seq=args.frames_per_seq, seed=args.seed)
    print(json.dumps(meta, indent=2))
