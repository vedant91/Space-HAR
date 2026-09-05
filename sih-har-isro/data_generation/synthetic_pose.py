"""
Procedural MediaPipe-style pose generator for the 8-step BAS protocol.

Produces (T, 132) sequences — 33 landmarks x (x, y, z, visibility) — with
distinct kinematics per experiment step so the LSTM has real signal to learn.
Includes microgravity drift, orientation variants, and sensor noise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 4  # 132

# MediaPipe Pose landmark indices
NOSE, L_EYE, R_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


def _ease(t: np.ndarray) -> np.ndarray:
    """Smoothstep easing."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a + (b - a) * t


def _base_floating_pose() -> np.ndarray:
    """Neutral microgravity pose, torso slightly pitched, legs relaxed.

    Returns (33, 4) array of x, y, z, visibility in MediaPipe normalized coords.
    """
    pose = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    pose[:, 3] = 0.95  # high visibility

    # Head / face
    pose[0, :3] = (0.50, 0.22, -0.05)          # nose
    pose[1, :3] = (0.48, 0.20, -0.04)          # left eye inner
    pose[2, :3] = (0.47, 0.20, -0.05)          # left eye
    pose[3, :3] = (0.46, 0.20, -0.04)          # left eye outer
    pose[4, :3] = (0.52, 0.20, -0.04)
    pose[5, :3] = (0.53, 0.20, -0.05)
    pose[6, :3] = (0.54, 0.20, -0.04)
    pose[7, :3] = (0.44, 0.21, -0.02)          # left ear
    pose[8, :3] = (0.56, 0.21, -0.02)
    pose[9, :3] = (0.48, 0.24, -0.06)          # mouth left
    pose[10, :3] = (0.52, 0.24, -0.06)

    # Torso
    pose[L_SHOULDER, :3] = (0.38, 0.34, -0.02)
    pose[R_SHOULDER, :3] = (0.62, 0.34, -0.02)
    pose[L_HIP, :3] = (0.42, 0.58, 0.00)
    pose[R_HIP, :3] = (0.58, 0.58, 0.00)

    # Arms rest slightly forward (typical ISS handrail-ready)
    pose[L_ELBOW, :3] = (0.30, 0.46, -0.04)
    pose[R_ELBOW, :3] = (0.70, 0.46, -0.04)
    pose[L_WRIST, :3] = (0.28, 0.56, -0.08)
    pose[R_WRIST, :3] = (0.72, 0.56, -0.08)

    # Hands
    for i, (wx, wy) in ((L_PINKY, (-0.03, 0.02)), (L_INDEX, (0.02, 0.02)), (L_THUMB, (0.01, -0.01))):
        pose[i, :3] = pose[L_WRIST, :3] + np.array([wx, wy, -0.01], dtype=np.float32)
    for i, (wx, wy) in ((R_PINKY, (0.03, 0.02)), (R_INDEX, (-0.02, 0.02)), (R_THUMB, (-0.01, -0.01))):
        pose[i, :3] = pose[R_WRIST, :3] + np.array([wx, wy, -0.01], dtype=np.float32)

    # Legs (slightly bent, floating)
    pose[L_KNEE, :3] = (0.40, 0.74, 0.04)
    pose[R_KNEE, :3] = (0.60, 0.74, 0.04)
    pose[L_ANKLE, :3] = (0.39, 0.88, 0.02)
    pose[R_ANKLE, :3] = (0.61, 0.88, 0.02)
    pose[L_HEEL, :3] = (0.38, 0.90, 0.03)
    pose[R_HEEL, :3] = (0.62, 0.90, 0.03)
    pose[L_FOOT, :3] = (0.40, 0.92, 0.00)
    pose[R_FOOT, :3] = (0.60, 0.92, 0.00)
    return pose


def _set_arm(pose: np.ndarray, side: str, wrist_xy, z: float = -0.10):
    """Place wrist and infer elbow as a bent IK midpoint."""
    sh_idx = L_SHOULDER if side == "L" else R_SHOULDER
    el_idx = L_ELBOW if side == "L" else R_ELBOW
    wr_idx = L_WRIST if side == "L" else R_WRIST
    shoulder = pose[sh_idx, :3].copy()
    wrist = np.array([wrist_xy[0], wrist_xy[1], z], dtype=np.float32)
    mid = 0.5 * (shoulder + wrist)
    # Bend elbow away from torso (outward + slightly down)
    outward = -0.08 if side == "L" else 0.08
    mid[0] += outward
    mid[1] += 0.04
    mid[2] += 0.02
    pose[el_idx, :3] = mid
    pose[wr_idx, :3] = wrist
    # Hands follow wrist
    if side == "L":
        pose[L_PINKY, :3] = wrist + np.array([-0.03, 0.02, -0.01], dtype=np.float32)
        pose[L_INDEX, :3] = wrist + np.array([0.02, 0.02, -0.01], dtype=np.float32)
        pose[L_THUMB, :3] = wrist + np.array([0.01, -0.01, 0.00], dtype=np.float32)
    else:
        pose[R_PINKY, :3] = wrist + np.array([0.03, 0.02, -0.01], dtype=np.float32)
        pose[R_INDEX, :3] = wrist + np.array([-0.02, 0.02, -0.01], dtype=np.float32)
        pose[R_THUMB, :3] = wrist + np.array([-0.01, -0.01, 0.00], dtype=np.float32)


# Wrist waypoints per step, in (left_xy, right_xy) at t=0, 0.5, 1.0
# Distinct enough that a temporal model can separate them.
_STEP_WAYPOINTS = {
    1: [  # Approach main box — both hands travel inward and down toward container
        ((0.28, 0.56), (0.72, 0.56)),
        ((0.36, 0.58), (0.64, 0.58)),
        ((0.42, 0.62), (0.58, 0.62)),
    ],
    2: [  # Open lid — hands at box then pull apart (unlatch)
        ((0.42, 0.60), (0.58, 0.60)),
        ((0.38, 0.52), (0.62, 0.52)),
        ((0.34, 0.46), (0.66, 0.46)),
    ],
    3: [  # Pick red — both hands to left-of-center, then lift
        ((0.40, 0.62), (0.50, 0.62)),
        ((0.38, 0.58), (0.46, 0.58)),
        ((0.36, 0.44), (0.44, 0.44)),
    ],
    4: [  # Examine red — hands together at chest, small rotation
        ((0.44, 0.42), (0.52, 0.42)),
        ((0.46, 0.38), (0.54, 0.40)),
        ((0.44, 0.40), (0.52, 0.38)),
    ],
    5: [  # Place red — move to left zone and lower
        ((0.36, 0.44), (0.44, 0.44)),
        ((0.26, 0.52), (0.34, 0.52)),
        ((0.22, 0.64), (0.30, 0.64)),
    ],
    6: [  # Pick yellow — both hands to right-of-center, then lift
        ((0.50, 0.62), (0.60, 0.62)),
        ((0.54, 0.58), (0.62, 0.58)),
        ((0.56, 0.44), (0.64, 0.44)),
    ],
    7: [  # Examine yellow — hands together slightly lower/right of red-examine
        ((0.48, 0.44), (0.56, 0.44)),
        ((0.50, 0.40), (0.58, 0.42)),
        ((0.48, 0.42), (0.56, 0.40)),
    ],
    8: [  # Place yellow — move to right zone and lower
        ((0.56, 0.44), (0.64, 0.44)),
        ((0.66, 0.52), (0.74, 0.52)),
        ((0.70, 0.64), (0.78, 0.64)),
    ],
}


def _interpolate_waypoints(waypoints, t: float):
    """t in [0,1] across the waypoint list."""
    n = len(waypoints) - 1
    x = t * n
    i = min(int(x), n - 1)
    local = _ease(np.array([x - i]))[0]
    l0, r0 = waypoints[i]
    l1, r1 = waypoints[i + 1]
    left = (_lerp(l0[0], l1[0], local), _lerp(l0[1], l1[1], local))
    right = (_lerp(r0[0], r1[0], local), _lerp(r0[1], r1[1], local))
    return left, right


def _rotate_xy(pose: np.ndarray, k: int):
    """Rotate x,y around (0.5, 0.5) by k*90 degrees (microgravity orientation)."""
    k = k % 4
    if k == 0:
        return
    xy = pose[:, :2] - 0.5
    for _ in range(k):
        x = xy[:, 0].copy()
        y = xy[:, 1].copy()
        xy[:, 0] = -y
        xy[:, 1] = x
    pose[:, :2] = xy + 0.5


def generate_sequence(
    step_id: int,
    n_frames: int = 60,
    rng: Optional[np.random.Generator] = None,
    orientation: int = 0,
    drift: bool = True,
) -> np.ndarray:
    """Return (n_frames, 132) pose features for one trial of a step."""
    if rng is None:
        rng = np.random.default_rng()
    if step_id not in _STEP_WAYPOINTS:
        raise ValueError(f"Unknown step_id {step_id}")

    waypoints = _STEP_WAYPOINTS[step_id]
    # Per-sequence identity jitter (different astronaut / camera)
    body_shift = rng.normal(0.0, 0.018, size=2).astype(np.float32)
    scale = float(rng.uniform(0.92, 1.08))
    z_bias = float(rng.normal(0.0, 0.02))

    frames = np.zeros((n_frames, NUM_LANDMARKS, 4), dtype=np.float32)
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        pose = _base_floating_pose()
        left, right = _interpolate_waypoints(waypoints, t)
        _set_arm(pose, "L", left, z=-0.10 + z_bias)
        _set_arm(pose, "R", right, z=-0.10 + z_bias)

        # Scale about torso center
        center = pose[[L_HIP, R_HIP, L_SHOULDER, R_SHOULDER], :2].mean(axis=0)
        pose[:, :2] = center + (pose[:, :2] - center) * scale
        pose[:, 0] += body_shift[0]
        pose[:, 1] += body_shift[1]

        if drift:
            # Slow whole-body float
            pose[:, 0] += 0.015 * np.sin(2 * np.pi * t + step_id)
            pose[:, 1] += 0.010 * np.cos(2 * np.pi * t * 0.7 + step_id)

        _rotate_xy(pose, orientation)

        # Landmark noise + visibility
        pose[:, :3] += rng.normal(0.0, 0.006, size=pose[:, :3].shape).astype(np.float32)
        pose[:, 3] = np.clip(pose[:, 3] + rng.normal(0.0, 0.03, size=NUM_LANDMARKS), 0.4, 1.0)
        pose[:, :2] = np.clip(pose[:, :2], 0.02, 0.98)
        frames[i] = pose

    return frames.reshape(n_frames, FEATURE_DIM)


def sequences_to_windows(
    seq: np.ndarray, label: int, window: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding windows from a (T, F) sequence."""
    xs, ys = [], []
    if len(seq) < window:
        return np.zeros((0, window, seq.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for start in range(0, len(seq) - window + 1, stride):
        xs.append(seq[start: start + window])
        ys.append(label)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def generate_dataset(
    output_dir: str,
    n_sequences_per_step: int = 40,
    frames_per_seq: int = 60,
    window: int = 30,
    stride: int = 15,
    seed: int = 7,
    include_orientations: bool = True,
    steps: Optional[list] = None,
) -> Dict:
    """Generate train-ready X_sequences.npy / y_labels.npy and return metadata."""
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = list(range(1, 9))

    orientations = (0, 1, 2, 3) if include_orientations else (0,)
    all_x, all_y = [], []
    per_step = {}

    for step_id in steps:
        step_windows = 0
        n_ori = len(orientations)
        per_ori = max(1, n_sequences_per_step // n_ori)
        for ori in orientations:
            for _ in range(per_ori):
                seq = generate_sequence(
                    step_id, n_frames=frames_per_seq, rng=rng, orientation=ori, drift=True
                )
                xw, yw = sequences_to_windows(seq, step_id, window, stride)
                if len(xw) == 0:
                    continue
                all_x.append(xw)
                all_y.append(yw)
                step_windows += len(xw)
        per_step[int(step_id)] = int(step_windows)
        logger.info("Step %d: %d windows", step_id, step_windows)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    np.save(str(out / "X_sequences.npy"), X)
    np.save(str(out / "y_labels.npy"), y)
    meta = {
        "total_windows": int(len(X)),
        "feature_dim": int(X.shape[-1]),
        "window": int(window),
        "source": "synthetic_pose",
        "seed": int(seed),
        "steps": [{"step_id": k, "windows": v} for k, v in per_step.items()],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s  X=%s y=%s", out, X.shape, y.shape)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_dataset("dataset/skeleton_sequences", n_sequences_per_step=48)
    generate_dataset("dataset/skeleton_sequences_test", n_sequences_per_step=16, seed=99)
