"""
3D Human Mesh Recovery Backend (Stage 2) — Optional
====================================================
Stage 1 (pipeline/rack_frame.py) makes the 2D skeleton rack-relative, but a
2D projection still loses out-of-plane depth. True orientation-agnostic
tracking needs 3D Human Mesh Recovery: an HMR model (HMR 2.0, WHAM, OS-X,
etc.) regresses a SMPL body mesh in a ROOT-RELATIVE canonical frame, which
is inherently gravity-free — no floor prior, no fixed 'up'.

This module is a thin, dependency-optional adapter:

  - `HMRBackend` tries to load a supported HMR package: hmr2, wham, or romp
    (the one actually installed/working in .venv-hmr).
  - If none is installed (or there is no GPU), it reports unavailable and the
    pipeline transparently falls back to MediaPipe 2D + rack-frame Stage 1.
  - Output is always the SAME 132-dim contract (33 landmarks x, y, z, vis),
    so the LSTM/state machine never changes, but the coordinate convention
    depends on backend:
      - hmr2/wham: root-relative 3D (meters), gravity-free.
      - romp: MediaPipe's own convention (x, y normalized image coords,
        z hip-relative and scaled to roughly match) via
        `romp_to_feature_vector` — NOT root-relative 3D. See that function
        and ROMP_CONTRACT_VERSION for the exact math.

Install (when GPU hardware is available):
    pip install git+https://github.com/shubham-goenka/hmr2.git   # or
    pip install wham                                              # or
    pip install simple_romp   # (run under D:/SIH-2026/.venv-hmr)

Nothing in this file is required for the current CPU pipeline to run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import HMR_BACKEND, SKELETON_FEATURES

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES  # single source of truth: config.SKELETON_FEATURES (132)

# SMPL -> MediaPipe slot mapping.
#
# Both index spaces are fixed and public, so this table is checkable rather
# than a matter of taste:
#
#   SMPL (24-joint body skeleton)
#     0 pelvis   1 L_hip     2 R_hip    3 spine1   4 L_knee   5 R_knee
#     6 spine2   7 L_ankle   8 R_ankle  9 spine3  10 L_foot  11 R_foot
#    12 neck    13 L_collar 14 R_collar 15 head   16 L_shoulder 17 R_shoulder
#    18 L_elbow 19 R_elbow  20 L_wrist 21 R_wrist 22 L_hand  23 R_hand
#
#   MediaPipe Pose (33 landmarks)
#     0 nose   11/12 shoulders  13/14 elbows  15/16 wrists  19/20 index
#    23/24 hips  25/26 knees  27/28 ankles  31/32 foot_index
#
# The previous table was wrong on the MediaPipe side for every single entry:
# it sent hips to 11/12 (which are the SHOULDERS), knees to 13/14 (ELBOWS),
# ankles to 15/16 (WRISTS), wrists to 17/18 (PINKY), and shoulders to 20/21
# (INDEX and THUMB). Every mapped joint landed one limb away from where it
# belonged, so an active HMR backend would have fed the LSTM a scrambled
# skeleton — silently, because the vector still had the right shape and the
# right magnitudes. It was dormant only because HMR_BACKEND defaults to
# "none".
_SMPL_TO_SLOT = {
    15: 0,    # head        -> nose (closest available head landmark)
    16: 11,   # L_shoulder  -> left_shoulder
    17: 12,   # R_shoulder  -> right_shoulder
    18: 13,   # L_elbow     -> left_elbow
    19: 14,   # R_elbow     -> right_elbow
    20: 15,   # L_wrist     -> left_wrist
    21: 16,   # R_wrist     -> right_wrist
    22: 19,   # L_hand      -> left_index
    23: 20,   # R_hand      -> right_index
    1: 23,    # L_hip       -> left_hip
    2: 24,    # R_hip       -> right_hip
    4: 25,    # L_knee      -> left_knee
    5: 26,    # R_knee      -> right_knee
    7: 27,    # L_ankle     -> left_ankle
    8: 28,    # R_ankle     -> right_ankle
    10: 31,   # L_foot      -> left_foot_index
    11: 32,   # R_foot      -> right_foot_index
}

# MediaPipe slots for the hips, used to root the output the same way
# MediaPipe itself does (z is expressed relative to the hip midpoint).
_L_HIP_SLOT, _R_HIP_SLOT = 23, 24

# ── ROMP (backend="romp") ────────────────────────────────────────────────────
# Unlike hmr2/wham above (root-relative metres, a DIFFERENT distribution from
# MediaPipe), the ROMP path emits MediaPipe's own coordinate convention, per
# the contract hmr-contract verified on blender_core:
#   x, y = ROMP's pj2d_org (image pixels) / frame W, H   - absolute, NOT hip-centred
#   z    = (J_z - hip_mid_z) * cam_s * max(W,H)/2 / W    - hip-relative, x-pixel units
#   vis  = 1.0 on the 27 slots with a ROMP joint, 0.0 on the 6 without one
# That makes rack normalisation (pipeline/rack_frame.py) behave on it exactly
# as on MediaPipe. It is still NOT MediaPipe: a model trained on MediaPipe
# must not be served on it, nor the reverse. check_pose_source() enforces that.
#
# MediaPipe slot -> ROMP joints54 index (romp/utils.py SMPL_ALL_54). SMPL left
# is MediaPipe left throughout.
_ROMP_SLOTS = {
    0: 24,                                  # Nose (joints54, beats SMPL 15 'Jaw')
    2: 26, 5: 25, 7: 28, 8: 27,             # L_Eye, R_Eye, L_Ear, R_Ear
    11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,   # shoulders, elbows, wrists
    17: 39, 18: 44,                         # L/R_Hand_pinky
    19: 22, 20: 23,                         # SMPL L/R_Hand -> index (approximate)
    21: 35, 22: 40,                         # L/R_Hand_thumb
    23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8,         # hips, knees, ankles
    29: 31, 30: 34,                         # L/R_Heel
    31: 10, 32: 11,                         # SMPL L/R_Toe -> foot_index (approximate)
}
# No ROMP joint at all (eye inner/outer, mouth corners). They copy the nearest
# measured point with vis 0: left at (0, 0) they would still be rotated and
# scaled by rack normalisation into a rack-dependent phantom trajectory, and
# would bias the z re-centring.
_ROMP_PROXY = {1: 2, 3: 2, 4: 5, 6: 5, 9: 0, 10: 0}
ROMP_CENTER_THRESH = 0.02    # default 0.25 finds nobody on these renders (hmr-env)
ROMP_CONTRACT_VERSION = 1    # bump when romp_to_feature_vector changes (keys the build cache)


def romp_to_feature_vector(joints54: np.ndarray, pj2d_org: np.ndarray, cam: np.ndarray,
                           wh: Tuple[int, int]) -> np.ndarray:
    """One ROMP person -> 132-dim vector in MediaPipe's coordinate convention."""
    w, h = wh
    slots = np.fromiter(_ROMP_SLOTS.keys(), dtype=np.int64)
    idx = np.fromiter(_ROMP_SLOTS.values(), dtype=np.int64)
    f = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    f[slots, 0] = pj2d_org[idx, 0] / w
    f[slots, 1] = pj2d_org[idx, 1] / h
    hip_z = 0.5 * (joints54[1, 2] + joints54[2, 2])
    f[slots, 2] = (joints54[idx, 2] - hip_z) * float(cam[0]) * max(w, h) / 2.0 / w
    f[slots, 3] = 1.0
    for slot, src in _ROMP_PROXY.items():
        f[slot, :3] = f[src, :3]
    return f.reshape(-1)


def trained_hmr_backend(meta: dict) -> str:
    """HMR backend a checkpoint / sequence set's pose came from, or "none" for
    MediaPipe-convention 2D (MediaPipe, Blender ground truth, synthetic).
    Unrecorded = "none": everything built before --pose-source hmr existed."""
    return str(meta.get("hmr_backend", "none")) if meta.get("pose_source") == "hmr" else "none"


def check_pose_source(name: str, meta: dict, runtime_backend: str) -> None:
    """Refuse to serve a model on a pose source it was not trained on.

    `runtime_backend` is the HMR backend ACTUALLY producing pose ("none" when
    MediaPipe does, including when HMR_BACKEND asked for one that failed to
    load). Both sides fill the same 132 slots, so a mismatch never throws on
    its own; it just makes the model quietly worse."""
    trained = trained_hmr_backend(meta)
    if trained == runtime_backend:
        return
    side = lambda b: "MediaPipe 2D (no HMR backend)" if b == "none" else f"HMR backend {b!r}"
    raise RuntimeError(
        f"{name}: pose source mismatch. The model was TRAINED on {side(trained)} "
        f"(pose_source={meta.get('pose_source', 'unrecorded')!r}) but this runtime SERVES "
        f"{side(runtime_backend)} (config HMR_BACKEND={HMR_BACKEND!r}). Same 132 slots, "
        f"different coordinates: refusing to run rather than degrade silently. Set "
        f"HMR_BACKEND={trained!r}{' (with its package importable)' if trained != 'none' else ''}, "
        f"or serve a model built "
        f"with tools/build_sequences.py --pose-source "
        f"{'hmr' if runtime_backend != 'none' else 'mediapipe'}.")


class HMRBackend:
    """
    Optional 3D HMR adapter with graceful degradation.

    Example:
        backend = HMRBackend()           # never raises
        feats, name = backend.get_pose_features(frame_rgb)
        # name == "mediapipe" → Stage-1 rack-frame path handles it
    """

    def __init__(self, backend: str = "auto", device: Optional[str] = None):
        self.requested = (backend or "none").lower()
        self.device = device
        self.model = None
        self.backend_name = "none"
        if self.requested in ("none", "mediapipe"):
            return
        if self.requested == "romp":
            try:
                self.model = self._load_romp()
                self.backend_name = "romp"
                logger.info("HMR backend active: romp (MediaPipe-convention 2D + hip-relative z)")
            except Exception as e:   # ImportError included: same graceful fallback as below
                logger.warning("HMR backend 'romp' unavailable (%s); falling back to MediaPipe.", e)
            return
        self._try_load()

    # ── loading ───────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        for pkg, loader in (("hmr2", self._load_hmr2), ("wham", self._load_wham)):
            try:
                self.model = loader()
                self.backend_name = pkg
                logger.info("HMR backend active: %s (3D root-relative mesh)", pkg)
                return
            except ImportError:
                continue
            except Exception as e:  # weights/IO problems must not kill the pipeline
                logger.warning("HMR package '%s' found but failed to load: %s", pkg, e)
        logger.warning(
            "No 3D HMR backend available (requested='%s'). "
            "Falling back to MediaPipe 2D + rack-frame normalization. "
            "Install one with: pip install hmr2  (GPU required).",
            self.requested,
        )

    def _load_hmr2(self):
        import torch  # noqa: F401  (import proves the runtime exists)
        from hmr2.models import load_hmr2  # type: ignore
        model, _ = load_hmr2()
        model.eval()
        return model

    def _load_wham(self):
        import torch  # noqa: F401
        from wham_api import WHAM  # type: ignore
        return WHAM()

    def _load_romp(self):
        import os
        # simple_romp shells out to `pip install ...` resolved via PATH, not
        # sys.executable: put this interpreter's Scripts dir first so any such
        # call lands in this environment and never in the system Python.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        import romp  # type: ignore
        from romp.main import romp_settings  # type: ignore
        return romp.ROMP(romp_settings(["--center_thresh", str(ROMP_CENTER_THRESH)]))

    def _romp_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        import contextlib
        import io
        import cv2
        with contextlib.redirect_stdout(io.StringIO()):   # ROMP prints a timing line per call
            out = self.model(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if out is None or len(out.get("center_confs", [])) == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)       # nobody: same as MediaPipe
        # ponytail: most confident centre wins; per-frame confidence is not a
        # reliable gate here (AUC 0.64), so unreliable takes are dropped upstream.
        i = int(np.argmax(np.asarray(out["center_confs"]).reshape(-1)))
        h, w = frame_rgb.shape[:2]
        return romp_to_feature_vector(out["joints"][i], out["pj2d_org"][i], out["cam"][i], (w, h))

    # ── public API ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.model is not None

    def get_pose_features(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Returns (features132, backend_name).

        - HMR active:  root-relative 3D joints mapped into the 33-slot
          contract (x, y, z in meters; visibility=1 for mapped joints).
        - HMR absent:  zero features + "mediapipe"; the caller must run its
          MediaPipe path (this keeps the pipeline's latency contract intact).
        """
        if not self.available:
            return np.zeros(FEATURE_DIM, dtype=np.float32), "mediapipe"
        try:
            if self.backend_name == "romp":
                return self._romp_features(frame_rgb), "romp"
            joints = self._infer_joints(frame_rgb)  # (J, 3) root-relative meters
            return self._to_feature_vector(joints), self.backend_name
        except Exception as e:
            logger.warning("HMR inference failed (%s); caller should fall back.", e)
            return np.zeros(FEATURE_DIM, dtype=np.float32), "mediapipe"

    # ── internals ─────────────────────────────────────────────────────────

    def _infer_joints(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch
        import cv2

        img = cv2.resize(frame_rgb, (256, 256))
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        if self.device:
            x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            # hmr2 returns dict with 'smpl_vertices'/'joints3d'; wham returns tensor
            if isinstance(out, dict):
                joints = out.get("joints3d", out.get("smpl_joints3d"))
            else:
                joints = out
        return joints[0].detach().cpu().numpy().astype(np.float32)

    def _to_feature_vector(self, joints: np.ndarray) -> np.ndarray:
        """(J, 3) root-relative 3D → 132-dim MediaPipe-style contract."""
        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        smpl = np.asarray(joints, dtype=np.float32).reshape(-1, 3)

        for smpl_idx, slot in _SMPL_TO_SLOT.items():
            if smpl_idx < len(smpl):
                feats[slot, :3] = smpl[smpl_idx]
                feats[slot, 3] = 1.0

        # Root on the hip midpoint, not the mean of all joints. MediaPipe's
        # own convention is hip-centred, and a mean over whatever subset of
        # joints happens to be mapped would move the origin whenever the
        # mapping changed — making features from two builds incomparable.
        if feats[_L_HIP_SLOT, 3] > 0.0 and feats[_R_HIP_SLOT, 3] > 0.0:
            origin = 0.5 * (feats[_L_HIP_SLOT, :3] + feats[_R_HIP_SLOT, :3])
        else:
            mapped = feats[feats[:, 3] > 0.0, :3]
            origin = mapped.mean(axis=0) if len(mapped) else np.zeros(3, dtype=np.float32)
        feats[feats[:, 3] > 0.0, :3] -= origin
        return feats.reshape(-1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = HMRBackend(backend="auto")
    feats, name = b.get_pose_features(np.zeros((720, 1280, 3), dtype=np.uint8))
    print(f"backend={name} available={b.available} feats_shape={feats.shape}")
    assert feats.shape == (FEATURE_DIM,) and np.isfinite(feats).all()

    # ROMP contract: pixel -> normalised image coords, hips at the right slots.
    j54 = np.zeros((54, 3), np.float32); p2d = np.zeros((54, 2), np.float32)
    p2d[1], p2d[2], p2d[16] = (640, 360), (660, 360), (600, 200)   # L_hip, R_hip, L_shoulder
    j54[16, 2], j54[1, 2], j54[2, 2] = 0.3, 0.1, 0.1
    f = romp_to_feature_vector(j54, p2d, np.array([0.5, 0, 0]), (1280, 720)).reshape(33, 4)
    assert np.allclose(f[23, :2], (0.5, 0.5)) and np.allclose(f[11, :2], (600 / 1280, 200 / 720))
    assert np.isclose(f[11, 2], 0.2 * 0.5 * 1280 / 2 / 1280) and f[23, 2] == 0.0
    assert f[:, 3].sum() == 27 and f[[1, 3, 4, 6, 9, 10], 3].sum() == 0
    assert np.allclose(f[1, :3], f[2, :3])                           # proxy copies neighbour

    # Pose-source guard: match passes, either mismatch direction raises.
    check_pose_source("m", {}, "none")
    check_pose_source("m", {"pose_source": "hmr", "hmr_backend": "romp"}, "romp")
    for meta, rt in (({"pose_source": "hmr", "hmr_backend": "romp"}, "none"), ({}, "romp")):
        try:
            check_pose_source("m", meta, rt)
            raise AssertionError("mismatch not refused")
        except RuntimeError:
            pass
    print("hmr_backend smoke test passed (graceful fallback, ROMP contract, pose-source guard).")
