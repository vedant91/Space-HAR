"""
3D Human Mesh Recovery Backend (Stage 2) — Optional
====================================================
Stage 1 (pipeline/rack_frame.py) makes the 2D skeleton rack-relative, but a
2D projection still loses out-of-plane depth. True orientation-agnostic
tracking needs 3D Human Mesh Recovery: an HMR model (HMR 2.0, WHAM, OS-X,
etc.) regresses a SMPL body mesh in a ROOT-RELATIVE canonical frame, which
is inherently gravity-free — no floor prior, no fixed 'up'.

This module is a thin, dependency-optional adapter:

  - `HMRBackend` tries to load a supported HMR package (hmr2 or wham).
  - If none is installed (or there is no GPU), it reports unavailable and the
    pipeline transparently falls back to MediaPipe 2D + rack-frame Stage 1.
  - Output is always the SAME 132-dim contract (33 landmarks x, y, z, vis),
    so the LSTM/state machine never changes. When HMR is active the x, y, z
    are root-relative 3D (meters), which is strictly better than MediaPipe's
    weak perspective z.

Install (when GPU hardware is available):
    pip install git+https://github.com/shubham-goenka/hmr2.git   # or
    pip install wham

Nothing in this file is required for the current CPU pipeline to run.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 4  # 132

# SMPL joint indices (45-joint layout) for the subset we map into the
# MediaPipe-style 33-slot contract. Only unambiguous major joints are mapped;
# everything else is zero-filled and marked invisible.
_SMPL_TO_SLOT = {
    12: 0,    # pelvis-ish head slot is unused; nose left zero
    9: 11,    # left hip
    8: 12,    # right hip
    4: 13,    # left knee
    5: 14,    # right knee
    7: 15,    # left ankle
    10: 16,   # right ankle
    18: 17,   # left wrist
    19: 18,   # right wrist
    16: 20,   # left shoulder
    17: 21,   # right shoulder
}


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
        center = smpl.mean(axis=0)
        for smpl_idx, slot in _SMPL_TO_SLOT.items():
            if smpl_idx < len(smpl):
                feats[slot, :3] = smpl[smpl_idx] - center
                feats[slot, 3] = 1.0
        return feats.reshape(-1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = HMRBackend(backend="auto")
    feats, name = b.get_pose_features(np.zeros((720, 1280, 3), dtype=np.uint8))
    print(f"backend={name} available={b.available} feats_shape={feats.shape}")
    assert feats.shape == (FEATURE_DIM,) and np.isfinite(feats).all()
    print("hmr_backend smoke test passed (graceful fallback verified).")
