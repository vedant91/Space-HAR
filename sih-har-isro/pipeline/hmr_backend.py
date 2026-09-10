"""
3D Human Mesh Recovery Backend (Stage 2) — RETIRED
====================================================
Stage 1 (pipeline/rack_frame.py) makes the 2D skeleton rack-relative, but a
2D projection still loses out-of-plane depth. True orientation-agnostic
tracking would need 3D Human Mesh Recovery: an HMR model (HMR 2.0, WHAM,
OS-X, etc.) regressing a SMPL body mesh in a root-relative canonical frame.

That HMR model would itself be a third-party pretrained network — exactly
what this project's "no open-source/pretrained model for detecting objects
or movement" requirement rules out. So Stage 2 is retired rather than wired
to one: this module is kept only as an inert shim (`available` is always
False) so any config/import referencing HMR_BACKEND doesn't break. The
project's actual depth signal is the custom PoseNet's own z-regression head
(pipeline/pose_net.py, train/train_posenet.py) — trained from scratch on
this project's own synthetic ground truth, same as every other model here.

Output stays the SAME 132-dim contract (33 landmarks x, y, z, vis) either
way, so the LSTM/state machine never needs to change.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import SKELETON_FEATURES

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES  # single source of truth: config.SKELETON_FEATURES (132)


class HMRBackend:
    """
    Retired 3D HMR adapter. Always reports unavailable — the caller falls
    back to the custom PoseNet's 2D (+ its own z head) path, unconditionally.

    Example:
        backend = HMRBackend()           # never raises
        feats, name = backend.get_pose_features(frame_rgb)
        # name == "posenet" → caller's normal PoseNet path handles it
    """

    def __init__(self, backend: str = "auto", device: Optional[str] = None):
        self.requested = (backend or "none").lower()
        self.device = device
        self.model = None
        self.backend_name = "none"
        if self.requested not in ("none", "posenet", "mediapipe"):
            logger.warning(
                "HMR backend '%s' requested but 3D HMR support is retired in this project "
                "(it would require a third-party pretrained model — see this module's "
                "docstring). Falling back to the custom PoseNet 2D path.", self.requested)

    @property
    def available(self) -> bool:
        return False

    def get_pose_features(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, str]:
        """Always unavailable — caller must run its own PoseNet path."""
        return np.zeros(FEATURE_DIM, dtype=np.float32), "posenet"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    b = HMRBackend(backend="auto")
    feats, name = b.get_pose_features(np.zeros((720, 1280, 3), dtype=np.uint8))
    print(f"backend={name} available={b.available} feats_shape={feats.shape}")
    assert feats.shape == (FEATURE_DIM,) and np.isfinite(feats).all() and not b.available
    print("hmr_backend smoke test passed (retired shim verified inert).")
