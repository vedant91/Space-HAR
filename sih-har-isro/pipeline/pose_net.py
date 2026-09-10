"""
Custom Pose Model — Runtime Wrapper (replaces MediaPipe)
==========================================================
Drop-in replacement for the old OptimizedMPWrapper. Loads the from-scratch
HARPoseNet (train/train_posenet.py), preferring ONNX Runtime for speed
(same convention as the LSTM/CNN — see pipeline/har_pipeline.py).

No pretrained/third-party model is loaded anywhere in this file: HARPoseNet
is trained entirely on this project's own synthetic renderer ground truth
(Stage 1) plus classical-CV pseudo-labels on real footage (Stage 2) — see
train/train_posenet.py's module docstring.

Usage mirrors the old wrapper's contract:
    wrapper = PoseNetWrapper()
    features132 = wrapper.process(frame_rgb_full)   # np.float32 (132,)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    SKELETON_FEATURES, POSE_JOINT_SLOTS, POSE_NUM_JOINTS, POSE_INPUT_SIZE,
    POSE_MIN_CONFIDENCE, POSE_DECODE_RADIUS, POSE_DECODE_BETA,
    POSENET_PATH, POSENET_ONNX_PATH,
)

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
POSE_FEATURE_DIM = SKELETON_FEATURES  # 132, single source of truth: config
JOINT_SLOTS = list(POSE_JOINT_SLOTS)
J = POSE_NUM_JOINTS
IN_SIZE = POSE_INPUT_SIZE

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


def _preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 (any size) -> (1,3,IN_SIZE,IN_SIZE) float32 in [-1,1]."""
    img = cv2.resize(frame_rgb, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose(2, 0, 1)[None, ...]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _local_soft_argmax(heatmap: np.ndarray, radius: int = POSE_DECODE_RADIUS,
                       beta: float = POSE_DECODE_BETA):
    """Sub-pixel joint location via a softmax-weighted centroid in a small
    window around the argmax — see train.train_posenet._local_soft_argmax
    (kept as an independent numpy copy here so ONNX inference never needs
    torch). Returns (x, y, peak_val) in heatmap-pixel units."""
    size = heatmap.shape[-1]
    iy, ix = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    peak_val = float(np.clip(heatmap[iy, ix], 0.0, 1.0))
    y0, y1 = max(0, iy - radius), min(size, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(size, ix + radius + 1)
    patch = heatmap[y0:y1, x0:x1]
    w = np.exp((patch - patch.max()) * beta)
    w /= w.sum()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((w * xx).sum()), float((w * yy).sum()), peak_val


def _decode(heatmaps: np.ndarray, z: np.ndarray, vis_logits: np.ndarray) -> np.ndarray:
    """heatmaps (J,H,W), z (J,), vis_logits (J,) -> full (132,) feature vector.
    Pure-numpy re-implementation of train.train_posenet.decode_heatmaps so this
    module never needs torch just to run ONNX inference."""
    hm_size = heatmaps.shape[-1]
    out = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    vis = _sigmoid(vis_logits)
    for j, slot in enumerate(JOINT_SLOTS):
        x, y, peak_val = _local_soft_argmax(heatmaps[j])
        conf = float(vis[j]) * peak_val
        if conf < POSE_MIN_CONFIDENCE:
            continue
        out[slot, 0] = (x + 0.5) / hm_size
        out[slot, 1] = (y + 0.5) / hm_size
        out[slot, 2] = float(z[j])
        out[slot, 3] = conf
    return out.reshape(-1).astype(np.float32)


class PoseNetWrapper:
    """
    Loads HARPoseNet (ONNX preferred, PyTorch fallback) and exposes a single
    `.process(frame_rgb_full) -> (132,) np.float32` call — same output shape/
    semantics as the retired MediaPipe path, so state_machine, rack_frame,
    the LSTM and the CNN ensemble need zero changes.
    """

    def __init__(self, onnx_path: str = POSENET_ONNX_PATH, pt_path: str = POSENET_PATH):
        self._sess: Optional["ort.InferenceSession"] = None
        self._torch_model = None
        self._input_name = None
        self.backend = "none"

        if ORT_AVAILABLE and Path(onnx_path).exists():
            try:
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                                  providers=["CPUExecutionProvider"])
                self._input_name = self._sess.get_inputs()[0].name
                self.backend = "onnx"
                logger.info("PoseNet ONNX session loaded: %s", onnx_path)
            except Exception as e:
                logger.warning("PoseNet ONNX load failed (%s); trying PyTorch.", e)

        if self._sess is None and Path(pt_path).exists():
            try:
                import torch
                from train.train_posenet import HARPoseNet
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                model = HARPoseNet(num_joints=ckpt["num_joints"])
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._torch_model = model
                self.backend = "pytorch"
                logger.info("PoseNet PyTorch checkpoint loaded: %s (val_pck=%.3f)",
                           pt_path, ckpt.get("val_pck", 0.0) or 0.0)
            except Exception as e:
                logger.warning("PoseNet PyTorch load failed: %s", e)

        if self.backend == "none":
            logger.warning(
                "No trained PoseNet found at %s / %s — pose features will be all-zero "
                "until `python train/train_posenet.py` is run.", onnx_path, pt_path)

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def process(self, frame_rgb_full: np.ndarray) -> np.ndarray:
        """Full pipeline: resize + normalize + infer + decode. Returns (132,) float32."""
        if not self.available:
            return np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        x = _preprocess(frame_rgb_full)

        if self.backend == "onnx":
            heatmaps, z, vis_logits = self._sess.run(None, {self._input_name: x})
            return _decode(heatmaps[0], z[0], vis_logits[0])

        import torch
        with torch.no_grad():
            heatmaps, z, vis_logits = self._torch_model(torch.from_numpy(x))
        return _decode(heatmaps[0].numpy(), z[0].numpy(), vis_logits[0].numpy())

    def close(self):
        """No persistent resources to release — kept for interface symmetry
        with the retired MediaPipe wrapper (har_pipeline.py calls .close())."""
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    w = PoseNetWrapper()
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    feats = w.process(dummy)
    print(f"backend={w.backend} available={w.available} feats_shape={feats.shape}")
    assert feats.shape == (POSE_FEATURE_DIM,) and np.isfinite(feats).all()
    print("pose_net smoke test passed.")
