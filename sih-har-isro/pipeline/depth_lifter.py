"""Single-frame BAS skeleton depth in camera-width units, relative to the hips."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "bas_depth_lifter.pt"
FORMAT = "bas-depth-xyv-v1"


def encode(pose, size):
    """Canonicalize measured xy and visibility. Depth never enters the network.

    Inputs may be batched. Converting y to width units before rotation makes a
    change of camera roll or raster aspect independent of the distance scale.
    """
    width, height = size
    p = np.asarray(pose, dtype=np.float32).reshape(-1, 33, 4)
    xy = p[..., :2] * np.array([1, height / width], dtype=np.float32)
    vis = np.clip(p[..., 3], 0, 1)
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(vis)
    vis = np.where(finite, vis, 0)
    xy = np.where(finite[..., None], xy, 0)
    hip = xy[:, [23, 24]].mean(axis=1)
    down = hip - xy[:, [11, 12]].mean(axis=1)
    scale = np.linalg.norm(down, axis=1)
    torso_vis = vis[:, [11, 12, 23, 24]]
    valid = (scale > 1e-3) & (torso_vis.min(axis=1) >= 0.15)
    scale = np.maximum(scale, 1e-3)
    down /= scale[:, None]
    right = np.stack([down[:, 1], -down[:, 0]], axis=-1)
    axes = np.stack([right, down], axis=-1)
    local = (xy - hip[:, None]) @ axes / scale[:, None, None]
    local = np.where((vis >= 0.15)[..., None], local.clip(-5, 5), 0)
    return np.concatenate([local, vis[..., None]], axis=-1), scale, valid


def refine_image_pose(pose, rgb_pose, size):
    """Fuse nearby image measurements and recover weak joints from the RGB regressor."""
    p = np.asarray(pose).reshape(-1, 33, 4).copy()
    r = np.asarray(rgb_pose).reshape(p.shape)
    torso = (p[:, [11, 12], :2].mean(1) - p[:, [23, 24], :2].mean(1)) * size
    length = np.linalg.norm(torso, axis=1)
    distance = np.linalg.norm((r[..., :2] - p[..., :2]) * size, axis=-1)
    use = (r[..., 3] >= 0.32) & ((distance <= length[:, None]) | (p[..., 3] < 0.4))
    p[..., :2] = np.where(use[..., None], (p[..., :2] + r[..., :2]) * 0.5, p[..., :2])
    p[..., 3] = np.where(use, np.maximum(p[..., 3], r[..., 3]), p[..., 3])
    return p.reshape(np.shape(pose))


class DepthLifter(nn.Module):
    def __init__(self):
        super().__init__()
        self.skeleton_weight = 1.0
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(99, 256), nn.LayerNorm(256),
                                 nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 33))

    def forward(self, features):
        depth = self.net(features)
        return depth - depth[:, [23, 24]].mean(dim=1, keepdim=True)

    @classmethod
    def load(cls, path=WEIGHTS):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT:
            raise ValueError(f"Unsupported depth checkpoint at {path}")
        model = cls()
        model.skeleton_weight = float(checkpoint["skeleton_weight"])
        model.load_state_dict(checkpoint["model_state"])
        return model.eval()

    @torch.inference_mode()
    def lift(self, pose, size, rgb_pose=None):
        """Replace depth only; return None when the torso cannot anchor a lift."""
        encoded, scale, valid = encode(pose, size)
        if len(valid) != 1:
            raise ValueError("lift expects one frame")
        if not valid[0]:
            return None
        result = np.asarray(pose, dtype=np.float32).reshape(33, 4).copy()
        result[~np.isfinite(result).all(axis=1)] = 0
        result[:, 2] = self(torch.from_numpy(encoded))[0].numpy() * scale[0]
        if rgb_pose is not None:
            rgb = np.asarray(rgb_pose).reshape(33, 4)
            if np.isfinite(rgb).all() and np.any(rgb[:, 3] > 0):
                result = refine_image_pose(result, rgb, size)
                rgb_z = rgb[:, 2] - rgb[[23, 24], 2].mean()
                result[:, 2] = self.skeleton_weight * result[:, 2] + (1 - self.skeleton_weight) * rgb_z
        return result.reshape(-1)
