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


class DepthLifter(nn.Module):
    def __init__(self):
        super().__init__()
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
        model.load_state_dict(checkpoint["model_state"])
        return model.eval()

    @torch.inference_mode()
    def lift(self, pose, size):
        """Replace depth only; return None when the torso cannot anchor a lift."""
        encoded, scale, valid = encode(pose, size)
        if len(valid) != 1:
            raise ValueError("lift expects one frame")
        if not valid[0]:
            return None
        result = np.asarray(pose, dtype=np.float32).reshape(33, 4).copy()
        result[~np.isfinite(result).all(axis=1)] = 0
        result[:, 2] = self(torch.from_numpy(encoded))[0].numpy() * scale[0]
        return result.reshape(-1)
