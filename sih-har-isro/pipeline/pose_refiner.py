"""Calibrate the fused BAS pose using paired camera observations."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "bas_pose_refiner.pt"


def width_units(size):
    return np.array([1, size[1] / size[0], 1], dtype=np.float32)


def observations(skeleton, rgb, fused, size):
    """Express all three poses in common image-width units."""
    scale = width_units(size)
    inputs = []
    for pose in (skeleton, rgb, fused):
        p = np.asarray(pose, dtype=np.float32).reshape(-1, 33, 4).copy()
        p[..., :3] *= scale
        p[..., :2] -= scale[:2] / 2
        p = np.where(np.isfinite(p), p, 0)
        inputs.append(p.clip(-2, 2))
    return np.concatenate(inputs, axis=-1)


class PoseRefiner(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(33 * 12, width),
                                 nn.LayerNorm(width), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(width, width), nn.LayerNorm(width),
                                 nn.GELU(), nn.Dropout(.1), nn.Linear(width, 99))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        delta = self.net(inputs).reshape(-1, 33, 3)
        pose = inputs[..., 8:11] + delta
        depth = pose[..., 2] - pose[:, [23, 24], 2].mean(dim=1, keepdim=True)
        return torch.cat((pose[..., :2], depth.unsqueeze(-1)), dim=-1)

    @classmethod
    def load(cls, path=WEIGHTS):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format"] != "bas-pose-refiner-v1":
            raise ValueError(f"Unknown pose refiner format in {path}")
        model = cls(state["width"])
        model.load_state_dict(state["model"])
        return model.eval()

    @torch.inference_mode()
    def refine(self, skeleton, rgb, fused, size):
        if rgb is None:
            return fused
        joints = np.asarray(rgb).reshape(33, 4)
        if not np.isfinite(joints).all() or not np.any(joints[:, 3] > 0):
            return fused
        x = observations(skeleton, rgb, fused, size)
        result = fused.copy().reshape(33, 4)
        p = self(torch.from_numpy(x))[0].numpy()
        scale = width_units(size)
        result[:, :3] = p / scale
        result[:, :2] += .5
        return result.reshape(-1)
