"""
Custom 3D pose regressor - replaces ROMP for the HMR backend (Stage 2).

Why this exists: ROMP's pretrained SMPL regression network was trained on
real photographs of bare-skinned or normally-clothed people, almost all
upright. Measured on our own data (tools/measure_hmr_crop.py), it fails at
moderate roll (30-90 deg) and dim lighting regardless of input framing, and
is internally inconsistent on pitch even with a correctly-cropped, correctly-
rotated view - a domain-gap problem in ROMP's own weights, not something crop
engineering can fix (see docs/architecture.html section 3 for the measured
evidence). This module trains a small network on exactly the domain ROMP
never saw: a rendered, suited astronaut, at every orientation the Blender
digital twin covers, supervised by exact synthetic ground truth
(dataset/blender_core/*/payload_a/pose_2d.npy) that ROMP's original training
data never had for this subject.

Contract: given a square RGB crop (produced by HMRBackend._crop, the SAME
tracked-box-and-angle crop ROMP used), predict a 33x4 vector in the crop's
OWN local coordinate frame: (x, y) normalized to [0, 1] of the crop, z
hip-relative in the same physically-motivated units groundtruth.py already
uses (metric depth difference / world-width-at-hip-distance - unaffected by
the crop's in-plane rotation, since rotating a 2D image does not change true
depth from the camera), and a visibility-like score in [0, 1]. HMRBackend
maps (x, y) back to full-frame coordinates with the crop's own inverse
affine (the same _to_frame() already used for ROMP's output) before slotting
into the shared 132-dim contract - this module never needs to know about
full-frame coordinates at all.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

NUM_LANDMARKS = 33
OUT_DIM = NUM_LANDMARKS * 4          # x, y, z, vis per joint
CROP_SIZE = 224                      # matches config.CNN_IMG_SIZE; MobileNetV3's native training res
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HMR3DNet(nn.Module):
    """MobileNetV3-Small backbone (ImageNet-pretrained) + a small regression
    head. ~2.5M backbone params: fast enough for the live loop's adaptive
    schedule (it only runs on the same throttled fraction of frames ROMP
    did), small enough to train in a couple of hours on 21k synthetic frames
    without overfitting immediately."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features                 # conv stack, no classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[0].in_features      # 576 for mobilenet_v3_small
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.Hardswish(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, CROP_SIZE, CROP_SIZE), ImageNet-normalized.
        Returns (B, 33, 4): x, y in [0,1]-ish (raw linear output, not
        sigmoid-clamped - a joint just outside the crop from a slightly loose
        box is a real, useful signal, not something to clip away), z
        unrestricted, vis intended to sit near the training data's own
        {0.04, 0.32, 0.97} range but not hard-constrained either."""
        f = self.pool(self.features(x)).flatten(1)
        return self.head(f).view(-1, NUM_LANDMARKS, 4)


def preprocess_crop(crop_rgb_uint8) -> torch.Tensor:
    """(H, W, 3) uint8 RGB, any size -> (3, CROP_SIZE, CROP_SIZE) normalized
    float tensor. Shared by training (on cached crops) and inference (on a
    live HMRBackend crop), so the two paths can never silently diverge."""
    import cv2
    import numpy as np
    img = cv2.resize(crop_rgb_uint8, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1))


def weighted_pose_loss(pred: torch.Tensor, target: torch.Tensor, target_vis: torch.Tensor
                       ) -> Tuple[torch.Tensor, dict]:
    """pred, target: (B, 33, 4) with columns (x, y, z, vis). target_vis is
    target[..., 3] pulled out separately so the weighting is explicit rather
    than re-derived from a column index buried in the loss.

    Position (x, y, z) loss is weighted by ground-truth visibility per joint:
    an occluded joint's true position is exact (Blender knows the whole
    skeleton regardless of what the camera sees), but the network has no
    visual evidence for it, so forcing a precise prediction there teaches it
    to hallucinate from context rather than see. Visibility itself is always
    supervised, unweighted, on all 33 joints - "is this joint visible" is
    itself a real, always-known target."""
    pos_err = (pred[..., :3] - target[..., :3]).pow(2).sum(dim=-1)     # (B, 33)
    pos_loss = (pos_err * target_vis).sum() / target_vis.sum().clamp_min(1.0)
    vis_loss = (pred[..., 3] - target_vis).pow(2).mean()
    total = pos_loss + vis_loss
    return total, {"pos_loss": float(pos_loss.detach()), "vis_loss": float(vis_loss.detach())}


if __name__ == "__main__":
    import numpy as np
    net = HMR3DNet(pretrained=False)   # no download needed for a shape/gradient smoke test
    x = preprocess_crop(np.zeros((300, 300, 3), dtype=np.uint8)).unsqueeze(0)
    out = net(x)
    assert out.shape == (1, NUM_LANDMARKS, 4)

    target = torch.rand(2, NUM_LANDMARKS, 4)
    pred = torch.rand(2, NUM_LANDMARKS, 4, requires_grad=True)
    loss, parts = weighted_pose_loss(pred, target, target[..., 3])
    assert loss.requires_grad and loss.item() > 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    # An occluded (vis=0) joint's position error must not move the gradient;
    # a clearly-visible (vis=1) one must.
    target2 = torch.zeros(1, NUM_LANDMARKS, 4)
    target2[0, 0, 3] = 1.0     # joint 0 visible, rest occluded
    pred2 = torch.zeros(1, NUM_LANDMARKS, 4, requires_grad=True)
    with torch.no_grad():
        pred2[0, 1, :3] = 5.0   # a big position error on an OCCLUDED joint
    loss2, _ = weighted_pose_loss(pred2, target2, target2[..., 3])
    loss2.backward()
    assert torch.isclose(pred2.grad[0, 1, :3], torch.zeros(3), atol=1e-6).all(), \
        "occluded joint's position error must be masked out of the gradient"

    print("hmr3d_model smoke test passed (shapes, backward pass, visibility masking).")
