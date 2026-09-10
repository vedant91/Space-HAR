"""
Custom Pose Heatmap Network — Training Script
==============================================
Replaces MediaPipe entirely. HARPoseNet is a small from-scratch CNN
(no pretrained backbone, no third-party weights) that regresses 2D heatmaps
for the 13 joints that actually drive step classification (see
config.POSE_JOINT_SLOTS), plus a lightweight per-joint depth (z) and
visibility head. Output is re-assembled into the same 33x4=132 "MediaPipe
contract" every downstream module (rack_frame.py, train_lstm.py) expects, so
nothing downstream needs to know the pose came from a different model.

Ground truth for Stage 1 (pretraining) comes for free and *exactly* from
simulation/renderer.py + data_generation/synthetic_pose.py: the renderer
draws the astronaut from a known pose array, so every rendered frame's joint
pixel positions are known without any manual annotation or external model.

Stage 2 (fine-tune) adapts to the real "gravitational mimic" footage in
`new video data/` using classical-CV pseudo-labels (skin-tone + motion wrist
tracking — see data_generation/real_video_autolabel.py), masked so loss is
only computed on joints the pseudo-labeler actually found. No pretrained
network is used at any point in this file.

Usage:
    python train/train_posenet.py                       # synthetic pretrain only
    python train/train_posenet.py --finetune-real        # + real-video fine-tune
    python train/train_posenet.py --n-samples 1500 --epochs 6   # quick smoke test
"""

from __future__ import annotations

import inspect
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    FRAME_WIDTH, FRAME_HEIGHT, POSE_JOINT_SLOTS, POSE_NUM_JOINTS,
    POSE_INPUT_SIZE, POSE_HEATMAP_SIZE, POSE_HEATMAP_SIGMA, POSE_MIN_CONFIDENCE,
    POSE_DECODE_RADIUS, POSE_DECODE_BETA,
    POSENET_EPOCHS, POSENET_BATCH_SIZE, POSENET_LR, POSENET_SYNTH_SAMPLES,
    POSENET_VAL_SAMPLES, POSENET_FINETUNE_EPOCHS, POSENET_FINETUNE_LR,
    POSENET_PCK_THRESHOLD, POSENET_PATH, POSENET_ONNX_PATH,
    SKELETON_FEATURES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# macOS (and Windows) default multiprocessing to spawn, which would re-pickle
# this dataset's in-memory image tensor to every worker process — expensive
# for no benefit at this dataset size. Same convention as train/train_cnn.py.
_NUM_WORKERS = 0 if platform.system() in ("Windows", "Darwin") else 4

NUM_LANDMARKS = 33
JOINT_SLOTS = list(POSE_JOINT_SLOTS)
J = POSE_NUM_JOINTS
IN_SIZE = POSE_INPUT_SIZE
HM_SIZE = POSE_HEATMAP_SIZE
STRIDE = IN_SIZE // HM_SIZE


# Torso landmark slots (within the 33-landmark array), used for PCK scale —
# mirrors pipeline/rack_frame.py's own choice of torso as the scale anchor.
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24


# ═══════════════════════════════════════════════════════════════
# Ground-truth target construction
# ═══════════════════════════════════════════════════════════════

def _gaussian_heatmap(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """(size,size) float32 heatmap, peak 1.0 at (cx,cy) in heatmap-pixel units."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)[:, None]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def pose_to_targets(pose_flat: np.ndarray, valid_mask: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pose_flat: (132,) full MediaPipe-contract pose (normalized [0,1] x,y; z; vis)
    valid_mask: optional (J,) 0/1 — which of the J tracked joints have real
        ground truth this sample (all-ones for synthetic; partial for real
        pseudo-labels). Defaults to all-valid.

    Returns (heatmaps (J,H,W), z (J,), vis (J,), valid (J,)).
    """
    pose = np.asarray(pose_flat, dtype=np.float32).reshape(NUM_LANDMARKS, 4)
    if valid_mask is None:
        valid_mask = np.ones(J, dtype=np.float32)
    heatmaps = np.zeros((J, HM_SIZE, HM_SIZE), dtype=np.float32)
    z = np.zeros(J, dtype=np.float32)
    vis = np.zeros(J, dtype=np.float32)
    for j, slot in enumerate(JOINT_SLOTS):
        if valid_mask[j] < 0.5:
            continue
        x, y, zz, vv = pose[slot]
        cx, cy = float(np.clip(x, 0, 1)) * (HM_SIZE - 1), float(np.clip(y, 0, 1)) * (HM_SIZE - 1)
        heatmaps[j] = _gaussian_heatmap(HM_SIZE, cx, cy, POSE_HEATMAP_SIGMA)
        z[j] = zz
        vis[j] = float(np.clip(vv, 0.0, 1.0))
    return heatmaps, z, vis, valid_mask.astype(np.float32)


def _local_soft_argmax(heatmap: np.ndarray, radius: int = POSE_DECODE_RADIUS,
                       beta: float = POSE_DECODE_BETA) -> Tuple[float, float, float]:
    """
    Sub-pixel joint location: softmax-weighted centroid over a small window
    around the argmax, rather than the argmax pixel itself. A plain argmax on
    a HM_SIZE x HM_SIZE grid quantizes to a whole heatmap pixel, which alone
    can burn most of a PCK@0.10*torso error budget at this resolution — see
    config.POSE_DECODE_RADIUS/BETA. Returns (x, y, peak_val) in heatmap-pixel
    units (peak_val clipped to [0,1], used as the confidence signal).
    """
    size = heatmap.shape[-1]
    iy, ix = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    peak_val = float(np.clip(heatmap[iy, ix], 0.0, 1.0))
    y0, y1 = max(0, iy - radius), min(size, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(size, ix + radius + 1)
    patch = heatmap[y0:y1, x0:x1]
    w = np.exp((patch - patch.max()) * beta)
    w /= w.sum()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    x = float((w * xx).sum())
    y = float((w * yy).sum())
    return x, y, peak_val


def decode_heatmaps(heatmaps: np.ndarray, z: np.ndarray, vis_logits: np.ndarray
                     ) -> np.ndarray:
    """
    heatmaps: (J,H,W) numpy, z: (J,), vis_logits: (J,) raw logits.
    Returns a full (132,) MediaPipe-contract feature vector: the J tracked
    joints filled in at their slot, everything else left at zero (never
    claimed — this model was never trained to localize those landmarks).
    """
    out = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    vis = 1.0 / (1.0 + np.exp(-vis_logits))
    for j, slot in enumerate(JOINT_SLOTS):
        x, y, peak_val = _local_soft_argmax(heatmaps[j])
        conf = float(vis[j]) * peak_val
        if conf < POSE_MIN_CONFIDENCE:
            continue  # not claiming this joint this frame
        out[slot, 0] = (x + 0.5) / HM_SIZE
        out[slot, 1] = (y + 0.5) / HM_SIZE
        out[slot, 2] = float(z[j])
        out[slot, 3] = conf
    return out.reshape(-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Model — 100% from scratch, no pretrained weights
# ═══════════════════════════════════════════════════════════════

class _ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class HARPoseNet(nn.Module):
    """
    From-scratch heatmap-regression pose net.

    Input:  (B, 3, IN_SIZE, IN_SIZE) RGB in [-1, 1]
    Output: heatmaps (B, J, HM_SIZE, HM_SIZE), z (B, J), vis_logits (B, J)

    Encoder (192->96->48->24) with a skip connection from the 48x48 stage
    into a single-step decoder back to 48x48 heatmaps (stride 4 overall) —
    a minimal hourglass, not a generic classifier backbone, sized for CPU
    real-time use (~1.6M params).
    """

    def __init__(self, num_joints: int = J):
        super().__init__()
        self.num_joints = num_joints
        self.stem = _ConvBNReLU(3, 32, k=7, s=2, p=3)          # 192 -> 96
        self.enc1 = nn.Sequential(_ResBlock(32, 64, stride=2),  # 96 -> 48
                                  _ResBlock(64, 64, stride=1))
        self.enc2 = nn.Sequential(_ResBlock(64, 128, stride=2),  # 48 -> 24
                                  _ResBlock(128, 128, stride=1),
                                  _ResBlock(128, 128, stride=1))

        self.up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 24 -> 48
        self.fuse = _ConvBNReLU(64 + 64, 96, k=3, s=1, p=1)
        self.heatmap_head = nn.Conv2d(96, num_joints, kernel_size=1)

        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_joints * 2),  # z + vis_logit per joint
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s = self.stem(x)          # (B,32,96,96)
        e1 = self.enc1(s)         # (B,64,48,48)
        e2 = self.enc2(e1)        # (B,128,24,24)

        u = self.up(e2)           # (B,64,48,48)
        fused = self.fuse(torch.cat([u, e1], dim=1))
        heatmaps = self.heatmap_head(fused)   # (B,J,48,48)

        aux = self.aux_head(e2)                          # (B, 2J)
        z, vis_logits = aux[:, :self.num_joints], aux[:, self.num_joints:]
        return heatmaps, z, vis_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (any size) -> (3, IN_SIZE, IN_SIZE) float32 in [-1, 1]."""
    img = cv2.resize(frame_bgr, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose(2, 0, 1)


# ═══════════════════════════════════════════════════════════════
# Synthetic dataset (Stage 1 — free, exact ground truth)
# ═══════════════════════════════════════════════════════════════

class SyntheticPoseDataset(Dataset):
    """
    Pre-renders `n_samples` (frame, pose) pairs via simulation/renderer.py,
    storing frames already resized to IN_SIZE (keeps memory bounded — a
    full 1280x720 uint8 frame is ~30x bigger than a 192x192 one).
    """

    def __init__(self, n_samples: int, seed: int = 11):
        from simulation.renderer import render_frame
        from data_generation.synthetic_pose import generate_sequence

        rng = np.random.default_rng(seed)
        steps = list(range(1, 9))
        per_step = max(1, n_samples // len(steps))
        self.images: List[np.ndarray] = []
        self.poses: List[np.ndarray] = []

        t0 = time.time()
        for step_id in steps:
            made = 0
            while made < per_step:
                ori = int(rng.integers(0, 4))
                n_frames = int(rng.integers(12, 30))
                seq = generate_sequence(step_id, n_frames=n_frames, rng=rng,
                                        orientation=ori, drift=True)
                # Sample a handful of frames from this trial rather than every
                # one — cheaper, and consecutive frames in one trial are
                # near-duplicates anyway.
                pick = rng.choice(n_frames, size=min(4, n_frames), replace=False)
                for i in pick:
                    if made >= per_step:
                        break
                    t = float(i) / max(n_frames - 1, 1)
                    frame, meta = render_frame(seq[i], step_id, t,
                                               width=FRAME_WIDTH, height=FRAME_HEIGHT, rng=rng)
                    small = cv2.resize(frame, (IN_SIZE, IN_SIZE), interpolation=cv2.INTER_AREA)
                    self.images.append(small)
                    self.poses.append(meta["pose"].copy())
                    made += 1
        logger.info("Rendered %d synthetic pose samples in %.1fs",
                   len(self.images), time.time() - t0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_t = torch.from_numpy(_preprocess(img))
        heatmaps, z, vis, valid = pose_to_targets(self.poses[idx])
        return (img_t, torch.from_numpy(heatmaps), torch.from_numpy(z),
                torch.from_numpy(vis), torch.from_numpy(valid))


class RealPseudoPoseDataset(Dataset):
    """
    Stage 2 fine-tune dataset — real 'gravitational mimic' frames with
    partial (masked) pseudo-labels from data_generation/real_video_autolabel.py.

    Expects `pseudo_dir` to contain, per source video, a `frames/*.jpg` folder
    and a `poses.npz` with arrays `pose` (N,132) and `valid` (N,J) — see
    real_video_autolabel.save_pseudo_labels().
    """

    def __init__(self, pseudo_dir: str):
        self.samples: List[Tuple[Path, np.ndarray, np.ndarray]] = []
        root = Path(pseudo_dir)
        if not root.exists():
            logger.warning("Real pseudo-label dir not found: %s", pseudo_dir)
            return
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            npz_path = video_dir / "poses.npz"
            frames_dir = video_dir / "frames"
            if not npz_path.exists() or not frames_dir.exists():
                continue
            data = np.load(str(npz_path))
            pose, valid = data["pose"], data["valid"]
            frame_files = sorted(frames_dir.glob("*.jpg"))
            n = min(len(frame_files), len(pose))
            for i in range(n):
                if valid[i].sum() < 1:
                    continue  # nothing usable this frame — skip rather than train on noise
                self.samples.append((frame_files[i], pose[i], valid[i]))
        logger.info("RealPseudoPoseDataset: %d usable frames from %s", len(self.samples), pseudo_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pose, valid = self.samples[idx]
        frame = cv2.imread(str(path))
        if frame is None:
            frame = np.zeros((IN_SIZE, IN_SIZE, 3), dtype=np.uint8)
        img_t = torch.from_numpy(_preprocess(frame))
        heatmaps, z, vis, valid_out = pose_to_targets(pose, valid_mask=valid)
        return (img_t, torch.from_numpy(heatmaps), torch.from_numpy(z),
                torch.from_numpy(vis), torch.from_numpy(valid_out))


# ═══════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════

def _masked_loss(heatmaps_pred, heatmaps_gt, z_pred, z_gt, vis_pred, vis_gt, valid):
    """valid: (B,J) 0/1 — zero out loss contributions for untracked/unlabeled joints."""
    m = valid.unsqueeze(-1).unsqueeze(-1)  # (B,J,1,1)
    hm_diff2 = (heatmaps_pred - heatmaps_gt) ** 2 * m
    hm_loss = hm_diff2.sum() / m.sum().clamp_min(1.0) / heatmaps_gt.shape[-1] / heatmaps_gt.shape[-2]

    z_loss_elem = nn.functional.smooth_l1_loss(z_pred, z_gt, reduction="none") * valid
    z_loss = z_loss_elem.sum() / valid.sum().clamp_min(1.0)

    vis_loss_elem = nn.functional.binary_cross_entropy_with_logits(
        vis_pred, vis_gt, reduction="none") * valid
    vis_loss = vis_loss_elem.sum() / valid.sum().clamp_min(1.0)

    return hm_loss + 0.1 * z_loss + 0.5 * vis_loss, hm_loss.item(), z_loss.item(), vis_loss.item()


# ═══════════════════════════════════════════════════════════════
# PCK evaluation
# ═══════════════════════════════════════════════════════════════

def _torso_scale(pose_flat: np.ndarray) -> float:
    pose = np.asarray(pose_flat, dtype=np.float32).reshape(NUM_LANDMARKS, 4)
    sh_mid = pose[[_L_SHOULDER, _R_SHOULDER], :2].mean(axis=0)
    hp_mid = pose[[_L_HIP, _R_HIP], :2].mean(axis=0)
    return float(max(np.linalg.norm(sh_mid - hp_mid), 1e-3))


@torch.no_grad()
def evaluate_pck(model: "HARPoseNet", dataset: Dataset, device,
                  threshold: float = POSENET_PCK_THRESHOLD, batch_size: int = 64) -> Dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    per_joint_correct = np.zeros(J)
    per_joint_total = np.zeros(J)
    idx = 0
    poses = getattr(dataset, "poses", None)
    for img, hm_gt, z_gt, vis_gt, valid in loader:
        img = img.to(device)
        hm_pred, z_pred, vis_logits = model(img)
        hm_pred_np = hm_pred.cpu().numpy()
        z_np = z_pred.cpu().numpy()
        vis_np = vis_logits.cpu().numpy()
        bsz = img.shape[0]
        for b in range(bsz):
            gt_pose = poses[idx + b] if poses is not None else None
            if gt_pose is None:
                idx += bsz
                continue
            torso = _torso_scale(gt_pose)
            pred_feats = decode_heatmaps(hm_pred_np[b], z_np[b], vis_np[b]).reshape(NUM_LANDMARKS, 4)
            gt = np.asarray(gt_pose, dtype=np.float32).reshape(NUM_LANDMARKS, 4)
            for j, slot in enumerate(JOINT_SLOTS):
                if valid[b, j] < 0.5:
                    continue
                dist = float(np.linalg.norm(pred_feats[slot, :2] - gt[slot, :2]))
                ok = dist <= threshold * torso
                correct += int(ok)
                total += 1
                per_joint_correct[j] += int(ok)
                per_joint_total[j] += 1
        idx += bsz
    pck = correct / max(total, 1)
    per_joint = {int(JOINT_SLOTS[j]): float(per_joint_correct[j] / max(per_joint_total[j], 1))
                 for j in range(J)}
    return {"pck": float(pck), "n": int(total), "per_joint_slot_pck": per_joint}


# ═══════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════

def train_posenet(n_samples: int = POSENET_SYNTH_SAMPLES,
                  n_val: int = POSENET_VAL_SAMPLES,
                  epochs: int = POSENET_EPOCHS,
                  batch_size: int = POSENET_BATCH_SIZE,
                  lr: float = POSENET_LR,
                  output_path: str = POSENET_PATH,
                  seed: int = 11) -> Dict:
    """Stage 1: pretrain purely on synthetic renderer ground truth."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SyntheticPoseDataset(n_samples, seed=seed)
    val_ds = SyntheticPoseDataset(n_val, seed=seed + 1000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=_NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HARPoseNet().to(device)
    logger.info("HARPoseNet params: %s | device=%s", f"{model.count_parameters():,}", device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_pck = -1.0
    patience_counter = 0
    max_patience = 6
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = tot_hm = tot_z = tot_vis = 0.0
        n_batches = 0
        for img, hm_gt, z_gt, vis_gt, valid in train_loader:
            img, hm_gt, z_gt, vis_gt, valid = (t.to(device) for t in
                                               (img, hm_gt, z_gt, vis_gt, valid))
            optimizer.zero_grad(set_to_none=True)
            hm_pred, z_pred, vis_logits = model(img)
            loss, hm_l, z_l, vis_l = _masked_loss(hm_pred, hm_gt, z_pred, z_gt,
                                                  vis_logits, vis_gt, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            tot_loss += loss.item(); tot_hm += hm_l; tot_z += z_l; tot_vis += vis_l
            n_batches += 1
        scheduler.step()

        metrics = evaluate_pck(model, val_ds, device)
        elapsed = time.time() - t0
        logger.info(
            "Epoch %02d/%02d | loss=%.4f (hm=%.4f z=%.4f vis=%.4f) | val_PCK@%.2f=%.3f | %.1fs",
            epoch, epochs, tot_loss / max(n_batches, 1), tot_hm / max(n_batches, 1),
            tot_z / max(n_batches, 1), tot_vis / max(n_batches, 1),
            POSENET_PCK_THRESHOLD, metrics["pck"], elapsed,
        )
        history.append({"epoch": epoch, "train_loss": tot_loss / max(n_batches, 1),
                        "val_pck": metrics["pck"]})

        if metrics["pck"] > best_pck:
            best_pck = metrics["pck"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "num_joints": J,
                "joint_slots": JOINT_SLOTS,
                "input_size": IN_SIZE,
                "heatmap_size": HM_SIZE,
                "val_pck": best_pck,
                "epoch": epoch,
                "architecture": "HARPoseNet-scratch",
                "stage": "synthetic",
            }, output_path)
            logger.info("  best checkpoint saved (val_PCK=%.3f)", best_pck)
        else:
            patience_counter += 1
            if patience_counter >= max_patience and epoch >= 10:
                logger.info("Early stopping at epoch %d (best val_PCK=%.3f).", epoch, best_pck)
                break

    logger.info("Stage-1 (synthetic) training complete. Best val PCK=%.3f", best_pck)
    _export_onnx(output_path)
    return {"val_pck": best_pck, "history": history, "n_train": len(train_ds), "n_val": len(val_ds)}


def _freeze_batchnorm(model: "HARPoseNet") -> None:
    """Keep BatchNorm layers in eval mode (fixed running mean/var, frozen
    affine params) throughout fine-tuning. Without this, ~800 real frames —
    tiny, and with very different pixel statistics from the synthetic
    renders — quickly drag the running stats to a real-only distribution;
    at eval time on synthetic (or even just a *different* real clip) the
    mismatched normalization can collapse the whole network's output
    (measured: synthetic PCK 0.66 -> 0.005 with BN left trainable). Standard
    "freeze BN when fine-tuning on a small/out-of-domain set" practice."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def _configure_finetune_trainable(model: "HARPoseNet") -> None:
    """Freeze the ENTIRE shared feature trunk (stem/enc1/enc2/up/fuse) and
    everything in aux_head except its final Linear — only `heatmap_head`
    (a 1x1 conv, one independent output-channel weight vector per joint) and
    aux_head's last Linear (one independent output-row per z/vis value) stay
    trainable.

    Why: real pseudo-labels only ever supervise the two wrist joints (see
    data_generation/real_video_autolabel.py), so only wrist-related output
    rows get gradient. Even with BatchNorm frozen (_freeze_batchnorm), that
    gradient still flows back through the SHARED trunk that feeds every
    joint's output — measured result: 92% synthetic-PCK regression after
    just 8 epochs at lr=1e-4, i.e. still-catastrophic forgetting of the other
    12 joints' localization ability. Freezing the trunk entirely makes that
    literally impossible: other joints' output rows and the features feeding
    them never change, so this fine-tune can only ever improve (or leave
    unchanged) how well the model reads its existing, frozen, general-purpose
    features to localize wrists specifically — it cannot regress anything
    else by construction. The tradeoff is a lower ceiling (frozen
    synthetic-only features may not describe real images as well as further
    encoder adaptation could) — an intentional, safer trade validated by the
    regression guard in finetune_on_real() either way.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.heatmap_head.parameters():
        p.requires_grad_(True)
    for p in model.aux_head[-1].parameters():   # final Linear(128, 2*num_joints)
        p.requires_grad_(True)


def finetune_on_real(pseudo_dir: str = "dataset/real_pseudo",
                     input_path: str = POSENET_PATH,
                     output_path: str = POSENET_PATH,
                     epochs: int = POSENET_FINETUNE_EPOCHS,
                     batch_size: int = 16,
                     lr: float = POSENET_FINETUNE_LR,
                     val_holdout_frac: float = 0.15,
                     max_pck_regression_frac: float = 0.25,
                     n_regression_check_samples: int = 300,
                     seed: int = 13) -> Dict:
    """Stage 2: fine-tune the synthetic-pretrained PoseNet on real, classical-CV
    pseudo-labeled 'gravitational mimic' frames (masked loss — only joints the
    pseudo-labeler actually found contribute gradient). BatchNorm is frozen
    (see _freeze_batchnorm) and a synthetic-PCK regression guard refuses to
    overwrite `output_path` if the fine-tuned model lost more than
    `max_pck_regression_frac` of the pre-finetune synthetic PCK — catching a
    bad fine-tune run instead of silently shipping a collapsed model."""
    torch.manual_seed(seed)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"No base PoseNet checkpoint at {input_path} — run "
                                "train_posenet() (Stage 1) first.")

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    pre_finetune_pck = float(ckpt.get("val_pck") or 0.0)
    model = HARPoseNet(num_joints=ckpt["num_joints"])
    model.load_state_dict(ckpt["model_state"])

    # Preserve the pre-finetune checkpoint — Stage 2 overwrites `output_path`
    # (usually == input_path) on success, and without this a bad run has no
    # way back to the known-good Stage-1 weights.
    backup_path = str(Path(input_path).with_name(Path(input_path).stem + "_pre_finetune.pt"))
    import shutil as _shutil
    _shutil.copyfile(input_path, backup_path)
    logger.info("Backed up pre-finetune checkpoint to %s (val_pck=%.3f)",
               backup_path, pre_finetune_pck)

    full_ds = RealPseudoPoseDataset(pseudo_dir)
    if len(full_ds) < 10:
        logger.warning("Only %d usable real pseudo-labeled frames — skipping fine-tune "
                       "(need real footage processed by real_video_autolabel.py first).",
                       len(full_ds))
        return {"skipped": True, "n_real_samples": len(full_ds)}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(full_ds))
    n_val = max(1, int(len(full_ds) * val_holdout_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    from torch.utils.data import Subset
    train_ds, val_ds = Subset(full_ds, train_idx.tolist()), Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(len(train_ds), 1)),
                              shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _configure_finetune_trainable(model)   # freeze the shared trunk — see its docstring
    _freeze_batchnorm(model)                # belt-and-suspenders: BN stats never drift either
    model.eval()  # stays in eval mode for the whole fine-tune — see _configure_finetune_trainable;
                  # .eval() only changes BN/Dropout forward behavior, not autograd/backward
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    logger.info("Fine-tune trainable params: %d (of %d total) — heatmap_head + aux_head final layer only",
               sum(p.numel() for p in model.parameters() if p.requires_grad),
               sum(p.numel() for p in model.parameters()))

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        tot_loss, n_batches = 0.0, 0
        for img, hm_gt, z_gt, vis_gt, valid in train_loader:
            img, hm_gt, z_gt, vis_gt, valid = (t.to(device) for t in
                                               (img, hm_gt, z_gt, vis_gt, valid))
            optimizer.zero_grad(set_to_none=True)
            hm_pred, z_pred, vis_logits = model(img)
            loss, *_ = _masked_loss(hm_pred, hm_gt, z_pred, z_gt, vis_logits, vis_gt, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            tot_loss += loss.item(); n_batches += 1
        avg_loss = tot_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            vn = 0
            for img, hm_gt, z_gt, vis_gt, valid in val_loader:
                img, hm_gt, z_gt, vis_gt, valid = (t.to(device) for t in
                                                   (img, hm_gt, z_gt, vis_gt, valid))
                hm_pred, z_pred, vis_logits = model(img)
                l, *_ = _masked_loss(hm_pred, hm_gt, z_pred, z_gt, vis_logits, vis_gt, valid)
                val_loss += l.item(); vn += 1
        val_loss /= max(vn, 1)
        logger.info("Fine-tune epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f",
                   epoch, epochs, avg_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        logger.warning("Fine-tune produced no improving checkpoint — leaving %s untouched.", output_path)
        return {"skipped": True, "reason": "no improving epoch", "n_real_samples": len(full_ds)}

    # ── Regression guard ────────────────────────────────────────────────
    # Load the best fine-tuned weights into a fresh model and check it
    # hasn't collapsed on the ORIGINAL synthetic task before trusting it.
    candidate = HARPoseNet(num_joints=ckpt["num_joints"])
    candidate.load_state_dict(best_state)
    candidate.eval()
    check_ds = SyntheticPoseDataset(n_regression_check_samples, seed=seed + 5000)
    post_metrics = evaluate_pck(candidate, check_ds, torch.device("cpu"))
    post_pck = post_metrics["pck"]
    regression = (pre_finetune_pck - post_pck) / max(pre_finetune_pck, 1e-6)

    if pre_finetune_pck > 0 and regression > max_pck_regression_frac:
        attempt_path = str(Path(output_path).with_name(Path(output_path).stem + "_finetune_attempt.pt"))
        torch.save({"model_state": best_state, **{k: v for k, v in ckpt.items() if k != "model_state"},
                   "real_finetune_val_loss": best_loss, "post_finetune_synthetic_pck": post_pck,
                   "stage": "synthetic+real_finetune_REJECTED"}, attempt_path)
        logger.error(
            "REJECTED fine-tune result: synthetic PCK %.3f -> %.3f (%.0f%% regression, limit %.0f%%). "
            "%s left untouched; the rejected attempt was saved to %s for inspection. "
            "Likely cause: too-aggressive adaptation to a small/narrow real set even with BN frozen — "
            "try a lower --lr or fewer epochs.",
            pre_finetune_pck, post_pck, regression * 100, max_pck_regression_frac * 100,
            output_path, attempt_path)
        return {"rejected": True, "pre_finetune_pck": pre_finetune_pck, "post_finetune_pck": post_pck,
               "regression_frac": regression, "attempt_path": attempt_path,
               "n_real_samples": len(full_ds)}

    torch.save({
        "model_state": best_state,
        "num_joints": ckpt["num_joints"],
        "joint_slots": ckpt["joint_slots"],
        "input_size": ckpt["input_size"],
        "heatmap_size": ckpt["heatmap_size"],
        "val_pck": pre_finetune_pck,                 # pre-finetune synthetic PCK (still meaningful)
        "post_finetune_synthetic_pck": post_pck,      # confirms how much (if any) it moved
        "real_finetune_val_loss": best_loss,
        "epoch": epoch,
        "architecture": "HARPoseNet-scratch",
        "stage": "synthetic+real_finetune",
    }, output_path)
    logger.info("Stage-2 (real fine-tune) complete and ACCEPTED. Best val_loss=%.4f, n_real=%d, "
               "synthetic PCK %.3f -> %.3f", best_loss, len(full_ds), pre_finetune_pck, post_pck)
    _export_onnx(output_path)
    return {"best_val_loss": best_loss, "n_real_samples": len(full_ds),
           "n_train": len(train_ds), "n_val": len(val_ds),
           "pre_finetune_pck": pre_finetune_pck, "post_finetune_pck": post_pck}


def _export_onnx(model_pt_path: str):
    """Export to ONNX for CPU-optimized inference (mirrors train_lstm.py /
    train_cnn.py's version-gated dynamo=False workaround for PyTorch 2.6+)."""
    try:
        ckpt = torch.load(model_pt_path, map_location="cpu", weights_only=False)
        model = HARPoseNet(num_joints=ckpt["num_joints"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        dummy = torch.randn(1, 3, ckpt["input_size"], ckpt["input_size"])
        export_kwargs = dict(
            export_params=True,
            opset_version=17,
            input_names=["frame"],
            output_names=["heatmaps", "z", "vis_logits"],
            dynamic_axes={"frame": {0: "batch"}, "heatmaps": {0: "batch"},
                          "z": {0: "batch"}, "vis_logits": {0: "batch"}},
        )
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        torch.onnx.export(model, dummy, POSENET_ONNX_PATH, **export_kwargs)
        logger.info("PoseNet ONNX saved: %s", POSENET_ONNX_PATH)
    except Exception as e:
        logger.error("PoseNet ONNX export failed: %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train HARPoseNet from scratch")
    parser.add_argument("--n-samples", type=int, default=POSENET_SYNTH_SAMPLES)
    parser.add_argument("--n-val", type=int, default=POSENET_VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=POSENET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=POSENET_BATCH_SIZE)
    parser.add_argument("--finetune-real", action="store_true",
                        help="Also run the Stage-2 real-video fine-tune after Stage 1")
    parser.add_argument("--finetune-only", action="store_true",
                        help="Skip Stage 1, fine-tune an existing checkpoint on real data only")
    parser.add_argument("--pseudo-dir", type=str, default="dataset/real_pseudo")
    parser.add_argument("--finetune-epochs", type=int, default=POSENET_FINETUNE_EPOCHS)
    parser.add_argument("--finetune-lr", type=float, default=POSENET_FINETUNE_LR)
    args = parser.parse_args()

    result = {}
    if not args.finetune_only:
        result["stage1"] = train_posenet(n_samples=args.n_samples, n_val=args.n_val,
                                         epochs=args.epochs, batch_size=args.batch_size)
    if args.finetune_real or args.finetune_only:
        result["stage2"] = finetune_on_real(pseudo_dir=args.pseudo_dir,
                                            epochs=args.finetune_epochs, lr=args.finetune_lr)

    print(json.dumps(result, indent=2, default=str))
