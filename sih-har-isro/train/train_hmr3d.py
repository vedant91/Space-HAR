"""
Train HMR3DNet (train/hmr3d_model.py) on the cache built by
tools/prepare_hmr3d_dataset.py, replacing ROMP as the HMR backend.

Split by TAKE, never by frame (GroupShuffleSplit) - the same rule
train/train_lstm.py already follows and for the same reason: two frames a
few hundred ms apart from the same take are nearly identical, so a
frame-level split lets the model memorise a take instead of generalising to
an unseen one. take_0022 ("heldout_view") is always forced into the held-out
set: it is the one take rendered from a different camera entirely, the
closest thing this dataset has to a genuine unseen-viewpoint test.

Evaluation metric matches tools/measure_hmr_crop.py's ROMP scoring exactly
(median upper-body joint error / torso length, "tracked" = under 0.5 torso),
computed directly in the cached crop-local coordinates. This is a valid
like-for-like comparison because HMRBackend._crop only rotates and
translates (square_crop uses scale=1.0) - it never rescales - so the
error/torso RATIO is identical whether measured in crop-local or full-frame
pixels; only the common scale factor differs, and it cancels out of a ratio.

Run under system Python (needs torch + torchvision; does not need mediapipe
or ultralytics - the cache is already built):
    python train/train_hmr3d.py --epochs 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.hmr3d_model import HMR3DNet, preprocess_crop, weighted_pose_loss, CROP_SIZE  # noqa: E402

UPPER = [11, 12, 13, 14, 15, 16, 23, 24]   # shoulders, elbows, wrists, hips - matches measure_hmr_crop.py
HELDOUT_VIEW_TAKE = 22                      # different camera entirely; always held out


class CropDataset(Dataset):
    def __init__(self, cache_dir: Path, fnames: np.ndarray, targets: np.ndarray):
        self.cache_dir = cache_dir
        self.fnames = fnames
        self.targets = targets

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img = cv2.imread(str(self.cache_dir / "crops" / self.fnames[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess_crop(img)   # already CROP_SIZE from prepare_hmr3d_dataset.py, but resize is a no-op if so
        y = torch.from_numpy(self.targets[i].astype(np.float32))
        return x, y


def group_split(take_idx: np.ndarray, tags: np.ndarray, val_fraction: float, seed: int,
                force_val: set = frozenset({HELDOUT_VIEW_TAKE})):
    """Held-out TAKES (not frames). Every take in `force_val` is always in
    val (default: just take_0022, the different-camera take); the rest
    filled by random takes up to val_fraction of all takes. Pass the 8
    ROMP_UNRELIABLE_TAGS take indices here for a fair, apples-to-apples
    comparison against ROMP's numbers - those takes must be UNSEEN during
    training, or "this model beats ROMP on the same orientations" is
    measuring interpolation, not generalisation."""
    takes = sorted(set(take_idx.tolist()))
    rng = np.random.default_rng(seed)
    others = [t for t in takes if t not in force_val]
    rng.shuffle(others)
    n_val = max(len(force_val), int(round(val_fraction * len(takes)))) - len(force_val)
    val_takes = set(force_val) | set(others[:max(n_val, 0)])
    train_mask = ~np.isin(take_idx, list(val_takes))
    val_mask = np.isin(take_idx, list(val_takes))
    return train_mask, val_mask, val_takes


def evaluate(model, loader, device) -> dict:
    """Median upper-body joint error / torso, and the fraction of frames
    "tracked" (median error < 0.5 torso) - same definition as
    tools/measure_hmr_crop.py's frame_error(), applied per-frame here."""
    model.eval()
    medians = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            gt = y.numpy()
            for p, g in zip(pred, gt):
                torso = np.linalg.norm(g[[11, 12], :2].mean(0) - g[[23, 24], :2].mean(0))
                if torso < 1e-6:
                    continue
                e = np.linalg.norm(p[UPPER, :2] - g[UPPER, :2], axis=1) / torso
                medians.append(np.median(e))
    medians = np.array(medians)
    return {"n": len(medians), "tracked": float(np.mean(medians < 0.5)) if len(medians) else 0.0,
            "median_err": float(np.median(medians)) if len(medians) else float("inf")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "dataset" / "hmr3d_cache"))
    ap.add_argument("--out", default=str(REPO / "models" / "hmr3d_net.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force-holdout", default=str(HELDOUT_VIEW_TAKE),
                    help="comma-separated take indices always held out of training, on top of the "
                         "random split. Default is just take_0022. For a fair comparison against "
                         "ROMP's numbers, pass the 8 HMR_UNRELIABLE_TAGS takes: "
                         "1,2,3,4,7,13,15,17")
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    d = np.load(cache_dir / "manifest.npz", allow_pickle=True)
    fnames, take_idx, tags, targets = d["fnames"], d["take_idx"], d["tags"], d["targets"]

    force_val = {int(t) for t in args.force_holdout.split(",")}
    train_mask, val_mask, val_takes = group_split(take_idx, tags, args.val_fraction, args.seed, force_val)
    val_tags = sorted(set(tags[val_mask].tolist()))
    print(f"train frames: {train_mask.sum()}  val frames: {val_mask.sum()}")
    print(f"held-out takes: {sorted(val_takes)}  tags: {val_tags}")

    train_ds = CropDataset(cache_dir, fnames[train_mask], targets[train_mask])
    val_ds = CropDataset(cache_dir, fnames[val_mask], targets[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = HMR3DNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_tracked = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"pos_loss": 0.0, "vis_loss": 0.0, "n": 0}
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss, parts = weighted_pose_loss(pred, y, y[..., 3])
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in ("pos_loss", "vis_loss"):
                running[k] += parts[k] * len(x)
            running["n"] += len(x)
        sched.step()

        metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"pos_loss={running['pos_loss']/running['n']:.4f}  "
              f"vis_loss={running['vis_loss']/running['n']:.4f}  "
              f"val_tracked={metrics['tracked']:.3f}  val_median_err={metrics['median_err']:.3f}  "
              f"({dt:.0f}s)")
        history.append({"epoch": epoch + 1, **metrics})

        if metrics["tracked"] > best_tracked:
            best_tracked = metrics["tracked"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "val_tracked": metrics["tracked"],
                "val_median_err": metrics["median_err"],
                "held_out_takes": sorted(val_takes),
                "held_out_tags": val_tags,
                "epoch": epoch + 1,
                "crop_size": CROP_SIZE,
            }, args.out)
            print(f"  -> new best, saved to {args.out}")

    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nDone. Best val_tracked={best_tracked:.3f}. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
