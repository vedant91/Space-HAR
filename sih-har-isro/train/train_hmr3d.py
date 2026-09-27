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
WRIST = [15, 16]
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


def evaluate_per_joint(model, loader, device, joints=UPPER) -> dict:
    """Same 'tracked' definition as evaluate(), but broken out per joint
    instead of pooled into one median across UPPER. A fine-tune that
    regresses only the wrists (exactly what real pseudo-labels bias toward)
    can still look fine in evaluate()'s pooled median if the other 6 UPPER
    joints are untouched — this is what actually catches that, matching
    train_posenet.py's finetune_on_real() per-joint regression check, which
    exists because the pooled metric alone let a wrist PCK 0.96 -> 0.01
    collapse through undetected."""
    model.eval()
    errs = {j: [] for j in joints}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            gt = y.numpy()
            for p, g in zip(pred, gt):
                torso = np.linalg.norm(g[[11, 12], :2].mean(0) - g[[23, 24], :2].mean(0))
                if torso < 1e-6:
                    continue
                for j in joints:
                    if g[j, 3] < 0.5:   # ground truth says this joint wasn't even visible here
                        continue
                    e = np.linalg.norm(p[j, :2] - g[j, :2]) / torso
                    errs[j].append(e)
    return {j: (float(np.mean(np.array(v) < 0.5)) if v else None) for j, v in errs.items()}


def masked_real_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Position AND visibility loss both weighted by the SAME per-joint
    weight (target[..., 3], which prepare_hmr3d_real_cache.py leaves at 0 on
    every joint the pseudo-labeler didn't attempt, confidence on the wrist(s)
    it did). Deliberately NOT train/hmr3d_model.py's weighted_pose_loss: that
    function's vis_loss is unweighted across all 33 joints, correct for
    synthetic ground truth (which really does know every joint's occlusion
    state) but wrong here — it would teach the model every unlabeled real
    joint is invisible, when the truth is just 'not measured this frame'
    (same distinction data_generation/real_video_autolabel.py's docstring
    draws for the pose labels themselves)."""
    w = target[..., 3].clamp(0.0, 1.0)
    pos_err = (pred[..., :2] - target[..., :2]).pow(2).sum(dim=-1)   # x, y only — no depth signal
    pos_loss = (pos_err * w).sum() / w.sum().clamp_min(1.0)
    vis_loss = ((pred[..., 3] - target[..., 3]).pow(2) * w).sum() / w.sum().clamp_min(1.0)
    return pos_loss + vis_loss


def _freeze_hmr3d_for_finetune(model: HMR3DNet) -> None:
    """Freeze everything except the final Linear of model.head. Mirrors
    train_posenet.py's _configure_finetune_trainable — same reason: real
    pseudo-labels only ever supervise 2 of 33 joints, and any trainable
    layer upstream of the last one mixes wrist-only gradient into features
    every OTHER joint's output also reads, corrupting them even though their
    own loss term is exactly zero (train_posenet.py measured this: 92%
    synthetic-PCK regression from "just" unfreezing more than the last
    layer, even with BatchNorm frozen). The final Linear (head[-1]) is safe
    to leave fully trainable, not just its wrist rows: masked_real_loss's
    per-joint weighting already zeroes dL/d(output) for every non-wrist row,
    and a Linear layer's per-row gradient depends only on that row's own
    output gradient, so the other 31 joints' rows get exactly zero update
    from this loss with no extra masking needed."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head[-1].parameters():
        p.requires_grad_(True)


def finetune_on_real(real_cache: str = str(REPO / "dataset" / "hmr3d_real_cache"),
                     synthetic_cache: str = str(REPO / "dataset" / "hmr3d_cache"),
                     input_path: str = str(REPO / "models" / "hmr3d_net.pt"),
                     output_path: str = str(REPO / "models" / "hmr3d_net.pt"),
                     epochs: int = 15,
                     batch_size: int = 16,
                     lr: float = 3e-4,
                     val_holdout_frac: float = 0.15,
                     max_tracked_regression_frac: float = 0.25,
                     max_joint_tracked_drop: float = 0.10,
                     seed: int = 13) -> dict:
    """Stage 2: fine-tune the synthetic-only HMR3DNet on real pseudo-labeled
    wrist positions (tools/prepare_hmr3d_real_cache.py's cache), the same
    masked-loss / frozen-trunk / synthetic-replay / regression-guard shape
    train_posenet.py's finetune_on_real() already validated for PoseNet's
    exact same real-pseudo-label source. See _freeze_hmr3d_for_finetune and
    masked_real_loss for what's different about HMR3DNet's architecture."""
    torch.manual_seed(seed)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"No base HMR3D checkpoint at {input_path} — run "
                                "train_hmr3d.py's stage-1 trainer first.")

    real_dir = Path(real_cache)
    real_manifest = real_dir / "manifest.npz"
    if not real_manifest.exists():
        print(f"No real cache at {real_manifest} — run tools/prepare_hmr3d_real_cache.py first. Skipping.")
        return {"skipped": True, "reason": "no real cache"}
    rd = np.load(real_manifest, allow_pickle=True)
    real_fnames, real_targets = rd["fnames"], rd["targets"]
    if len(real_fnames) < 10:
        print(f"Only {len(real_fnames)} usable real pseudo-labeled frames — skipping "
             "(need more real footage through data_generation/real_video_autolabel.py first).")
        return {"skipped": True, "n_real_samples": int(len(real_fnames))}

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    pre_metrics = {"tracked": float(ckpt.get("val_tracked") or 0.0),
                   "median_err": float(ckpt.get("val_median_err") or float("inf"))}
    held_out_takes = set(ckpt.get("held_out_takes") or [])
    model = HMR3DNet(pretrained=False)
    model.load_state_dict(ckpt["model_state"])

    backup_path = str(Path(input_path).with_name(Path(input_path).stem + "_pre_real_finetune.pt"))
    import shutil as _shutil
    _shutil.copyfile(input_path, backup_path)
    print(f"Backed up pre-finetune checkpoint to {backup_path} "
         f"(val_tracked={pre_metrics['tracked']:.3f})")

    # Real train/val split — by FRAME here (not by take/video): unlike the
    # synthetic set, adjacent real frames come from only 3-4 short clips
    # total, so a video-level split would leave whole clips with zero
    # representation in either train or val on a set this small.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real_fnames))
    n_val = max(1, int(len(real_fnames) * val_holdout_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    real_train_ds = CropDataset(real_dir, real_fnames[train_idx], real_targets[train_idx])
    real_val_ds = CropDataset(real_dir, real_fnames[val_idx], real_targets[val_idx])

    # Synthetic replay: mirrors train_posenet.py's finding that fine-tuning
    # only on real frames drifts the trainable layer entirely to the real
    # domain even with the trunk frozen (measured there: synthetic wrist PCK
    # 0.96 -> 0.01). Replay uses TRAIN-split synthetic takes only (every take
    # except the checkpoint's own held_out_takes) so the regression check
    # below stays a genuinely unseen comparison.
    synth_dir = Path(synthetic_cache)
    sd = np.load(synth_dir / "manifest.npz", allow_pickle=True)
    s_fnames, s_take_idx, s_targets = sd["fnames"], sd["take_idx"], sd["targets"]
    synth_train_mask = ~np.isin(s_take_idx, list(held_out_takes))
    synth_check_mask = np.isin(s_take_idx, list(held_out_takes))
    synth_replay_ds = CropDataset(synth_dir, s_fnames[synth_train_mask], s_targets[synth_train_mask])
    synth_check_ds = CropDataset(synth_dir, s_fnames[synth_check_mask], s_targets[synth_check_mask])
    print(f"real: {len(real_train_ds)} train / {len(real_val_ds)} val  |  "
         f"synthetic replay: {len(synth_replay_ds)}  |  synthetic regression check "
         f"(held-out takes {sorted(held_out_takes)}): {len(synth_check_ds)}")

    real_loader = DataLoader(real_train_ds, batch_size=min(batch_size, max(len(real_train_ds), 1)),
                             shuffle=True, num_workers=0, drop_last=len(real_train_ds) > batch_size)
    real_val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    replay_loader = DataLoader(synth_replay_ds, batch_size=min(batch_size, max(len(synth_replay_ds), 1)),
                               shuffle=True, num_workers=0, drop_last=len(synth_replay_ds) > batch_size)
    check_loader = DataLoader(synth_check_ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _freeze_hmr3d_for_finetune(model)
    model.eval()   # stays in eval for the whole fine-tune (frozen BN/dropout) — see _freeze_hmr3d_for_finetune
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Fine-tune trainable params: {trainable} (of {total} total) — head[-1] only")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        replay_iter = iter(replay_loader)
        tot_loss, n_batches = 0.0, 0
        for x, y in real_loader:
            x, y = x.to(device), y.to(device)
            try:
                xs, ys = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                xs, ys = next(replay_iter)
            xs, ys = xs.to(device), ys.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = masked_real_loss(model(x), y) + weighted_pose_loss(model(xs), ys, ys[..., 3])[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            tot_loss += float(loss.item()); n_batches += 1
        avg_loss = tot_loss / max(n_batches, 1)

        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for x, y in real_val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += float(masked_real_loss(model(x), y).item()); vn += 1
        val_loss /= max(vn, 1)
        print(f"epoch {epoch:2d}/{epochs}  train_loss={avg_loss:.4f}  real_val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print(f"Fine-tune produced no improving checkpoint — leaving {output_path} untouched.")
        return {"skipped": True, "reason": "no improving epoch", "n_real_samples": len(real_fnames)}

    # ── Regression guard ─────────────────────────────────────────────────
    candidate = HMR3DNet(pretrained=False)
    candidate.load_state_dict(best_state)
    candidate = candidate.to(device)
    post_agg = evaluate(candidate, check_loader, device)
    post_pj = evaluate_per_joint(candidate, check_loader, device, joints=UPPER)

    base_model = HMR3DNet(pretrained=False)
    base_model.load_state_dict(ckpt["model_state"])
    base_model = base_model.to(device)
    base_pj = evaluate_per_joint(base_model, check_loader, device, joints=UPPER)

    regression = (pre_metrics["tracked"] - post_agg["tracked"]) / max(pre_metrics["tracked"], 1e-6)
    joint_drops = {j: (base_pj[j] - post_pj[j]) for j in UPPER
                   if base_pj[j] is not None and post_pj[j] is not None}
    worst_joint = max(joint_drops, key=joint_drops.get) if joint_drops else None
    worst_drop = joint_drops.get(worst_joint, 0.0)
    print("Per-joint synthetic 'tracked' fraction after fine-tune (joint: before -> after): " +
         ", ".join(f"{j}: {base_pj[j]}->{post_pj[j]}" for j in UPPER))

    if worst_drop > max_joint_tracked_drop or \
            (pre_metrics["tracked"] > 0 and regression > max_tracked_regression_frac):
        attempt_path = str(Path(output_path).with_name(Path(output_path).stem + "_real_finetune_attempt.pt"))
        torch.save({"model_state": best_state, "crop_size": CROP_SIZE,
                   "real_finetune_val_loss": best_val_loss,
                   "post_finetune_synthetic_tracked": post_agg["tracked"],
                   "stage": "synthetic+real_finetune_REJECTED"}, attempt_path)
        print(f"REJECTED: synthetic tracked {pre_metrics['tracked']:.3f} -> {post_agg['tracked']:.3f} "
             f"({regression*100:.0f}% regression, limit {max_tracked_regression_frac*100:.0f}%); "
             f"worst joint {worst_joint} dropped {worst_drop:.2f} (limit {max_joint_tracked_drop:.2f}). "
             f"{output_path} left untouched; rejected attempt saved to {attempt_path}.")
        return {"rejected": True, "pre_finetune_tracked": pre_metrics["tracked"],
               "post_finetune_tracked": post_agg["tracked"], "regression_frac": regression,
               "worst_joint": worst_joint, "worst_joint_drop": worst_drop,
               "attempt_path": attempt_path, "n_real_samples": len(real_fnames)}

    torch.save({
        "model_state": best_state,
        "crop_size": CROP_SIZE,
        "val_tracked": pre_metrics["tracked"],                      # pre-finetune synthetic (still meaningful)
        "val_median_err": pre_metrics["median_err"],
        "post_finetune_synthetic_tracked": post_agg["tracked"],
        "real_finetune_val_loss": best_val_loss,
        "held_out_takes": sorted(held_out_takes),
        "stage": "synthetic+real_finetune",
    }, output_path)
    print(f"\nACCEPTED. real_val_loss={best_val_loss:.4f}, n_real={len(real_fnames)}, "
         f"synthetic tracked {pre_metrics['tracked']:.3f} -> {post_agg['tracked']:.3f}")
    return {"best_val_loss": best_val_loss, "n_real_samples": len(real_fnames),
           "pre_finetune_tracked": pre_metrics["tracked"], "post_finetune_tracked": post_agg["tracked"],
           "worst_joint_drop": worst_drop}


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
    ap.add_argument("--finetune-real", action="store_true",
                    help="Stage 2: fine-tune an existing --out checkpoint on "
                         "tools/prepare_hmr3d_real_cache.py's real wrist pseudo-labels instead of "
                         "running the stage-1 synthetic trainer. See finetune_on_real().")
    ap.add_argument("--real-cache", default=str(REPO / "dataset" / "hmr3d_real_cache"))
    ap.add_argument("--finetune-epochs", type=int, default=15)
    args = ap.parse_args()

    if args.finetune_real:
        result = finetune_on_real(real_cache=args.real_cache, synthetic_cache=args.cache,
                                  input_path=args.out, output_path=args.out,
                                  epochs=args.finetune_epochs, lr=args.lr, seed=args.seed)
        print(json.dumps(result, indent=2, default=str))
        return

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
