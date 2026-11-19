"""Train and score BAS depth on disjoint Blender animations and camera observations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.depth_lifter import DepthLifter, FORMAT, WEIGHTS, encode

BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def batches(root, split, measured):
    for path in sorted((root / split).glob("take_*/pose.npz")):
        with np.load(path) as data:
            target, size, metres = data["pose"], data["size"], data["metres_per_width"]
        old = None
        if measured:
            with np.load(path.with_name("observations.npz")) as obs:
                inp, old, ix = obs["skeleton"], obs["rgb_hmr"], obs["offset"]
            target, metres = target[ix], metres[ix]
        else:
            inp = target
        yield path.parent.name, inp, target, size, metres, old


@torch.inference_mode()
def evaluate(model, root, split):
    model.eval()
    totals = {k: [] for k in ("lift", "rgb_hmr", "mediapipe", "flat")}
    per_take = {}
    per_joint = {j: [] for j in BODY}
    total_frames = lifted = common = 0
    for name, inp, gt, size, metres, old in batches(root, split, True):
        x, scale, valid = encode(inp, size)
        prediction = model(torch.from_numpy(x)).numpy() * scale[:, None]
        mask = valid & (old[..., 3].max(axis=1) > 0)
        total_frames += len(gt)
        lifted += int(valid.sum())
        common += int(mask.sum())
        visible = (gt[..., 3] >= 0.5) & mask[:, None]
        estimates = {"lift": prediction, "rgb_hmr": old[..., 2], "mediapipe": inp[..., 2],
                     "flat": np.zeros_like(prediction)}
        row = {"frames": len(gt), "lifted": int(valid.sum()), "common": int(mask.sum())}
        for method, depths in estimates.items():
            depths = depths - depths[:, [23, 24]].mean(axis=1, keepdims=True)
            errors = abs(depths - gt[..., 2]) * metres[:, None] * 100
            selected = errors[:, BODY][visible[:, BODY]]
            totals[method].extend(selected.tolist())
            row[method] = float(selected.mean()) if selected.size else None
            if method == "lift":
                for j in BODY:
                    per_joint[j].extend(errors[visible[:, j], j].tolist())
        per_take[name] = row
    return {"frames": total_frames, "lifted": lifted, "common": common,
            "visible_joint_observations": len(totals["lift"]),
            "depth_mae_cm": {m: float(np.mean(v)) if v else None for m, v in totals.items()},
            "joint_mae_cm": {j: float(np.mean(v)) if v else None for j, v in per_joint.items()},
            "takes": per_take}


def training_arrays(root, measured):
    xs, ys, ws = [], [], []
    for _, inp, gt, size, _, _ in batches(root, "train", measured):
        x, scale, valid = encode(inp, size)
        xs.append(x[valid])
        ys.append(gt[valid, :, 2] / scale[valid, None])
        ws.append(gt[valid, :, 3].clip(.1, 1))
    arrays = [torch.from_numpy(np.concatenate(a)) for a in (xs, ys, ws)]
    if not len(arrays[0]):
        raise ValueError("Training requires at least one anchored pose in each input source")
    return arrays


def train(root, output, epochs, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = DepthLifter()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    clean, observed = training_arrays(root, False), training_arrays(root, True)
    best, best_epoch = float("inf"), None
    joint_weight = torch.full((33,), 0.25)
    joint_weight[BODY] = 1
    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(max(1, len(clean[0]) // 32)):
            batch = []
            for source in (clean, observed):
                idx = torch.randint(len(source[0]), (32,))
                batch.append(tuple(t[idx] for t in source))
            x, y, w = (torch.cat(items) for items in zip(*batch))
            x[..., :2] += torch.randn_like(x[..., :2]) * 0.035
            hidden = torch.rand_like(w) < .06
            x[hidden] = 0
            weight = w * joint_weight
            error = F.smooth_l1_loss(model(x), y, reduction="none", beta=.15)
            loss = (error * weight).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = evaluate(model, root, "val")["depth_mae_cm"]["lift"]
        if score is not None and score < best:
            best, best_epoch = score, epoch
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": FORMAT, "model_state": model.state_dict(), "seed": seed,
                        "epoch": epoch, "validation_depth_mae_cm": score}, output)
    if best_epoch is None:
        raise ValueError("No visible, paired validation joints")
    print(f"Selected epoch {best_epoch} of {epochs}, validation depth MAE {best:.2f} cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=WEIGHTS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--evaluate", action="store_true", help="Score existing weights on test takes")
    args = ap.parse_args()
    if not args.evaluate:
        train(args.dataset, args.out, args.epochs, args.seed)
    else:
        print(json.dumps(evaluate(DepthLifter.load(args.out), args.dataset, "test"), indent=2))
