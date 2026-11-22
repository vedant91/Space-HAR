"""Fit a 3D residual refiner on rendered camera observations and rig labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.depth_lifter import DepthLifter, encode, refine_image_pose
from pipeline.pose_viz import HipDepthFilter
from pipeline.pose_refiner import PoseRefiner, WEIGHTS, observations, width_units
from train.train_depth_lifter import BODY, batches, camera_joints


def load_takes(root, split, lifter):
    for name, skeleton, truth, size, metres, rgb, fov in batches(root, split, True):
        x, scale, valid = encode(skeleton, size)
        keep = valid & (rgb[..., 3].max(axis=1) > 0)
        with torch.inference_mode():
            depth = lifter(torch.from_numpy(x)).numpy() * scale[:, None]
        base = refine_image_pose(skeleton, rgb, size)
        rgb_depth = rgb[..., 2] - rgb[:, [23, 24], 2].mean(axis=1, keepdims=True)
        base[..., 2] = depth * lifter.skeleton_weight + rgb_depth * (1 - lifter.skeleton_weight)
        features = observations(skeleton, rgb, base, size)
        yield name, features[keep], truth[keep], base[keep], size, metres[keep], fov, len(truth)


@torch.inference_mode()
def score(model, takes, motion=False):
    totals = {"published": [], "refined": []}
    depths = {"published": [], "refined": []}
    detections = {"published": [], "refined": []}
    by_take = {}
    by_joint = {key: [[] for _ in BODY] for key in totals}
    motion_errors = {key: [] for key in totals}
    frames = matched = 0
    for name, x, gt, base, size, metres, fov, total in takes:
        frames += total
        matched += len(x)
        if not len(x):
            continue
        pred = base.copy()
        pred[..., :3] = model(torch.from_numpy(x)).numpy() / width_units(size)
        pred[..., :2] += .5
        mask = gt[:, BODY, 3] >= .5
        truth_3d = camera_joints(gt, gt[..., 2], size, metres, fov)
        torso = np.linalg.norm((gt[:, [11, 12], :2].mean(1) -
                                gt[:, [23, 24], :2].mean(1)) * size, axis=1)
        by_take[name] = {}
        for key, pose in (("published", base), ("refined", pred)):
            xyz = camera_joints(pose, pose[..., 2], size, metres, fov)
            errors = np.linalg.norm(xyz - truth_3d, axis=-1)[:, BODY][mask] * 1000
            totals[key].extend(errors.tolist())
            joint_errors = np.linalg.norm(xyz - truth_3d, axis=-1)[:, BODY] * 1000
            for index in range(len(BODY)):
                by_joint[key][index].extend(joint_errors[mask[:, index], index].tolist())
            delta_z = abs(pose[..., 2] - gt[..., 2]) * metres[:, None] * 100
            depths[key].extend(delta_z[:, BODY][mask].tolist())
            if motion and len(x) == total:
                filt = HipDepthFilter()
                filtered = np.stack([filt(z, i / 10) for i, z in enumerate(pose[..., 2])])
                signed = (filtered - gt[..., 2]) * metres[:, None] * 100
                adjacent = mask[1:] & mask[:-1]
                motion_errors[key].extend(abs(np.diff(signed[:, BODY], axis=0))[adjacent].tolist())
            delta_xy = np.linalg.norm((pose[..., :2] - gt[..., :2]) * size, axis=-1)
            hit = (delta_xy[:, BODY] < .5 * torso[:, None]) & (pose[:, BODY, 3] >= .15)
            detections[key].extend(hit[mask].tolist())
            by_take[name][key] = float(errors.mean()) if len(errors) else None
    def means(values):
        return {k: float(np.mean(v)) if len(v) else None for k, v in values.items()}
    if not totals["refined"]:
        raise ValueError("No paired visible joints to score")
    return {"frames": frames, "matched_frames": matched, "joints": len(totals["refined"]),
            "mpjpe_mm": means(totals), "depth_mae_cm": means(depths),
            "pck_half_torso": means(detections),
            "mpjpe_90th_percentile_mm": {k: float(np.percentile(v, 90)) for k, v in totals.items()},
            "mpjpe_by_joint_mm": {k: means(dict(zip(BODY, v))) for k, v in by_joint.items()},
            "filtered_depth_change_mae_cm_per_100ms": means(motion_errors) if motion else {},
            "mpjpe_by_take_mm": by_take}


def train(model, root, lifter, epochs, output, width):
    training = list(load_takes(root, "train", lifter)) + list(load_takes(root, "refine_train", lifter))
    validation = list(load_takes(root, "val", lifter))
    inputs, targets, weights = [], [], []
    for _, x, truth, _, size, metres, _, _ in training:
        goal = truth[..., :3] * width_units(size)
        goal[..., :2] -= width_units(size)[:2] / 2
        weight = truth[..., 3].clip(.15, 1)
        weight[:, BODY] *= 2
        inputs.append(x)
        targets.append(goal)
        weights.append(weight * metres[:, None])
    x, y, weight = [torch.from_numpy(np.concatenate(a)).float() for a in (inputs, targets, weights)]
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for k in torch.randperm(len(x)).split(64):
            batch = x[k] + torch.randn_like(x[k]) * .007
            pred = model(batch)
            loss = (F.smooth_l1_loss(pred, y[k], reduction="none", beta=.03)
                    * weight[k, :, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        result = score(model.eval(), validation)
        error = result["mpjpe_mm"]["refined"]
        if error < best:
            best = error
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(format="bas-pose-refiner-v1", width=width, epoch=epoch,
                            model=model.state_dict()), output)
            print(f"Epoch {epoch}: validation MPJPE {error:.1f} mm", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=WEIGHTS)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--evaluate", choices=("val", "refine_test", "motion"))
    args = parser.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(3)
    lifter = DepthLifter.load()
    if args.evaluate:
        print(json.dumps(score(PoseRefiner.load(args.out),
                               load_takes(args.dataset, args.evaluate, lifter),
                               motion=args.evaluate == "motion"), indent=2))
    else:
        train(PoseRefiner(args.width), args.dataset, lifter, args.epochs, args.out, args.width)
