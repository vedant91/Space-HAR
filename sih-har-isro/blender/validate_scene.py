"""
Numeric checks on the animated scene, so staging problems are found by
measurement instead of by squinting at renders.

Three failure modes this catches, all of which look plausible in a still:

  reach      An IK goal placed further from the shoulder than the arm is long.
             The solver just straightens the arm and stops, so the hand never
             arrives - the render shows a stiff, fully-extended arm pointing
             at nothing, and every wrist landmark for those frames is wrong.

  converge   The solved wrist not actually landing on its goal, even when the
             goal is in range (pole-vector fighting, constraint conflicts).
             This one is invisible in a still but corrupts the ground truth,
             because the exported landmark is the *bone*, not the goal.

  framing    Landmarks or payload leaving the camera frustum. A dataset where
             the hands are out of frame during the step that is defined by
             those hands is worse than no dataset.

Run:
    blender --background --factory-startup --python blender/validate_scene.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_scene                                   # noqa: E402
from bas_har import astronaut as A                   # noqa: E402
from bas_har import groundtruth as GT                # noqa: E402
from bas_har import render as R                      # noqa: E402


def _bone_len(name: str) -> float:
    head, tail, _ = A.BONES[name]
    return (Vector(tail) - Vector(head)).length


def validate(built: dict, cameras=("payload_a",), stride: int = 3) -> dict:
    scene = built["scene"]
    rig = built["crew"]["rig"]
    crew = built["crew"]
    meta = built["meta"]
    labels = meta["labels"]

    arm_reach = {
        "L": _bone_len("upperarm.L") + _bone_len("forearm.L"),
        "R": _bone_len("upperarm.R") + _bone_len("forearm.R"),
    }

    per_step = defaultdict(lambda: {
        "frames": 0, "reach_violations": 0, "max_reach_ratio": 0.0,
        "max_converge_err": 0.0,
    })
    framing = {cam: defaultdict(int) for cam in cameras}
    payload_seen = {cam: defaultdict(int) for cam in cameras}

    frames = list(range(scene.frame_start, scene.frame_end + 1, stride))
    for frame in frames:
        scene.frame_set(frame)
        step = labels[frame - scene.frame_start] if frame - scene.frame_start < len(labels) else 0
        rec = per_step[step]
        rec["frames"] += 1

        for side in ("L", "R"):
            goal = crew["ik"][f"hand.{side}"].matrix_world.translation
            shoulder = rig.matrix_world @ rig.pose.bones[f"upperarm.{side}"].head
            wrist = rig.matrix_world @ rig.pose.bones[f"hand.{side}"].head

            need = (goal - shoulder).length
            ratio = need / arm_reach[side]
            rec["max_reach_ratio"] = max(rec["max_reach_ratio"], ratio)
            if ratio > 0.99:
                rec["reach_violations"] += 1

            err = (wrist - goal).length
            rec["max_converge_err"] = max(rec["max_converge_err"], err)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        for cam_name in cameras:
            scene.camera = built["cameras"][cam_name]
            sampler = GT.GroundTruthSampler(scene, scene.camera, rig, built["payload"])
            feats, _world = sampler.sample_pose(depsgraph)
            arr = feats.reshape(GT.NUM_LANDMARKS, 4)
            # Wrists and shoulders are the landmarks the protocol is defined
            # by; if those leave frame the sample is useless regardless of
            # what the rest of the body is doing.
            for idx, tag in ((15, "wrist_L"), (16, "wrist_R"),
                             (11, "shoulder_L"), (12, "shoulder_R")):
                x, y = float(arr[idx, 0]), float(arr[idx, 1])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    framing[cam_name][tag] += 1
            pay = sampler.sample_payload(depsgraph)
            for key, info in pay.items():
                if info["visible"]:
                    payload_seen[cam_name][key] += 1

    out = {
        "arm_reach_m": {k: round(v, 4) for k, v in arm_reach.items()},
        "sampled_frames": len(frames),
        "per_step": {
            int(k): {
                "frames": v["frames"],
                "reach_violations": v["reach_violations"],
                "max_reach_ratio": round(v["max_reach_ratio"], 3),
                "max_converge_err_m": round(v["max_converge_err"], 4),
            } for k, v in sorted(per_step.items())
        },
        "offscreen_counts": {c: dict(d) for c, d in framing.items()},
        "payload_visible_frames": {c: dict(d) for c, d in payload_seen.items()},
    }
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--frames-per-step", type=int, default=72)
    ap.add_argument("--cameras", default="payload_a,payload_b,payload_wide,payload_over")
    ap.add_argument("--roll", type=float, default=0.0)
    args = ap.parse_args(argv)

    built = build_scene.build(seed=args.seed, body_roll_deg=args.roll,
                              frames_per_step=args.frames_per_step,
                              resolution=(1280, 720), samples=8)
    report = validate(built, cameras=tuple(args.cameras.split(",")), stride=args.stride)
    print("VALIDATE " + json.dumps(report))


if __name__ == "__main__":
    main()
