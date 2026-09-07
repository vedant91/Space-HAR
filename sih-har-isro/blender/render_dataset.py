"""
Render the BAS HAR dataset: photoreal frames plus exact per-frame ground truth.

Each "take" is one performance of the protocol under one set of conditions
(crew orientation, lighting, clutter, sequence), rendered from one or more
fixed payload cameras. Written per take/camera:

    take_XXXX/meta.json
    take_XXXX/<camera>/frames/00000.png ...
    take_XXXX/<camera>/pose_2d.npy      (T, 132)  MediaPipe convention
    take_XXXX/<camera>/pose_world.npy   (T, 33, 3) metric, world frame
    take_XXXX/<camera>/labels.npy       (T,)      step id per frame
    take_XXXX/<camera>/payload.json     per-frame true boxes + visibility

The variation axes are chosen to attack the specific weaknesses of the
existing pipeline rather than to add generic diversity:

  body_roll   The whole point of the "orientation-agnostic" requirement. A
              crew member at 0, 45, 90 and 180 degrees relative to the rack
              is the case a floor-referenced pose model cannot handle, and
              the case `pipeline/rack_frame.py` was written for but has never
              had data to be validated against.
  lighting    Nominal / dim / one-luminaire-failed. A model trained under a
              single fixed light rig learns the rig.
  distractors Whether anything else in frame sits near the payload HSV bands.
              "harsh" uses ISS-accurate pale-gold handrails, which genuinely
              do fall close to HSV_YELLOW - that measures how much of the
              detector's reported recall is an artifact of a clean scene.
  sequence    Nominal 1..8, a skip, and a skip-then-correct. The state
              machine's hold/recovery path currently only ever sees
              hand-written perfect predictions.

Resumable: a take whose meta.json already exists is skipped, so a long run
can be interrupted and restarted without losing work.

Run:
    blender --background --factory-startup --python blender/render_dataset.py -- \
        --out dataset/blender --plan smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_scene                                   # noqa: E402
from bas_har import groundtruth as GT                # noqa: E402
from bas_har import materials as M                   # noqa: E402
from bas_har import render as R                      # noqa: E402

NOMINAL = list(range(1, 9))
SKIP_STEP2 = [1, 3, 4, 5, 6, 7, 8]
RECOVER_STEP2 = [1, 3, 2, 3, 4, 5, 6, 7, 8]


def build_plan(name: str) -> list:
    """Return a list of take specifications."""
    if name == "smoke":
        return [
            dict(seed=11, roll=0.0, lighting="nominal", distractors="moderate",
                 sequence=NOMINAL, cameras=["payload_a"], frames_per_step=20),
        ]

    if name == "orientation":
        # One axis at a time, so an accuracy drop is attributable.
        return [
            dict(seed=100 + i, roll=r, lighting="nominal", distractors="moderate",
                 sequence=NOMINAL, cameras=["payload_a"], frames_per_step=45)
            for i, r in enumerate((0.0, 25.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0))
        ]

    if name == "study":
        # The plan actually rendered for this project, sized to the available
        # hardware (no CUDA GPU here, so EEVEE on an 8-thread CPU at ~4 s a
        # frame). Every take earns its render time by isolating one variable:
        #
        #   0-5   crew orientation sweep. Train/test are split BY TAKE, so a
        #         held-out orientation is genuinely unseen - the thing the
        #         orientation-agnostic requirement is actually about.
        #   6-7   lighting degradation.
        #   8-9   colour clutter, for honest HSV recall.
        #  10-11  protocol errors, for the state machine's hold/recovery path.
        plan = []
        for i, roll in enumerate((0.0, 45.0, 90.0, 135.0, 180.0, -45.0)):
            plan.append(dict(seed=200 + i, roll=roll, lighting="nominal",
                             distractors="moderate", sequence=NOMINAL,
                             cameras=["payload_a"], frames_per_step=32,
                             tag=f"orient_{int(roll)}"))
        for i, (light, roll) in enumerate((("dim", 0.0), ("single_fail", 90.0))):
            plan.append(dict(seed=300 + i, roll=roll, lighting=light,
                             distractors="moderate", sequence=NOMINAL,
                             cameras=["payload_a"], frames_per_step=32,
                             tag=f"light_{light}"))
        for i, dis in enumerate(("none", "harsh")):
            plan.append(dict(seed=400 + i, roll=0.0, lighting="nominal",
                             distractors=dis, sequence=NOMINAL,
                             cameras=["payload_a"], frames_per_step=32,
                             tag=f"clutter_{dis}"))
        for i, (seq, tag) in enumerate(((SKIP_STEP2, "skip"),
                                        (RECOVER_STEP2, "recover"))):
            plan.append(dict(seed=500 + i, roll=0.0, lighting="nominal",
                             distractors="moderate", sequence=seq, tag=tag,
                             cameras=["payload_a"], frames_per_step=32))
        return plan

    if name == "core":
        plan = []
        # Nominal runs across orientation and camera - the bulk of the training
        # signal.
        for i, roll in enumerate((0.0, 30.0, 60.0, 90.0, 180.0, -30.0, -60.0, -90.0)):
            plan.append(dict(seed=200 + i, roll=roll, lighting="nominal",
                             distractors="moderate", sequence=NOMINAL,
                             cameras=["payload_a", "payload_b"], frames_per_step=45))
        # Lighting stress, upright and inverted.
        for i, (light, roll) in enumerate((("dim", 0.0), ("dim", 90.0),
                                           ("single_fail", 0.0), ("single_fail", 180.0))):
            plan.append(dict(seed=300 + i, roll=roll, lighting=light,
                             distractors="moderate", sequence=NOMINAL,
                             cameras=["payload_a"], frames_per_step=45))
        # Colour-distractor stress for the HSV detector.
        for i, dis in enumerate(("none", "harsh")):
            plan.append(dict(seed=400 + i, roll=0.0, lighting="nominal",
                             distractors=dis, sequence=NOMINAL,
                             cameras=["payload_a"], frames_per_step=45))
        # Protocol error cases for the state machine.
        for i, (seq, tag) in enumerate(((SKIP_STEP2, "skip"),
                                        (RECOVER_STEP2, "recover"))):
            plan.append(dict(seed=500 + i, roll=0.0, lighting="nominal",
                             distractors="moderate", sequence=seq, tag=tag,
                             cameras=["payload_a"], frames_per_step=45))
        # Held-out geometry: a camera the training takes never use.
        for i, roll in enumerate((0.0, 90.0)):
            plan.append(dict(seed=600 + i, roll=roll, lighting="nominal",
                             distractors="moderate", sequence=NOMINAL,
                             cameras=["payload_wide", "payload_over"],
                             frames_per_step=45, tag="heldout_view"))
        return plan

    raise SystemExit(f"unknown plan '{name}'")


def render_take(spec: dict, out_root: Path, resolution, samples: int,
                write_frames: bool = True, gt_only: bool = False) -> dict:
    take_dir = out_root / f"take_{spec['index']:04d}"
    meta_path = take_dir / "meta.json"
    if meta_path.exists():
        print(f"SKIP take_{spec['index']:04d} (already done)", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    t0 = time.time()
    built = build_scene.build(
        seed=spec["seed"],
        body_roll_deg=spec["roll"],
        lighting=spec["lighting"],
        distractors=spec["distractors"],
        sequence=spec["sequence"],
        frames_per_step=spec["frames_per_step"],
        resolution=resolution,
        samples=samples,
    )
    scene = built["scene"]
    meta = built["meta"]
    rig = built["crew"]["rig"]
    payload = built["payload"]

    take_dir.mkdir(parents=True, exist_ok=True)
    per_camera = {}

    for cam_name in spec["cameras"]:
        cam = built["cameras"][cam_name]
        scene.camera = cam
        cam_dir = take_dir / cam_name
        frame_dir = cam_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Ground truth first, in one pass over the timeline. Doing it before
        # rendering means a run interrupted mid-render still leaves usable
        # labels, and it keeps the (cheap) sampling out of the (expensive)
        # render loop's timing.
        gt = GT.collect_take(scene, cam, rig, payload, meta["labels"],
                             scene.frame_start, scene.frame_end,
                             progress_every=0)
        np.save(cam_dir / "pose_2d.npy", gt["pose_2d"])
        np.save(cam_dir / "pose_world.npy", gt["pose_world"])
        np.save(cam_dir / "labels.npy", gt["labels"])
        (cam_dir / "payload.json").write_text(
            json.dumps(gt["payload"]), encoding="utf-8")

        if write_frames and not gt_only:
            # Frame-level resume. Sustained EEVEE work on an integrated GPU
            # can take the OpenGL driver down mid-run (observed here as an
            # EXCEPTION_ACCESS_VIOLATION inside ig9icd64.dll after a few
            # hundred frames). Skipping frames that already exist means the
            # supervising loop in tools/render_dataset_resilient.sh can just
            # relaunch Blender and continue, instead of losing the take.
            done = 0
            for offset, frame in enumerate(range(scene.frame_start, scene.frame_end + 1)):
                target = frame_dir / f"{offset:05d}.png"
                if target.exists() and target.stat().st_size > 0:
                    done += 1
                    continue
                scene.frame_set(frame)
                scene.render.filepath = str((frame_dir / f"{offset:05d}").resolve())
                bpy.ops.render.render(write_still=True)
                done += 1
                if done % 25 == 0:
                    print(f"  {cam_name} {done}/{meta['n_frames']} "
                          f"({time.time() - t0:.0f}s)", flush=True)

        per_camera[cam_name] = {
            "frames": int(meta["n_frames"]),
            "pose_2d": "pose_2d.npy",
            "labels": "labels.npy",
        }

    payload_hsv = M.verify_payload_hsv()
    take_meta = {
        "index": spec["index"],
        "seed": spec["seed"],
        "body_roll_deg": spec["roll"],
        "lighting": spec["lighting"],
        "distractors": spec["distractors"],
        "sequence": spec["sequence"],
        "tag": spec.get("tag", "nominal"),
        "cameras": list(spec["cameras"]),
        "frames_per_step": spec["frames_per_step"],
        "n_frames": int(meta["n_frames"]),
        "fps": meta["fps"],
        "spans": meta["spans"],
        "resolution": list(resolution),
        "samples": samples,
        "view_transform": scene.view_settings.view_transform,
        "payload_hsv_check": payload_hsv,
        "per_camera": per_camera,
        "frames_written": bool(write_frames and not gt_only),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(take_meta, indent=2), encoding="utf-8")
    print(f"TAKE {spec['index']:04d} done in {take_meta['elapsed_sec']}s "
          f"({meta['n_frames']} frames x {len(spec['cameras'])} cam)", flush=True)
    return take_meta


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/blender")
    ap.add_argument("--plan", default="smoke",
                    choices=["smoke", "orientation", "study", "core"])
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Render at most N takes")
    ap.add_argument("--start", type=int, default=0, help="First take index")
    ap.add_argument("--gt-only", action="store_true",
                    help="Write ground truth but skip image rendering")
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.plan)
    for i, spec in enumerate(plan):
        spec["index"] = i

    selected = plan[args.start:]
    if args.limit:
        selected = selected[:args.limit]

    print(f"PLAN {args.plan}: {len(plan)} takes, running {len(selected)}", flush=True)
    index = []
    t0 = time.time()
    for spec in selected:
        index.append(render_take(spec, out_root, (args.width, args.height),
                                 args.samples, gt_only=args.gt_only))

    (out_root / "index.json").write_text(json.dumps({
        "plan": args.plan,
        "resolution": [args.width, args.height],
        "samples": args.samples,
        "takes": index,
        "total_elapsed_sec": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"DATASET_OK {len(index)} takes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
