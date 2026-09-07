"""
Build the BAS payload-rack scene and (optionally) render preview stills.

Run:
    blender --background --factory-startup --python blender/build_scene.py -- \
        --out build/bas_scene.blend --preview build/preview --seed 3

Everything downstream (render_dataset.py, the showcase renders) calls
`build(...)` from here, so the scene has exactly one definition.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bas_har import astronaut as A          # noqa: E402
from bas_har import common as C             # noqa: E402
from bas_har import materials as M          # noqa: E402
from bas_har import module as MOD           # noqa: E402
from bas_har import protocol as P           # noqa: E402
from bas_har import render as R             # noqa: E402


def build(seed: int = 3,
          body_roll_deg: float = 0.0,
          facing_deg: float = 0.0,
          lighting: str = "nominal",
          distractors: str = "moderate",
          crew_offset=(0.0, -0.50, 1.00),
          sequence=None,
          frames_per_step: int = 72,
          resolution=(1280, 720),
          samples: int = 48,
          animate: bool = True) -> dict:
    """Construct the full scene. Returns the handles other scripts need."""
    C.reset_scene()
    rng = random.Random(seed)

    mats = M.build_library(distractors)
    coll = C.get_collection("bas")

    module_parts = MOD.build_module(mats, coll, rng)
    payload = MOD.build_payload(mats, coll)
    crew = A.build_astronaut(mats, coll, location=crew_offset,
                             body_roll_deg=body_roll_deg,
                             facing_deg=facing_deg, rng=rng)
    lights = MOD.build_lighting(mats, coll, rng, lighting)
    cameras = R.build_camera_rig(coll)

    scene = bpy.context.scene
    R.setup_render(scene, resolution=resolution, samples=samples)
    scene.camera = cameras["payload_a"]

    meta = {"n_frames": 1, "labels": [], "sequence": [], "spans": [], "fps": P.FPS}
    if animate:
        animator = P.ProtocolAnimator(crew, payload, rng, frames_per_step=frames_per_step)
        meta = animator.animate(sequence)

    return {
        "scene": scene,
        "collection": coll,
        "materials": mats,
        "module": module_parts,
        "payload": payload,
        "crew": crew,
        "lights": lights,
        "cameras": cameras,
        "meta": meta,
        "seed": seed,
        "body_roll_deg": body_roll_deg,
        "lighting": lighting,
        "distractors": distractors,
    }


def _parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser(description="Build the BAS HAR scene")
    ap.add_argument("--out", default=None, help="Save the .blend here")
    ap.add_argument("--preview", default=None,
                    help="Directory for one preview still per camera")
    ap.add_argument("--preview-frame", type=int, default=None,
                    help="Frame to preview (default: middle of step 4)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--roll", type=float, default=0.0,
                    help="Crew body roll about the rack normal, degrees")
    ap.add_argument("--lighting", default="nominal",
                    choices=["nominal", "dim", "single_fail"])
    ap.add_argument("--distractors", default="moderate",
                    choices=["none", "moderate", "harsh"])
    ap.add_argument("--frames-per-step", type=int, default=72)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--showcase", action="store_true",
                    help="Use AgX + high samples + DOF for presentation stills")
    return ap.parse_args(argv)


def main():
    args = _parse_args(sys.argv)

    built = build(seed=args.seed,
                  body_roll_deg=args.roll,
                  lighting=args.lighting,
                  distractors=args.distractors,
                  frames_per_step=args.frames_per_step,
                  resolution=(args.width, args.height),
                  samples=args.samples)

    scene = built["scene"]
    meta = built["meta"]

    print("PAYLOAD_HSV " + json.dumps(M.verify_payload_hsv()))
    print(f"FRAMES {meta['n_frames']}")
    print(f"OBJECTS {len(bpy.data.objects)}")
    suit = built["crew"]["body"]
    print(f"SUIT_VERTS {len(suit.data.vertices)}")

    if args.showcase:
        R.setup_showcase_render(scene, resolution=(args.width, args.height),
                                samples=max(args.samples, 128))

    if args.preview:
        # Absolute. Blender resolves a bare relative render path against its
        # own notion of the working directory, which is not reliably the shell
        # cwd in --background; a relative path here silently writes nothing
        # useful while still reporting success.
        out_dir = Path(args.preview).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = args.preview_frame
        if frame is None:
            # Middle of step 4 (examine red box): both hands engaged, the box
            # is clear of the container, and the lid is open - the single
            # frame that shows the most of what the scene can do.
            spans = [s for s in meta["spans"] if s["step_id"] == 4]
            frame = ((spans[0]["start_frame"] + spans[0]["end_frame"]) // 2
                     if spans else scene.frame_start)
        scene.frame_set(frame)
        names = list(R.CAMERA_PRESETS) if args.showcase else list(R.DATASET_CAMERAS)
        for name in names:
            scene.camera = built["cameras"][name]
            scene.render.filepath = str(out_dir / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"PREVIEW {name} -> {scene.render.filepath}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()))
        print(f"SAVED {out}")

    print("BUILD_OK")


if __name__ == "__main__":
    main()
