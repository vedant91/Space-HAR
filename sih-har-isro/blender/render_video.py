"""
Render the animated 8-step payload protocol as a watchable showcase clip -
frames now, muxed to .mp4 by `tools/frames_to_video.py` afterwards.

Two steps because the Blender build shipped on this machine (5.2.1 LTS) was
compiled without the FFMPEG muxer - `image_settings.file_format` has no
`FFMPEG` member - so Blender can only write an image sequence. The project
venv has OpenCV, which does the encode.

The scene, rig and protocol are exactly the ones the dataset uses (same
`build_scene.build()`), so the clip shows what the pipeline actually sees. By
default it renders from the cinematic `showcase` camera through AgX; pass
`--camera payload_a` for the real fixed-payload-camera view.

Run:
    blender --background --factory-startup --python blender/render_video.py -- \
        --out build/showcase/protocol.mp4 --frames-per-step 24 --samples 24
    python tools/frames_to_video.py build/showcase/protocol_frames build/showcase/protocol.mp4 --fps 30

`render_video.py` prints the exact second command with the right fps when it
finishes.

Screen-space raytracing is OFF by default: heaviest EEVEE feature, and on the
integrated-GPU machine this was built on the one that takes the driver down
over a long render (blender/README.md). Pass `--raytracing` on a real GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_scene                                    # noqa: E402
from bas_har import render as R                       # noqa: E402

SEQUENCES = {
    "nominal": list(range(1, 9)),
    "skip": [1, 3, 4, 5, 6, 7, 8],
    "recover": [1, 3, 2, 3, 4, 5, 6, 7, 8],
}


def _selfcheck() -> None:
    """Smallest thing that fails if the wiring below breaks: build a tiny
    animated scene, assert the timeline is non-empty and the frame count the
    encoder will be told matches what Blender will render. `-- --check`."""
    built = build_scene.build(frames_per_step=4, resolution=(64, 36), samples=1)
    scene = built["scene"]
    assert scene.frame_end > scene.frame_start, "empty timeline"
    assert built["meta"]["n_frames"] == scene.frame_end - scene.frame_start + 1
    assert built["meta"]["fps"] > 0
    print("SELFCHECK_OK")


def _parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description="Render the protocol to frames")
    ap.add_argument("--out", default="build/showcase/protocol.mp4",
                    help="Final .mp4 path (frames go in <stem>_frames/ beside it)")
    ap.add_argument("--camera", default="showcase", choices=list(R.CAMERA_PRESETS))
    ap.add_argument("--sequence", default="nominal", choices=list(SEQUENCES))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--lighting", default="nominal",
                    choices=["nominal", "dim", "single_fail"])
    ap.add_argument("--distractors", default="moderate",
                    choices=["none", "moderate", "harsh"])
    ap.add_argument("--frames-per-step", type=int, default=24)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--raytracing", action="store_true",
                    help="Enable screen-space raytracing (needs a real GPU)")
    ap.add_argument("--check", action="store_true",
                    help="Run the self-check and exit without rendering")
    return ap.parse_args(argv)


def main():
    args = _parse_args(sys.argv)

    if args.check:
        _selfcheck()
        return

    built = build_scene.build(
        seed=args.seed,
        body_roll_deg=args.roll,
        lighting=args.lighting,
        distractors=args.distractors,
        sequence=SEQUENCES[args.sequence],
        frames_per_step=args.frames_per_step,
        resolution=(args.width, args.height),
        samples=args.samples,
    )
    scene = built["scene"]
    meta = built["meta"]

    R.setup_showcase_render(scene, resolution=(args.width, args.height),
                            samples=args.samples)
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = args.raytracing

    cam = built["cameras"][args.camera]
    if args.camera != "showcase":
        cam.data.dof.use_dof = False
    scene.camera = cam

    out = Path(args.out).resolve()
    frame_dir = out.parent / (out.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("*.png"):
        stale.unlink()

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frame_dir) + "/"

    n, fps = meta["n_frames"], meta["fps"]
    print(f"VIDEO {n} frames @ {fps} fps ({args.camera}, {args.sequence}) "
          f"-> {frame_dir}", flush=True)
    bpy.ops.render.render(animation=True)

    written = sorted(frame_dir.glob("*.png"))
    if not written:
        raise SystemExit("render produced no frames")
    print(f"FRAMES_OK {len(written)} in {frame_dir}", flush=True)
    print(f"NEXT python tools/frames_to_video.py "
          f"{frame_dir} {out} --fps {fps}", flush=True)


if __name__ == "__main__":
    main()
