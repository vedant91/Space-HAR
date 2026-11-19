"""Export sparse, synchronized landmarks from the BAS rig for depth lifting.

Run with Blender's Python or blender --background --python ... -- --out DIR.
A take (animation seed, body attitude and camera) belongs to just one split.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene import build
from bas_har.groundtruth import GroundTruthSampler

BOX_NAMES = ("red_box", "yellow_box", "main_box")


# (roll, yaw, pitch) in degrees. Validation and test use independent animations.
ATTITUDES = {
    "train": [(0, 0, 0), (45, 0, 0), (-90, 0, 0), (180, 0, 0),
              (90, 40, 10), (-45, -45, 0), (0, 65, 0), (0, -65, 0),
              (30, 0, 45), (-30, 0, -45), (145, 30, 30), (-145, -30, -30)],
    "val": [(60, 0, 0), (0, -30, 15), (160, 20, -20)],
    "test": [(75, 0, 0), (-110, 25, 0), (0, 55, 35), (180, -20, -30)],
}


# Extra animation seeds cover mixed attitudes and out-of-camera-plane motion.
# Keep the original seed indices stable for users who already have those takes.
for split, count, rng_seed in (("train", 30, 5921), ("val", 6, 5951), ("audit", 12, 5981)):
    rng = np.random.default_rng(rng_seed)
    attitudes = ATTITUDES.setdefault(split, [])
    for roll in np.linspace(-175, 175, count):
        attitudes.append((float(roll), float(rng.uniform(-70, 70)), float(rng.uniform(-55, 55))))


def export(out, previews=False, only_split=None, start=0):
    width, height = 640, 360
    for split_id, (split, attitudes) in enumerate(ATTITUDES.items()):
        if only_split and split != only_split:
            continue
        for index, (roll, yaw, pitch) in enumerate(attitudes):
            if index < start:
                continue
            seed = 8100 + 100 * split_id + index
            root = out / split / f"take_{seed}"
            root.mkdir(parents=True, exist_ok=True)
            lighting = ("nominal", "dim", "single_fail")[index % 3]
            built = build(seed=seed, body_roll_deg=roll, facing_deg=yaw, body_pitch_deg=pitch,
                          frames_per_step=24, lighting=lighting, resolution=(width, height), samples=4)
            scene = built["scene"]
            camera_name = "payload_a"
            if split == "test" and index == 3:
                camera_name = "payload_wide"
            elif (split == "train" and index >= 12) or (split == "val" and index >= 3) or split == "audit":
                camera_name = ("payload_a", "payload_b", "payload_wide", "payload_over")[index % 4]
            camera = built["cameras"][camera_name]
            scene.camera = camera
            sampler = GroundTruthSampler(scene, camera, built["crew"]["rig"], built["payload"])
            boxes = []
            poses, worlds, steps, metres_per_width = [], [], [], []
            for frame in range(scene.frame_start, scene.frame_end + 1, 3):
                scene.frame_set(frame)
                features, world = sampler.sample_pose(bpy.context.evaluated_depsgraph_get())
                steps.append(built["meta"]["labels"][frame - scene.frame_start])
                poses.append(features.reshape(33, 4))
                worlds.append(world)
                hip = world[[23, 24]].mean(axis=0)
                view = np.array(camera.matrix_world.inverted())
                distance = -(view[2, :3] @ hip + view[2, 3])
                if distance <= 0:
                    raise ValueError(f"Rig behind camera at frame {frame} in {root}")
                metres_per_width.append(distance * camera.data.sensor_width / camera.data.lens)
                if previews and (len(poses) - 1) % 8 == 0:
                    offset = len(poses) - 1
                    payload = sampler.sample_payload(bpy.context.evaluated_depsgraph_get())
                    lines = []
                    annotations = []
                    for cls, name in enumerate(BOX_NAMES):
                        if payload[name]["visible"]:
                            x1, y1, x2, y2 = payload[name]["bbox"]
                            annotations.append(dict(label=name, bbox_xywh=[x1, y1, x2-x1, y2-y1]))
                            lines.append(f"{cls} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} "
                                         f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
                    (root / f"{offset:04d}.txt").write_text("\n".join(lines))
                    boxes.append(dict(image=f"{offset:04d}.png", annotations=annotations))
                    scene.render.filepath = str((root / f"{len(poses)-1:04d}.png").resolve())
                    bpy.ops.render.render(write_still=True)
            if previews:
                (root / "boxes.json").write_text(json.dumps(boxes))
            np.savez_compressed(root / "pose.npz", pose=poses, world=worlds, step=steps,
                                size=[width, height], attitude=[roll, yaw, pitch], seed=seed, camera=camera_name, lighting=lighting,
                                metres_per_width=metres_per_width,
                                sensor_over_focal=camera.data.sensor_width / camera.data.lens)
            print(f"{split} {seed}: {len(poses)} frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--previews", action="store_true", help="Render sparse aligned RGB for detector evaluation")
    parser.add_argument("--split", choices=ATTITUDES)
    parser.add_argument("--start", type=int, default=0, help="First take index within split")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    export(args.out, args.previews, args.split, args.start)
