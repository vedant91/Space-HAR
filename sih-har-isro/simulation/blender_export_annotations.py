"""Export camera-space labels from the Blender BAS digital twin.

Run this file with Blender after ``blender_build_bas_scene.py``.  It writes
COCO-style per-frame labels for the astronaut and the two payload boxes.  The
labels are *simulation ground truth*; they are useful for training/evaluation,
but must never be displayed as live detector output.
"""
from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "blender_twin"
SCENE = bpy.context.scene
CAMERA = bpy.data.objects["Payload_Camera_Fixed"]
WIDTH, HEIGHT = SCENE.render.resolution_x, SCENE.render.resolution_y


def projected_box(obj):
    """Return an unclipped [x, y, width, height] pixel box or None.

    This intentionally exports geometric ground-truth boxes.  Occlusion and
    visibility are handled later by the training augmentation/evaluation split.
    """
    corners = [obj.matrix_world @ v.co for v in obj.data.vertices]
    points = [world_to_camera_view(SCENE, CAMERA, point) for point in corners]
    visible = [p for p in points if p.z > 0]
    if not visible:
        return None
    xs = [max(0.0, min(1.0, p.x)) for p in visible]
    ys = [max(0.0, min(1.0, p.y)) for p in visible]
    x0, x1 = min(xs) * WIDTH, max(xs) * WIDTH
    # Blender camera coordinates start bottom-left; image labels use top-left.
    y0, y1 = (1.0 - max(ys)) * HEIGHT, (1.0 - min(ys)) * HEIGHT
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


def combined_box(objects):
    boxes = [projected_box(obj) for obj in objects if obj and obj.type == "MESH"]
    boxes = [box for box in boxes if box]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


astronaut_parts = [obj for obj in bpy.data.objects if obj.name == "astronaut"]
tracked = [("astronaut", astronaut_parts), ("red_payload", [bpy.data.objects.get("red_box")]),
           ("yellow_payload", [bpy.data.objects.get("yellow_box")])]
frames = []
for frame in range(SCENE.frame_start, SCENE.frame_end + 1):
    SCENE.frame_set(frame)
    annotations = []
    for label, objects in tracked:
        box = combined_box(objects)
        if box:
            annotations.append({"label": label, "bbox_xywh": box})
    frames.append({
        "frame": frame,
        "image": f"raw_frames/frame_{frame:04d}.png",
        "annotations": annotations,
        "coordinate_frame": "fixed_payload_camera",
    })

payload = {
    "schema": "bas-har-synthetic-ground-truth/v1",
    "label_provenance": "Blender geometric ground truth; not live inference",
    "image_size": [WIDTH, HEIGHT],
    "fps": SCENE.render.fps,
    "categories": ["astronaut", "red_payload", "yellow_payload"],
    "frames": frames,
}
(OUT / "annotations_ground_truth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"frames": len(frames), "output": str(OUT / "annotations_ground_truth.json")}, indent=2))
