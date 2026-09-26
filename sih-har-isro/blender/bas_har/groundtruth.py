"""
Per-frame ground truth in exactly the format the HAR pipeline consumes.

What comes out of here is the reason the Blender work is worth doing at all.
For every rendered frame we can state, exactly:

  * where all 33 MediaPipe pose landmarks are, in MediaPipe's own normalised
    image convention (so a 132-dim feature vector is directly comparable
    with what `mp.solutions`/Tasks returns on the same frame);
  * whether each landmark is actually visible or occluded, by ray-casting
    the real geometry rather than asserting it;
  * the true pixel bounding box and visible fraction of the red box, the
    yellow box and the main container.

Those last two are what let the project finally measure the things it
currently only asserts:

  - MediaPipe's own landmark error, by comparing its output on the rendered
    frame against this ground truth. The repo has never measured this - the
    simulation injects ground-truth pose straight into the LSTM
    (`run_pipeline_on_clip(inject_pose=True)`), so the camera-to-pose stage
    is untested.
  - Real HSV detector recall, against true boxes. The current 1.0 is
    tautological: `simulation/renderer.py` paints the boxes in colours
    chosen to sit inside the detector's own thresholds, on top of everything
    else so nothing can occlude them.

MediaPipe landmark conventions reproduced here:
  x, y   normalised to the image, origin at TOP-left, y increasing downward
  z      depth relative to the hip midpoint, in roughly the same units as x,
         negative toward the camera
  visibility  0..1
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 4

# ── Landmark -> rig definition ───────────────────────────────────────────────
# Each entry is (bone, end, local_offset). `end` picks the bone's head or
# tail; local_offset is applied in the bone's own space for the landmarks
# that have no bone of their own (face, fingers, heel).
#
# The .L bones are the ones on -X, which is the crew member's own LEFT
# (see astronaut.py's frame note) - so they map to MediaPipe's *left*
# indices. Getting this backwards silently mirrors every exported label,
# and would not show up as an error anywhere downstream.

_HEAD_R = 0.135

LANDMARKS: List[Tuple[str, str, Tuple[float, float, float]]] = [
    ("head", "mid", (0.000, _HEAD_R, 0.010)),           # 0  nose
    ("head", "mid", (-0.028, _HEAD_R * 0.88, 0.050)),   # 1  left eye inner
    ("head", "mid", (-0.045, _HEAD_R * 0.86, 0.052)),   # 2  left eye
    ("head", "mid", (-0.062, _HEAD_R * 0.82, 0.050)),   # 3  left eye outer
    ("head", "mid", (0.028, _HEAD_R * 0.88, 0.050)),    # 4  right eye inner
    ("head", "mid", (0.045, _HEAD_R * 0.86, 0.052)),    # 5  right eye
    ("head", "mid", (0.062, _HEAD_R * 0.82, 0.050)),    # 6  right eye outer
    ("head", "mid", (-0.115, 0.010, 0.030)),            # 7  left ear
    ("head", "mid", (0.115, 0.010, 0.030)),             # 8  right ear
    ("head", "mid", (-0.030, _HEAD_R * 0.90, -0.048)),  # 9  mouth left
    ("head", "mid", (0.030, _HEAD_R * 0.90, -0.048)),   # 10 mouth right

    ("upperarm.L", "head", (0.0, 0.0, 0.0)),            # 11 left shoulder
    ("upperarm.R", "head", (0.0, 0.0, 0.0)),            # 12 right shoulder
    ("forearm.L", "head", (0.0, 0.0, 0.0)),             # 13 left elbow
    ("forearm.R", "head", (0.0, 0.0, 0.0)),             # 14 right elbow
    ("hand.L", "head", (0.0, 0.0, 0.0)),                # 15 left wrist
    ("hand.R", "head", (0.0, 0.0, 0.0)),                # 16 right wrist
    ("hand.L", "tail", (-0.030, 0.012, 0.0)),           # 17 left pinky
    ("hand.R", "tail", (0.030, 0.012, 0.0)),            # 18 right pinky
    ("hand.L", "tail", (0.018, 0.022, 0.0)),            # 19 left index
    ("hand.R", "tail", (-0.018, 0.022, 0.0)),           # 20 right index
    ("hand.L", "mid", (-0.030, 0.026, 0.0)),            # 21 left thumb
    ("hand.R", "mid", (0.030, 0.026, 0.0)),             # 22 right thumb

    ("thigh.L", "head", (0.0, 0.0, 0.0)),               # 23 left hip
    ("thigh.R", "head", (0.0, 0.0, 0.0)),               # 24 right hip
    ("shin.L", "head", (0.0, 0.0, 0.0)),                # 25 left knee
    ("shin.R", "head", (0.0, 0.0, 0.0)),                # 26 right knee
    ("foot.L", "head", (0.0, 0.0, 0.0)),                # 27 left ankle
    ("foot.R", "head", (0.0, 0.0, 0.0)),                # 28 right ankle
    ("foot.L", "head", (0.0, -0.045, 0.0)),             # 29 left heel
    ("foot.R", "head", (0.0, -0.045, 0.0)),             # 30 right heel
    ("foot.L", "tail", (0.0, 0.030, 0.0)),              # 31 left foot index
    ("foot.R", "tail", (0.0, 0.030, 0.0)),              # 32 right foot index
]

LEFT_HIP, RIGHT_HIP = 23, 24

# Visibility levels. MediaPipe's `visibility` is a model confidence, not a
# hard occlusion flag, so an occluded-but-inferable landmark keeps a
# non-trivial value rather than dropping to zero - that is what the real
# detector reports, and training against a hard 0/1 would teach the sequence
# model a signal MediaPipe never produces.
VIS_CLEAR = 0.97
VIS_OCCLUDED = 0.32
VIS_OFFSCREEN = 0.04


def _bone_point(rig: bpy.types.Object, bone_name: str, end: str,
                offset: Sequence[float]) -> Vector:
    """World position of a point defined in a pose bone's local space."""
    bone = rig.pose.bones.get(bone_name)
    if bone is None:
        return rig.matrix_world.translation.copy()

    if end == "head":
        base = bone.head
    elif end == "tail":
        base = bone.tail
    else:
        base = (bone.head + bone.tail) * 0.5

    if any(offset):
        # Pose-bone matrix columns are the bone's local axes in armature
        # space; y runs head->tail.
        m = bone.matrix
        local = (m.col[0].xyz * offset[0]
                 + m.col[1].xyz * offset[2]
                 + m.col[2].xyz * offset[1])
        base = base + local
    return rig.matrix_world @ base


class GroundTruthSampler:
    """Samples landmark + payload ground truth for the current frame."""

    def __init__(self, scene: bpy.types.Scene, camera: bpy.types.Object,
                 rig: bpy.types.Object, payload: Dict):
        self.scene = scene
        self.camera = camera
        self.rig = rig
        self.payload = payload
        self.width = scene.render.resolution_x
        self.height = scene.render.resolution_y

    # ── projection ───────────────────────────────────────────────────────

    def _project(self, world: Vector) -> Tuple[float, float, float]:
        """World point -> (x, y, depth) with MediaPipe's top-left origin."""
        co = world_to_camera_view(self.scene, self.camera, world)
        return float(co.x), float(1.0 - co.y), float(co.z)

    def _occluded(self, depsgraph, world: Vector,
                  ignore: Optional[set] = None,
                  self_surface_tolerance: float = 0.16) -> bool:
        """Ray-cast camera -> point; True if the landmark is genuinely hidden.

        The subtlety is that a pose landmark is a *joint centre*, which lies
        inside the body. On a pressure suit the shell sits 5-9 cm out from the
        bone, so a naive "did the ray hit anything before reaching the point"
        test reports essentially every landmark as occluded - the suit
        occludes its own skeleton.

        MediaPipe's notion of visibility is whether the body part is visible
        in the image, not whether a line of sight reaches the joint centre. So
        the rule here is: find the first hit; if it is within
        `self_surface_tolerance` of the landmark, we are looking at the
        surface of that very limb and the landmark counts as visible. A hit
        substantially in front of the landmark is a real occluder (the rack,
        the container, the torso in front of an arm).
        """
        origin = self.camera.matrix_world.translation
        delta = world - origin
        distance = delta.length
        if distance < 1e-6:
            return False
        direction = delta / distance

        hit, location, _nrm, _idx, obj, _mat = self.scene.ray_cast(
            depsgraph, origin + direction * 0.01, direction,
            distance=distance)
        if not hit:
            return False
        if ignore and obj is not None and obj.name in ignore:
            return False
        return (location - world).length > self_surface_tolerance

    # ── pose ─────────────────────────────────────────────────────────────

    def sample_pose(self, depsgraph) -> Tuple[np.ndarray, np.ndarray]:
        """Return (features_132, world_xyz_33x3) for the current frame."""
        world_pts: List[Vector] = [
            _bone_point(self.rig, bone, end, off) for bone, end, off in LANDMARKS
        ]

        projected = [self._project(p) for p in world_pts]
        hip_depth = 0.5 * (projected[LEFT_HIP][2] + projected[RIGHT_HIP][2])

        # Convert metric depth difference into MediaPipe's x-like units: the
        # world width the frame spans at the hip plane.
        sensor_fit = self.camera.data.sensor_width
        focal = self.camera.data.lens
        frame_world_width = max(hip_depth * sensor_fit / max(focal, 1e-6), 1e-6)

        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        for i, ((x, y, depth), world) in enumerate(zip(projected, world_pts)):
            feats[i, 0] = x
            feats[i, 1] = y
            feats[i, 2] = (depth - hip_depth) / frame_world_width

            on_screen = (-0.05 <= x <= 1.05) and (-0.05 <= y <= 1.05) and depth > 0.0
            if not on_screen:
                feats[i, 3] = VIS_OFFSCREEN
            elif self._occluded(depsgraph, world):
                feats[i, 3] = VIS_OCCLUDED
            else:
                feats[i, 3] = VIS_CLEAR

        world_arr = np.array([[p.x, p.y, p.z] for p in world_pts], dtype=np.float32)
        return feats.reshape(-1), world_arr

    # ── payload ──────────────────────────────────────────────────────────

    def sample_payload(self, depsgraph) -> Dict:
        """True pixel bbox + visible fraction for each detector target."""
        out: Dict[str, Dict] = {}
        for key, obj in (("red_box", self.payload["red"]),
                         ("yellow_box", self.payload["yellow"]),
                         ("main_box", self.payload["container"])):
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            centre = sum(corners, Vector((0.0, 0.0, 0.0))) / 8.0
            samples = corners + [centre]

            projected = [self._project(p) for p in samples]
            xs = [p[0] for p in projected]
            ys = [p[1] for p in projected]
            in_front = all(p[2] > 0.0 for p in projected)

            # A corner sits exactly on the surface, so it self-occludes on
            # roughly half of any convex object no matter what. Nudging each
            # sample toward the camera by 12 mm tests the volume, not the skin.
            cam_pos = self.camera.matrix_world.translation
            visible = 0
            for p in samples:
                d = (cam_pos - p)
                if d.length < 1e-6:
                    continue
                probe = p + d.normalized() * 0.012
                if not self._occluded(depsgraph, probe, ignore={obj.name}):
                    visible += 1
            frac = visible / float(len(samples))

            x1 = max(0.0, min(xs)) * self.width
            x2 = min(1.0, max(xs)) * self.width
            y1 = max(0.0, min(ys)) * self.height
            y2 = min(1.0, max(ys)) * self.height
            on_screen = in_front and x2 > x1 and y2 > y1

            out[key] = {
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "visible_fraction": round(frac, 3),
                # The detector is expected to find it only when a real,
                # unoccluded patch of colour is on screen. 0.15 is the
                # smallest fraction that still yields a contour above
                # MIN_BOX_AREA_PX at 1280x720.
                "visible": bool(on_screen and frac >= 0.15),
                "on_screen": bool(on_screen),
            }
        return out


def collect_take(scene: bpy.types.Scene, camera: bpy.types.Object,
                 rig: bpy.types.Object, payload: Dict,
                 labels: Sequence[int],
                 frame_start: int, frame_end: int,
                 progress_every: int = 0) -> Dict:
    """Step the timeline and sample ground truth for every frame.

    Returns arrays aligned frame-for-frame with the rendered images, so a
    caller can render and sample in the same pass without re-deriving which
    frame is which.
    """
    sampler = GroundTruthSampler(scene, camera, rig, payload)

    poses: List[np.ndarray] = []
    worlds: List[np.ndarray] = []
    payloads: List[Dict] = []

    for idx, frame in enumerate(range(frame_start, frame_end + 1)):
        scene.frame_set(frame)
        # Re-fetch the depsgraph after frame_set: the cached one still holds
        # the previous frame's evaluated transforms, which would offset every
        # ray-cast occlusion test by one frame.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        feats, world = sampler.sample_pose(depsgraph)
        poses.append(feats)
        worlds.append(world)
        payloads.append(sampler.sample_payload(depsgraph))
        if progress_every and idx % progress_every == 0:
            print(f"  ground truth {idx + 1}/{frame_end - frame_start + 1}", flush=True)

    return {
        "pose_2d": np.stack(poses, axis=0),
        "pose_world": np.stack(worlds, axis=0),
        "payload": payloads,
        "labels": np.array(list(labels[:len(poses)]), dtype=np.int64),
    }
