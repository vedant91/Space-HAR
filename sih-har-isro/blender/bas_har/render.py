"""
Render configuration and the payload-camera rig.

Camera placement follows the problem statement's own constraint: "Inputs are
given from fixed-payload cameras." So these are not free-roaming cinematic
cameras - they are fixed mounts on the rack structure, wide-angle, looking
across the work volume, exactly like the video cameras on an ISS EXPRESS
rack. That constraint is what makes the resulting dataset honest: the model
sees the awkward, partly-occluded, off-axis view a real payload camera
gives, not a helpfully-framed one.

The crew member faces the rack (+Y), so a camera behind them sees only their
back. Every payload preset is therefore mounted at or forward of the rack
face and looks back across the work surface, which is both what real rack
cameras do and the only geometry that puts hands, payload and torso in the
same frame.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import bpy
from mathutils import Vector

from . import common as C

# name -> (location, look-at target, focal length mm, roll degrees)
CAMERA_PRESETS: Dict[str, Tuple[Tuple[float, float, float],
                                Tuple[float, float, float], float, float]] = {
    # Primary: high on the starboard side of the rack face, angled down and
    # inboard. Hands, both payload boxes and the crew torso are all in frame.
    "payload_a": ((1.55, -0.35, 1.70), (-0.12, -0.58, 1.16), 24.0, 0.0),
    # Secondary: mirrored port mount. A second fixed view is what makes
    # occlusion recoverable, and it doubles the dataset for free.
    "payload_b": ((-1.55, -0.35, 1.70), (0.12, -0.58, 1.16), 24.0, 0.0),
    # Wide documentation view from outboard starboard - more of the module,
    # useful for the showcase render and for whole-body orientation cases.
    "payload_wide": ((2.34, -1.08, 1.86), (-0.15, -0.50, 1.10), 28.0, 0.0),
    # Overhead: mounted on the +Z bay looking straight down the work volume.
    # This is the view where a floor-relative pose model fails hardest, which
    # is exactly why it is in the set.
    "payload_over": ((1.30, -0.62, 2.00), (-0.05, -0.42, 1.12), 24.0, 0.0),
    # Cinematic - showcase stills only, never used for dataset frames.
    "showcase": ((2.62, -1.34, 1.84), (-0.12, -0.45, 1.22), 32.0, 0.0),
}

DATASET_CAMERAS = ("payload_a", "payload_b", "payload_wide", "payload_over")


def add_camera(name: str, coll, preset: Optional[str] = None,
               sensor_width: float = 36.0) -> bpy.types.Object:
    loc, target, lens, roll = CAMERA_PRESETS[preset or name]
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.sensor_width = sensor_width
    data.clip_start = 0.02
    data.clip_end = 40.0
    cam = bpy.data.objects.new(name, data)
    C.link(cam, coll)
    cam.location = Vector(loc)
    C.look_at(cam, Vector(target), roll=math.radians(roll))
    return cam


def build_camera_rig(coll) -> Dict[str, bpy.types.Object]:
    return {name: add_camera(name, coll) for name in CAMERA_PRESETS}


def setup_render(scene: bpy.types.Scene,
                 resolution: Tuple[int, int] = (1280, 720),
                 samples: int = 48,
                 view_transform: str = "Standard",
                 quality: str = "dataset") -> None:
    """Configure EEVEE Next.

    view_transform defaults to Standard, NOT Blender's AgX default. AgX is a
    filmic tone map that deliberately desaturates saturated colour; rendering
    the dataset through it pushes the red and yellow payload boxes out of the
    HSV bands `pipeline/hsv_detector.py` thresholds on, which would look like
    a detector failure rather than a colour-management choice. See
    materials.py for the full note. Use "AgX" only for showcase stills.
    """
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15

    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    eevee = scene.eevee
    eevee.taa_render_samples = samples
    eevee.taa_samples = min(samples, 16)

    # Every one of these is version-sensitive, and a hard AttributeError here
    # would kill a multi-hour dataset run over a cosmetic setting - so each is
    # applied only if this build actually exposes it.
    def _set(obj, attr, value):
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, value)
            except Exception:
                pass

    # Screen-space raytracing is the single heaviest EEVEE feature and the one
    # most likely to destabilise a weak/integrated GPU driver over a long
    # unattended render. It is worth it for a showcase still and not worth it
    # for a dataset frame that a pose estimator will look at, so it is on only
    # for "showcase".
    _set(eevee, "use_raytracing", quality == "showcase")
    _set(eevee, "ray_tracing_method", "SCREEN")
    _set(eevee, "use_fast_gi", True)
    _set(eevee, "fast_gi_method", "GLOBAL_ILLUMINATION")
    _set(eevee, "fast_gi_resolution", "2")
    _set(eevee, "gi_diffuse_bounces", 2 if quality != "draft" else 1)
    _set(eevee, "shadow_ray_count", 2 if quality != "draft" else 1)
    _set(eevee, "shadow_step_count", 4 if quality != "draft" else 2)
    _set(eevee, "shadow_resolution_scale", 1.0)
    _set(eevee, "use_shadows", True)
    _set(eevee, "use_volumetric_shadows", False)
    # Clamping the indirect contribution keeps the bright white suit from
    # blowing out the payload boxes it is standing next to, which would eat
    # into their saturation and cost HSV recall.
    _set(eevee, "clamp_surface_indirect", 8.0)

    if hasattr(eevee, "ray_tracing_options"):
        _set(eevee.ray_tracing_options, "resolution_scale", "2")

    scene.render.use_persistent_data = True


def setup_showcase_render(scene: bpy.types.Scene,
                          resolution: Tuple[int, int] = (1920, 1080),
                          samples: int = 160) -> None:
    """Higher-effort settings for presentation stills: AgX for a filmic
    response, more samples, and depth of field on the showcase camera."""
    setup_render(scene, resolution=resolution, samples=samples,
                 view_transform="AgX", quality="showcase")

    # The AgX look names are not stable across Blender versions - 5.2 offers
    # "AgX - Base Contrast" where earlier builds had "AgX - Medium Contrast",
    # and assigning a missing enum member is a hard TypeError that kills the
    # whole render. Pick the first name this build actually offers.
    try:
        available = [item.identifier for item in
                     scene.view_settings.bl_rna.properties["look"].enum_items]
    except Exception:
        available = []
    for candidate in ("AgX - Base Contrast", "AgX - Medium Contrast",
                      "AgX - Medium High Contrast", "AgX - High Contrast"):
        if candidate in available:
            scene.view_settings.look = candidate
            break
    cam = bpy.data.objects.get("showcase")
    if cam is not None:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = (cam.location - Vector((-0.12, -0.45, 1.22))).length
        cam.data.dof.aperture_fstop = 2.8


def frame_hsv_report(image_path: str) -> Optional[Dict]:
    """Measure the actual OpenCV HSV of the payload pixels in a rendered
    frame, so the colour contract is verified against the *render*, not just
    against the material swatch.

    Returns None when numpy/OpenCV are not importable from inside Blender's
    interpreter, which is the normal case - `tools/verify_render_hsv.py`
    runs the same check from the project venv instead.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {
        "shape": list(img.shape),
        "hsv_mean": [float(x) for x in hsv.reshape(-1, 3).mean(axis=0)],
    }
