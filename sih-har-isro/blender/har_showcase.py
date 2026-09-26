"""
BAS HAR 3D showcase for Blender 5.2 LTS (EEVEE). Single self-contained script.

Builds a Columbus-style Bharatiya Antariksh Station module, an ISPR payload rack
with the red/yellow sort experiment, a rigged pressure-suited crew member, a
MediaPipe-33 HAR overlay and a holographic HUD, animates the 8-step protocol
and renders a PNG sequence.

    blender --background --factory-startup --python har_showcase.py -- --quick --camera hero

Options after "--":
    --out DIR          output directory (default build/har3d)
    --width/--height   resolution (default 1280x720, --quick 960x540)
    --samples N        EEVEE samples (default 32, --quick 8)
    --fps N            frame rate (default 30)
    --quick            960x540, 8 samples, 24 frames per step
    --build-only       build and save the .blend, skip rendering
    --camera NAME      hero | payload_a | payload_b | helmet | tunnel | all
    --no-ik            pure forward-kinematic arms (hands will not meet the boxes)

Blender 5.2 has no EEVEE bloom setting, so bloom is a compositor Glare node.
The HAR skeleton renders in its own scene and is composited over the beauty
pass, the way a detector draws landmarks over the camera image; in the same
pass the suit would hide every joint centre.
"""

import argparse
import math
import random
import sys
import warnings
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

TAG = "[HAR3D]"
STEPS = 8
STEP_NAMES = [
    "APPROACH MAIN BOX", "OPEN MAIN BOX", "PICK RED BOX", "EXAMINE RED BOX",
    "PLACE RED BOX", "PICK YELLOW BOX", "EXAMINE YELLOW BOX", "PLACE YELLOW BOX",
]

# Scene frame: rack front face at y=0, crew works at y<0, module axis along Y.
TARGET = (0.0, -0.18, 0.70)
ZC = 0.95                       # module axis height
SHELL_R = 2.1
MODULE_Y = (-5.2, 1.2)          # 6.4 m
AISLE = 1.10                    # wall-rack faces sit this far from the axis
FLOOR_Z = ZC - AISLE
DOME_R = 1.78
WINDOW_ANGLE = math.radians(20.0)
WINDOW_Y = -0.55
WINDOW_R = 0.21

RACK_W, RACK_H, RACK_D = 1.046, 2.013, 0.858
SHELF_TOP = 0.65
BAY_TOP = 1.12
BAY_DEPTH = 0.08                # recess behind the work surface so the lid clears the rack face
CONTAINER = (0.28, 0.22, 0.20)
CONTAINER_CENTER = (0.0, -0.18, SHELF_TOP + CONTAINER[2] / 2)
LID_HINGE = (0.0, CONTAINER_CENTER[1] + CONTAINER[1] / 2, SHELF_TOP + CONTAINER[2])
INNER_BOX = (0.11, 0.10, 0.085)
ZONE_PAD_TOP = 0.74 - INNER_BOX[2] / 2

ASTRO_START = (0.0, -0.85, 0.70)
ASTRO_WORK = (0.0, -0.72, 0.70)

MP_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]

# name -> (location, lens mm, f-stop). All aim at TARGET unless CAMERA_AIM overrides.
# Three spec positions were unusable once rendered, so they are moved (lens/f-stop kept):
#   hero      (-1.00,-1.85,1.10): at hip height and 1.3 m the 50 mm frame held only the crew member's back.
#   payload_a (0,-0.60,0.20) sat between the crew member's shins, facing the shelf underside;
#             it is now a floor-level worm's-eye from the front-left of the bay.
#   helmet    (-0.25,-1.10,1.45) sat behind the left shoulder; it now faces the visor and tracks the head.
CAMERAS = {
    "hero":      ((-1.00, -2.35, 1.75), 50, 2.4),
    "payload_a": ((-0.95, -0.25, 0.05), 35, 2.8),
    "payload_b": ((0.00, -0.18, 2.32), 28, 4.0),
    "helmet":    ((-0.85, -0.10, 1.50), 85, 1.8),
    "tunnel":    ((0.00, -3.50, 1.00), 24, 5.6),
}
HELMET = "helmet_centre"
CAMERA_AIM = {"hero": (0.22, -0.35, 1.20), "payload_a": (0.0, -0.70, 1.05), "helmet": HELMET}
EXPOSURE = -0.6

# name, type, watts, colour, size or cone degrees, location, rotation degrees, aim override
LIGHTS = [
    ("warm_key", "AREA", 280, (1.00, 0.82, 0.60), 0.9, (-0.8, -1.8, 1.7), (35, -25, 0), None),
    ("cool_fill", "AREA", 80, (0.72, 0.84, 1.00), 1.2, (1.4, -0.5, 1.0), (0, 50, 0), None),
    ("rim", "SPOT", 220, (0.85, 0.95, 1.00), 38, (0.0, 0.0, 2.2), (-40, 0, 0), None),
    # The listed -65 deg X would point the task spot away from the bench; it is aimed at it instead.
    ("task", "SPOT", 160, (1.00, 0.97, 0.88), 28, (0.0, -0.5, 2.1), (-65, 0, 0), TARGET),
    ("sub_fill", "AREA", 35, (0.60, 0.65, 0.72), 1.8, (0.0, -2.0, 0.4), (70, 0, 0), None),
]

BLOOM_THRESHOLD = 0.8
BLOOM_INTENSITY = 0.15

NEUTRAL = {
    "pelvis": (-8, 0, 0), "spine": (5, 0, 0), "chest": (8, 0, 0),
    "neck": (0, 0, 0), "head": (0, 0, 0),
    "upperarm.L": (-15, 0, 20), "upperarm.R": (-15, 0, -20),
    "forearm.L": (12, 0, 0), "forearm.R": (12, 0, 0),
    "hand.L": (0, 0, 0), "hand.R": (0, 0, 0),
    "thigh.L": (42, 0, 8), "thigh.R": (42, 0, -8),
    "shin.L": (-38, 0, 0), "shin.R": (-38, 0, 0),
    "foot.L": (0, 0, 0), "foot.R": (0, 0, 0),
}
ARMS_OUT = {"upperarm.L": (-20, 0, 20), "upperarm.R": (-20, 0, -20),
            "forearm.L": (30, 0, 0), "forearm.R": (30, 0, 0)}
CUPPED = {"upperarm.L": (-20, 0, 20), "upperarm.R": (-20, 0, -20),
          "forearm.L": (40, 0, 0), "forearm.R": (40, 0, 0), "head": (-12, 0, 0)}
R_RELAXED = {"upperarm.R": (-15, 0, -20), "forearm.R": (12, 0, 0)}
L_RELAXED = {"upperarm.L": (-15, 0, 20), "forearm.L": (12, 0, 0)}

# (step, fraction of step, start from NEUTRAL?, overrides). Unlisted bones hold.
POSE_TRACK = [
    (1, 0.00, True, {}),
    (1, 1.00, True, ARMS_OUT),
    (2, 0.35, False, {"upperarm.R": (-28, 0, -20), "forearm.R": (45, 0, 0)}),
    (2, 1.00, False, {"upperarm.R": (-15, 0, -20), "forearm.R": (20, 0, 0)}),
    (3, 0.35, False, {"upperarm.L": (-20, 0, 10), "forearm.L": (55, 0, 0), "head": (-6, 0, 0), **R_RELAXED}),
    (3, 1.00, False, {"forearm.L": (30, 0, 0)}),
    (4, 0.50, False, CUPPED),
    (4, 1.00, False, {}),
    (5, 0.50, False, {"upperarm.L": (-25, 0, 15), "forearm.L": (45, 0, 0), "head": (-6, 0, 0), **R_RELAXED}),
    (5, 1.00, True, {}),
    (6, 0.35, True, {"upperarm.R": (-20, 0, -10), "forearm.R": (55, 0, 0), "head": (-6, 0, 0)}),
    (6, 1.00, False, {"forearm.R": (30, 0, 0)}),
    (7, 0.50, False, CUPPED),
    (7, 1.00, False, {}),
    (8, 0.50, False, {"upperarm.R": (-25, 0, -15), "forearm.R": (45, 0, 0), "head": (-6, 0, 0), **L_RELAXED}),
    (8, 1.00, True, {}),
]

# (step, fraction, location, rotation degrees). Extra waypoints lift boxes clear of the container walls.
RED_TRACK = [
    (1, 0.00, (-0.07, -0.18, 0.70), (0, 0, 0)),
    (3, 0.35, (-0.07, -0.18, 0.70), (0, 0, 0)),
    (3, 0.55, (-0.07, -0.18, 0.915), (3, 0, 0)),
    (3, 0.78, (-0.07, -0.36, 0.905), (6, 0, 0)),
    (3, 1.00, (-0.07, -0.40, 0.82), (8, 0, 0)),
    (4, 1.00, (-0.02, -0.50, 0.90), (8, 45, 0)),
    (5, 0.50, (-0.19, -0.30, 0.93), (4, 22.5, 0)),
    (5, 0.85, (-0.36, -0.10, 0.74), (0, 0, 0)),
]
YELLOW_TRACK = [
    (1, 0.00, (0.07, -0.18, 0.70), (0, 0, 0)),
    (6, 0.35, (0.07, -0.18, 0.70), (0, 0, 0)),
    (6, 0.55, (0.07, -0.18, 0.915), (0, 0, 0)),
    (6, 0.78, (0.07, -0.36, 0.905), (0, 0, 0)),
    (6, 1.00, (0.07, -0.40, 0.82), (0, 0, 0)),
    (7, 1.00, (0.04, -0.48, 0.88), (0, -30, 0)),
    (8, 0.50, (0.20, -0.29, 0.93), (0, -15, 0)),
    (8, 0.85, (0.36, -0.10, 0.74), (0, 0, 0)),
]
LID_TRACK = [(1, 0.00, 0.0), (2, 0.20, 0.0), (2, 0.90, -115.0)]

# Fingertip contact points in the held object's local frame.
GRIP = (0.0, -0.02, 0.055)
SIDE_L = (-0.068, -0.01, 0.0)
SIDE_R = (0.068, -0.01, 0.0)
LID_GRIP = (0.03, -0.235, 0.02)
# Influence ramps (0 -> 1 -> 1 -> 0) and offset keys, per hand.
IK_PLAN = {
    "R": [
        ("main_box_lid", ((2, 0.08), (2, 0.20), (2, 0.62), (2, 0.76)), [((2, 0.0), LID_GRIP)]),
        ("red_box", ((4, 0.00), (4, 0.15), (5, 0.00), (5, 0.14)), [((4, 0.0), SIDE_R)]),
        ("yellow_box", ((6, 0.20), (6, 0.35), (8, 0.84), (8, 0.98)),
         [((6, 0.0), GRIP), ((7, 0.0), GRIP), ((7, 0.15), SIDE_R), ((8, 0.0), SIDE_R), ((8, 0.15), GRIP)]),
    ],
    "L": [
        ("red_box", ((3, 0.20), (3, 0.35), (5, 0.84), (5, 0.98)),
         [((3, 0.0), GRIP), ((4, 0.0), GRIP), ((4, 0.15), SIDE_L), ((5, 0.0), SIDE_L), ((5, 0.15), GRIP)]),
        ("yellow_box", ((7, 0.00), (7, 0.15), (8, 0.00), (8, 0.14)), [((7, 0.0), SIDE_L)]),
    ],
}


def log(msg):
    print(f"{TAG} {msg}", flush=True)


def rgba(c, a=1.0):
    return tuple(c) if len(c) == 4 else (c[0], c[1], c[2], a)


def deg3(v):
    return tuple(math.radians(x) for x in v)


def frange(a, b, step):
    n = max(1, int(round((b - a) / step)))
    return [a + (b - a) * i / n for i in range(n + 1)]


def align_z(vec):
    v = Vector(vec).normalized()
    return v.to_track_quat("Z", "Y" if abs(v.y) < 0.99 else "X").to_matrix().to_4x4()


def rot4(rot):
    if rot is None:
        return Matrix.Identity(4)
    if isinstance(rot, Matrix):
        return rot.to_4x4()
    return Euler(rot, "XYZ").to_matrix().to_4x4()


def frame_matrix(origin, x, y, z):
    return Matrix((
        (x[0], y[0], z[0], origin[0]),
        (x[1], y[1], z[1], origin[1]),
        (x[2], y[2], z[2], origin[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))


def look_rotation(loc, target):
    d = Vector(target) - Vector(loc)
    if abs(d.normalized().z) > 0.999:
        return Euler((0.0, 0.0, 0.0)) if d.z < 0 else Euler((math.pi, 0.0, 0.0))
    return d.to_track_quat("-Z", "Y").to_euler()


def iter_fcurves(obj):
    anim = obj.animation_data
    if anim is None or anim.action is None:
        return
    action = anim.action
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in action.layers:
        for strip in layer.strips:
            bags = getattr(strip, "channelbags", None)
            if bags is None:
                bag = getattr(strip, "channelbag", None)
                bags = [bag] if bag else []
            for bag in bags:
                yield from bag.fcurves


class Timeline:
    def __init__(self, frames_per_step):
        self.fps_step = frames_per_step
        self.last = STEPS * frames_per_step

    def t(self, step, frac):
        return min(self.last, 1 + int(round((step - 1 + frac) * self.fps_step)))


# ── Materials ─────────────────────────────────────────────────────────────────

def _tree(mat):
    if mat.node_tree is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat.use_nodes = True
    return mat.node_tree


def set_blend(mat):
    for attr, value in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
        try:
            setattr(mat, attr, value)
        except (AttributeError, TypeError):
            pass


def mat_principled(name, color, metallic=0.0, roughness=0.5, inputs=None):
    mat = bpy.data.materials.new(name)
    bsdf = _tree(mat).nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = rgba(color)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    for key, value in (inputs or {}).items():
        bsdf.inputs[key].default_value = value
    mat.diffuse_color = rgba(color)
    return mat


def mat_emission(name, color, strength, alpha=1.0):
    mat = bpy.data.materials.new(name)
    nt = _tree(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.name = "Emission"
    emit.inputs["Color"].default_value = rgba(color)
    emit.inputs["Strength"].default_value = strength
    if alpha < 1.0:
        transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Factor"].default_value = alpha
        nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
        nt.links.new(emit.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    set_blend(mat)
    mat.diffuse_color = rgba(color)
    return mat


def add_noise_roughness(mat, scale, lo, hi, stretch=(1.0, 1.0, 1.0)):
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coords = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = stretch
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 6.0
    remap = nt.nodes.new("ShaderNodeMapRange")
    remap.inputs["To Min"].default_value = lo
    remap.inputs["To Max"].default_value = hi
    nt.links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], remap.inputs["Value"])
    nt.links.new(remap.outputs["Result"], bsdf.inputs["Roughness"])


def add_fabric_bump(mat, scale=320.0, strength=0.12):
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 2.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def mat_earth(name):
    mat = bpy.data.materials.new(name)
    nt = _tree(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.4
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.05, 0.18, 0.55, 1.0)
    ramp.color_ramp.elements[1].position = 0.70
    ramp.color_ramp.elements[1].color = (0.92, 0.95, 1.0, 1.0)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_materials():
    m = {}
    m["shell"] = mat_principled("module_interior", (0.85, 0.86, 0.88), 0.0, 0.55)
    add_noise_roughness(m["shell"], 8.0, 0.45, 0.65)
    m["rib"] = mat_principled("module_rib", (0.72, 0.73, 0.75), 0.6, 0.35)
    m["floor"] = mat_principled("floor_panel", (0.58, 0.60, 0.63), 0.2, 0.6)
    m["seat_track"] = mat_principled("seat_track", (0.55, 0.56, 0.58), 0.9, 0.3)
    m["tray"] = mat_principled("cable_tray", (0.50, 0.52, 0.55), 0.8, 0.4)
    m["cable_black"] = mat_principled("cable_black", (0.03, 0.03, 0.035), 0.0, 0.5)
    m["cable_orange"] = mat_principled("cable_orange", (0.90, 0.35, 0.05), 0.0, 0.45)
    m["cable_blue"] = mat_principled("cable_blue", (0.05, 0.20, 0.65), 0.0, 0.45)
    m["rack_body"] = mat_principled("wall_rack_body", (0.70, 0.71, 0.72), 0.3, 0.5)
    m["panel_a"] = mat_principled("wall_panel_light", (0.80, 0.80, 0.76), 0.0, 0.5)
    m["panel_b"] = mat_principled("wall_panel_blue", (0.28, 0.36, 0.46), 0.1, 0.45)
    m["handle"] = mat_principled("handle_alu", (0.75, 0.76, 0.78), 0.9, 0.25)
    m["led_green"] = mat_emission("status_led_green", (0.2, 1.0, 0.3), 6.0)
    m["led_amber"] = mat_emission("status_led_amber", (1.0, 0.55, 0.05), 5.0)
    m["lamp"] = mat_emission("luminaire", (1.0, 0.96, 0.88), 2.5)
    m["glass"] = mat_principled("viewport_glass", (0.35, 0.55, 0.85), 0.0, 0.05,
                                {"Transmission Weight": 0.85, "IOR": 1.45, "Alpha": 0.55})
    set_blend(m["glass"])
    m["bezel"] = mat_principled("window_bezel", (0.20, 0.21, 0.23), 0.7, 0.35)
    m["earth"] = mat_earth("earth_glow")

    m["alu"] = mat_principled("brushed_aluminium", (0.62, 0.63, 0.65), 0.9, 0.2, {"Anisotropic": 0.5})
    add_noise_roughness(m["alu"], 1.0, 0.16, 0.26, stretch=(220.0, 220.0, 3.0))
    m["alu_panel"] = mat_principled("rack_front_panel", (0.70, 0.71, 0.73), 0.85, 0.3)
    m["work_surface"] = mat_principled("work_surface", (0.30, 0.32, 0.35), 0.1, 0.55)
    m["screen"] = mat_emission("rack_display", (0.10, 0.60, 0.70), 1.6)
    m["screen_bezel"] = mat_principled("display_bezel", (0.03, 0.03, 0.035), 0.2, 0.15)
    m["knob"] = mat_principled("knob", (0.08, 0.08, 0.09), 0.3, 0.35)
    m["label"] = mat_principled("label_ink", (0.05, 0.06, 0.08), 0.0, 0.6)
    m["zone_pad"] = mat_principled("zone_pad", (0.16, 0.17, 0.19), 0.0, 0.8)
    m["zone_red"] = mat_principled("zone_rim_red", (0.85, 0.07, 0.05), 0.0, 0.4)
    m["zone_yellow"] = mat_principled("zone_rim_yellow", (0.95, 0.70, 0.02), 0.0, 0.4)
    m["main_box"] = mat_principled("main_container", (0.92, 0.92, 0.90), 0.0, 0.35)
    m["red_box"] = mat_principled("red_box", (0.85, 0.07, 0.05), 0.0, 0.4)
    m["yellow_box"] = mat_principled("yellow_box", (0.95, 0.70, 0.02), 0.0, 0.4)

    m["suit"] = mat_principled("suit_fabric", (0.88, 0.88, 0.88), 0.0, 0.75, {"Sheen Weight": 0.3})
    add_fabric_bump(m["suit"])
    m["hut"] = mat_principled("suit_hard_torso", (0.88, 0.88, 0.88), 0.0, 0.45)
    m["ring"] = mat_principled("suit_convolute", (0.82, 0.82, 0.80), 0.3, 0.4)
    m["helmet"] = mat_principled("helmet_shell", (0.90, 0.90, 0.90), 0.0, 0.25)
    m["visor"] = mat_principled("visor", (0.02, 0.02, 0.03), 0.0, 0.02,
                                {"Transmission Weight": 0.6, "IOR": 1.5})
    m["dark"] = mat_principled("comm_cap", (0.05, 0.05, 0.06), 0.0, 0.6)
    m["chest_box"] = mat_principled("chest_module", (0.35, 0.37, 0.40), 0.4, 0.4)
    m["sole"] = mat_principled("boot_sole", (0.15, 0.15, 0.16), 0.0, 0.7)
    m["saffron"] = mat_principled("flag_saffron", (1.0, 0.45, 0.10), 0.0, 0.6)
    m["flag_white"] = mat_principled("flag_white", (0.95, 0.95, 0.95), 0.0, 0.6)
    m["flag_green"] = mat_principled("flag_green", (0.05, 0.45, 0.05), 0.0, 0.6)

    m["har_joint"] = mat_emission("har_landmark", (0.0, 0.9, 1.0), 3.5, alpha=0.9)
    m["har_limb"] = mat_emission("har_connector", (0.0, 0.9, 1.0), 2.0, alpha=0.75)
    m["hud_bg"] = mat_emission("hud_background", (0.0, 0.35, 0.45), 0.9, alpha=0.65)
    m["hud_edge"] = mat_emission("hud_border", (0.0, 0.9, 1.0), 2.0)
    m["hud_led"] = mat_emission("gravity_led", (0.2, 1.0, 0.3), 5.0)
    m["hud_text"] = mat_emission("hud_text", (0.75, 0.97, 1.0), 3.0)
    m["hud_title"] = mat_emission("hud_title", (0.0, 0.9, 1.0), 3.0)
    m["hud_ok"] = mat_emission("hud_ok", (0.2, 1.0, 0.5), 3.0)
    return m


# ── Geometry ──────────────────────────────────────────────────────────────────

class MeshBuilder:
    """Accumulates primitives into one bmesh; faces carry a material slot index."""

    def __init__(self):
        self.bm = bmesh.new()

    def _finish(self, verts, matrix, mat, smooth):
        bmesh.ops.transform(self.bm, matrix=matrix, verts=verts)
        faces = list({f for v in verts for f in v.link_faces})
        for f in faces:
            f.material_index = mat
            f.smooth = smooth
        return faces

    def box(self, center, size, rot=None, mat=0, space=None):
        verts = bmesh.ops.create_cube(self.bm, size=1.0)["verts"]
        m = (space or Matrix.Identity(4)) @ Matrix.Translation(center) @ rot4(rot) \
            @ Matrix.Diagonal((size[0], size[1], size[2], 1.0))
        return self._finish(verts, m, mat, False)

    def ellipsoid(self, center, radii, rot=None, mat=0, seg=32, rings=16, space=None):
        verts = bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=rings, radius=1.0)["verts"]
        m = (space or Matrix.Identity(4)) @ Matrix.Translation(center) @ rot4(rot) \
            @ Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
        return self._finish(verts, m, mat, True)

    def cylinder(self, p0, p1, radius, mat=0, seg=24, space=None):
        p0, p1 = Vector(p0), Vector(p1)
        axis = p1 - p0
        verts = bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg,
                                      radius1=radius, radius2=radius, depth=axis.length)["verts"]
        m = (space or Matrix.Identity(4)) @ Matrix.Translation((p0 + p1) / 2) @ align_z(axis)
        faces = self._finish(verts, m, mat, True)
        n = (m.to_3x3() @ Vector((0, 0, 1))).normalized()
        for f in faces:
            f.normal_update()
            if abs(f.normal.dot(n)) > 0.9:
                f.smooth = False
        return faces

    def lathe(self, profile, seg=32, matrix=None, mat=0, smooth=True, closed=False):
        """Surface of revolution about local Z; profile is [(z, r)], r<=0 makes a pole."""
        m = matrix or Matrix.Identity(4)
        angles = [2 * math.pi * j / seg for j in range(seg)]
        rings = []
        for z, r in profile:
            if r <= 1e-7:
                rings.append([self.bm.verts.new(m @ Vector((0.0, 0.0, z)))])
            else:
                rings.append([self.bm.verts.new(m @ Vector((r * math.cos(a), r * math.sin(a), z)))
                              for a in angles])
        pairs = list(zip(rings[:-1], rings[1:]))
        if closed:
            pairs.append((rings[-1], rings[0]))
        faces = []
        for ra, rb in pairs:
            for j in range(seg):
                k = (j + 1) % seg
                if len(ra) == 1 and len(rb) == 1:
                    continue
                if len(ra) == 1:
                    vs = (ra[0], rb[k], rb[j])
                elif len(rb) == 1:
                    vs = (ra[j], ra[k], rb[0])
                else:
                    vs = (ra[j], ra[k], rb[k], rb[j])
                f = self.bm.faces.new(vs)
                f.material_index = mat
                f.smooth = smooth
                faces.append(f)
        return faces

    def capsule(self, p0, p1, radius, mat=0, seg=24, rings=6):
        length = (Vector(p1) - Vector(p0)).length
        prof = [(radius * math.sin(math.radians(-90 + 90 * i / rings)),
                 radius * math.cos(math.radians(-90 + 90 * i / rings))) for i in range(rings + 1)]
        prof += [(length + radius * math.sin(math.radians(90 * i / rings)),
                  radius * math.cos(math.radians(90 * i / rings))) for i in range(rings + 1)]
        return self.lathe(prof, seg, Matrix.Translation(p0) @ align_z(Vector(p1) - Vector(p0)), mat)

    def torus(self, center, axis, major, minor, mat=0, seg=32, tube=10, flatten=1.0):
        prof = [(minor * flatten * math.sin(2 * math.pi * i / tube), major + minor * math.cos(2 * math.pi * i / tube))
                for i in range(tube)]
        return self.lathe(prof, seg, Matrix.Translation(center) @ align_z(axis), mat, closed=True)

    def orient_faces(self, faces, inward_point_fn):
        flip = []
        for f in faces:
            f.normal_update()
            c = f.calc_center_median()
            if f.normal.dot(inward_point_fn(c) - c) < 0:
                flip.append(f)
        if flip:
            bmesh.ops.reverse_faces(self.bm, faces=flip)

    def to_object(self, name, coll, mats):
        me = bpy.data.meshes.new(name)
        self.bm.normal_update()
        self.bm.to_mesh(me)
        self.bm.free()
        for mat in mats:
            me.materials.append(mat)
        obj = bpy.data.objects.new(name, me)
        coll.objects.link(obj)
        return obj


def add_bevel(obj, width, segments=2):
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    return mod


def make_text(name, body, size, mat, coll, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), align="LEFT"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.materials.append(mat)
    obj = bpy.data.objects.new(name, cu)
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    obj.location = loc
    obj.rotation_euler = deg3(rot)
    return obj


def new_collection(name, parent):
    coll = bpy.data.collections.new(name)
    parent.children.link(coll)
    return coll


# ── 1. Module interior ────────────────────────────────────────────────────────

def module_axis_matrix(y, forward=True):
    """Local Z of the result runs along +Y (or -Y) through the module axis at depth y."""
    return Matrix.Translation((0.0, y, ZC)) @ Matrix.Rotation(math.radians(-90 if forward else 90), 4, "X")


def window_center():
    return Vector((SHELL_R * math.cos(WINDOW_ANGLE), WINDOW_Y, ZC + SHELL_R * math.sin(WINDOW_ANGLE)))


def build_module(coll, M, rng):
    y0, y1 = MODULE_Y
    axis_point = lambda c: Vector((0.0, c.y, ZC))
    centre = Vector((0.0, (y0 + y1) / 2, ZC))

    shell = MeshBuilder()
    faces = shell.lathe([(z, SHELL_R) for z in frange(0.0, y1 - y0, 0.1)], seg=128,
                        matrix=module_axis_matrix(y0), mat=0)
    shell.orient_faces(faces, axis_point)
    wc = window_center()
    hole = [f for f in faces if (f.calc_center_median() - wc).length < WINDOW_R + 0.03]
    bmesh.ops.delete(shell.bm, geom=hole, context="FACES")
    for y, fwd in ((y1, True), (y0, False)):
        ring = shell.lathe([(0.0, DOME_R), (0.0, SHELL_R)], seg=128, matrix=module_axis_matrix(y, fwd))
        shell.orient_faces(ring, lambda c: centre)
        dome = shell.lathe([(DOME_R * math.sin(math.radians(a)), DOME_R * math.cos(math.radians(a)))
                            for a in range(0, 91, 6)], seg=96, matrix=module_axis_matrix(y, fwd))
        shell.orient_faces(dome, lambda c: centre)
    shell.to_object("bas_module_shell", coll, [M["shell"]])

    ribs = MeshBuilder()
    for y in (-4.35, -3.25, -2.15, -0.95, 0.15, 1.0):
        ribs.torus((0.0, y, ZC), (0, 1, 0), SHELL_R - 0.05, 0.035, seg=96, tube=8)
    ribs.to_object("module_ribs", coll, [M["rib"]])

    floor = MeshBuilder()
    tracks = MeshBuilder()
    spans = [(-5.10, -3.62), (-3.58, -2.10), (-2.06, -0.58), (-0.54, 0.94)]
    for a, b in spans:
        floor.box((0.0, (a + b) / 2, FLOOR_Z - 0.02), (2.0, b - a, 0.04))
    for x in (-0.9, -0.45, 0.45, 0.9):
        tracks.box((x, (y0 + 0.1 + 0.94) / 2, FLOOR_Z + 0.004), (0.03, 0.94 - y0 - 0.1, 0.008))
    add_bevel(floor.to_object("floor_panels", coll, [M["floor"]]), 0.006)
    tracks.to_object("seat_tracks", coll, [M["seat_track"]])

    trays = MeshBuilder()
    cables = MeshBuilder()
    for ang in (30, 150, 210, 330):
        th = math.radians(ang)
        r = 1.90
        c = Vector((r * math.cos(th), 0.0, ZC + r * math.sin(th)))
        rot = Matrix.Rotation(-(th + math.pi / 2), 4, "Y")
        length = y1 - y0 - 0.2
        mid_y = (y0 + y1) / 2
        radial = Vector((math.cos(th), 0.0, math.sin(th)))
        tangent = Vector((-math.sin(th), 0.0, math.cos(th)))
        trays.box(c + Vector((0, mid_y, 0)), (0.12, length, 0.008), rot=rot)
        for side in (-1, 1):
            trays.box(c + Vector((0, mid_y, 0)) + tangent * side * 0.06 - radial * 0.02,
                      (0.006, length, 0.04), rot=rot)
        for i, mi in enumerate((0, 1, 2)):
            p = c - radial * 0.02 + tangent * (-0.03 + 0.03 * i)
            cables.cylinder(p + Vector((0, y0 + 0.1, 0)), p + Vector((0, y1 - 0.1, 0)), 0.011, mat=mi, seg=10)
    add_bevel(trays.to_object("cable_trays", coll, [M["tray"]]), 0.002)
    cables.to_object("cable_bundles", coll, [M["cable_black"], M["cable_orange"], M["cable_blue"]])

    build_wall_racks(coll, M, rng)

    win = MeshBuilder()
    n = (wc - Vector((0.0, wc.y, ZC))).normalized()
    win.torus(wc - n * 0.015, n, WINDOW_R + 0.06, 0.06, seg=48, tube=12, flatten=0.5)
    win.to_object("viewport_bezel", coll, [M["bezel"]])
    glass = MeshBuilder()
    glass.cylinder(wc - n * 0.02, wc - n * 0.012, WINDOW_R + 0.025, seg=48)
    g = glass.to_object("viewport_glass", coll, [M["glass"]])
    g.visible_shadow = False
    earth = MeshBuilder()
    earth.ellipsoid(wc + n * 14.0 + Vector((0.0, 3.0, -9.0)), (12.0, 12.0, 12.0), seg=64, rings=32)
    earth.to_object("earth", coll, [M["earth"]])


def build_wall_racks(coll, M, rng):
    B = {k: MeshBuilder() for k in ("body", "panel_a", "panel_b", "handle", "led_green", "led_amber", "lamp", "rail")}
    layouts = [(0.25, 0.25, 0.25, 0.25), (0.40, 0.30, 0.30), (0.20, 0.50, 0.30),
               (0.34, 0.33, 0.33), (0.15, 0.35, 0.25, 0.25)]
    walls = [
        # origin-fn, width axis, height axis, normal, face height, depth
        (lambda yc: (-AISLE, yc, ZC), (0, 1, 0), (0, 0, 1), (1, 0, 0), 1.70, 0.60),
        (lambda yc: (AISLE, yc, ZC), (0, -1, 0), (0, 0, 1), (-1, 0, 0), 1.70, 0.60),
        (lambda yc: (0.0, yc, ZC + AISLE), (0, 1, 0), (1, 0, 0), (0, 0, -1), 1.60, 0.55),
    ]
    racks = [(-1.60, RACK_W), (-2.70, RACK_W), (-3.80, RACK_W), (-4.78, 0.80)]
    for origin_fn, w, h, nrm, height, depth in walls:
        for yc, width in racks:
            sp = frame_matrix(origin_fn(yc), w, h, nrm)
            B["body"].box((0, 0, -depth / 2), (width - 0.01, height, depth), space=sp)
            top = height / 2 - 0.03
            for frac in rng.choice(layouts):
                ph = frac * (height - 0.06)
                cy = top - ph / 2
                key = "panel_b" if rng.random() < 0.3 else "panel_a"
                B[key].box((0, cy, 0.008), (width - 0.07, ph - 0.018, 0.016), space=sp)
                hx = width / 2 - 0.08
                for sx in (-1, 1):
                    B["handle"].cylinder((sx * hx, cy - ph * 0.28, 0.03), (sx * hx, cy + ph * 0.28, 0.03),
                                         0.007, seg=10, space=sp)
                    for sy in (-1, 1):
                        B["handle"].cylinder((sx * hx, cy + sy * ph * 0.26, 0.016),
                                             (sx * hx, cy + sy * ph * 0.26, 0.03), 0.005, seg=8, space=sp)
                if rng.random() < 0.6:
                    led = "led_green" if rng.random() < 0.75 else "led_amber"
                    B[led].box((width / 2 - 0.14, cy + ph / 2 - 0.035, 0.018), (0.012, 0.012, 0.006), space=sp)
                top -= ph
    for sx in (-1, 1):
        for yc, _ in racks[:3]:
            B["lamp"].box((sx * 0.88, yc, ZC + AISLE - 0.03), (0.10, 0.9, 0.02),
                          rot=(0.0, math.radians(sx * 35), 0.0))
        x = sx * (AISLE - 0.05)
        B["rail"].cylinder((x, -1.10, 1.30), (x, -4.90, 1.30), 0.012, seg=12)
        for yc in (-1.3, -2.4, -3.5, -4.6):
            B["rail"].cylinder((x, yc, 1.30), (sx * AISLE, yc, 1.30), 0.008, seg=8)
    add_bevel(B["body"].to_object("wall_rack_bodies", coll, [M["rack_body"]]), 0.006)
    add_bevel(B["panel_a"].to_object("wall_rack_panels_light", coll, [M["panel_a"]]), 0.004)
    add_bevel(B["panel_b"].to_object("wall_rack_panels_blue", coll, [M["panel_b"]]), 0.004)
    B["handle"].to_object("wall_rack_handles", coll, [M["handle"]])
    B["led_green"].to_object("wall_rack_leds_green", coll, [M["led_green"]])
    B["led_amber"].to_object("wall_rack_leds_amber", coll, [M["led_amber"]])
    B["lamp"].to_object("module_luminaires", coll, [M["lamp"]])
    B["rail"].to_object("handrails", coll, [M["handle"]])


# ── 2-3. Payload rack and experiment boxes ────────────────────────────────────

def build_payload(coll, M):
    alu, front, dark, screen, bezel, knob = (MeshBuilder() for _ in range(6))
    hw = RACK_W / 2
    z0, z1 = FLOOR_Z, FLOOR_Z + RACK_H
    zm = (z0 + z1) / 2
    for x in (-hw + 0.0225, hw - 0.0225):
        for y in (0.0225, RACK_D - 0.0225):
            alu.box((x, y, zm), (0.045, 0.045, RACK_H))
    for x in (-hw + 0.005, hw - 0.005):
        alu.box((x, RACK_D / 2, zm), (0.01, RACK_D - 0.05, RACK_H - 0.04))
    alu.box((0, RACK_D / 2, z1 - 0.01), (RACK_W, RACK_D, 0.02))
    alu.box((0, RACK_D / 2, z0 + 0.01), (RACK_W, RACK_D, 0.02))
    alu.box((0, RACK_D - 0.005, zm), (RACK_W - 0.05, 0.01, RACK_H - 0.04))

    lower = frange(z0 + 0.03, SHELF_TOP - 0.025, (SHELF_TOP - 0.025 - z0 - 0.03) / 3)
    for za, zb in zip(lower[:-1], lower[1:]):
        front.box((0, 0.01, (za + zb) / 2), (RACK_W - 0.09, 0.02, zb - za - 0.012))
        alu.cylinder((-0.18, -0.03, zb - 0.045), (0.18, -0.03, zb - 0.045), 0.008, seg=12)
        for x in (-0.18, 0.18):
            alu.cylinder((x, 0.0, zb - 0.045), (x, -0.03, zb - 0.045), 0.006, seg=10)

    bay_mid = (SHELF_TOP + BAY_TOP) / 2
    bay_h = BAY_TOP - SHELF_TOP
    dark.box((0, BAY_DEPTH + 0.005, bay_mid), (0.80, 0.01, bay_h))
    for sx in (-1, 1):
        alu.box((sx * 0.405, BAY_DEPTH / 2, bay_mid), (0.01, BAY_DEPTH, bay_h))
        front.box((sx * 0.4415, 0.01, bay_mid), (0.063, 0.02, bay_h))
    alu.box((0, BAY_DEPTH / 2, BAY_TOP + 0.005), (0.82, BAY_DEPTH, 0.01))

    split = 1.47
    front.box((0, 0.01, (BAY_TOP + split) / 2 + 0.005), (RACK_W - 0.09, 0.02, split - BAY_TOP - 0.02))
    front.box((0, 0.01, (split + z1 - 0.03) / 2), (RACK_W - 0.09, 0.02, z1 - 0.03 - split - 0.012))
    bezel.box((-0.12, -0.004, 1.30), (0.40, 0.012, 0.22))
    screen.box((-0.12, -0.011, 1.30), (0.36, 0.002, 0.18))
    for i, x in enumerate((0.18, 0.26, 0.34)):
        knob.cylinder((x, 0.0, 1.34), (x, -0.028, 1.34), 0.018, seg=20)
        knob.box((x, -0.004, 1.24), (0.02, 0.01, 0.035))
    for i in range(6):
        knob.box((-0.36 + i * 0.07, -0.004, 1.62), (0.04, 0.01, 0.06))
    for x in (0.22, 0.30, 0.38):
        knob.cylinder((x, 0.0, 1.62), (x, -0.02, 1.62), 0.012, seg=16)

    dark.box((0, (-0.40 + BAY_DEPTH) / 2, SHELF_TOP - 0.01), (1.0, 0.40 + BAY_DEPTH, 0.02))
    alu.box((0, -0.405, SHELF_TOP - 0.012), (1.0, 0.012, 0.028))
    for x in (-0.47, 0.47):
        alu.cylinder((x, -0.045, 0.72), (x, -0.045, 1.60), 0.006, seg=16)
        for z in (0.74, 1.58):
            alu.cylinder((x, 0.0, z), (x, -0.045, z), 0.005, seg=10)

    add_bevel(alu.to_object("ispr_rack_frame", coll, [M["alu"]]), 0.003)
    add_bevel(front.to_object("ispr_front_panels", coll, [M["alu_panel"]]), 0.003)
    add_bevel(dark.to_object("ispr_work_surface", coll, [M["work_surface"]]), 0.003)
    screen.to_object("ispr_display", coll, [M["screen"]])
    bezel.to_object("ispr_display_bezel", coll, [M["screen_bezel"]])
    knob.to_object("ispr_controls", coll, [M["knob"]])
    make_text("ispr_label", "ISPR  BAS-HAR-03   PAYLOAD RACK", 0.022, M["label"], coll,
              loc=(-0.44, -0.001, 1.79), rot=(90, 0, 0))

    for sx, rim, label in ((-1, "zone_red", "RED ZONE"), (1, "zone_yellow", "YELLOW ZONE")):
        pad, frame = MeshBuilder(), MeshBuilder()
        cx, cy, h = sx * 0.36, -0.10, ZONE_PAD_TOP - SHELF_TOP
        pad.box((cx, cy, SHELF_TOP + h / 2), (0.13, 0.12, h))
        for dx, dy, sx_, sy_ in ((0, 0.058, 0.134, 0.008), (0, -0.058, 0.134, 0.008),
                                 (0.063, 0, 0.008, 0.124), (-0.063, 0, 0.008, 0.124)):
            frame.box((cx + dx, cy + dy, ZONE_PAD_TOP + 0.0015), (sx_, sy_, 0.003))
        add_bevel(pad.to_object(f"zone_pad_{label.split()[0].lower()}", coll, [M["zone_pad"]]), 0.004)
        frame.to_object(f"zone_rim_{label.split()[0].lower()}", coll, [M[rim]])
        make_text(f"zone_label_{label.split()[0].lower()}", label, 0.016, M[rim], coll,
                  loc=(cx, -0.20, SHELF_TOP + 0.0006), align="CENTER")

    container = MeshBuilder()
    container.box((0, 0, 0), CONTAINER)
    top = [f for f in container.bm.faces if f.calc_center_median().z > CONTAINER[2] / 2 - 1e-4]
    bmesh.ops.delete(container.bm, geom=top, context="FACES_ONLY")
    box = container.to_object("main_box", coll, [M["main_box"]])
    box.location = CONTAINER_CENTER
    sol = box.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0075
    sol.offset = -1.0
    add_bevel(box, 0.0025)

    lid_mb = MeshBuilder()
    lid_mb.box((0, -CONTAINER[1] / 2, 0.006), (CONTAINER[0] + 0.006, CONTAINER[1] + 0.004, 0.012))
    lid_mb.box((0, -CONTAINER[1] + 0.02, 0.018), (0.08, 0.014, 0.012))
    lid = lid_mb.to_object("main_box_lid", coll, [M["main_box"]])
    lid.location = LID_HINGE
    add_bevel(lid, 0.002)

    boxes = {}
    for name, loc in (("red_box", RED_TRACK[0][2]), ("yellow_box", YELLOW_TRACK[0][2])):
        mb = MeshBuilder()
        mb.box((0, 0, 0), INNER_BOX)
        obj = mb.to_object(name, coll, [M[name]])
        obj.location = loc
        add_bevel(obj, 0.004, 3)
        boxes[name] = obj
    return {"lid": lid, **boxes}


# ── 4. Astronaut ──────────────────────────────────────────────────────────────

def astronaut_skeleton():
    """Armature-local rest joints (origin at pelvis, facing +Y).

    Arms rest in a neutral-body-posture crouch (elbow bent 120 deg) so the listed
    euler angles keep free hands at chest height instead of inside the container.
    """
    s = Vector((-0.23, 0.0, 0.46))
    du = Vector((-0.201, 0.259, -0.945)).normalized()
    e = s + du * 0.30
    df = Vector((-0.08, 0.707, 0.707)).normalized()
    w = e + df * 0.28
    dh = Vector((-0.07, 0.90, 0.30)).normalized()
    ht = w + dh * 0.10
    j = {"S": s, "E": e, "W": w, "Ht": ht, "dU": du, "dF": df, "dH": dh,
         "hip": Vector((-0.10, 0.0, -0.02)), "knee": Vector((-0.10, 0.0, -0.44)),
         "ankle": Vector((-0.10, 0.0, -0.84)), "toe": Vector((-0.10, 0.17, -0.90)),
         "HC": Vector((0.0, 0.03, 0.74))}
    mirror = lambda v: Vector((-v.x, v.y, v.z))
    joints = {"L": j, "R": {k: (mirror(v) if k != "HC" else v) for k, v in j.items()}}
    # name -> (head, tail, parent, roll-align vector). Rolls are chosen so the
    # listed euler signs read anatomically: +X flexes spine/hips, -X raises the
    # upper arm, +X straightens the elbow, -X tilts the head forward.
    bones = {
        "pelvis": ((0, 0, 0), (0, 0, 0.12), None, (0, 1, 0)),
        "spine": ((0, 0, 0.12), (0, 0, 0.30), "pelvis", (0, 1, 0)),
        "chest": ((0, 0, 0.30), (0, 0, 0.52), "spine", (0, 1, 0)),
        "neck": ((0, 0.01, 0.52), (0, 0.02, 0.60), "chest", (0, -1, 0)),
        "head": ((0, 0.02, 0.60), (0, 0.02, 0.84), "neck", (0, -1, 0)),
    }
    for side, jj in joints.items():
        bones[f"upperarm.{side}"] = (jj["S"], jj["E"], "chest", (0, -1, 0))
        bones[f"forearm.{side}"] = (jj["E"], jj["W"], f"upperarm.{side}", (0, 0.7, -0.7))
        bones[f"hand.{side}"] = (jj["W"], jj["Ht"], f"forearm.{side}", (0, 0.3, -1))
        bones[f"thigh.{side}"] = (jj["hip"], jj["knee"], "pelvis", (0, 1, 0))
        bones[f"shin.{side}"] = (jj["knee"], jj["ankle"], f"thigh.{side}", (0, 1, 0))
        bones[f"foot.{side}"] = (jj["ankle"], jj["toe"], f"shin.{side}", (0, 0, 1))
    return bones, joints


SUIT_SLOTS = ["suit", "hut", "ring", "helmet", "visor", "dark", "chest_box", "sole",
              "saffron", "flag_white", "flag_green"]


def _slot(name):
    return SUIT_SLOTS.index(name)


def perpendicular(axis, hint):
    a = Vector(axis).normalized()
    v = Vector(hint) - a * Vector(hint).dot(a)
    return v.normalized()


def build_astronaut(coll, M):
    bones, joints = astronaut_skeleton()
    arm_data = bpy.data.armatures.new("astronaut_rig")
    arm = bpy.data.objects.new("astronaut_rig", arm_data)
    arm.location = ASTRO_START
    coll.objects.link(arm)
    vl = bpy.context.view_layer
    vl.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail, parent, roll) in bones.items():
        eb = arm_data.edit_bones.new(name)
        eb.head, eb.tail = head, tail
        eb.use_connect = False
        if parent:
            eb.parent = arm_data.edit_bones[parent]
        eb.align_roll(Vector(roll))
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"

    parts = {name: MeshBuilder() for name in bones}
    S = _slot
    parts["pelvis"].ellipsoid((0, 0, 0.0), (0.17, 0.145, 0.13), mat=S("suit"))
    for z in (0.100, 0.125):
        parts["pelvis"].torus((0, 0, z), (0, 0, 1), 0.165, 0.012, mat=S("ring"))
    parts["spine"].capsule((0, 0, 0.06), (0, 0, 0.18), 0.15, mat=S("suit"))
    parts["chest"].ellipsoid((0, 0, 0.38), (0.22, 0.20, 0.28), mat=S("hut"), seg=40, rings=20)
    parts["chest"].torus((0, 0.01, 0.60), (0, 0, 1), 0.125, 0.022, mat=S("ring"), tube=12)
    parts["chest"].box((0, 0.195, 0.33), (0.17, 0.06, 0.10), mat=S("chest_box"))
    for x in (-0.05, 0.0, 0.05):
        parts["chest"].cylinder((x, 0.22, 0.35), (x, 0.24, 0.35), 0.011, mat=S("ring"), seg=12)

    hc = joints["L"]["HC"]
    hb = parts["head"]
    helmet = hb.ellipsoid(hc, (0.16, 0.16, 0.16), mat=S("helmet"), seg=48, rings=24)
    cut = [v for f in helmet for v in f.verts
           if (v.co - hc).y > 0.05 and -0.06 < (v.co - hc).z < 0.085]
    bmesh.ops.delete(hb.bm, geom=list(set(cut)), context="VERTS")
    visor = hb.ellipsoid(hc, (0.1635, 0.1635, 0.1635), mat=S("visor"), seg=48, rings=24)
    keep = lambda v: (v.co - hc).y > 0.03 and -0.08 < (v.co - hc).z < 0.10
    drop = [v for f in visor for v in f.verts if not keep(v)]
    bmesh.ops.delete(hb.bm, geom=list(set(drop)), context="VERTS")
    hb.ellipsoid(hc + Vector((0, 0.0, -0.01)), (0.10, 0.10, 0.11), mat=S("dark"))

    for side, jj in joints.items():
        sgn = -1 if side == "L" else 1
        ua, fa, hd = parts[f"upperarm.{side}"], parts[f"forearm.{side}"], parts[f"hand.{side}"]
        ua.capsule(jj["S"], jj["E"], 0.062, mat=S("suit"))
        for d in (0.03, 0.055, 0.08):
            ua.torus(jj["S"] + jj["dU"] * d, jj["dU"], 0.066, 0.011, mat=S("ring"))
        ua.torus(jj["E"] - jj["dU"] * 0.025, jj["dU"], 0.060, 0.010, mat=S("ring"))
        if side == "L":
            out = perpendicular(jj["dU"], (-1, 0, 0))
            across = jj["dU"].cross(out).normalized()
            rot = frame_matrix((0, 0, 0), across, -jj["dU"], out)
            base = jj["S"] + jj["dU"] * 0.13 + out * 0.064
            for i, key in enumerate(("saffron", "flag_white", "flag_green")):
                ua.box(base + jj["dU"] * (-0.0117 + 0.0117 * i), (0.052, 0.0117, 0.003), rot=rot, mat=S(key))
        fa.capsule(jj["E"], jj["W"], 0.054, mat=S("suit"))
        for d in (0.02, 0.045):
            fa.torus(jj["E"] + jj["dF"] * d, jj["dF"], 0.058, 0.010, mat=S("ring"))
        for d in (0.012, 0.032):
            fa.torus(jj["W"] - jj["dF"] * d, jj["dF"], 0.056, 0.009, mat=S("ring"))
        palm_z = perpendicular(jj["dH"], (0, 0, -1))
        palm_x = jj["dH"].cross(palm_z).normalized()
        hd.ellipsoid(jj["W"] + jj["dH"] * 0.055, (0.047, 0.068, 0.028),
                     rot=frame_matrix((0, 0, 0), palm_x, jj["dH"], palm_z), mat=S("suit"))
        inward = Vector((-sgn, 0.0, 0.0))
        hd.capsule(jj["W"] + jj["dH"] * 0.025 + inward * 0.035, jj["W"] + jj["dH"] * 0.075 + inward * 0.05,
                   0.016, mat=S("suit"), seg=12, rings=4)
        parts[f"thigh.{side}"].capsule(jj["hip"], jj["knee"], 0.085, mat=S("suit"))
        parts[f"shin.{side}"].capsule(jj["knee"], jj["ankle"], 0.072, mat=S("suit"))
        foot = parts[f"foot.{side}"]
        foot.capsule(jj["ankle"], jj["toe"], 0.062, mat=S("suit"))
        pitch = math.atan2(jj["toe"].z - jj["ankle"].z, jj["toe"].y - jj["ankle"].y)
        foot.box((jj["ankle"] + jj["toe"]) / 2 + Vector((0, 0, -0.055)), (0.10, 0.22, 0.02),
                 rot=(pitch, 0.0, 0.0), mat=S("sole"))

    mats = [M[k] for k in SUIT_SLOTS]
    segments = []
    for name, mb in parts.items():
        if not mb.bm.verts:
            mb.bm.free()
            continue
        obj = mb.to_object(f"suit_{name}", coll, mats)
        obj.parent = arm
        vg = obj.vertex_groups.new(name=name)
        vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        segments.append(obj)

    ik = {}
    for side in ("L", "R"):
        target = bpy.data.objects.new(f"ik_target.{side}", None)
        pole = bpy.data.objects.new(f"ik_pole.{side}", None)
        for e in (target, pole):
            e.empty_display_size = 0.04
            coll.objects.link(e)
        pole.parent = arm
        pole.location = (0.60 if side == "R" else -0.60, -0.20, 0.05)
        con = arm.pose.bones[f"hand.{side}"].constraints.new("IK")
        con.name = "IK"
        con.target = target
        con.pole_target = pole
        con.chain_count = 3
        con.use_tail = True
        con.use_stretch = False
        con.iterations = 500
        con.influence = 0.0
        ik[side] = (con, target, pole)
    return arm, bones, joints, ik


def calibrate_ik_poles(arm, ik, joints):
    """Pick each pole angle so the solved elbow actually bends toward its pole target."""
    vl = bpy.context.view_layer
    set_pose(arm, NEUTRAL)
    for side, (con, target, pole) in ik.items():
        sgn = -1 if side == "L" else 1
        target.location = arm.matrix_world @ Vector((sgn * 0.12, 0.36, 0.14))
        con.influence = 1.0
        best = (-2.0, 0.0)
        for deg in range(-180, 180, 5):
            con.pole_angle = math.radians(deg)
            vl.update()
            ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            s = ev.pose.bones[f"upperarm.{side}"].head
            e = ev.pose.bones[f"forearm.{side}"].head
            w = ev.pose.bones[f"hand.{side}"].tail
            p = arm.matrix_world.inverted() @ pole.matrix_world.translation
            u = (w - s).normalized()
            e_perp = (e - s) - u * (e - s).dot(u)
            p_perp = (p - s) - u * (p - s).dot(u)
            if e_perp.length < 1e-6:
                continue
            score = e_perp.normalized().dot(p_perp.normalized())
            if score > best[0]:
                best = (score, deg)
        con.pole_angle = math.radians(best[1])
        con.influence = 0.0
    vl.update()


def set_pose(arm, pose):
    for name, rot in pose.items():
        arm.pose.bones[name].rotation_euler = Euler(deg3(rot), "XYZ")


# ── 5. HAR skeleton overlay ───────────────────────────────────────────────────

def landmark_table(joints):
    """MediaPipe-33 index -> (bone, armature-local rest position). 'Left' is the crew member's left (-X)."""
    hc = joints["L"]["HC"]
    face = {
        0: (0.0, 0.105, -0.005),
        1: (-0.018, 0.095, 0.025), 2: (-0.035, 0.09, 0.028), 3: (-0.05, 0.08, 0.025),
        4: (0.018, 0.095, 0.025), 5: (0.035, 0.09, 0.028), 6: (0.05, 0.08, 0.025),
        7: (-0.085, 0.01, 0.01), 8: (0.085, 0.01, 0.01),
        9: (-0.022, 0.095, -0.045), 10: (0.022, 0.095, -0.045),
    }
    table = {i: ("head", hc + Vector(off)) for i, off in face.items()}
    for side, base in (("L", 0), ("R", 1)):
        jj = joints[side]
        sgn = -1 if side == "L" else 1
        table[11 + base] = (f"upperarm.{side}", jj["S"])
        table[13 + base] = (f"forearm.{side}", jj["E"])
        table[15 + base] = (f"hand.{side}", jj["W"])
        table[17 + base] = (f"hand.{side}", jj["Ht"] + Vector((sgn * 0.028, -0.01, 0.0)))
        table[19 + base] = (f"hand.{side}", jj["Ht"] + Vector((-sgn * 0.018, 0.005, 0.0)))
        table[21 + base] = (f"hand.{side}", jj["W"] + jj["dH"] * 0.045 + Vector((-sgn * 0.045, 0.0, 0.01)))
        table[23 + base] = (f"thigh.{side}", jj["hip"])
        table[25 + base] = (f"shin.{side}", jj["knee"])
        table[27 + base] = (f"foot.{side}", jj["ankle"])
        table[29 + base] = (f"foot.{side}", jj["ankle"] + Vector((0.0, -0.045, -0.06)))
        table[31 + base] = (f"foot.{side}", jj["toe"])
    return table


def build_har_overlay(coll, M, arm, joints):
    sphere = MeshBuilder()
    sphere.ellipsoid((0, 0, 0), (0.018, 0.018, 0.018), seg=16, rings=8)
    sphere_me = sphere.to_object("_tmp_sphere", coll, [M["har_joint"]])
    sphere_mesh = sphere_me.data
    bpy.data.objects.remove(sphere_me)
    rod = MeshBuilder()
    rod.cylinder((0, 0, 0), (0, 1, 0), 0.007, seg=10)
    rod_obj = rod.to_object("_tmp_rod", coll, [M["har_limb"]])
    rod_mesh = rod_obj.data
    bpy.data.objects.remove(rod_obj)

    marks = {}
    for idx, (bone_name, pos) in sorted(landmark_table(joints).items()):
        obj = bpy.data.objects.new(f"har_lm_{idx:02d}", sphere_mesh)
        coll.objects.link(obj)
        bone = arm.data.bones[bone_name]
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.matrix_parent_inverse = Matrix.Identity(4)
        tail_frame = bone.matrix_local @ Matrix.Translation((0.0, bone.length, 0.0))
        obj.matrix_basis = tail_frame.inverted() @ Matrix.Translation(pos)
        obj.visible_shadow = False
        marks[idx] = obj
    for a, b in MP_EDGES:
        obj = bpy.data.objects.new(f"har_link_{a:02d}_{b:02d}", rod_mesh)
        coll.objects.link(obj)
        cl = obj.constraints.new("COPY_LOCATION")
        cl.target = marks[a]
        st = obj.constraints.new("STRETCH_TO")
        st.target = marks[b]
        st.rest_length = 1.0
        st.volume = "NO_VOLUME"
        obj.visible_shadow = False
    return marks


# ── 6. HUD ────────────────────────────────────────────────────────────────────

def build_hud(coll, M):
    mb = MeshBuilder()
    verts = [mb.bm.verts.new(v) for v in ((-0.25, -0.175, 0), (0.25, -0.175, 0), (0.25, 0.175, 0), (-0.25, 0.175, 0))]
    mb.bm.faces.new(verts)
    panel = mb.to_object("hud_panel", coll, [M["hud_bg"]])
    panel.location = (0.72, -0.25, 1.45)
    panel.rotation_euler = deg3((90, 0, -12))

    def child(obj, loc):
        obj.parent = panel
        obj.location = loc
        obj.visible_shadow = False
        return obj

    edge = MeshBuilder()
    for x, y, sx, sy in ((0, 0.175, 0.506, 0.004), (0, -0.175, 0.506, 0.004),
                         (0.25, 0, 0.004, 0.354), (-0.25, 0, 0.004, 0.354)):
        edge.box((x, y, 0.001), (sx, sy, 0.002))
    child(edge.to_object("hud_border", coll, [M["hud_edge"]]), (0, 0, 0))

    segments = []
    for k in range(STEPS):
        seg_mat = mat_emission(f"hud_step_{k + 1}", (0.05, 0.12, 0.12), 0.3)
        cube = MeshBuilder()
        cube.box((0, 0, 0), (0.06, 0.016, 0.012))
        segments.append(child(cube.to_object(f"hud_progress_{k + 1}", coll, [seg_mat]),
                              ((k - 3.5) * 0.075, -0.21, 0.0)))
    led = MeshBuilder()
    led.ellipsoid((0, 0, 0), (0.025, 0.025, 0.025), seg=24, rings=12)
    child(led.to_object("hud_gravity_led", coll, [M["hud_led"]]), (0.205, -0.132, 0.02))
    child(make_text("hud_title", "BAS  HAR MONITOR", 0.026, M["hud_title"], coll), (-0.232, 0.13, 0.002))
    child(make_text("hud_gravity_label", "MICROGRAVITY STABLE", 0.015, M["hud_ok"], coll, align="RIGHT"),
          (0.172, -0.135, 0.002))
    child(make_text("hud_sequence", "SEQUENCE VALID", 0.015, M["hud_ok"], coll), (-0.232, -0.135, 0.002))
    step_texts = []
    for k in range(STEPS):
        txt = make_text(f"hud_step_text_{k + 1}", f"STEP {k + 1} / {STEPS}\n{STEP_NAMES[k]}", 0.042,
                        M["hud_text"], coll)
        cu = txt.data
        cu.space_line = 1.1
        step_texts.append(child(txt, (-0.232, -0.01, 0.002)))
    return {"panel": panel, "segments": segments, "texts": step_texts}


# ── 7. Lighting and world ─────────────────────────────────────────────────────

def setup_lighting(coll):
    for name, kind, watts, color, size, loc, rot, aim in LIGHTS:
        data = bpy.data.lights.new(name, kind)
        data.energy = watts
        data.color = color
        if kind == "AREA":
            data.shape = "SQUARE"
            data.size = size
        else:
            data.spot_size = math.radians(size)
            data.spot_blend = 0.3
            data.shadow_soft_size = 0.08
        obj = bpy.data.objects.new(name, data)
        coll.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = look_rotation(loc, aim) if aim else Euler(deg3(rot), "XYZ")

    world = bpy.data.worlds.new("deep_space")
    if world.node_tree is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:
        nt.nodes.clear()
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (0.004, 0.004, 0.006, 1.0)
    bg.inputs["Strength"].default_value = 0.3
    world.color = (0.004, 0.004, 0.006)
    bpy.context.scene.world = world

    # Without a reflection probe the visor and aluminium reflect the near-black world.
    probe_data = bpy.data.lightprobes.new("work_area_reflections", "SPHERE")
    probe_data.influence_distance = 2.6
    probe = bpy.data.objects.new("work_area_reflections", probe_data)
    coll.objects.link(probe)
    probe.location = (0.0, -0.60, 1.10)


# ── 8. Cameras ────────────────────────────────────────────────────────────────

def setup_cameras(scene, arm, joints):
    cams = {}
    helmet = bpy.data.objects.new(HELMET, None)
    helmet.empty_display_size = 0.05
    scene.collection.objects.link(helmet)
    head = arm.data.bones["head"]
    helmet.parent = arm
    helmet.parent_type = "BONE"
    helmet.parent_bone = "head"
    # Visor surface, not helmet centre: at f/1.8 the 10 cm difference blurs the face landmarks.
    helmet.matrix_basis = (head.matrix_local @ Matrix.Translation((0.0, head.length, 0.0))).inverted() \
        @ Matrix.Translation(joints["L"]["HC"] + Vector((0.0, 0.10, 0.0)))
    focus = {HELMET: helmet}
    for name, (loc, lens, fstop) in CAMERAS.items():
        aim = CAMERA_AIM.get(name, TARGET)
        if aim not in focus:
            empty = bpy.data.objects.new(f"focus_{len(focus)}", None)
            empty.location = aim
            empty.empty_display_size = 0.05
            scene.collection.objects.link(empty)
            focus[aim] = empty
        data = bpy.data.cameras.new(f"cam_{name}")
        data.lens = lens
        data.clip_start = 0.02
        data.clip_end = 60.0
        data.dof.use_dof = True
        data.dof.aperture_fstop = fstop
        data.dof.focus_object = focus[aim]
        obj = bpy.data.objects.new(f"cam_{name}", data)
        scene.collection.objects.link(obj)
        obj.location = loc
        if aim == HELMET:
            track = obj.constraints.new("TRACK_TO")
            track.target = helmet
            track.track_axis = "TRACK_NEGATIVE_Z"
            track.up_axis = "UP_Y"
        else:
            obj.rotation_euler = look_rotation(loc, aim)
        cams[name] = obj
    return cams


# ── Animation ─────────────────────────────────────────────────────────────────

def key_object(obj, frame, loc=None, rot=None):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert("location", frame=frame)
    if rot is not None:
        obj.rotation_euler = Euler(deg3(rot), "XYZ")
        obj.keyframe_insert("rotation_euler", frame=frame)


def animate(scene, tl, arm, ik, objs, hud, use_ik):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    key_object(arm, tl.t(1, 0.0), loc=ASTRO_START)
    key_object(arm, tl.t(1, 1.0), loc=ASTRO_WORK)
    pose = dict(NEUTRAL)
    for step, frac, reset, over in POSE_TRACK:
        if reset:
            pose = dict(NEUTRAL)
        pose.update(over)
        frame = tl.t(step, frac)
        set_pose(arm, pose)
        for name in pose:
            arm.pose.bones[name].keyframe_insert("rotation_euler", frame=frame)

    for step, frac, angle in LID_TRACK:
        key_object(objs["lid"], tl.t(step, frac), rot=(angle, 0, 0))
    for name, track in (("red_box", RED_TRACK), ("yellow_box", YELLOW_TRACK)):
        for step, frac, loc, rot in track:
            key_object(objs[name], tl.t(step, frac), loc=loc, rot=rot)

    if use_ik:
        bake_ik(scene, tl, ik, objs)

    for k, cube in enumerate(hud["segments"], start=1):
        sock = cube.data.materials[0].node_tree.nodes["Emission"].inputs
        for frame, color, strength in ((1, (0.05, 0.12, 0.12), 0.3),
                                       (tl.t(k, 0.9), (0.05, 0.12, 0.12), 0.3),
                                       (tl.t(k, 1.0), (0.0, 1.0, 0.6), 4.0)):
            sock["Color"].default_value = rgba(color)
            sock["Strength"].default_value = strength
            sock["Color"].keyframe_insert("default_value", frame=frame)
            sock["Strength"].keyframe_insert("default_value", frame=frame)
    for k, txt in enumerate(hud["texts"], start=1):
        first = tl.t(k, 0.0)
        last = tl.t(k + 1, 0.0) - 1 if k < STEPS else tl.last
        for frame, hidden in ((first - 1, True), (first, False), (last, False), (last + 1, True)):
            if 1 <= frame <= tl.last:
                txt.hide_render = hidden
                txt.keyframe_insert("hide_render", frame=frame)
        txt.hide_render = k != 1

    ids = list(bpy.data.objects) + [m.node_tree for m in bpy.data.materials if m.node_tree]
    for idb in ids:
        for fc in iter_fcurves(idb):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
    scene.frame_set(1)


def bake_ik(scene, tl, ik, objs):
    """Key each wrist target onto the held object every frame it matters, then ramp IK influence."""
    by_name = {"main_box_lid": objs["lid"], "red_box": objs["red_box"], "yellow_box": objs["yellow_box"]}
    work = {}
    for side, windows in IK_PLAN.items():
        con = ik[side][0]
        con.influence = 0.0
        con.keyframe_insert("influence", frame=1)
        for src, ramp, offsets in windows:
            a, b, c, d = (tl.t(*r) for r in ramp)
            for frame, value in ((a, 0.0), (b, 1.0), (c, 1.0), (d, 0.0)):
                con.influence = value
                con.keyframe_insert("influence", frame=frame)
            offs = [(tl.t(*when), Vector(off)) for when, off in offsets]
            for f in range(max(1, a - 1), min(tl.last, d + 1) + 1):
                work.setdefault(f, []).append((side, by_name[src], offs))
    for f in sorted(work):
        scene.frame_set(f)
        for side, src, offs in work[f]:
            target = ik[side][1]
            target.location = src.matrix_world @ interp_offset(offs, f)
            target.keyframe_insert("location", frame=f)


def interp_offset(offs, frame):
    if frame <= offs[0][0]:
        return offs[0][1]
    for (f0, v0), (f1, v1) in zip(offs[:-1], offs[1:]):
        if frame <= f1:
            return v0.lerp(v1, (frame - f0) / max(1, f1 - f0))
    return offs[-1][1]


# ── Render setup ──────────────────────────────────────────────────────────────

def _try_set(obj, attr, value):
    try:
        setattr(obj, attr, value)
    except (AttributeError, TypeError, ValueError):
        pass


def setup_overlay_scene(scene, har_coll, args, tl):
    """The HAR skeleton renders in its own scene through a DOF-free twin of the active camera.

    Detector overlays are drawn on the image, not photographed through the lens, so
    they must stay crisp even in the f/1.8 helmet shot.
    """
    ov = bpy.data.scenes.new("HAR_Overlay")
    scene.collection.children.unlink(har_coll)
    ov.collection.children.link(har_coll)
    r = ov.render
    r.engine = "BLENDER_EEVEE"
    r.resolution_x, r.resolution_y, r.resolution_percentage = args.width, args.height, 100
    r.fps = args.fps
    r.film_transparent = True
    r.use_motion_blur = False
    ov.frame_start, ov.frame_end = 1, tl.last
    ov.eevee.taa_render_samples = args.samples
    data = bpy.data.cameras.new("cam_overlay")
    data.dof.use_dof = False
    cam = bpy.data.objects.new("cam_overlay", data)
    ov.collection.objects.link(cam)
    cam.constraints.new("COPY_TRANSFORMS")
    ov.camera = cam
    return ov


def sync_overlay_camera(overlay, main_cam):
    cam = overlay.camera
    cam.constraints[0].target = main_cam
    for attr in ("lens", "sensor_width", "sensor_height", "sensor_fit", "clip_start", "clip_end"):
        setattr(cam.data, attr, getattr(main_cam.data, attr))


def setup_render(scene, overlay, args, tl):
    r = scene.render
    r.engine = "BLENDER_EEVEE"
    r.resolution_x, r.resolution_y = args.width, args.height
    r.resolution_percentage = 100
    r.fps = args.fps
    r.film_transparent = True
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.35
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.image_settings.compression = 15
    scene.frame_start, scene.frame_end = 1, tl.last

    ee = scene.eevee
    ee.taa_render_samples = args.samples
    ee.taa_samples = min(args.samples, 16)
    _try_set(ee, "use_raytracing", True)
    _try_set(ee, "ray_tracing_method", "SCREEN")
    _try_set(ee, "use_fast_gi", True)
    _try_set(ee, "use_shadows", True)

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = EXPOSURE
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        log("look 'AgX - Medium High Contrast' unavailable, using AgX base look")

    ng = bpy.data.node_groups.new("HAR3D_Compositor", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl_beauty = ng.nodes.new("CompositorNodeRLayers")
    rl_beauty.scene, rl_beauty.layer = scene, scene.view_layers[0].name
    rl_overlay = ng.nodes.new("CompositorNodeRLayers")
    rl_overlay.scene, rl_overlay.layer = overlay, overlay.view_layers[0].name
    over = ng.nodes.new("CompositorNodeAlphaOver")
    glare = ng.nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Threshold"].default_value = BLOOM_THRESHOLD
    glare.inputs["Strength"].default_value = BLOOM_INTENSITY
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl_beauty.outputs["Image"], over.inputs["Background"])
    ng.links.new(rl_overlay.outputs["Image"], over.inputs["Foreground"])
    ng.links.new(over.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs[0])
    scene.compositing_node_group = ng
    r.use_compositing = True


# ── Output ────────────────────────────────────────────────────────────────────

def render_camera(scene, overlay, cam, frames_dir, total):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("[0-9][0-9][0-9][0-9].png"):
        stale.unlink()
    scene.camera = cam
    sync_overlay_camera(overlay, cam)
    scene.render.filepath = str(frames_dir / "####")
    bpy.ops.render.render(animation=True)
    return sum(1 for f in range(1, total + 1) if (frames_dir / f"{f:04d}.png").exists())


def encode_helper():
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for cand in (here.parent / "tools" / "frames_to_video.py", Path.cwd() / "tools" / "frames_to_video.py",
                 here / "frames_to_video.py"):
        if cand.exists():
            try:
                return Path(cand.resolve().relative_to(Path.cwd().resolve())).as_posix()
            except ValueError:
                return cand.as_posix()
    return "frames_to_video.py"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="har_showcase.py", description="BAS HAR 3D showcase renderer")
    p.add_argument("--out", default="build/har3d")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--camera", default="hero", choices=[*CAMERAS, "all"])
    p.add_argument("--no-ik", action="store_true")
    a = p.parse_args(argv)
    a.width = a.width or (960 if a.quick else 1280)
    a.height = a.height or (540 if a.quick else 720)
    a.samples = a.samples or (8 if a.quick else 32)
    a.frames_per_step = 24 if a.quick else 48
    return a


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    tl = Timeline(args.frames_per_step)
    rng = random.Random(7)
    M = build_materials()

    root = new_collection("BAS_Scene", scene.collection)
    c_module = new_collection("BAS_Module", root)
    c_payload = new_collection("BAS_Payload", root)
    c_astro = new_collection("BAS_Astronaut", root)
    c_hud = new_collection("BAS_HUD", root)
    c_lights = new_collection("BAS_Lights", root)
    c_har = new_collection("HAR_Overlay", scene.collection)

    log("Building BAS module interior …")
    build_module(c_module, M, rng)
    log("Building payload rack + boxes …")
    objs = build_payload(c_payload, M)
    log("Building astronaut …")
    arm, bones, joints, ik = build_astronaut(c_astro, M)
    calibrate_ik_poles(arm, ik, joints)
    log("Building HAR skeleton overlay …")
    build_har_overlay(c_har, M, arm, joints)
    log("Building HUD panel …")
    hud = build_hud(c_hud, M)
    log("Setting up lighting …")
    setup_lighting(c_lights)
    log("Setting up cameras …")
    cams = setup_cameras(scene, arm, joints)
    log("Animating 8-step protocol …")
    animate(scene, tl, arm, ik, objs, hud, use_ik=not args.no_ik)
    overlay = setup_overlay_scene(scene, c_har, args, tl)
    setup_render(scene, overlay, args, tl)
    scene.camera = cams["hero" if args.camera == "all" else args.camera]
    sync_overlay_camera(overlay, scene.camera)
    log(f"Total frames: {tl.last}")

    out = Path(args.out)
    out_abs = out if out.is_absolute() else Path.cwd() / out
    out_abs.mkdir(parents=True, exist_ok=True)
    blend = out / "har_showcase.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out_abs / "har_showcase.blend"), check_existing=False)
    log(f".blend saved: {blend.as_posix()}")
    if args.build_only:
        log("SHOWCASE_DONE")
        return

    names = list(CAMERAS) if args.camera == "all" else [args.camera]
    step = max(1, tl.last // 8)

    def progress(scn, *_):
        if scn.frame_current % step == 0 or scn.frame_current == tl.last:
            log(f"  frame {scn.frame_current}/{tl.last}")

    bpy.app.handlers.render_write.append(progress)
    encodes = []
    failed = False
    for name in names:
        rel = out / "frames" / name if len(names) > 1 else out / "frames"
        log(f"Rendering [{name}] {tl.last} frames …")
        ok = render_camera(scene, overlay, cams[name], out_abs / rel.relative_to(out), tl.last)
        if ok == tl.last:
            log(f"FRAMES_OK {ok} in {rel.as_posix()}")
        else:
            log(f"FRAMES_MISSING {tl.last - ok} of {tl.last} in {rel.as_posix()}")
            failed = True
        mp4 = f"{out.as_posix()}.mp4" if len(names) == 1 else f"{out.as_posix()}_{name}.mp4"
        encodes.append(f"python {encode_helper()} {rel.as_posix()} {mp4} --fps {args.fps}")
    bpy.app.handlers.render_write.remove(progress)
    log("SHOWCASE_DONE")
    for cmd in encodes:
        log(f"Encode: {cmd}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
