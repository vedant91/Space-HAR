"""
Bharatiya Antariksh Station science-module interior and payload racks.

Geometry follows real International Standard Payload Rack (ISPR) dimensions
- 2.013 m tall, 1.046 m wide, 0.858 m deep - because those proportions are
what make a rendered rack read as spaceflight hardware rather than a
generic locker. The module is a 2.1 m-radius cylinder with rack bays on
*four* sides: in microgravity every wall is a working surface, and a scene
with a single floor-mounted bench would quietly reintroduce the gravity
prior this whole project exists to remove.

World frame convention used by every other module in this package:
    +X  along the module axis (down the tunnel)
    +Y  into the experiment rack face
    +Z  "up" relative to the rack, i.e. the rack's own long axis

The experiment rack's front face therefore lies in the XZ plane at y = 0,
payload sits at y > 0 on the work surface, and the crew member works from
y < 0. The camera looks along +Y. Nothing in the scene assumes -Z is down.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import bpy
from mathutils import Vector

from . import common as C

# ── Real ISPR dimensions (metres) ────────────────────────────────────────────
RACK_W = 1.046
RACK_H = 2.013
RACK_D = 0.858

MODULE_RADIUS = 2.10
MODULE_LENGTH = 6.40

# The four rack faces enclose a square working cavity. Its centre line *is*
# the module axis, and every rack is the same local build rotated about it -
# that is what makes "there is no floor" true structurally rather than just
# asserted in a comment. The experiment rack is the theta = 0 bay, so the
# axis sits one cavity-half-width behind its face (which lies at y = 0) and
# at its mid-height.
CAVITY_HALF = 1.03
AXIS_Y = -CAVITY_HALF
AXIS_Z = RACK_H * 0.5

# The rack face is the plane y = 0. Structure lives BEHIND it (y > 0) and
# everything the crew touches lives IN FRONT of it (y < 0). Keeping that sign
# convention strict is what stops the payload from ending up embedded inside
# the rack shell, and it is why the shell starts at +SHELL_INSET rather than
# at zero: the drawer fronts, panels and grilles occupy the millimetres
# either side of y = 0 and have to stand proud of the shell to read as
# separate hardware rather than as a flat slab.
SHELL_INSET = 0.02

# Work volume in front of the experiment rack
WORK_SURFACE_Z = 1.02          # top face of the work shelf
WORK_SURFACE_DEPTH = 0.34
WORK_SURFACE_Y = -0.19         # centre of the shelf, forward of the face
CONTAINER_SIZE = (0.30, 0.24, 0.17)
INNER_BOX_SIZE = (0.105, 0.095, 0.085)

# Restraint zones the protocol places boxes into
LEFT_ZONE = Vector((-0.40, WORK_SURFACE_Y, WORK_SURFACE_Z + 0.058))
RIGHT_ZONE = Vector((0.40, WORK_SURFACE_Y, WORK_SURFACE_Z + 0.058))
CONTAINER_ORIGIN = Vector((0.0, WORK_SURFACE_Y,
                           WORK_SURFACE_Z + 0.013 + CONTAINER_SIZE[2] * 0.5))


# ── Small detail primitives ──────────────────────────────────────────────────


def _handrail(name: str, length: float, location, rotation, mats, coll):
    """ISS-pattern handrail: a tube standing off the surface on two feet.

    The standoff matters visually - a rail flush against a panel reads as a
    moulding, whereas the shadow gap under a real handrail is instantly
    recognisable.
    """
    parts = []
    tube = C.cylinder(f"{name}_tube", radius=0.0165, depth=length, segments=20,
                      rotation=(0.0, math.pi / 2.0, 0.0), collection=coll)
    C.assign_material(tube, mats["handrail"])
    parts.append(tube)
    for sx in (-1.0, 1.0):
        foot = C.box(f"{name}_foot", (0.030, 0.048, 0.062),
                     location=(sx * (length * 0.5 - 0.035), 0.0, -0.036),
                     collection=coll)
        C.add_bevel(foot, 0.003, 2)
        C.assign_material(foot, mats["rack_frame"])
        parts.append(foot)
    rail = C.join(parts, name)
    rail.location = Vector(location)
    rail.rotation_euler = rotation
    C.shade_smooth(rail, 34.0)
    return rail


def _louvre_grille(name: str, width: float, height: float, location, mats, coll,
                   blades: int = 9):
    """Avionics cooling grille - angled blades in a recessed frame."""
    parts = []
    frame = C.box(f"{name}_frame", (width, 0.016, height), collection=coll)
    C.add_bevel(frame, 0.002, 2)
    C.assign_material(frame, mats["rack_frame"])
    parts.append(frame)
    inner = C.box(f"{name}_back", (width - 0.018, 0.008, height - 0.018),
                  location=(0.0, 0.006, 0.0), collection=coll)
    C.assign_material(inner, mats["seal_black"])
    parts.append(inner)
    pitch = (height - 0.026) / blades
    for i in range(blades):
        z = -height * 0.5 + 0.013 + pitch * (i + 0.5)
        blade = C.box(f"{name}_b{i}", (width - 0.024, 0.010, pitch * 0.62),
                      location=(0.0, -0.004, z), rotation=(math.radians(28.0), 0.0, 0.0),
                      collection=coll)
        C.assign_material(blade, mats["rack_panel"])
        parts.append(blade)
    grille = C.join(parts, name)
    grille.location = Vector(location)
    return grille


def _cam_latch(name: str, location, mats, coll):
    """Captive quarter-turn fastener - the small repeated detail that makes a
    panel read as removable flight hardware."""
    body = C.cylinder(f"{name}_body", 0.0135, 0.012, segments=16,
                      rotation=(math.pi / 2.0, 0.0, 0.0), collection=coll)
    C.assign_material(body, mats["latch"])
    slot = C.box(f"{name}_slot", (0.019, 0.006, 0.0035),
                 location=(0.0, -0.006, 0.0), collection=coll)
    C.assign_material(slot, mats["seal_black"])
    latch = C.join([body, slot], name)
    latch.location = Vector(location)
    C.shade_smooth(latch, 30.0)
    return latch


def _drawer_face(name: str, width: float, height: float, location, mats, coll,
                 with_label: bool = True):
    """A rack drawer front: recessed pull, corner fasteners, ID placard."""
    parts = []
    face = C.box(f"{name}_face", (width, 0.030, height), collection=coll)
    C.add_bevel(face, 0.0035, 3)
    C.assign_material(face, mats["rack_face"])
    parts.append(face)

    pull = C.box(f"{name}_pull", (width * 0.30, 0.026, 0.030),
                 location=(0.0, -0.016, -height * 0.5 + 0.055), collection=coll)
    C.add_bevel(pull, 0.004, 3)
    C.assign_material(pull, mats["rack_panel"])
    parts.append(pull)
    pull_shadow = C.box(f"{name}_pullrec", (width * 0.34, 0.012, 0.042),
                        location=(0.0, -0.004, -height * 0.5 + 0.055), collection=coll)
    C.assign_material(pull_shadow, mats["seal_black"])
    parts.append(pull_shadow)

    if with_label:
        label = C.box(f"{name}_label", (width * 0.42, 0.004, 0.036),
                      location=(0.0, -0.016, height * 0.5 - 0.048), collection=coll)
        C.add_bevel(label, 0.001, 2)
        C.assign_material(label, mats["placard"])
        parts.append(label)

    drawer = C.join(parts, name)
    drawer.location = Vector(location)
    C.shade_smooth(drawer, 36.0)
    return drawer


# ── Racks ────────────────────────────────────────────────────────────────────


def build_rack(name: str, mats: Dict, coll, rng: random.Random,
               kind: str = "storage") -> bpy.types.Object:
    """One ISPR bay, built at the origin with its face in the XZ plane (y=0)
    and its body extending to +Y. Callers place/rotate it into a bay.
    """
    parts: List[bpy.types.Object] = []

    # Structural shell, set back so face hardware stands proud of it --------
    shell = C.box(f"{name}_shell", (RACK_W, RACK_D, RACK_H),
                  location=(0.0, SHELL_INSET + RACK_D * 0.5, RACK_H * 0.5),
                  collection=coll)
    C.add_bevel(shell, 0.006, 3)
    C.assign_material(shell, mats["rack_panel"])
    parts.append(shell)

    # Front frame rails ------------------------------------------------------
    for sx in (-1.0, 1.0):
        post = C.box(f"{name}_post", (0.042, 0.052, RACK_H),
                     location=(sx * (RACK_W * 0.5 - 0.021), 0.010, RACK_H * 0.5),
                     collection=coll)
        C.add_bevel(post, 0.004, 3)
        C.assign_material(post, mats["rack_frame"])
        parts.append(post)
    for z in (0.026, RACK_H - 0.026):
        rail = C.box(f"{name}_rail", (RACK_W, 0.052, 0.052),
                     location=(0.0, 0.010, z), collection=coll)
        C.add_bevel(rail, 0.004, 3)
        C.assign_material(rail, mats["rack_frame"])
        parts.append(rail)

    inner_w = RACK_W - 0.10

    if kind == "storage":
        # Stack of drawers of varying height - uniform drawers look like a
        # texture, mixed heights look like equipment.
        heights = [0.30, 0.22, 0.22, 0.30, 0.26, 0.34]
        rng.shuffle(heights)
        z = 0.075
        idx = 0
        for h in heights:
            if z + h > RACK_H - 0.075:
                break
            parts.append(_drawer_face(f"{name}_dr{idx}", inner_w, h - 0.012,
                                      (0.0, -0.001, z + h * 0.5), mats, coll,
                                      with_label=rng.random() < 0.75))
            for sx in (-1.0, 1.0):
                parts.append(_cam_latch(
                    f"{name}_l{idx}{sx}",
                    (sx * (inner_w * 0.5 - 0.022), -0.016, z + h * 0.5), mats, coll))
            z += h
            idx += 1

    elif kind == "avionics":
        parts.append(_louvre_grille(f"{name}_g0", inner_w, 0.46,
                                    (0.0, 0.004, 0.42), mats, coll))
        parts.append(_louvre_grille(f"{name}_g1", inner_w, 0.34,
                                    (0.0, 0.004, 1.62), mats, coll))
        panel = C.box(f"{name}_panel", (inner_w, 0.028, 0.60),
                      location=(0.0, -0.001, 1.05), collection=coll)
        C.add_bevel(panel, 0.003, 3)
        C.assign_material(panel, mats["rack_face"])
        parts.append(panel)
        screen = C.box(f"{name}_screen", (0.30, 0.006, 0.20),
                       location=(-0.16, -0.018, 1.16), collection=coll)
        C.assign_material(screen, mats["screen"])
        parts.append(screen)
        for i in range(6):
            led = C.cylinder(f"{name}_led{i}", 0.006, 0.006, segments=10,
                             location=(0.22 + (i % 3) * 0.035, -0.020,
                                       1.22 - (i // 3) * 0.045),
                             rotation=(math.pi / 2.0, 0.0, 0.0), collection=coll)
            C.assign_material(led, mats["indicator_green"])
            parts.append(led)

    else:  # "soft" - cargo transfer bags behind a mesh restraint
        for i in range(3):
            bag = C.box(f"{name}_ctb{i}", (inner_w * 0.9, 0.30, 0.42),
                        location=(0.0, 0.20, 0.30 + i * 0.60), collection=coll)
            C.add_bevel(bag, 0.018, 3)
            C.assign_material(bag, mats["soft_goods"])
            parts.append(bag)
        for i in range(7):
            strap = C.box(f"{name}_strap{i}", (inner_w, 0.006, 0.020),
                          location=(0.0, 0.030, 0.14 + i * 0.28), collection=coll)
            C.assign_material(strap, mats["velcro"])
            parts.append(strap)

    rack = C.join(parts, name)
    C.shade_smooth(rack, 36.0)
    return rack


def build_experiment_rack(mats: Dict, coll, rng: random.Random) -> Dict:
    """The instrumented rack the protocol is performed at.

    Returns the rack object plus the named empties the camera rig and the
    protocol animation anchor to, so no other module has to re-derive these
    positions from magic numbers.
    """
    parts: List[bpy.types.Object] = []

    shell = C.box("exp_shell", (RACK_W, RACK_D, RACK_H),
                  location=(0.0, SHELL_INSET + RACK_D * 0.5, RACK_H * 0.5), collection=coll)
    C.add_bevel(shell, 0.006, 3)
    C.assign_material(shell, mats["rack_panel"])
    parts.append(shell)

    for sx in (-1.0, 1.0):
        post = C.box("exp_post", (0.042, 0.052, RACK_H),
                     location=(sx * (RACK_W * 0.5 - 0.021), 0.010, RACK_H * 0.5),
                     collection=coll)
        C.add_bevel(post, 0.004, 3)
        C.assign_material(post, mats["rack_frame"])
        parts.append(post)
    for z in (0.026, RACK_H - 0.026):
        rail = C.box("exp_rail", (RACK_W, 0.052, 0.052),
                     location=(0.0, 0.010, z), collection=coll)
        C.add_bevel(rail, 0.004, 3)
        C.assign_material(rail, mats["rack_frame"])
        parts.append(rail)

    # Work surface the protocol happens on --------------------------------
    surface = C.box("exp_surface", (RACK_W - 0.06, WORK_SURFACE_DEPTH, 0.026),
                    location=(0.0, WORK_SURFACE_Y, WORK_SURFACE_Z), collection=coll)
    C.add_bevel(surface, 0.004, 3)
    C.assign_material(surface, mats["rack_face"])
    parts.append(surface)
    for sx in (-1.0, 1.0):
        gusset = C.box("exp_gusset", (0.020, 0.30, 0.10),
                       location=(sx * (RACK_W * 0.5 - 0.055), WORK_SURFACE_Y,
                                 WORK_SURFACE_Z - 0.062), collection=coll)
        C.add_bevel(gusset, 0.003, 2)
        C.assign_material(gusset, mats["rack_frame"])
        parts.append(gusset)

    # Restraint zones - shallow recessed trays with a raised lip. These are
    # what "place the box in the designated zone" actually means physically.
    for tag, zone in (("left", LEFT_ZONE), ("right", RIGHT_ZONE)):
        tray = C.box(f"exp_zone_{tag}", (0.19, 0.17, 0.012),
                     location=(zone.x, zone.y, WORK_SURFACE_Z + 0.019),
                     collection=coll)
        C.add_bevel(tray, 0.003, 2)
        C.assign_material(tray, mats["rack_frame"])
        parts.append(tray)
        for sx in (-1.0, 1.0):
            lip = C.box(f"exp_lip_{tag}", (0.008, 0.17, 0.026),
                        location=(zone.x + sx * 0.095, zone.y,
                                  WORK_SURFACE_Z + 0.032), collection=coll)
            C.add_bevel(lip, 0.002, 2)
            C.assign_material(lip, mats["rack_frame"])
            parts.append(lip)
        marker = C.box(f"exp_zonelabel_{tag}", (0.11, 0.004, 0.024),
                       location=(zone.x, zone.y - 0.088, WORK_SURFACE_Z + 0.034),
                       collection=coll)
        C.assign_material(marker, mats["placard"])
        parts.append(marker)

    # Upper instrument panel ------------------------------------------------
    panel = C.box("exp_panel", (RACK_W - 0.10, 0.030, 0.44),
                  location=(0.0, -0.001, 1.62), collection=coll)
    C.add_bevel(panel, 0.004, 3)
    C.assign_material(panel, mats["rack_face"])
    parts.append(panel)
    screen = C.box("exp_screen", (0.34, 0.006, 0.22),
                   location=(-0.14, -0.019, 1.66), collection=coll)
    C.assign_material(screen, mats["screen"])
    parts.append(screen)
    for i in range(4):
        led = C.cylinder(f"exp_led{i}", 0.0055, 0.006, segments=10,
                         location=(0.22, -0.021, 1.74 - i * 0.042),
                         rotation=(math.pi / 2.0, 0.0, 0.0), collection=coll)
        C.assign_material(led, mats["indicator_green"])
        parts.append(led)

    parts.append(_louvre_grille("exp_grille", RACK_W - 0.10, 0.30,
                                (0.0, 0.004, 0.30), mats, coll))

    # Lower stowage drawers --------------------------------------------------
    for i, z in enumerate((0.62, 0.84)):
        parts.append(_drawer_face(f"exp_dr{i}", RACK_W - 0.10, 0.20,
                                  (0.0, -0.001, z), mats, coll))

    # Cable run + clamps -----------------------------------------------------
    for i in range(4):
        clamp = C.box(f"exp_clamp{i}", (0.022, 0.020, 0.014),
                      location=(RACK_W * 0.5 - 0.03, 0.004, 0.36 + i * 0.34),
                      collection=coll)
        C.assign_material(clamp, mats["rack_frame"])
        parts.append(clamp)
    cable = C.cylinder("exp_cable", 0.009, 1.24, segments=12,
                       location=(RACK_W * 0.5 - 0.03, 0.006, 0.87),
                       collection=coll)
    C.assign_material(cable, mats["cable"])
    parts.append(cable)

    if "cable_tie_warm" in mats:
        for i in range(3):
            tie = C.box(f"exp_tie{i}", (0.014, 0.012, 0.006),
                        location=(RACK_W * 0.5 - 0.03, -0.004, 0.52 + i * 0.40),
                        collection=coll)
            C.assign_material(tie, mats["cable_tie_warm"])
            parts.append(tie)
    if "caution_placard" in mats:
        plac = C.box("exp_caution", (0.075, 0.004, 0.030),
                     location=(-RACK_W * 0.5 + 0.10, -0.006, 1.40), collection=coll)
        C.assign_material(plac, mats["caution_placard"])
        parts.append(plac)

    rack = C.join(parts, "experiment_rack")
    C.shade_smooth(rack, 36.0)

    # Handrails flanking the bay - the crew member's restraint points -------
    rails = [
        _handrail("exp_rail_l", 0.53, (-0.60, -0.24, 1.32),
                  (0.0, math.pi / 2.0, 0.0), mats, coll),
        _handrail("exp_rail_r", 0.53, (0.60, -0.24, 1.32),
                  (0.0, math.pi / 2.0, 0.0), mats, coll),
        _handrail("exp_rail_b", 0.72, (0.0, -0.03, 0.16), (0.0, 0.0, 0.0), mats, coll),
    ]

    return {"rack": rack, "handrails": rails}


# ── Payload the protocol manipulates ─────────────────────────────────────────


def build_payload(mats: Dict, coll) -> Dict:
    """Main container (separate hinged lid) plus the red and yellow boxes.

    Each returned object has its origin at its own centre so protocol.py can
    keyframe location/rotation directly without compensating for an offset
    pivot - except the lid, whose origin is deliberately moved to its hinge
    line so opening it is a single rotation keyframe.
    """
    w, d, h = CONTAINER_SIZE

    body = C.box("main_box_body", (w, d, h), location=CONTAINER_ORIGIN, collection=coll)
    C.add_bevel(body, 0.005, 3)
    C.assign_material(body, mats["container_white"])
    C.shade_smooth(body, 34.0)

    # Lid: modelled around the origin, then the mesh is shifted so the object
    # origin sits on the rear hinge axis.
    lid_h = 0.028
    lid = C.box("main_box_lid", (w + 0.006, d + 0.006, lid_h), collection=coll)
    C.add_bevel(lid, 0.004, 3)
    C.assign_material(lid, mats["container_shell"])
    C.shade_smooth(lid, 34.0)
    for vert in lid.data.vertices:
        vert.co.y -= d * 0.5
        vert.co.z -= lid_h * 0.5
    lid.location = (CONTAINER_ORIGIN.x,
                    CONTAINER_ORIGIN.y + d * 0.5,
                    CONTAINER_ORIGIN.z + h * 0.5 + lid_h * 0.5)

    iw, idp, ih = INNER_BOX_SIZE
    red = C.box("red_box", (iw, idp, ih), collection=coll)
    C.add_bevel(red, 0.004, 3)
    C.assign_material(red, mats["payload_red"])
    C.shade_smooth(red, 34.0)
    red.location = (CONTAINER_ORIGIN.x - 0.062, CONTAINER_ORIGIN.y,
                    CONTAINER_ORIGIN.z - 0.012)

    yellow = C.box("yellow_box", (iw, idp, ih), collection=coll)
    C.add_bevel(yellow, 0.004, 3)
    C.assign_material(yellow, mats["payload_yellow"])
    C.shade_smooth(yellow, 34.0)
    yellow.location = (CONTAINER_ORIGIN.x + 0.062, CONTAINER_ORIGIN.y,
                       CONTAINER_ORIGIN.z - 0.012)

    return {"container": body, "lid": lid, "red": red, "yellow": yellow}


# ── Module shell ─────────────────────────────────────────────────────────────


def build_module(mats: Dict, coll, rng: random.Random) -> Dict:
    """Cylindrical pressure shell with rack bays on four sides.

    The experiment rack occupies the -Y bay (its face at y=0 looking toward
    -Y); the other three bays are dressed with storage / avionics / soft-goods
    racks so the background is real hardware rather than empty wall. Those
    three are what a HAR model must learn to ignore.
    """
    created: Dict[str, object] = {}
    axis = Vector((0.0, AXIS_Y, AXIS_Z))

    # Pressure shell - an open-ended tube, normals flipped inward.
    shell = C.cylinder("module_shell", MODULE_RADIUS, MODULE_LENGTH, segments=64,
                       rotation=(0.0, math.pi / 2.0, 0.0),
                       location=axis, cap=False, collection=coll)
    shell.data.flip_normals()
    C.assign_material(shell, mats["module_wall"])
    C.shade_smooth(shell, 60.0)
    created["shell"] = shell

    # Circumferential standoff ribs - the repeated structure that gives the
    # tunnel depth cues and keeps the background from looking like a backdrop.
    ribs = []
    for i in range(7):
        x = -MODULE_LENGTH * 0.5 + 0.45 + i * 0.90
        rib = C.torus(f"module_rib{i}", MODULE_RADIUS - 0.03, 0.035,
                      major_segments=56, minor_segments=10,
                      location=(x, AXIS_Y, AXIS_Z), rotation=(0.0, math.pi / 2.0, 0.0),
                      collection=coll)
        C.assign_material(rib, mats["rack_frame"])
        C.shade_smooth(rib, 50.0)
        ribs.append(rib)
    created["ribs"] = C.join(ribs, "module_ribs")

    # The experiment bay (theta = 0) ----------------------------------------
    exp = build_experiment_rack(mats, coll, rng)
    created["experiment"] = exp["rack"]
    created["handrails"] = exp["handrails"]

    # Three dressing bays at 90/180/270 degrees about the module axis.
    # Each is the identical local build (face in the XZ plane at y=0, body
    # toward +Y, base at z=0) rotated about the axis: parent empty sits on the
    # axis carrying the roll, child sits at the axis-relative offset, so at
    # theta=0 the child lands exactly where the experiment rack is.
    # The experiment bay's own wall gets flanking racks too. Without them the
    # +Y wall is a 1 m rack floating in a 6 m gap, which reads as a set rather
    # than a module - and, more importantly for the dataset, it leaves the
    # background either empty or showing bare hull, so a frame classifier
    # could separate the steps on background alone.
    bay_specs = [
        ("bay_fwd", "storage", 0.0),
        ("bay_port", "storage", math.pi * 0.5),
        ("bay_deck", "avionics", math.pi),
        ("bay_stbd", "soft", math.pi * 1.5),
    ]
    dressing = []
    for name, kind, roll in bay_specs:
        pivot = bpy.data.objects.new(f"{name}_pivot", None)
        C.link(pivot, coll)
        pivot.location = axis
        pivot.rotation_euler = (roll, 0.0, 0.0)
        for j, x_off in enumerate((-1.12, 1.12)):
            kind_here = kind
            if name == "bay_fwd":
                # Vary them so the row isn't two identical copies flanking the
                # experiment rack.
                kind_here = "avionics" if j == 0 else "soft"
            rack = build_rack(f"{name}{j}", mats, coll, rng, kind=kind_here)
            rack.parent = pivot
            rack.location = (x_off, -AXIS_Y, -AXIS_Z)
            dressing.append(rack)
    created["dressing"] = dressing

    # Endcone hatch at the far end - a strong depth anchor down the tunnel.
    far_x = MODULE_LENGTH * 0.5
    hatch_ring = C.torus("hatch_ring", 0.52, 0.075, 48, 12,
                         location=(far_x - 0.05, AXIS_Y, AXIS_Z),
                         rotation=(0.0, math.pi / 2.0, 0.0), collection=coll)
    C.assign_material(hatch_ring, mats["rack_frame"])
    C.shade_smooth(hatch_ring, 50.0)
    hatch = C.cylinder("hatch_plate", 0.50, 0.05, segments=48,
                       location=(far_x - 0.02, AXIS_Y, AXIS_Z),
                       rotation=(0.0, math.pi / 2.0, 0.0), collection=coll)
    C.assign_material(hatch, mats["rack_face"])
    C.shade_smooth(hatch, 40.0)
    endcap = C.cylinder("module_endcap", MODULE_RADIUS, 0.06, segments=64,
                        location=(far_x, AXIS_Y, AXIS_Z),
                        rotation=(0.0, math.pi / 2.0, 0.0), collection=coll)
    C.assign_material(endcap, mats["module_wall"])
    created["endcone"] = [hatch_ring, hatch, endcap]

    return created


def build_lighting(mats: Dict, coll, rng: random.Random,
                   variant: str = "nominal") -> List[bpy.types.Object]:
    """General module illumination.

    Three named variants exist so the dataset covers the lighting the PS
    actually implies: routine ops, a dimmed sleep-cycle period, and a
    single-luminaire-failed case. A HAR model trained under one fixed light
    rig learns that rig, not the activity.
    """
    lights: List[bpy.types.Object] = []

    # Energies are set so that nothing in the nominal scene clips to pure
    # white. That matters more here than it would in a normal render: the
    # dataset is graded with the Standard view transform (no filmic highlight
    # roll-off, see render.py), so an over-driven key does not compress - it
    # hard-clips, and a clipped highlight on the red or yellow box loses the
    # saturation the HSV detector keys on. Bright-but-unclipped is the target.
    if variant == "dim":
        strip_energy, fill_energy, panel_energy = 11.0, 4.0, 7.0
    elif variant == "single_fail":
        strip_energy, fill_energy, panel_energy = 26.0, 2.5, 15.0
    else:
        strip_energy, fill_energy, panel_energy = 32.0, 9.0, 17.0

    # Luminaire strips in the cavity corners, running along the module axis
    # and aimed back at the axis. Placed at +-52 degrees off the experiment
    # bay so the work volume gets two-sided key light without either strip
    # sitting in the payload camera's frame.
    axis = Vector((0.0, AXIS_Y, AXIS_Z))
    for i, roll in enumerate((math.radians(52.0), math.radians(-52.0))):
        if variant == "single_fail" and i == 1:
            continue
        data = bpy.data.lights.new(f"strip_{i}", type="AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = 0.16, 3.60
        data.energy = strip_energy
        data.color = (0.94, 0.96, 1.0)
        lamp = bpy.data.objects.new(f"strip_{i}", data)
        C.link(lamp, coll)
        r = CAVITY_HALF - 0.10
        lamp.location = axis + Vector((0.0, math.cos(roll) * r, math.sin(roll) * r))
        C.look_at(lamp, Vector((0.0, -0.55, 1.05)))
        lights.append(lamp)

    # Task light on the experiment rack - a practical, so the work volume has
    # its own directional key independent of the module ambient.
    data = bpy.data.lights.new("task_light", type="AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = 0.60, 0.10
    data.energy = panel_energy
    data.color = (1.0, 0.98, 0.94)
    task = bpy.data.objects.new("task_light", data)
    C.link(task, coll)
    task.location = (0.0, -0.30, 1.88)
    C.look_at(task, Vector((0.0, WORK_SURFACE_Y, WORK_SURFACE_Z)))
    lights.append(task)

    # Broad bounce fill from the opposite (deck) bay, standing in for the
    # light the real module's pale interior surfaces would kick back.
    data = bpy.data.lights.new("fill", type="AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = 2.4, 2.0
    data.energy = fill_energy
    data.color = (0.88, 0.92, 1.0)
    fill = bpy.data.objects.new("fill", data)
    C.link(fill, coll)
    fill.location = (0.0, AXIS_Y - CAVITY_HALF + 0.12, AXIS_Z)
    C.look_at(fill, Vector((0.0, -0.45, WORK_SURFACE_Z + 0.15)))
    lights.append(fill)

    # A dark, very slightly blue world keeps unlit crevices from going pure
    # black without washing out the directional lighting.
    world = bpy.data.worlds.new("module_world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.035, 0.040, 0.050, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world

    return lights
