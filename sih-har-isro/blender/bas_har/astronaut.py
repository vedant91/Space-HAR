"""
Suited crew member: segmented pressure-suit mesh on a deforming armature.

Design notes
------------
The suit is built as *rigid segments* rather than a smooth-skinned body,
because that is what a real pressure garment is: hard upper torso, hard
helmet, and soft limb sections separated by convolute (accordion) rings at
every joint. Modelling it that way is both more accurate and more robust -
each segment is weighted 100% to one bone, so there is no weight-painting to
go wrong, no candy-wrapper twist at the elbows, and the joint geometry stays
watertight at any bend angle because overlapping capsule caps plus the
convolute rings physically cover the seam.

The convolute rings are the single detail that makes the figure read as a
spacesuit rather than a mannequin, so they are not optional dressing.

Local frame (before the object is placed in the module):
    origin  = pelvis
    +Z      = head-ward
    +Y      = the direction the crew member faces
    +X      = the crew member's own RIGHT

That last line matters: MediaPipe's landmark names are anatomical, so
`groundtruth.py` maps bones on -X to the *left* landmark indices. Getting
this backwards silently mirrors every exported label.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Euler, Matrix, Vector

from . import common as C

# ── Skeleton definition (metres, pelvis-relative) ────────────────────────────
# A 1.71 m crew member. Segment lengths are anthropometric medians, not
# guesses: upper arm 0.28, forearm 0.26, hand 0.09, thigh 0.44, shin 0.42.

SHOULDER_X = 0.175
HIP_X = 0.095

BONES: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], Optional[str]]] = {
    "pelvis":      ((0.0, 0.0, 0.0),          (0.0, 0.0, 0.16),          None),
    "spine":       ((0.0, 0.0, 0.16),         (0.0, 0.0, 0.32),          "pelvis"),
    "chest":       ((0.0, 0.0, 0.32),         (0.0, 0.0, 0.47),          "spine"),
    "neck":        ((0.0, 0.0, 0.47),         (0.0, 0.0, 0.55),          "chest"),
    "head":        ((0.0, 0.0, 0.55),         (0.0, 0.0, 0.78),          "neck"),

    "clavicle.L":  ((0.0, 0.0, 0.44),         (-SHOULDER_X, 0.0, 0.455), "chest"),
    "upperarm.L":  ((-SHOULDER_X, 0.0, 0.455), (-0.190, 0.030, 0.180),   "clavicle.L"),
    "forearm.L":   ((-0.190, 0.030, 0.180),   (-0.196, 0.105, -0.070),   "upperarm.L"),
    "hand.L":      ((-0.196, 0.105, -0.070),  (-0.198, 0.140, -0.155),   "forearm.L"),

    "clavicle.R":  ((0.0, 0.0, 0.44),         (SHOULDER_X, 0.0, 0.455),  "chest"),
    "upperarm.R":  ((SHOULDER_X, 0.0, 0.455), (0.190, 0.030, 0.180),     "clavicle.R"),
    "forearm.R":   ((0.190, 0.030, 0.180),    (0.196, 0.105, -0.070),    "upperarm.R"),
    "hand.R":      ((0.196, 0.105, -0.070),   (0.198, 0.140, -0.155),    "forearm.R"),

    "thigh.L":     ((-HIP_X, 0.0, -0.020),    (-0.100, 0.045, -0.455),   "pelvis"),
    "shin.L":      ((-0.100, 0.045, -0.455),  (-0.105, 0.010, -0.875),   "thigh.L"),
    "foot.L":      ((-0.105, 0.010, -0.875),  (-0.105, 0.135, -0.925),   "shin.L"),

    "thigh.R":     ((HIP_X, 0.0, -0.020),     (0.100, 0.045, -0.455),    "pelvis"),
    "shin.R":      ((0.100, 0.045, -0.455),   (0.105, 0.010, -0.875),    "thigh.R"),
    "foot.R":      ((0.105, 0.010, -0.875),   (0.105, 0.135, -0.925),    "shin.R"),
}

# Bones whose orientation IK should not be allowed to spin freely.
IK_CHAINS = {"hand.L": ("forearm.L", 2), "hand.R": ("forearm.R", 2)}


def _seg(a: str) -> Tuple[Vector, Vector]:
    head, tail, _ = BONES[a]
    return Vector(head), Vector(tail)


# ── Suit segment builders ────────────────────────────────────────────────────


def _limb_segment(name: str, bone: str, radius_head: float, radius_tail: float,
                  mats: Dict, coll, material: str = "suit_fabric",
                  rings: int = 0, ring_at_head: bool = True) -> List[Tuple[bpy.types.Object, str]]:
    """A tapered soft-goods limb section aligned to `bone`, plus optional
    convolute rings clustered at the joint end."""
    head, tail = _seg(bone)
    axis = tail - head
    length = axis.length
    mid = (head + tail) * 0.5

    body = C.capsule(f"{name}", radius=max(radius_head, radius_tail), length=length * 0.86,
                     segments=20, rings=6, collection=coll)
    # Taper along the local Z before orienting: scaling the object afterwards
    # would also scale the convolute rings' apparent thickness.
    ratio = radius_tail / max(radius_head, 1e-5)
    for vert in body.data.vertices:
        t = (vert.co.z / max(length * 0.5, 1e-5)) * 0.5 + 0.5
        s = (1.0 - t) + t * ratio
        vert.co.x *= s
        vert.co.y *= s
    quat = axis.normalized().to_track_quat("Z", "Y")
    body.rotation_mode = "QUATERNION"
    body.rotation_quaternion = quat
    body.location = mid
    C.assign_material(body, mats[material])
    C.shade_smooth(body, 45.0)
    out = [(body, bone)]

    for i in range(rings):
        frac = 0.10 + i * 0.075
        pos = head + axis * frac if ring_at_head else tail - axis * frac
        r = radius_head if ring_at_head else radius_tail
        ring = C.torus(f"{name}_cvl{i}", major_radius=r * 1.02, minor_radius=r * 0.20,
                       major_segments=22, minor_segments=8, collection=coll)
        ring.rotation_mode = "QUATERNION"
        ring.rotation_quaternion = quat
        ring.location = pos
        C.assign_material(ring, mats["suit_fabric"])
        C.shade_smooth(ring, 45.0)
        out.append((ring, bone))
    return out


def _build_torso(mats: Dict, coll) -> List[Tuple[bpy.types.Object, str]]:
    out: List[Tuple[bpy.types.Object, str]] = []

    # Hard Upper Torso - the rigid fibreglass shell the arms and helmet bolt
    # onto. Slightly barrelled and wider at the shoulders than the waist.
    hut = C.box("hut", (0.395, 0.270, 0.235), location=(0.0, 0.0, 0.385), collection=coll)
    C.add_bevel(hut, 0.030, 4)
    C.add_subsurf(hut, 2)
    C.assign_material(hut, mats["suit_hut"])
    C.shade_smooth(hut, 50.0)
    out.append((hut, "chest"))

    # Shoulder bearings - the hard rings the arm assemblies rotate in.
    for side, sx in (("L", -1.0), ("R", 1.0)):
        bearing = C.torus(f"shoulder_bearing.{side}", 0.075, 0.021, 24, 10,
                          location=(sx * SHOULDER_X, 0.0, 0.452),
                          rotation=(0.0, math.pi / 2.0, 0.0), collection=coll)
        C.assign_material(bearing, mats["neck_ring"])
        C.shade_smooth(bearing, 45.0)
        out.append((bearing, f"clavicle.{side}"))

    # Display and Control Module on the chest.
    dcm = C.box("dcm", (0.150, 0.070, 0.110), location=(0.0, 0.150, 0.395), collection=coll)
    C.add_bevel(dcm, 0.010, 3)
    C.assign_material(dcm, mats["suit_hut"])
    C.shade_smooth(dcm, 40.0)
    out.append((dcm, "chest"))
    dcm_face = C.box("dcm_face", (0.110, 0.008, 0.056),
                     location=(0.0, 0.186, 0.408), collection=coll)
    C.assign_material(dcm_face, mats["screen"])
    out.append((dcm_face, "chest"))
    for i in range(3):
        sw = C.cylinder(f"dcm_sw{i}", 0.008, 0.014, 10,
                        location=(-0.040 + i * 0.040, 0.188, 0.362),
                        rotation=(math.pi / 2.0, 0.0, 0.0), collection=coll)
        C.assign_material(sw, mats["latch"])
        C.shade_smooth(sw, 40.0)
        out.append((sw, "chest"))

    # Primary Life Support System backpack.
    plss = C.box("plss", (0.330, 0.185, 0.400), location=(0.0, -0.205, 0.360), collection=coll)
    C.add_bevel(plss, 0.024, 4)
    C.add_subsurf(plss, 1)
    C.assign_material(plss, mats["plss"])
    C.shade_smooth(plss, 45.0)
    out.append((plss, "chest"))

    # Waist / brief section.
    brief = C.box("brief", (0.320, 0.235, 0.230), location=(0.0, 0.0, 0.115), collection=coll)
    C.add_bevel(brief, 0.032, 4)
    C.add_subsurf(brief, 2)
    C.assign_material(brief, mats["suit_fabric"])
    C.shade_smooth(brief, 50.0)
    out.append((brief, "spine"))

    waist_ring = C.torus("waist_ring", 0.150, 0.020, 26, 10,
                         location=(0.0, 0.0, 0.245), collection=coll)
    C.assign_material(waist_ring, mats["neck_ring"])
    C.shade_smooth(waist_ring, 45.0)
    out.append((waist_ring, "spine"))

    # Mission identity stripe - a flash of saturated colour on an otherwise
    # white suit, which also gives the pose estimator a shoulder-level
    # contrast edge to latch onto.
    for side, sx in (("L", -1.0), ("R", 1.0)):
        stripe = C.box(f"stripe.{side}", (0.030, 0.150, 0.075),
                       location=(sx * 0.185, 0.020, 0.430), collection=coll)
        C.add_bevel(stripe, 0.006, 2)
        C.assign_material(stripe, mats["suit_accent"])
        C.shade_smooth(stripe, 40.0)
        out.append((stripe, "chest"))
    return out


def _build_head(mats: Dict, coll) -> List[Tuple[bpy.types.Object, str]]:
    out: List[Tuple[bpy.types.Object, str]] = []
    centre = Vector((0.0, 0.010, 0.655))

    # Neck ring first - it hides the helmet/HUT intersection.
    ring = C.torus("neck_ring", 0.098, 0.024, 26, 10,
                   location=(0.0, 0.0, 0.500), collection=coll)
    C.assign_material(ring, mats["neck_ring"])
    C.shade_smooth(ring, 45.0)
    out.append((ring, "neck"))

    shell = C.capsule("helmet", radius=0.135, length=0.055, segments=32, rings=12,
                      collection=coll)
    shell.location = centre
    C.assign_material(shell, mats["helmet_shell"])
    C.shade_smooth(shell, 60.0)
    out.append((shell, "head"))

    # Visor: a spherical cap carved from a slightly larger sphere, facing +Y.
    visor = C.capsule("visor", radius=0.139, length=0.045, segments=32, rings=12,
                      collection=coll)
    keep = [v for v in visor.data.vertices if v.co.y > 0.052]
    if keep:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(visor.data)
        bm.verts.ensure_lookup_table()
        doomed = [v for v in bm.verts if v.co.y <= 0.052]
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        bm.to_mesh(visor.data)
        bm.free()
    visor.location = centre
    C.assign_material(visor, mats["visor"])
    C.shade_smooth(visor, 60.0)
    out.append((visor, "head"))

    # Sun-shade brow over the visor and a helmet light on each side.
    brow = C.box("visor_brow", (0.230, 0.100, 0.045),
                 location=(0.0, 0.075, 0.760), rotation=(math.radians(-18.0), 0.0, 0.0),
                 collection=coll)
    C.add_bevel(brow, 0.010, 3)
    C.assign_material(brow, mats["helmet_shell"])
    C.shade_smooth(brow, 40.0)
    out.append((brow, "head"))
    for sx in (-1.0, 1.0):
        lamp = C.cylinder("helmet_lamp", 0.028, 0.050, 16,
                          location=(sx * 0.140, 0.060, 0.700),
                          rotation=(math.pi / 2.0, 0.0, 0.0), collection=coll)
        C.assign_material(lamp, mats["rack_frame"])
        C.shade_smooth(lamp, 40.0)
        out.append((lamp, "head"))
    return out


def _build_hand(side: str, mats: Dict, coll) -> List[Tuple[bpy.types.Object, str]]:
    """Gloved hand: palm block, a fused finger mass in a light grip curl, and
    an opposed thumb. Individual fingers are deliberately not modelled - a
    pressurised EVA glove genuinely cannot splay them, and four extra
    articulated chains would add render cost while changing no MediaPipe
    landmark (the pose model only reports wrist/pinky/index/thumb)."""
    bone = f"hand.{side}"
    head, tail = _seg(bone)
    axis = (tail - head).normalized()
    quat = axis.to_track_quat("Z", "Y")
    sx = -1.0 if side == "L" else 1.0
    out: List[Tuple[bpy.types.Object, str]] = []

    palm = C.box(f"palm.{side}", (0.052, 0.090, 0.098), collection=coll)
    C.add_bevel(palm, 0.016, 4)
    C.add_subsurf(palm, 2)
    palm.rotation_mode = "QUATERNION"
    palm.rotation_quaternion = quat
    palm.location = head + axis * 0.042
    C.assign_material(palm, mats["glove"])
    C.shade_smooth(palm, 50.0)
    out.append((palm, bone))

    fingers = C.box(f"fingers.{side}", (0.048, 0.070, 0.052), collection=coll)
    C.add_bevel(fingers, 0.020, 4)
    C.add_subsurf(fingers, 2)
    fingers.rotation_mode = "QUATERNION"
    fingers.rotation_quaternion = quat
    fingers.location = head + axis * 0.098 + Vector((0.0, 0.022, 0.0))
    C.assign_material(fingers, mats["glove"])
    C.shade_smooth(fingers, 50.0)
    out.append((fingers, bone))

    thumb = C.capsule(f"thumb.{side}", radius=0.017, length=0.042, segments=14,
                      rings=5, collection=coll)
    thumb.rotation_mode = "QUATERNION"
    thumb.rotation_quaternion = (Euler((0.0, math.radians(sx * 55.0), 0.0)).to_quaternion()
                                 @ quat)
    thumb.location = head + axis * 0.062 + Vector((sx * 0.030, 0.026, 0.0))
    C.assign_material(thumb, mats["glove"])
    C.shade_smooth(thumb, 50.0)
    out.append((thumb, bone))

    grip = C.box(f"grip.{side}", (0.040, 0.014, 0.070), collection=coll)
    C.add_bevel(grip, 0.005, 2)
    grip.rotation_mode = "QUATERNION"
    grip.rotation_quaternion = quat
    grip.location = head + axis * 0.055 + Vector((0.0, 0.046, 0.0))
    C.assign_material(grip, mats["glove_grip"])
    C.shade_smooth(grip, 40.0)
    out.append((grip, bone))

    wrist_ring = C.torus(f"wrist_ring.{side}", 0.045, 0.013, 20, 8, collection=coll)
    wrist_ring.rotation_mode = "QUATERNION"
    wrist_ring.rotation_quaternion = quat
    wrist_ring.location = head
    C.assign_material(wrist_ring, mats["neck_ring"])
    C.shade_smooth(wrist_ring, 45.0)
    out.append((wrist_ring, bone))
    return out


def _build_boot(side: str, mats: Dict, coll) -> List[Tuple[bpy.types.Object, str]]:
    bone = f"foot.{side}"
    head, tail = _seg(bone)
    sx = -1.0 if side == "L" else 1.0
    boot = C.box(f"boot.{side}", (0.105, 0.245, 0.095),
                 location=(sx * 0.105, 0.055, -0.900), collection=coll)
    C.add_bevel(boot, 0.026, 4)
    C.add_subsurf(boot, 2)
    C.assign_material(boot, mats["boot"])
    C.shade_smooth(boot, 45.0)
    cuff = C.torus(f"boot_cuff.{side}", 0.062, 0.018, 20, 8,
                   location=(sx * 0.105, 0.010, -0.845), collection=coll)
    C.assign_material(cuff, mats["suit_fabric"])
    C.shade_smooth(cuff, 45.0)
    return [(boot, bone), (cuff, bone)]


# ── Armature ─────────────────────────────────────────────────────────────────


def _build_armature(coll) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new("crew_rig")
    rig = bpy.data.objects.new("crew_rig", arm_data)
    C.link(rig, coll)

    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    for name, (head, tail, parent) in BONES.items():
        bone = edit_bones.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        bone.use_deform = True
    for name, (_h, _t, parent) in BONES.items():
        if parent:
            edit_bones[name].parent = edit_bones[parent]
            # Connected bones would force head==parent.tail; the clavicles and
            # thighs deliberately branch from a point partway along their
            # parent, so connection is left off throughout.
            edit_bones[name].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = prev_active

    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
    return rig


def _add_ik(rig: bpy.types.Object, coll) -> Dict[str, bpy.types.Object]:
    """Two-bone IK on each arm with a pole target, so protocol.py can drive a
    hand to a 3-D point on the payload instead of hand-authoring shoulder and
    elbow rotations for 8 steps x N variations."""
    targets: Dict[str, bpy.types.Object] = {}
    for side, sx in (("L", -1.0), ("R", 1.0)):
        _h, tail, _p = BONES[f"forearm.{side}"]

        goal = bpy.data.objects.new(f"ik_hand.{side}", None)
        goal.empty_display_type = "PLAIN_AXES"
        goal.empty_display_size = 0.06
        C.link(goal, coll)
        goal.location = Vector(tail)
        targets[f"hand.{side}"] = goal

        pole = bpy.data.objects.new(f"ik_pole.{side}", None)
        pole.empty_display_type = "SPHERE"
        pole.empty_display_size = 0.04
        C.link(pole, coll)
        # Behind and outboard of the elbow, so the arm bends the way a human
        # elbow does rather than inverting through the torso.
        pole.location = Vector((sx * 0.55, -0.45, 0.30))
        targets[f"pole.{side}"] = pole

        con = rig.pose.bones[f"forearm.{side}"].constraints.new("IK")
        con.target = goal
        con.pole_target = pole
        con.chain_count = 2
        con.pole_angle = math.radians(-90.0 if sx < 0 else -90.0)

        # The hand tracks the goal's own rotation, so a step can specify wrist
        # orientation (examining a box requires rotating it, not just holding
        # it at a point).
        copy_rot = rig.pose.bones[f"hand.{side}"].constraints.new("COPY_ROTATION")
        copy_rot.target = goal
        copy_rot.influence = 0.85
    return targets


# ── Assembly ─────────────────────────────────────────────────────────────────


def apply_neutral_body_posture(rig: bpy.types.Object,
                               rng: Optional["random.Random"] = None,
                               tuck: float = 1.0) -> None:
    """Pose the legs and spine into the microgravity neutral body posture.

    A relaxed human in freefall does not hang straight: with no ground
    reaction force the body settles into a a semi-foetal crouch - hips flexed
    ~40 deg, knees ~35 deg, shoulders slightly forward. NASA-STD-3001
    documents this as the neutral body posture and it is one of the most
    recognisable things about how people actually look on orbit.

    Leaving the legs hanging vertically would put a gravity pose in every
    frame of a dataset whose entire premise is that there is no gravity - and
    a HAR model would happily learn that vertical-legs cue.
    """
    import math as _m

    jitter = (lambda s: rng.gauss(0.0, s) if rng is not None else 0.0)

    pose = rig.pose.bones
    for side in ("L", "R"):
        # Hip flexion swings the thigh forward (+Y) from its rest -Z axis.
        thigh = pose.get(f"thigh.{side}")
        if thigh:
            thigh.rotation_mode = "XYZ"
            thigh.rotation_euler = (
                _m.radians(-42.0 * tuck) + jitter(0.05),
                jitter(0.04),
                _m.radians(6.0 if side == "L" else -6.0) + jitter(0.03),
            )
        shin = pose.get(f"shin.{side}")
        if shin:
            shin.rotation_mode = "XYZ"
            shin.rotation_euler = (_m.radians(38.0 * tuck) + jitter(0.05), 0.0, 0.0)
        foot = pose.get(f"foot.{side}")
        if foot:
            foot.rotation_mode = "XYZ"
            foot.rotation_euler = (_m.radians(-16.0 * tuck) + jitter(0.04), 0.0, 0.0)

    # Slight forward curl through the trunk, and the head pitched down toward
    # the work surface - a crew member looking at what their hands are doing.
    for name, rx in (("spine", -5.0), ("chest", -6.0), ("neck", 8.0), ("head", 6.0)):
        bone = pose.get(name)
        if bone:
            bone.rotation_mode = "XYZ"
            bone.rotation_euler = (_m.radians(rx * tuck) + jitter(0.02), 0.0, 0.0)


def build_astronaut(mats: Dict, coll,
                    location=(0.0, -0.50, 1.00),
                    body_roll_deg: float = 0.0,
                    facing_deg: float = 0.0,
                    rng=None,
                    leg_tuck: float = 1.0) -> Dict:
    """Build the rigged crew member and place them in the module.

    body_roll_deg rolls the whole body about the rack normal (+Y). This is
    the microgravity attitude control: at 0 the crew member is 'upright'
    relative to the rack, at 180 they are inverted, and the HAR features must
    survive both. It is the single most important dataset axis for this
    problem statement, so it is a first-class argument rather than something
    a caller has to construct by hand.
    """
    rig = _build_armature(coll)

    segments: List[Tuple[bpy.types.Object, str]] = []
    segments += _build_torso(mats, coll)
    segments += _build_head(mats, coll)

    for side in ("L", "R"):
        segments += _limb_segment(f"upperarm.{side}", f"upperarm.{side}",
                                  0.072, 0.058, mats, coll, rings=3, ring_at_head=True)
        segments += _limb_segment(f"forearm.{side}", f"forearm.{side}",
                                  0.058, 0.046, mats, coll, rings=3, ring_at_head=True)
        segments += _build_hand(side, mats, coll)
        segments += _limb_segment(f"thigh.{side}", f"thigh.{side}",
                                  0.092, 0.072, mats, coll, rings=3, ring_at_head=True)
        segments += _limb_segment(f"shin.{side}", f"shin.{side}",
                                  0.072, 0.052, mats, coll, rings=3, ring_at_head=True)
        segments += _build_boot(side, mats, coll)

    # Bake modifiers and stamp a single full-weight vertex group per segment
    # *before* joining, so the join carries the groups through.
    for obj, bone in segments:
        C.apply_all(obj)
        group = obj.vertex_groups.new(name=bone)
        group.add(range(len(obj.data.vertices)), 1.0, "REPLACE")

    body = C.join([obj for obj, _ in segments], "crew_suit")

    modifier = body.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    body.parent = rig

    # Posture before IK, so the arm goals are seeded from the posed skeleton
    # rather than from the straight-legged rest pose.
    apply_neutral_body_posture(rig, rng=rng, tuck=leg_tuck)
    bpy.context.view_layer.update()

    targets = _add_ik(rig, coll)

    # Root empty: everything (rig, mesh, IK goals and poles) hangs off this so
    # a single transform re-poses the whole crew member in the module without
    # disturbing the IK relationships.
    root = bpy.data.objects.new("crew_root", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 0.25
    C.link(root, coll)
    rig.parent = root
    for obj in targets.values():
        obj.parent = root

    root.location = Vector(location)
    root.rotation_euler = Euler((0.0,
                                 math.radians(body_roll_deg),
                                 math.radians(facing_deg)))
    bpy.context.view_layer.update()

    # The hand IK goals are deliberately re-parented out to world space, while
    # the pole targets stay on the root.
    #
    # A crew member working a rack is anchored by their hands (or a foot
    # restraint) and their torso drifts around that anchor - not the other way
    # round. If the goals rode on the root, body drift would drag the hands
    # off the payload and the arms would stay rigid, which is the single most
    # obvious way CG microgravity looks fake. Unparented goals give the
    # opposite, correct behaviour: the hands hold station on the hardware and
    # the arms visibly compensate as the body floats.
    for key, obj in targets.items():
        if key.startswith("hand."):
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world

    return {"root": root, "rig": rig, "body": body, "ik": targets, "bones": BONES}
