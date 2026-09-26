"""
ISRO BAS - Professional HAR Detection Showcase
===============================================

Single-command cinematic Blender render of the full HAR pipeline.

Imports the CC0 rigged astronaut, builds a high-fidelity BAS module,
animates the 8-step protocol, renders a glowing HAR skeleton overlay
and holographic HUD, with AgX tone-mapping, DOF and motion blur.

Run (quick draft):
    set BLENDER=D:\SIH-2026\tools\blender-5.2.0\blender-5.2.0-windows-x64\blender.exe
    %BLENDER% --background --factory-startup ^
        --python blender/render_har_showcase.py -- ^
        --out build/har_showcase --quick --save-blend

Run (full quality):
    %BLENDER% --background --factory-startup ^
        --python blender/render_har_showcase.py -- ^
        --out build/har_showcase --frames-per-step 48 --samples 48 ^
        --width 1920 --height 1080

    python tools/frames_to_video.py build/har_showcase/frames ^
        build/har_showcase.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
import bmesh
from mathutils import Euler, Vector

# Resolve repo root so imports work from --background
_HERE = Path(__file__).resolve().parent   # blender/
_ROOT = _HERE.parent                      # sih-har-isro/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from bas_har import common as C
from bas_har import materials as M
from bas_har import module as MOD
from bas_har import protocol as P
from bas_har import render as R


# ============================================================
# 1.  MediaPipe skeleton edges (33-landmark connectivity)
# ============================================================

_MP_EDGES = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
]

HAR_JOINT  = (0.00, 0.90, 1.00, 1.0)
HAR_LIMB   = (0.10, 0.40, 1.00, 1.0)
HAR_GREEN  = (0.00, 1.00, 0.60, 1.0)

STEP_NAMES = {
    1:"APPROACH PAYLOAD", 2:"OPEN LID", 3:"PICK RED BOX",
    4:"EXAMINE RED BOX",  5:"PLACE RED BOX", 6:"PICK YELLOW BOX",
    7:"EXAMINE YELLOW BOX", 8:"PLACE YELLOW BOX",
}

# ============================================================
# 2.  Material helpers
# ============================================================

def _emit(name: str, color, strength: float = 8.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out  = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value    = color
    emit.inputs["Strength"].default_value = strength
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = color
    return mat


def _holo(name: str, color, alpha: float = 0.30):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value     = color
    bsdf.inputs["Alpha"].default_value          = alpha
    bsdf.inputs["Emission Color"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = 3.0
    mat.blend_method = "BLEND"
    return mat


# ============================================================
# 3.  HAR skeleton overlay
# ============================================================

def build_har_overlay(rig: bpy.types.Object,
                      coll: bpy.types.Collection) -> Dict:
    """Glowing joint spheres + limb connectors parented to the rig."""
    from bas_har.groundtruth import LANDMARKS

    j_mat = _emit("har_joint", HAR_JOINT, 12.0)
    l_mat = _emit("har_limb",  HAR_LIMB,   6.0)

    root = bpy.data.objects.new("har_root", None)
    root.empty_display_size = 0.01
    C.link(root, coll)

    joints: List[bpy.types.Object] = []
    for idx, (bone_name, end, offset) in enumerate(LANDMARKS):
        bone = rig.pose.bones.get(bone_name)
        if bone is None:
            ph = bpy.data.objects.new(f"har_j{idx}", None)
            C.link(ph, coll)
            ph.parent = root
            joints.append(ph)
            continue

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.012)
        me = bpy.data.meshes.new(f"har_j{idx}_me")
        bm.to_mesh(me); bm.free()

        sp = bpy.data.objects.new(f"har_j{idx}", me)
        C.link(sp, coll)
        C.assign_material(sp, j_mat)
        sp.parent = rig

        con = sp.constraints.new("CHILD_OF")
        con.target    = rig
        con.subtarget = bone_name
        con.set_inverse_pending = True
        joints.append(sp)

    for a, b in _MP_EDGES:
        if a >= len(joints) or b >= len(joints):
            continue
        oa, ob = joints[a], joints[b]
        if oa.type != "MESH" or ob.type != "MESH":
            continue

        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                              segments=6, radius1=0.004, radius2=0.004, depth=1.0)
        me = bpy.data.meshes.new(f"har_l{a}_{b}_me")
        bm.to_mesh(me); bm.free()

        lmb = bpy.data.objects.new(f"har_l{a}_{b}", me)
        C.link(lmb, coll)
        C.assign_material(lmb, l_mat)
        lmb.parent = root

        dt = lmb.constraints.new("DAMPED_TRACK")
        dt.target     = ob
        dt.track_axis = "TRACK_Z"

        st = lmb.constraints.new("STRETCH_TO")
        st.target       = ob
        st.rest_length  = 0.10
        st.volume       = "NO_VOLUME"

        cp = lmb.constraints.new("COPY_LOCATION")
        cp.target = oa

    return {"root": root, "joints": joints}


# ============================================================
# 4.  Holographic HUD panel
# ============================================================

def build_hud(coll: bpy.types.Collection) -> Dict:
    """Floating 3D holographic panel: step label, confidence bar, G-status."""
    bg_mat  = _holo("hud_bg",    (0.05, 0.18, 0.35, 1.0), 0.28)
    fr_mat  = _emit("hud_frame", (0.00, 0.80, 1.00, 1.0),  4.0)
    bar_mat = _emit("hud_bar",   (0.00, 1.00, 0.40, 1.0), 10.0)
    gok_mat = _emit("hud_gok",   (0.00, 0.90, 0.30, 1.0),  8.0)
    lbl_mat = _emit("hud_lbl",   HAR_GREEN,                  6.0)

    root = bpy.data.objects.new("hud_root", None)
    root.empty_display_size = 0.05
    root.location = (-0.72, -0.55, 1.82)
    C.link(root, coll)

    # Background
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    me = bpy.data.meshes.new("hud_bg_me"); bm.to_mesh(me); bm.free()
    bg = bpy.data.objects.new("hud_bg", me)
    bg.scale = (0.22, 0.14, 1.0); bg.parent = root
    C.link(bg, coll); C.assign_material(bg, bg_mat)

    # Frame edges
    for dx, dy, sx, sy in [(0,0.135,0.44,0.004),(0,-0.135,0.44,0.004),
                            (-0.22,0,0.004,0.27),(0.22,0,0.004,0.27)]:
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
        me = bpy.data.meshes.new("hud_fr_me"); bm.to_mesh(me); bm.free()
        fr = bpy.data.objects.new("hud_fr", me)
        fr.location = (dx, dy, 0.001); fr.scale = (sx, sy, 1.0)
        fr.parent = root; C.link(fr, coll); C.assign_material(fr, fr_mat)

    # Confidence bar
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    me = bpy.data.meshes.new("hud_bar_me"); bm.to_mesh(me); bm.free()
    bar = bpy.data.objects.new("hud_bar", me)
    bar.location = (0, -0.06, 0.002); bar.scale = (0.18, 0.012, 1.0)
    bar.parent = root; C.link(bar, coll); C.assign_material(bar, bar_mat)

    # Gravity ring sphere
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=12, radius=0.024)
    me = bpy.data.meshes.new("hud_grav_me"); bm.to_mesh(me); bm.free()
    grav = bpy.data.objects.new("hud_grav", me)
    grav.location = (0.16, 0.06, 0.003); grav.parent = root
    C.link(grav, coll); C.assign_material(grav, gok_mat)

    # Header text
    cu = bpy.data.curves.new("hud_hdr_cu", type="FONT")
    cu.body = "HAR ACTIVE";  cu.size = 0.016; cu.align_x = "CENTER"
    hdr = bpy.data.objects.new("hud_hdr", cu)
    hdr.location = (0, 0.08, 0.002); hdr.parent = root
    C.link(hdr, coll); C.assign_material(hdr, lbl_mat)

    # Step text
    cu2 = bpy.data.curves.new("hud_step_cu", type="FONT")
    cu2.body = "STEP 1  APPROACH PAYLOAD"; cu2.size = 0.013; cu2.align_x = "CENTER"
    step_obj = bpy.data.objects.new("hud_step", cu2)
    step_obj.location = (0, 0.035, 0.002); step_obj.parent = root
    C.link(step_obj, coll); C.assign_material(step_obj, fr_mat)

    # G-label
    cu3 = bpy.data.curves.new("hud_grav_cu", type="FONT")
    cu3.body = "G: 0.002g  STABLE"; cu3.size = 0.011; cu3.align_x = "LEFT"
    grav_lbl = bpy.data.objects.new("hud_grav_lbl", cu3)
    grav_lbl.location = (-0.18, -0.085, 0.002); grav_lbl.parent = root
    C.link(grav_lbl, coll); C.assign_material(grav_lbl, gok_mat)

    return {"root": root, "bar": bar, "grav": grav, "step": step_obj,
            "grav_lbl": grav_lbl}


def _update_hud(hud: Dict, step: int, frame: int) -> None:
    name = STEP_NAMES.get(step, "HAR DETECTING")
    hud["step"].data.body = f"STEP {step}  {name}"
    hud["bar"].scale.x = 0.18 * (0.65 + step / 8.0 * 0.32)
    hud["bar"].keyframe_insert("scale", frame=frame)


# ============================================================
# 5.  CC0 astronaut import
# ============================================================

def import_cc0_astronaut(coll: bpy.types.Collection,
                         blend_path: Path,
                         location=(0.0, -0.50, 0.05),
                         scale: float = 1.0) -> Optional[bpy.types.Object]:
    if not blend_path.exists():
        print(f"WARN  astro.blend not found at {blend_path}")
        return None

    with bpy.data.libraries.load(str(blend_path), link=False) as (src, dst):
        dst.objects   = list(src.objects)
        dst.armatures = list(src.armatures)
        dst.meshes    = list(src.meshes)
        dst.materials = list(src.materials)
        dst.actions   = list(src.actions)

    rig = None
    for obj in dst.objects:
        if obj is None:
            continue
        for oc in list(obj.users_collection):
            oc.objects.unlink(obj)
        coll.objects.link(obj)
        if obj.type == "ARMATURE":
            rig = obj

    if rig is None:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                rig = obj; break

    if rig is None:
        print("WARN  no armature in astro.blend")
        return None

    rig.location = Vector(location)
    rig.scale    = (scale, scale, scale)

    for obj in dst.objects:
        if obj is None or obj.type != "MESH":
            continue
        if obj.parent is None:
            obj.parent = rig
            mod = obj.modifiers.get("Armature") or obj.modifiers.new("Armature","ARMATURE")
            mod.object = rig

    print(f"INFO  CC0 rig: {rig.name}  bones={len(rig.pose.bones)}")
    return rig


# ============================================================
# 6.  Pose helpers (multi-convention bone names)
# ============================================================

def _pose(rig, candidates: List[str], deg, frame=None):
    for name in candidates:
        pb = rig.pose.bones.get(name)
        if pb:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((math.radians(deg[0]),
                                       math.radians(deg[1]),
                                       math.radians(deg[2])))
            if frame is not None:
                pb.keyframe_insert("rotation_euler", frame=frame)
            return


def set_neutral_posture(rig, tuck: float = 1.0):
    """Microgravity neutral body posture (NASA-STD-3001)."""
    _pose(rig, ["spine","Spine","mixamorig:Spine","DEF-spine"],          (-6,0,0))
    _pose(rig, ["spine001","Spine1","mixamorig:Spine1","DEF-spine.001"], (-5,0,0))
    _pose(rig, ["chest","Spine2","mixamorig:Spine2","DEF-chest"],        (-4,0,0))
    for ss, sm, z in [("L","Left",6),("R","Right",-6)]:
        _pose(rig,[f"thigh.{ss}",f"mixamorig:{sm}UpLeg",
                   f"upper_leg.{ss}",f"DEF-thigh.{ss}"],(-42*tuck,0,z))
        _pose(rig,[f"shin.{ss}",f"mixamorig:{sm}Leg",
                   f"lower_leg.{ss}",f"DEF-shin.{ss}"],(38*tuck,0,0))
        _pose(rig,[f"foot.{ss}",f"mixamorig:{sm}Foot",
                   f"DEF-foot.{ss}"],(-16*tuck,0,0))
        _pose(rig,[f"upperarm.{ss}",f"mixamorig:{sm}Arm",
                   f"upper_arm.{ss}",f"DEF-upper_arm.{ss}"],
              (0,0,-30 if ss=="L" else 30))


# ============================================================
# 7.  CC0 protocol animation (bone-rotation keyframes)
# ============================================================

def animate_cc0(rig, payload: Dict, rng: random.Random,
                fps_per_step: int = 48) -> Dict:
    """
    Drive the 8-step protocol via direct bone rotation keyframes.

    Per-step arm targets (L_upper, L_fore, R_upper, R_fore) in XYZ deg.
    """
    STEP_ARMS = {
        1:((-40,0, 20),(-30,0,0),(-40,0,-20),(-30,0,0)),
        2:((-80,0, 10),(-60,0,0),(-80,0,-10),(-60,0,0)),
        3:((-70,0, 30),(-50,0,0),(-20,0,-10),(-10,0,0)),
        4:((-95,0, 15),(-80,0,0),(-70,0,-20),(-60,0,0)),
        5:((-60,0, 40),(-40,0,0),(-20,0,-10),(-10,0,0)),
        6:((-20,0, 15),(-10,0,0),(-70,0,-30),(-50,0,0)),
        7:((-70,0, 20),(-60,0,0),(-95,0,-15),(-80,0,0)),
        8:((-20,0, 10),(-10,0,0),(-60,0,-40),(-40,0,0)),
    }

    scene    = bpy.context.scene
    lid      = payload["lid"]
    red      = payload["red"]
    yellow   = payload["yellow"]
    red_home = red.location.copy()
    yel_home = yellow.location.copy()

    def ease(t):
        t = min(max(t,0.0),1.0)
        return t*t*t*(t*(t*6-15)+10)

    def lerp3(a, b, e):
        return tuple(a[i]+(b[i]-a[i])*e for i in range(3))

    set_neutral_posture(rig)
    bpy.context.view_layer.update()

    frame = 1; labels = []; spans = []; lid_ang = 0.0
    NEU_U, NEU_F = (-10,0,0), (-5,0,0)

    for step in range(1,9):
        n  = max(8, int(round(fps_per_step * rng.uniform(0.92, 1.08))))
        s0 = frame
        la, lf, ra, rf = STEP_ARMS[step]

        for i in range(n):
            t = i / max(n-1,1); e = ease(t)
            _pose(rig,["upperarm.L","mixamorig:LeftArm","upper_arm.L","DEF-upper_arm.L"],
                  lerp3(NEU_U,la,e), frame)
            _pose(rig,["forearm.L","mixamorig:LeftForeArm","lower_arm.L","DEF-forearm.L"],
                  lerp3(NEU_F,lf,e), frame)
            _pose(rig,["upperarm.R","mixamorig:RightArm","upper_arm.R","DEF-upper_arm.R"],
                  lerp3(NEU_U,ra,e), frame)
            _pose(rig,["forearm.R","mixamorig:RightForeArm","lower_arm.R","DEF-forearm.R"],
                  lerp3(NEU_F,rf,e), frame)

            if step == 2:
                lid_ang = P._lid_open_angle() * ease(t)
            lid.rotation_mode = "XYZ"
            lid.rotation_euler = Euler((lid_ang, 0.0, 0.0))
            lid.keyframe_insert("rotation_euler", frame=frame)

            if step == 3:
                pos = Vector((red_home.x-0.02, red_home.y-0.10, red_home.z+0.26*e))
                red.location = pos; red.keyframe_insert("location", frame=frame)
            elif step == 4:
                red.location = Vector((-0.05,-0.22,1.50))
                red.keyframe_insert("location", frame=frame)
            elif step == 5:
                pos = red_home + (MOD.LEFT_ZONE - red_home) * ease(min(1.0, t*1.6))
                red.location = pos; red.keyframe_insert("location", frame=frame)

            if step == 6:
                pos = Vector((yel_home.x+0.02, yel_home.y-0.10, yel_home.z+0.26*e))
                yellow.location = pos; yellow.keyframe_insert("location", frame=frame)
            elif step == 7:
                yellow.location = Vector((0.05,-0.22,1.48))
                yellow.keyframe_insert("location", frame=frame)
            elif step == 8:
                pos = yel_home + (MOD.RIGHT_ZONE - yel_home) * ease(min(1.0, t*1.6))
                yellow.location = pos; yellow.keyframe_insert("location", frame=frame)

            labels.append(step); frame += 1

        spans.append({"step_id":step,"occurrence":step-1,
                      "start_frame":s0,"end_frame":frame-1})

    scene.frame_start = 1
    scene.frame_end   = frame - 1
    scene.render.fps  = P.FPS

    # Flatten all F-curve interpolation to LINEAR.
    # Use C.iter_fcurves() which handles both the old flat action.fcurves
    # (Blender <= 4.3) and the new layered strips API (Blender 5.x).
    for obj in [rig, lid, red, yellow]:
        for fc in C.iter_fcurves(obj):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    return {"n_frames":frame-1,"labels":labels,"sequence":list(range(1,9)),
            "spans":spans,"fps":P.FPS}


# ============================================================
# 8.  Professional cameras
# ============================================================

CAMS = {
    "hero":     ((2.80,-1.50,1.92),(-0.10,-0.42,1.20),28.0, True),
    "payload_a":((1.55,-0.35,1.70),(-0.12,-0.58,1.16),24.0,False),
    "payload_b":((-1.55,-0.35,1.70),(0.12,-0.58,1.16),24.0,False),
    "overhead": ((0.15,-0.55,2.65),(0.05,-0.38,1.02), 22.0,False),
    "helmet":   ((0.85,-1.05,1.82),(0.00,-0.20,1.65), 55.0, True),
    "tunnel":   ((-3.80,-1.03,1.01),(0.40,-0.50,1.10),24.0,False),
}

def build_cameras(coll) -> Dict:
    result = {}
    for name,(loc,tgt,lens,dof) in CAMS.items():
        data = bpy.data.cameras.new(name)
        data.lens=lens; data.sensor_width=36.0
        data.clip_start=0.05; data.clip_end=50.0
        if dof:
            data.dof.use_dof=True
            data.dof.focus_distance=(Vector(loc)-Vector(tgt)).length
            data.dof.aperture_fstop=2.4
        cam = bpy.data.objects.new(name, data)
        C.link(cam, coll)
        cam.location = Vector(loc)
        C.look_at(cam, Vector(tgt))
        result[name] = cam
    return result


# ============================================================
# 9.  Cinematic lighting
# ============================================================

def build_lights(coll) -> List:
    lights = []

    def area(name, energy, color, sx, sy, loc, aim):
        ld = bpy.data.lights.new(name,"AREA")
        ld.shape="RECTANGLE"; ld.size=sx; ld.size_y=sy
        ld.energy=energy; ld.color=color
        o = bpy.data.objects.new(name, ld)
        C.link(o, coll); o.location=Vector(loc); C.look_at(o, Vector(aim))
        lights.append(o); return o

    area("key",  580, (1.00,0.95,0.86), 1.20,0.60, ( 2.60,-0.80,2.40),(-0.15,-0.40,1.15))
    area("fill", 180, (0.80,0.88,1.00), 2.40,1.80, (-2.20,-0.40,1.50),( 0.20,-0.50,1.10))
    area("rim",  420, (0.70,0.85,1.00), 0.60,0.12, ( 0.00, 0.52,2.30),( 0.00,-0.30,1.70))
    area("task", 280, (1.00,0.97,0.92), 0.60,0.08, ( 0.00,-0.28,1.92),( 0.00,-0.28,1.02))
    area("sub",   60, (0.85,0.90,1.00), 3.00,1.50, ( 0.00,-0.90,0.10),( 0.00,-0.30,1.20))

    world = bpy.data.worlds.new("showcase_world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value    = (0.020, 0.025, 0.038, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world
    return lights


# ============================================================
# 10.  Render settings (AgX + DOF + motion blur)
# ============================================================

def setup_render(scene, res, samples, motion_blur=True):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format  = "PNG"
    scene.render.image_settings.color_mode   = "RGB"
    scene.render.image_settings.compression  = 10
    scene.render.use_persistent_data         = True

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma    = 1.0
    try:
        avail = [i.identifier for i in
                 scene.view_settings.bl_rna.properties["look"].enum_items]
        for c in ("AgX - Base Contrast","AgX - Medium Contrast"):
            if c in avail:
                scene.view_settings.look = c; break
    except Exception:
        pass

    e = scene.eevee
    e.taa_render_samples = samples
    e.taa_samples         = min(samples, 16)

    def _s(o,a,v):
        if hasattr(o,a):
            try: setattr(o,a,v)
            except Exception: pass

    _s(e,"use_raytracing",True); _s(e,"ray_tracing_method","SCREEN")
    _s(e,"use_fast_gi",True);    _s(e,"fast_gi_method","GLOBAL_ILLUMINATION")
    _s(e,"gi_diffuse_bounces",3)
    _s(e,"shadow_ray_count",4);  _s(e,"shadow_step_count",8)
    _s(e,"shadow_resolution_scale",1.0); _s(e,"use_shadows",True)
    _s(e,"clamp_surface_indirect",12.0)
    if hasattr(e,"ray_tracing_options"):
        _s(e.ray_tracing_options,"resolution_scale","2")

    scene.render.use_motion_blur     = motion_blur
    scene.render.motion_blur_shutter = 0.35


# ============================================================
# 11.  Scene assembly
# ============================================================

def build_showcase(args) -> Dict:
    C.reset_scene()
    rng = random.Random(args.seed)

    mats = M.build_library("moderate")
    coll = C.get_collection("bas_showcase")

    MOD.build_module(mats, coll, rng)
    payload = MOD.build_payload(mats, coll)

    # Try CC0 astronaut first
    astro_blend = _ROOT/"assets"/"source"/"cc0_rigged_astronaut"/"astronaut"/"astro.blend"
    rig = import_cc0_astronaut(coll, astro_blend, location=(0.0,-0.50,0.05))
    use_cc0 = rig is not None

    if not use_cc0:
        print("INFO  building procedural suit")
        from bas_har import astronaut as A
        crew = A.build_astronaut(mats, coll, location=(0.0,-0.50,1.00))
        rig  = crew["rig"]

    har = build_har_overlay(rig, coll)
    hud = build_hud(coll)
    build_lights(coll)
    cams = build_cameras(coll)

    scene = bpy.context.scene
    scene.camera = cams["hero"]

    setup_render(scene, (args.width, args.height), args.samples,
                 motion_blur=not args.no_motion_blur)

    # Animate
    if use_cc0:
        meta = animate_cc0(rig, payload, rng, fps_per_step=args.frames_per_step)
    else:
        animator = P.ProtocolAnimator(
            {"root":crew["root"],"rig":crew["rig"],
             "body":crew["body"],"ik":crew["ik"]},
            payload, rng, frames_per_step=args.frames_per_step)
        meta = animator.animate()

    for span in meta["spans"]:
        mid = (span["start_frame"] + span["end_frame"]) // 2
        _update_hud(hud, span["step_id"], mid)

    return {"scene":scene,"cams":cams,"meta":meta,"payload":payload,
            "rig":rig,"har":har,"hud":hud,"use_cc0":use_cc0}


# ============================================================
# 12.  Render
# ============================================================

def render_frames(built: Dict, out: Path, cam_name: str) -> None:
    scene = built["scene"]; meta = built["meta"]; cams = built["cams"]
    cam = cams.get(cam_name, cams["hero"])
    scene.camera = cam

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frames_dir) + "/"

    n, fps = meta["n_frames"], meta["fps"]
    print(f"RENDER  cam={cam_name}  {n} frames @ {fps} fps -> {frames_dir}", flush=True)
    bpy.ops.render.render(animation=True)

    written = sorted(frames_dir.glob("*.png"))
    if not written:
        raise SystemExit("render produced no frames")
    print(f"FRAMES_OK  {len(written)}  in {frames_dir}")
    print(f"NEXT  python tools/frames_to_video.py "
          f"{frames_dir} {out.parent/(out.name+'.mp4')} --fps {fps}")


# ============================================================
# 13.  CLI
# ============================================================

def _parse(argv):
    argv = argv[argv.index("--")+1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description="ISRO BAS — Professional HAR Showcase")
    ap.add_argument("--out",             default="build/har_showcase")
    ap.add_argument("--cam",             default="hero", choices=list(CAMS))
    ap.add_argument("--seed",            type=int, default=7)
    ap.add_argument("--frames-per-step", type=int, default=48)
    ap.add_argument("--samples",         type=int, default=48)
    ap.add_argument("--width",           type=int, default=1920)
    ap.add_argument("--height",          type=int, default=1080)
    ap.add_argument("--no-motion-blur",  action="store_true")
    ap.add_argument("--save-blend",      action="store_true")
    ap.add_argument("--quick",           action="store_true",
                    help="Draft: 8 fps/step, 8 samples, 960x540")
    ap.add_argument("--check",           action="store_true")
    return ap.parse_args(argv)


def main():
    args = _parse(sys.argv)

    if args.quick:
        args.frames_per_step = 8; args.samples = 8
        args.width = 960;         args.height  = 540
        args.no_motion_blur = True
        print("QUICK MODE")

    print("="*60)
    print("  ISRO BAS — Professional HAR Showcase")
    print(f"  {args.width}x{args.height}  {args.samples} smp  {args.frames_per_step} fps/step")
    print("="*60, flush=True)

    built = build_showcase(args)
    out   = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.save_blend:
        bp = out / "har_showcase.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(bp))
        print(f"BLEND_SAVED  {bp}")

    render_frames(built, out, args.cam)
    print("SHOWCASE_DONE")


if __name__ == "__main__":
    main()
