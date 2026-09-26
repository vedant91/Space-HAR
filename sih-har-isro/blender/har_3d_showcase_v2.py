"""
ISRO BAS — Space HAR 3D Showcase  (Blender 5.x, EEVEE)
========================================================

Complete, self-contained Blender Python script.
Run with:

  set BLENDER=D:/SIH-2026/tools/blender-5.2.0\blender-5.2.0-windows-x64\blender.exe

  REM Quick draft  (~8 min, 960x540, 16 samples)
  %BLENDER% --background --factory-startup ^
      --python blender/har_3d_showcase_v2.py -- --quick

  REM Full HD  (~35 min, 1280x720, 48 samples)
  %BLENDER% --background --factory-startup ^
      --python blender/har_3d_showcase_v2.py -- ^
      --out build/har3d --width 1280 --height 720 --samples 48

  REM Open .blend interactively (no render)
  %BLENDER% --background --factory-startup ^
      --python blender/har_3d_showcase_v2.py -- --build-only

  REM Encode frames to MP4 after rendering
  python tools/frames_to_video.py build/har3d/frames build/har_showcase_v2.mp4 --fps 30
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_ASTRO_BLEND = _ROOT / "assets/source/cc0_rigged_astronaut/astronaut/astro.blend"

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from bas_har import common as C
from bas_har import materials as M

RACK_W, RACK_H, RACK_D = 1.046, 2.013, 0.858
MODULE_RADIUS = 2.10
MODULE_LENGTH = 6.40

STEPS = [
    {"id":1,"name":"Approach Main Box","dur":48},
    {"id":2,"name":"Open Main Box","dur":48},
    {"id":3,"name":"Pick Red Box","dur":48},
    {"id":4,"name":"Examine Red Box","dur":48},
    {"id":5,"name":"Place Red Box","dur":48},
    {"id":6,"name":"Pick Yellow Box","dur":48},
    {"id":7,"name":"Examine Yellow Box","dur":48},
    {"id":8,"name":"Place Yellow Box","dur":48},
]

_MP_EDGES = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,9),(12,10),
]

_BONE_TO_LM = {
    "head":0,"neck":9,"chest":11,
    "clavicle.L":11,"clavicle.R":12,
    "upperarm.L":13,"upperarm.R":14,
    "forearm.L":15,"forearm.R":16,
    "hand.L":17,"hand.R":18,
    "pelvis":23,
    "thigh.L":25,"thigh.R":26,
    "shin.L":27,"shin.R":28,
    "foot.L":31,"foot.R":32,
}


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/har3d")
    ap.add_argument("--width",  type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--samples",type=int, default=16)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--frames-per-step", type=int, default=48, dest="fps_step")
    ap.add_argument("--quick",      action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--camera", default="hero",
                    choices=["hero","payload_a","payload_b","helmet","tunnel","all"])
    return ap.parse_args(argv)


def _clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    for blk in (bpy.data.meshes,bpy.data.materials,bpy.data.lights,
                bpy.data.cameras,bpy.data.curves,bpy.data.armatures):
        for item in list(blk):
            blk.remove(item, do_unlink=True)


def _obj(name, data):
    o = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(o)
    return o


def _kf(obj, dp, frame, val):
    setattr(obj, dp, val)
    obj.keyframe_insert(data_path=dp, frame=frame)


def _bkf(arm, bone, frame, rot):
    pb = arm.pose.bones.get(bone)
    if pb is None: return
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = Euler(rot)
    pb.keyframe_insert("rotation_euler", frame=frame)


# ── BAS MODULE ────────────────────────────────────────────────────────────────
def build_module():
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=MODULE_RADIUS, depth=MODULE_LENGTH,
        location=(0, MODULE_LENGTH*0.5, 0),
        rotation=(math.radians(90),0,0))
    sh = bpy.context.active_object
    sh.name = "BAS_Shell"
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    mat = M.painted("MAT_Shell",(0.85,0.86,0.88,1.0),roughness=0.45)
    sh.data.materials.append(mat)

    pmat = M.painted("MAT_Floor",(0.55,0.56,0.58,1.0),roughness=0.70)
    for i in range(4):
        bpy.ops.mesh.primitive_plane_add(size=0.44, location=((i-1.5)*0.45,-MODULE_RADIUS+0.04,0))
        p = bpy.context.active_object
        p.name = f"Floor_{i}"
        p.data.materials.append(pmat)

    wmat = bpy.data.materials.new("MAT_Window")
    wmat.use_nodes = True
    bsdf = wmat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.05,0.15,0.35,1.0)
        bsdf.inputs["Transmission Weight"].default_value = 0.85
        bsdf.inputs["Roughness"].default_value = 0.0
    bpy.ops.mesh.primitive_circle_add(vertices=32, radius=0.21,
        location=(MODULE_RADIUS-0.05,1.0,0.3),rotation=(0,math.radians(90),0))
    w = bpy.context.active_object
    w.name = "BAS_Window"
    w.data.materials.append(wmat)


# ── PAYLOAD RACK ─────────────────────────────────────────────────────────────
def build_rack():
    rmat = M.brushed_metal("MAT_Rack",(0.72,0.73,0.74,1.0))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,RACK_D*0.5,RACK_H*0.5))
    rack = bpy.context.active_object
    rack.name = "PayloadRack"
    rack.scale = (RACK_W, RACK_D, RACK_H)
    bpy.ops.object.transform_apply(scale=True)
    rack.data.materials.append(rmat)

    def box(name, loc, scl, rgb):
        # M.painted needs 4-component RGBA tuple
        rgba = (*rgb, 1.0) if len(rgb) == 3 else tuple(rgb)
        mat = M.painted(f"MAT_{name}", rgba)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        o = bpy.context.active_object
        o.name = name
        o.scale = Vector(scl)
        bpy.ops.object.transform_apply(scale=True)
        o.data.materials.append(mat)
        return o

    MBRZ = RACK_H * 0.34
    _    = box("Main_Box", (0.0,  -0.18, MBRZ),         (0.28, 0.22, 0.20),  M.CONTAINER_WHITE_SRGB)
    lid  = box("Main_Lid", (0.0,  -0.18, MBRZ+0.103),   (0.28, 0.22, 0.01),  M.CONTAINER_WHITE_SRGB)
    red  = box("Red_Box",  (-0.07,-0.18, MBRZ),          (0.11, 0.10, 0.085), M.PAYLOAD_RED_SRGB)
    yel  = box("Yellow_Box",(0.07, -0.18, MBRZ),         (0.11, 0.10, 0.085), M.PAYLOAD_YELLOW_SRGB)
    return lid, red, yel


# ── ASTRONAUT ─────────────────────────────────────────────────────────────────
def build_astronaut():
    if _ASTRO_BLEND.exists():
        with bpy.data.libraries.load(str(_ASTRO_BLEND), link=False) as (src, dst):
            dst.objects = src.objects[:]
        rig = None
        for obj in dst.objects:
            if obj is None: continue
            bpy.context.collection.objects.link(obj)
            if obj.type == "ARMATURE": rig = obj
        if rig:
            rig.location = Vector((0.0,-0.85,RACK_H*0.35))
            print(f"[HAR3D] CC0 astronaut imported: {rig.name}")
            return rig
    # procedural
    from bas_har import astronaut as A
    rig = A.build()
    if rig is None: rig = _fallback_suit()
    rig.location = Vector((0.0,-0.85,RACK_H*0.35))
    return rig


def _fallback_suit():
    mat = bpy.data.materials.new("MAT_Suit")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    if b: b.inputs["Base Color"].default_value=(0.88,0.88,0.88,1.0)
    parts=[("Torso",(0,0,.35),(.22,.20,.28)),("Helmet",(0,0,.72),(.16,.16,.16)),
           ("UpperL",(-.28,0,.44),(.07,.07,.18)),("ForeL",(-.30,.08,.26),(.06,.06,.16)),
           ("HandL", (-.31,.13,.16),(.05,.05,.06)),
           ("UpperR",(.28,0,.44),(.07,.07,.18)),("ForeR",(.30,.08,.26),(.06,.06,.16)),
           ("HandR", (.31,.13,.16),(.05,.05,.06)),
           ("ThighL",(-.10,0,-.06),(.08,.08,.22)),("ThighR",(.10,0,-.06),(.08,.08,.22))]
    parent = None
    for nm,loc,sc in parts:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=10,ring_count=6,radius=0.5,location=loc)
        o = bpy.context.active_object
        o.name = f"Suit_{nm}"; o.scale=Vector(sc)
        bpy.ops.object.transform_apply(scale=True)
        o.data.materials.append(mat)
        if parent is None: parent=o
        else: o.parent=parent
    # visor
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10,ring_count=6,radius=0.13,location=(0,.04,.72))
    vis=bpy.context.active_object; vis.name="Suit_Visor"
    vm=bpy.data.materials.new("MAT_Visor"); vm.use_nodes=True
    vb=vm.node_tree.nodes.get("Principled BSDF")
    if vb:
        vb.inputs["Base Color"].default_value=(0.02,.06,.10,1.0)
        vb.inputs["Transmission Weight"].default_value=0.6
        vb.inputs["Roughness"].default_value=0.02
    vis.data.materials.append(vm); vis.parent=parent
    return parent


# ── HAR SKELETON OVERLAY ─────────────────────────────────────────────────────
def build_har_skeleton(rig):
    jmat = bpy.data.materials.new("MAT_HAR_Joint")
    jmat.use_nodes=True; jmat.blend_method="BLEND"
    jt=jmat.node_tree; jt.nodes.clear()
    jo=jt.nodes.new("ShaderNodeOutputMaterial")
    je=jt.nodes.new("ShaderNodeEmission")
    je.inputs["Color"].default_value=(0.0,0.9,1.0,1.0)
    je.inputs["Strength"].default_value=3.5
    jt.links.new(je.outputs["Emission"],jo.inputs["Surface"])

    cmat = bpy.data.materials.new("MAT_HAR_Conn")
    cmat.use_nodes=True; cmat.blend_method="BLEND"
    ct=cmat.node_tree; ct.nodes.clear()
    co=ct.nodes.new("ShaderNodeOutputMaterial")
    ce=ct.nodes.new("ShaderNodeEmission")
    ce.inputs["Color"].default_value=(0.0,0.75,0.95,1.0)
    ce.inputs["Strength"].default_value=2.0
    ct.links.new(ce.outputs["Emission"],co.inputs["Surface"])

    pos = _mp_positions(rig)

    for i,p in enumerate(pos):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=6,ring_count=4,radius=0.018,location=p)
        o=bpy.context.active_object; o.name=f"HAR_J{i:02d}"
        o.data.materials.append(jmat)

    for a,b in _MP_EDGES:
        if a>=len(pos) or b>=len(pos): continue
        pa,pb=pos[a],pos[b]
        mid=(pa+pb)*0.5; L=(pb-pa).length
        if L<0.001: continue
        d=(pb-pa).normalized()
        z=Vector((0,0,1)); ax=z.cross(d)
        if ax.length>1e-6:
            angle=math.acos(max(-1,min(1,z.dot(d))))
            rot=Matrix.Rotation(angle,4,ax.normalized()).to_euler()
        else:
            rot=Euler((0,0,0))
        bpy.ops.mesh.primitive_cylinder_add(vertices=6,radius=0.007,depth=L,location=mid)
        o=bpy.context.active_object; o.name=f"HAR_C{a}_{b}"
        o.rotation_euler=rot
        o.data.materials.append(cmat)


def _mp_positions(rig):
    origin = rig.location.copy()
    pos = [origin.copy() for _ in range(33)]
    if rig.type == "ARMATURE":
        for bn,li in _BONE_TO_LM.items():
            pb=rig.pose.bones.get(bn)
            if pb is None: pb=rig.pose.bones.get(bn.split(".")[0])
            if pb:
                wp=rig.matrix_world@pb.head
                pos[li]=wp.copy()
                mir={11:12,13:14,15:16,17:18,23:24,25:26,27:28,31:32}
                if li in mir:
                    mp=wp.copy(); mp.x*=-1; pos[mir[li]]=mp
    else:
        z_lo=rig.location.z-0.5; z_hi=rig.location.z+0.9
        h=z_hi-z_lo; cx,cy=rig.location.x,rig.location.y
        pos[0]=Vector((cx,cy,z_hi))
        pos[9]=pos[10]=Vector((cx,cy,z_hi-h*.12))
        pos[11]=Vector((cx-.18,cy,z_hi-h*.22))
        pos[12]=Vector((cx+.18,cy,z_hi-h*.22))
        pos[13]=Vector((cx-.28,cy+.04,z_hi-h*.38))
        pos[14]=Vector((cx+.28,cy+.04,z_hi-h*.38))
        pos[15]=Vector((cx-.30,cy+.10,z_hi-h*.52))
        pos[16]=Vector((cx+.30,cy+.10,z_hi-h*.52))
        pos[23]=Vector((cx-.10,cy,z_hi-h*.55))
        pos[24]=Vector((cx+.10,cy,z_hi-h*.55))
        pos[25]=Vector((cx-.11,cy+.05,z_hi-h*.75))
        pos[26]=Vector((cx+.11,cy+.05,z_hi-h*.75))
        pos[27]=Vector((cx-.11,cy+.01,z_hi-h*.95))
        pos[28]=Vector((cx+.11,cy+.01,z_hi-h*.95))
    return pos


# ── HUD PANEL ─────────────────────────────────────────────────────────────────
def build_hud():
    bpy.ops.mesh.primitive_plane_add(size=1.0,location=(0.72,-0.25,RACK_H*0.72))
    hud=bpy.context.active_object; hud.name="HAR_HUD"
    hud.scale=(0.50,0.35,1.0)
    hud.rotation_euler=Euler((0,0,math.radians(-12)))
    bpy.ops.object.transform_apply(scale=True)
    hm=bpy.data.materials.new("MAT_HUD"); hm.use_nodes=True; hm.blend_method="BLEND"
    nt=hm.node_tree; nt.nodes.clear()
    ho=nt.nodes.new("ShaderNodeOutputMaterial")
    he=nt.nodes.new("ShaderNodeEmission"); mx=nt.nodes.new("ShaderNodeMixShader")
    tr=nt.nodes.new("ShaderNodeBsdfTransparent")
    he.inputs["Color"].default_value=(0.0,0.35,0.45,1.0); he.inputs["Strength"].default_value=0.9
    mx.inputs["Fac"].default_value=0.65
    nt.links.new(tr.outputs["BSDF"],mx.inputs[1])
    nt.links.new(he.outputs["Emission"],mx.inputs[2])
    nt.links.new(mx.outputs["Shader"],ho.inputs["Surface"])
    hud.data.materials.append(hm)

    # Step progress bar
    bon=bpy.data.materials.new("MAT_Bar_ON"); bon.use_nodes=True
    bt=bon.node_tree; bt.nodes.clear()
    boo=bt.nodes.new("ShaderNodeOutputMaterial"); boe=bt.nodes.new("ShaderNodeEmission")
    boe.inputs["Color"].default_value=(0.0,1.0,0.6,1.0); boe.inputs["Strength"].default_value=4.0
    bt.links.new(boe.outputs["Emission"],boo.inputs["Surface"])

    boff=bpy.data.materials.new("MAT_Bar_OFF"); boff.use_nodes=True
    bft=boff.node_tree; bft.nodes.clear()
    bfo=bft.nodes.new("ShaderNodeOutputMaterial"); bfe=bft.nodes.new("ShaderNodeEmission")
    bfe.inputs["Color"].default_value=(0.05,0.12,0.12,1.0); bfe.inputs["Strength"].default_value=0.3
    bft.links.new(bfe.outputs["Emission"],bfo.inputs["Surface"])

    for i in range(8):
        bpy.ops.mesh.primitive_cube_add(size=1.0,location=(0.61+i*0.075,-0.25,RACK_H*0.58))
        seg=bpy.context.active_object; seg.name=f"HUD_Bar_{i}"
        seg.scale=(0.030,0.015,0.018); bpy.ops.object.transform_apply(scale=True)
        seg.data.materials.append(bon if i==0 else boff)

    # G-level LED
    bpy.ops.mesh.primitive_uv_sphere_add(segments=8,ring_count=6,radius=0.025,
                                          location=(0.61,-0.25,RACK_H*0.65))
    led=bpy.context.active_object; led.name="HUD_GLED"
    gm=bpy.data.materials.new("MAT_GLED"); gm.use_nodes=True
    gn=gm.node_tree; gn.nodes.clear()
    go=gn.nodes.new("ShaderNodeOutputMaterial"); ge=gn.nodes.new("ShaderNodeEmission")
    ge.inputs["Color"].default_value=(0.2,1.0,0.3,1.0); ge.inputs["Strength"].default_value=5.0
    gn.links.new(ge.outputs["Emission"],go.inputs["Surface"])
    led.data.materials.append(gm)


# ── LIGHTING ──────────────────────────────────────────────────────────────────
def build_lighting():
    def add_light(name, ltype, energy, color, loc, rot, **kw):
        d=bpy.data.lights.new(name,ltype)
        d.energy=energy; d.color=color
        for k,v in kw.items(): setattr(d,k,v)
        o=_obj(name,d); o.location=Vector(loc); o.rotation_euler=Euler(rot)
        return o
    R=RACK_H
    add_light("Lgt_Key","AREA",280,(1.0,.82,.60),(-0.8,-1.8,R*.85),
              (math.radians(35),math.radians(-25),0),size=0.90)
    add_light("Lgt_Fill","AREA",80,(0.72,.84,1.0),(1.4,-0.5,R*.5),
              (0,math.radians(50),0),size=1.20)
    add_light("Lgt_Rim","SPOT",220,(0.85,.95,1.0),(0.0,2.0,R*1.1),
              (math.radians(-40),0,0),spot_size=math.radians(38))
    add_light("Lgt_Task","SPOT",160,(1.0,.97,.88),(0.0,-0.5,R*1.05),
              (math.radians(-65),0,0),spot_size=math.radians(28))
    add_light("Lgt_Sub","AREA",35,(0.60,.65,.72),(0.0,-2.0,R*.2),
              (math.radians(70),0,0),size=1.8)
    world=bpy.context.scene.world
    if world is None:
        world=bpy.data.worlds.new("World"); bpy.context.scene.world=world
    world.use_nodes=True
    bg=world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value=(0.004,0.004,0.006,1.0)
        bg.inputs["Strength"].default_value=0.3


# ── CAMERAS ───────────────────────────────────────────────────────────────────
def build_cameras():
    cams={}
    tgt=Vector((0.0,-0.18,RACK_H*0.38))
    defs=[
        ("hero",      (-1.0,-1.85,RACK_H*.55), 50, 2.4),
        ("payload_a", ( 0.0,-0.60,RACK_H*.10), 35, 2.8),
        ("payload_b", ( 0.0,-0.18,RACK_H*1.15),28, 4.0),
        ("helmet",    (-0.25,-1.10,RACK_H*.72),85, 1.8),
        ("tunnel",    ( 0.0,-3.50,RACK_H*.50), 24, 5.6),
    ]
    for nm,loc,focal,fstop in defs:
        cd=bpy.data.cameras.new(f"Cam_{nm}")
        cd.lens=focal; cd.dof.use_dof=True; cd.dof.aperture_fstop=fstop
        co=_obj(f"Camera_{nm}",cd); co.location=Vector(loc)
        d=(tgt-co.location).normalized()
        co.rotation_euler=d.to_track_quat("-Z","Y").to_euler()
        cams[nm]=co
    return cams


# ── ANIMATION ─────────────────────────────────────────────────────────────────
def animate_protocol(rig, lid, red, yel, fps_step=48):
    scene=bpy.context.scene; scene.frame_start=1
    F=fps_step; MBRZ=RACK_H*0.34

    NP={
        "pelvis":   (math.radians(-8),0,0),
        "spine":    (math.radians( 5),0,0),
        "chest":    (math.radians( 8),0,0),
        "upperarm.L":(math.radians(-15),0,math.radians(20)),
        "upperarm.R":(math.radians(-15),0,math.radians(-20)),
        "forearm.L":(math.radians(12),0,0),
        "forearm.R":(math.radians(12),0,0),
        "thigh.L":  (math.radians(42),0,math.radians(8)),
        "thigh.R":  (math.radians(42),0,math.radians(-8)),
        "shin.L":   (math.radians(-38),0,0),
        "shin.R":   (math.radians(-38),0,0),
    }

    def neutral(fr):
        if rig.type=="ARMATURE":
            for bn,rot in NP.items(): _bkf(rig,bn,fr,rot)
        _kf(rig,"location",fr,(0.0,-0.85,RACK_H*0.35))

    def reach(fr, side, fwd, bend):
        if rig.type!="ARMATURE": return
        _bkf(rig,f"upperarm.{side}",fr,(math.radians(-10),0,math.radians(fwd)))
        _bkf(rig,f"forearm.{side}", fr,(math.radians(bend),0,0))
        _bkf(rig,f"hand.{side}",    fr,(math.radians(15),0,0))

    # Step 1: Approach
    neutral(1)
    _kf(rig,"location",F,(0.0,-0.72,RACK_H*0.35))
    if rig.type=="ARMATURE":
        for s,sg in (("L",1),("R",-1)):
            _bkf(rig,f"upperarm.{s}",F,(math.radians(-20),0,math.radians(sg*15)))
            _bkf(rig,f"forearm.{s}",F,(math.radians(30),0,0))

    # Step 2: Open lid
    _kf(lid,"rotation_euler",F+1,(0,0,0))
    _kf(lid,"rotation_euler",F+F//2,(math.radians(-115),0,0))
    _kf(lid,"rotation_euler",F*2,(math.radians(-115),0,0))
    reach(F+F//3,"R",5,45); reach(F*2,"R",-5,20)

    # Step 3: Pick red
    neutral(F*2+1)
    _kf(red,"location",F*2+1,(-0.07,-0.18,MBRZ))
    _kf(red,"location",F*3,  (-0.07,-0.40,MBRZ+0.12))
    reach(F*2+F//2,"L",10,55); reach(F*3,"L",8,30)

    # Step 4: Examine red
    _kf(red,"location",F*3+1,(-0.07,-0.40,MBRZ+0.12))
    _kf(red,"location",F*4,  (-0.02,-0.50,MBRZ+0.22))
    _kf(red,"rotation_euler",F*4,(math.radians(8),math.radians(45),0))
    reach(F*3+1,"L",12,40); reach(F*3+1,"R",12,40)
    if rig.type=="ARMATURE": _bkf(rig,"head",F*3+F//2,(math.radians(-12),0,0))

    # Step 5: Place red
    _kf(red,"location",F*4+1,(-0.02,-0.50,MBRZ+0.22))
    _kf(red,"location",F*5,  (-0.36,-0.10,MBRZ+0.04))
    _kf(red,"rotation_euler",F*5,(0,0,0))
    reach(F*4+F//2,"L",5,20); neutral(F*5)

    # Step 6: Pick yellow
    _kf(yel,"location",F*5+1,(0.07,-0.18,MBRZ))
    _kf(yel,"location",F*6,  (0.07,-0.40,MBRZ+0.12))
    reach(F*5+F//2,"R",10,55)

    # Step 7: Examine yellow
    _kf(yel,"location",F*6+1,(0.07,-0.40,MBRZ+0.12))
    _kf(yel,"location",F*7,  (0.04,-0.48,MBRZ+0.20))
    _kf(yel,"rotation_euler",F*7,(math.radians(5),math.radians(-30),0))
    reach(F*6+1,"R",12,40); reach(F*6+1,"L",12,40)

    # Step 8: Place yellow
    _kf(yel,"location",F*7+1,(0.04,-0.48,MBRZ+0.20))
    _kf(yel,"location",F*8,  (0.36,-0.10,MBRZ+0.04))
    _kf(yel,"rotation_euler",F*8,(0,0,0))
    reach(F*7+F//2,"R",5,20); neutral(F*8)

    scene.frame_end = F*8

    # Blender 5.x safe fcurves
    for obj in [rig,lid,red,yel]:
        for fc in C.iter_fcurves(obj):
            for kp in fc.keyframe_points:
                kp.interpolation="LINEAR"

    return F*8


# ── RENDER SETUP ──────────────────────────────────────────────────────────────
def setup_render(args, out_dir, cam):
    from bas_har import render as R
    scene = bpy.context.scene
    quality = "draft" if args.quick else "showcase"
    R.setup_render(scene,
                   resolution=(args.width, args.height),
                   samples=args.samples,
                   view_transform="AgX",
                   quality=quality)
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "PNG"
    if not args.quick:
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 0.35
    # AgX look
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    fd = out_dir / "frames"
    fd.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(fd / "####")
    scene.camera = cam


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args=_parse_args()
    if args.quick:
        args.width=960; args.height=540; args.samples=8; args.fps_step=24

    out_dir=Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    print("\n[HAR3D] ISRO BAS Space HAR 3D Showcase v2")

    _clear()
    print("[HAR3D] Building module …"); build_module()
    print("[HAR3D] Building rack …");   lid,red,yel=build_rack()
    print("[HAR3D] Building astronaut …"); rig=build_astronaut()
    print("[HAR3D] Building HAR overlay …"); build_har_skeleton(rig)
    print("[HAR3D] Building HUD …");     build_hud()
    print("[HAR3D] Lighting …");         build_lighting()
    print("[HAR3D] Cameras …");          cams=build_cameras()
    print("[HAR3D] Animating protocol …"); last=animate_protocol(rig,lid,red,yel,args.fps_step)
    print(f"[HAR3D] Total frames: {last}")

    blend_path=out_dir/"har_showcase_v2.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"[HAR3D] .blend saved: {blend_path}")

    if args.build_only:
        print("[HAR3D] --build-only: done."); return

    cam_names=[args.camera] if args.camera!="all" else list(cams.keys())
    for cn in cam_names:
        cam_obj=cams.get(cn)
        if not cam_obj: continue
        cam_out=out_dir/cn if len(cam_names)>1 else out_dir
        setup_render(args,cam_out,cam_obj)
        print(f"[HAR3D] Rendering [{cn}] {last} frames …")
        bpy.ops.render.render(animation=True,write_still=False)
        n=len(list((cam_out/"frames").glob("*.png")))
        print(f"[HAR3D] FRAMES_OK {n} in {cam_out}/frames")

    print(f"\n[HAR3D] SHOWCASE_DONE")
    print(f"[HAR3D] python tools/frames_to_video.py {out_dir}/frames {out_dir}.mp4 --fps {args.fps}")


if __name__=="__main__":
    main()


