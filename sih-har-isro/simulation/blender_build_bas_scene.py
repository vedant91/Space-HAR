"""Build/render a BAS digital-twin scene from the CC0 rigged astronaut source.

Run through Blender, never Python directly.  Output RGB footage is deliberately
clean: OpenCV/HAR overlays are applied after inference, not baked into truth.
"""
from pathlib import Path
import bpy
from mathutils import Vector
from math import radians

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "blender_twin"
OUT.mkdir(parents=True, exist_ok=True)

def mat(name, color, metallic=0.0, rough=.45):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes=True
    bsdf=m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=(*color,1)
    bsdf.inputs["Metallic"].default_value=metallic
    bsdf.inputs["Roughness"].default_value=rough
    m.diffuse_color=(*color,1); m.metallic=metallic; m.roughness=rough
    return m

WHITE=mat("Suit white",(.72,.75,.78),.12,.32); DARK=mat("Rack graphite",(.035,.05,.07),.75,.26)
RED=mat("Payload red",(.7,.012,.01),.08,.28); YELLOW=mat("Payload yellow",(.85,.55,.01),.12,.28)
GLOW=mat("Panel glow",(.03,.27,.38),.3,.2)

def cube(name, loc, scale, material, bevel=.03):
    bpy.ops.mesh.primitive_cube_add(location=loc); o=bpy.context.object; o.name=name; o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    bev=o.modifiers.new("edge rounding","BEVEL"); bev.width=bevel; bev.segments=3
    o.data.materials.append(material); return o

def cyl(name, loc, radius, depth, material, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=radius,depth=depth,location=loc,rotation=rot)
    o=bpy.context.object; o.name=name; o.data.materials.append(material); return o

# Clear cameras/lights only; retain the imported astronaut mesh and rig.
for o in list(bpy.data.objects):
    if o.type in {"CAMERA","LIGHT"}: bpy.data.objects.remove(o,do_unlink=True)
astronaut=[o for o in bpy.data.objects if o.name in {"astronaut","Armature"}]
root=bpy.data.objects.new("Astronaut_Rig_Root",None); bpy.context.collection.objects.link(root)
for o in astronaut:
    if o.parent is None: o.parent=root
# Source mesh is 5.8 Blender units tall; normalize it to an approximately
# 1.7 m IVA astronaut before placing it in the 4.5 m rack work volume.
root.location=(0,0,.52); root.rotation_euler=(radians(82),0,radians(8)); root.scale=(.30,.30,.30)
# Normalise any legacy imported materials so the source astronaut is visible
# under an IVA-style, low-glare payload-rack lighting environment.
for obj in astronaut:
    if obj.type == 'MESH':
        for slot in obj.material_slots:
            if slot.material:
                slot.material.use_nodes=True
                bsdf=slot.material.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value=(.38,.43,.47,1)
                    bsdf.inputs["Roughness"].default_value=.52

# Science rack: all geometry is fixed in the rack frame.
cube("Rack_Back",(0,.8,1.35),(2.25,.12,1.35),DARK,.06)
for x in (-2.0,2.0): cube("Rack_rail",(x,.63,1.35),(.06,.12,1.28),WHITE,.02)
for z in (.18,2.42): cyl("Handrail",(0,.42,z),.035,4.1,WHITE,(0,radians(90),0))
for x in (-1.4,-.7,.7,1.4): cube("Avionics",(x,.62,1.7),(.25,.08,.45),GLOW,.02)
main=cube("main_box",(0,.23,.55),(.55,.26,.32),WHITE,.06)
lid=cube("main_lid",(0,-.03,.88),(.57,.035,.33),WHITE,.04); lid.rotation_euler=(radians(-58),0,0)
red=cube("red_box",(-.18,-.02,.64),(.12,.10,.10),RED,.02)
yellow=cube("yellow_box",(.18,-.02,.64),(.12,.10,.10),YELLOW,.02)
for x in (-1.25,1.25): cube("target_zone",(x,.23,.54),(.22,.03,.22),GLOW,.015)

# Zero-g animation: object velocity persists after release, then rack latch holds it.
def k(obj, frame, loc, rot=None):
    obj.location=loc; obj.keyframe_insert("location",frame=frame)
    if rot is not None: obj.rotation_euler=rot; obj.keyframe_insert("rotation_euler",frame=frame)
k(root,1,(0,0,.55),(radians(82),0,radians(8))); k(root,90,(.06,.02,.62),(radians(90),radians(4),radians(12))); k(root,180,(-.04,-.01,.57),(radians(76),radians(-5),radians(-10)))
k(red,1,(-.18,-.02,.64)); k(red,52,(-.34,-.34,.98)); k(red,86,(-.82,.15,.59)); k(red,105,(-1.25,.23,.54))
k(yellow,1,(.18,-.02,.64)); k(yellow,112,(.35,-.34,.98)); k(yellow,148,(.84,.12,.59)); k(yellow,168,(1.25,.23,.54))
# Blender 5 action layers vary across releases; default Bezier interpolation is
# sufficient here and avoids relying on the legacy Action.fcurves API.

# Fixed payload camera, physically plausible studio lighting, and real-time renderer.
bpy.ops.object.camera_add(location=(0,-6.2,1.25),rotation=(radians(82),0,0))
cam=bpy.context.object; cam.name="Payload_Camera_Fixed"; bpy.context.scene.camera=cam; cam.data.lens=48
def track(obj, target): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
track(cam,(0,.2,.85)); cam.data.dof.use_dof=True; cam.data.dof.focus_object=main; cam.data.dof.aperture_fstop=7
for loc,energy,size in [((-3,-3,4),85,3),((3,-2,3),65,2),((0,1,4),75,2)]:
    bpy.ops.object.light_add(type='AREA',location=loc); l=bpy.context.object; l.data.energy=energy; l.data.shape='DISK'; l.data.size=size; track(l,(0,.2,1))

s=bpy.context.scene; s.render.engine='BLENDER_EEVEE'; s.render.resolution_x=960; s.render.resolution_y=540; s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'; (OUT/'raw_frames').mkdir(exist_ok=True); s.render.filepath=str(OUT/'raw_frames'/'frame_')
s.render.fps=30; s.frame_start=1; s.frame_end=180
s.world.color=(.006,.009,.015)
s.view_settings.look='None'; s.view_settings.exposure=-1.7
s.render.film_transparent=False
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'bas_har_digital_twin.blend'))
s.frame_set(90); s.render.filepath=str(OUT/'blender_preview_frame_0090.png'); bpy.ops.render.render(write_still=True)
s.render.filepath=str(OUT/'raw_frames'/'frame_')
bpy.ops.render.render(animation=True)
