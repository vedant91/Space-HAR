"""
Low-level Blender helpers shared by every builder in this package.

Everything here is deliberately operator-light: `bpy.ops` depends on context
(active object, mode, selected set) which is fragile in `--background` runs
where there is no window manager to keep that context coherent. Where an
operator is genuinely the only API (bevel/subsurf are modifiers, but
shade_auto_smooth is not), it is wrapped so the context is set explicitly
right before the call and restored after.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

# ── Scene / collection management ────────────────────────────────────────────


def reset_scene() -> None:
    """Delete every datablock so a rebuild never inherits the previous one.

    `--factory-startup` gives a clean file, but build_scene.py is also run
    repeatedly inside one session during development; orphaned meshes and
    materials would otherwise accumulate and silently change material lookups
    (Blender appends `.001` and `bpy.data.materials["suit"]` then resolves to
    a stale copy).
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (
        bpy.data.meshes, bpy.data.materials, bpy.data.armatures,
        bpy.data.cameras, bpy.data.lights, bpy.data.images,
        bpy.data.node_groups, bpy.data.actions, bpy.data.collections,
        bpy.data.worlds,
    ):
        for item in list(block):
            block.remove(item, do_unlink=True)


def get_collection(name: str) -> bpy.types.Collection:
    """Fetch-or-create a collection linked to the scene root."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def link(obj: bpy.types.Object, collection: bpy.types.Collection) -> bpy.types.Object:
    """Link `obj` to exactly `collection`, unlinking it from anywhere else."""
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


# ── Mesh construction ────────────────────────────────────────────────────────


def mesh_from_pydata(name: str, verts, faces,
                     collection: Optional[bpy.types.Collection] = None) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    link(obj, collection or bpy.context.scene.collection)
    return obj


def box(name: str,
        size: Sequence[float],
        location: Sequence[float] = (0.0, 0.0, 0.0),
        rotation: Sequence[float] = (0.0, 0.0, 0.0),
        collection: Optional[bpy.types.Collection] = None) -> bpy.types.Object:
    """Axis-aligned box of full extents `size`, centred on `location`.

    Built from pydata rather than `primitive_cube_add` so no operator context
    or post-hoc scaling is involved — the mesh comes out with unit object
    scale, which matters because a non-uniform object scale would distort
    every bevel width and modifier applied later.
    """
    sx, sy, sz = (s * 0.5 for s in size)
    verts = [
        (-sx, -sy, -sz), (+sx, -sy, -sz), (+sx, +sy, -sz), (-sx, +sy, -sz),
        (-sx, -sy, +sz), (+sx, -sy, +sz), (+sx, +sy, +sz), (-sx, +sy, +sz),
    ]
    faces = [
        (0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    obj = mesh_from_pydata(name, verts, faces, collection)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation)
    return obj


def cylinder(name: str,
             radius: float,
             depth: float,
             segments: int = 32,
             location: Sequence[float] = (0.0, 0.0, 0.0),
             rotation: Sequence[float] = (0.0, 0.0, 0.0),
             cap: bool = True,
             collection: Optional[bpy.types.Collection] = None) -> bpy.types.Object:
    """Z-aligned cylinder built with bmesh (no operator context needed)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=cap, cap_tris=False, segments=segments,
        radius1=radius, radius2=radius, depth=depth,
    )
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    link(obj, collection or bpy.context.scene.collection)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation)
    return obj


def capsule(name: str,
            radius: float,
            length: float,
            segments: int = 24,
            rings: int = 8,
            collection: Optional[bpy.types.Collection] = None) -> bpy.types.Object:
    """Z-aligned capsule (cylinder + hemispherical caps), centred on origin.

    Limb segments are capsules rather than cylinders so that a joint reads as
    a continuous rounded volume when two segments meet at an angle — a
    flat-capped cylinder shows a visible wedge gap at every bent elbow/knee.
    """
    half = max(length * 0.5, 1e-5)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings * 2, radius=radius)
    # Split the sphere at the equator and push each hemisphere out to form the
    # cylindrical mid-section.
    for vert in bm.verts:
        vert.co.z += half if vert.co.z > 1e-6 else (-half if vert.co.z < -1e-6 else 0.0)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    link(obj, collection or bpy.context.scene.collection)
    return obj


def torus(name: str,
          major_radius: float,
          minor_radius: float,
          major_segments: int = 32,
          minor_segments: int = 12,
          location: Sequence[float] = (0.0, 0.0, 0.0),
          rotation: Sequence[float] = (0.0, 0.0, 0.0),
          collection: Optional[bpy.types.Collection] = None) -> bpy.types.Object:
    """Z-axis torus — used for suit convolute rings and handrail returns."""
    bm = bmesh.new()
    for i in range(major_segments):
        phi = 2.0 * math.pi * i / major_segments
        centre = Vector((math.cos(phi) * major_radius, math.sin(phi) * major_radius, 0.0))
        radial = Vector((math.cos(phi), math.sin(phi), 0.0))
        for j in range(minor_segments):
            theta = 2.0 * math.pi * j / minor_segments
            offset = radial * (math.cos(theta) * minor_radius)
            offset.z += math.sin(theta) * minor_radius
            bm.verts.new(centre + offset)
    bm.verts.ensure_lookup_table()
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + j
            bm.faces.new((bm.verts[a], bm.verts[b], bm.verts[c], bm.verts[d]))
    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    link(obj, collection or bpy.context.scene.collection)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation)
    return obj


# ── Modifiers & shading ──────────────────────────────────────────────────────


def add_bevel(obj: bpy.types.Object, width: float = 0.004, segments: int = 2,
              angle_deg: float = 40.0) -> bpy.types.Object:
    """Bevel every hard edge. This is the single highest-value detail pass:
    a real object has no infinitely sharp edge, so an unbevelled cube reads as
    CG immediately no matter how good the material is — the specular
    highlight that a bevel catches along each edge is most of what sells
    'manufactured metal panel'."""
    mod = obj.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle_deg)
    mod.harden_normals = True
    mod.miter_outer = "MITER_ARC"
    return obj


def add_subsurf(obj: bpy.types.Object, levels: int = 2, render_levels: Optional[int] = None):
    mod = obj.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels if render_levels is None else render_levels
    return obj


def add_solidify(obj: bpy.types.Object, thickness: float, offset: float = -1.0):
    mod = obj.modifiers.new("Solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = offset
    return obj


def shade_smooth(obj: bpy.types.Object, angle_deg: float = 32.0) -> bpy.types.Object:
    """Smooth shading with an angle split.

    Blender 4.1 removed `mesh.use_auto_smooth`; the replacement is the
    `shade_auto_smooth` operator, which adds a 'Smooth by Angle' node group
    modifier. It needs a real active object, so the context is set explicitly
    here — relying on whatever happened to be active last is exactly what
    breaks in background renders.
    """
    for poly in obj.data.polygons:
        poly.use_smooth = True
    prev_active = bpy.context.view_layer.objects.active
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_auto_smooth(angle=math.radians(angle_deg))
    except Exception:
        # Not fatal: the per-polygon smooth flags above are already set, so
        # the object renders smooth, just without the sharp-edge split.
        pass
    finally:
        obj.select_set(False)
        bpy.context.view_layer.objects.active = prev_active
    return obj


def join(objects: Sequence[bpy.types.Object], name: str) -> bpy.types.Object:
    """Join meshes into one object, preserving each source's material slots.

    Done with bmesh instead of `bpy.ops.object.join` because the operator
    needs an active object plus a matching selection state, and it silently
    no-ops when run headless with an empty context.
    """
    target = bpy.data.meshes.new(name)
    bm = bmesh.new()
    material_index = {}
    for obj in objects:
        eval_mesh = obj.to_mesh()
        # Remap this source's slots onto the joined object's shared slot list.
        remap = {}
        for slot_i, slot in enumerate(obj.material_slots):
            mat = slot.material
            if mat is None:
                continue
            if mat.name not in material_index:
                material_index[mat.name] = len(material_index)
                target.materials.append(mat)
            remap[slot_i] = material_index[mat.name]
        tmp = bmesh.new()
        tmp.from_mesh(eval_mesh)
        tmp.transform(obj.matrix_world)
        for face in tmp.faces:
            face.material_index = remap.get(face.material_index, 0)
        merged = bpy.data.meshes.new(f"{name}_tmp")
        tmp.to_mesh(merged)
        tmp.free()
        bm.from_mesh(merged)
        bpy.data.meshes.remove(merged)
        obj.to_mesh_clear()

    bm.to_mesh(target)
    bm.free()
    joined = bpy.data.objects.new(name, target)
    link(joined, objects[0].users_collection[0] if objects[0].users_collection
         else bpy.context.scene.collection)
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    return joined


def assign_material(obj: bpy.types.Object, material: bpy.types.Material,
                    slot: Optional[int] = None) -> bpy.types.Object:
    if slot is None:
        obj.data.materials.append(material)
    else:
        while len(obj.data.materials) <= slot:
            obj.data.materials.append(None)
        obj.data.materials[slot] = material
    return obj


def apply_all(obj: bpy.types.Object) -> bpy.types.Object:
    """Bake every modifier into the mesh.

    Needed before joining or before reading vertex positions for ground
    truth — `to_mesh()` on an object with a Bevel modifier returns the
    unevaluated cage unless the depsgraph is consulted, which is a common
    source of 'the render looks right but the exported geometry is wrong'.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    baked = bpy.data.meshes.new_from_object(evaluated)
    obj.modifiers.clear()
    old = obj.data
    obj.data = baked
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def iter_fcurves(obj: bpy.types.Object):
    """Yield every F-curve on `obj`'s action, on both action APIs.

    Blender 4.4 introduced slotted/layered actions and removed the flat
    `action.fcurves` collection in 5.x; curves now live at
    action.layers[].strips[].channelbags[].fcurves. Scripts written against
    the old attribute fail with a bare AttributeError, so both shapes are
    handled here rather than at each call site.
    """
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
                bags = [bag] if bag is not None else []
            for bag in bags:
                yield from bag.fcurves


def look_at(obj: bpy.types.Object, target: Vector, roll: float = 0.0) -> None:
    """Aim a camera/light's -Z axis at `target`, +Y up, then roll about -Z."""
    direction = (Vector(target) - obj.location).normalized()
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = (quat @ Quaternion(Vector((0.0, 0.0, 1.0)), roll)).to_euler()
