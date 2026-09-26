import bpy
for o in bpy.data.objects:
    if o.type in {'MESH','ARMATURE'}:
        print(o.name, 'loc=',tuple(round(x,3) for x in o.location), 'dim=',tuple(round(x,3) for x in o.dimensions), 'scale=',tuple(round(x,3) for x in o.scale))
