import bpy

for obj in bpy.data.objects:
    print(f"OBJECT {obj.name} type={obj.type}")
    if obj.type == "ARMATURE":
        print("BONES", ",".join(b.name for b in obj.pose.bones))
