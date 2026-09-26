# BAS digital twin: data and presentation policy

`main.py --mode sim3d --render` produces the Blender-based fixed-payload-camera
scene. It is a synthetic test and data-generation asset, not evidence that the
real-time model has detected anything.

## Outputs

| File | Purpose |
| --- | --- |
| `dataset/blender_twin/bas_har_digital_twin.blend` | Editable scene: rack, camera, lighting, astronaut and payloads. |
| `dataset/blender_twin/raw_frames/` | Clean RGB frames. Feed these to the HAR pipeline. |
| `dataset/blender_twin/bas_blender_rgb.mp4` | Clean camera-feed video. |
| `dataset/blender_twin/annotations_ground_truth.json` | Blender-derived labels for synthetic training/evaluation. |
| `dataset/blender_twin/bas_blender_ground_truth_preview.mp4` | Review-only boxes, explicitly labelled as simulation ground truth. |

## What the scene models

- A fixed payload camera and payload-rack coordinate frame.
- Free-float object motion between handoff and rack target, followed by target
  latching. This is an intentionally simplified microgravity dynamics model.
- Red/yellow payload identities, object locations and per-frame 2D labels.

## What must be real in the final system

- Run object, pose/hand and temporal HAR inference on clean RGB only; draw HUD
  boxes only from those model outputs.
- Read gravity/acceleration from a spacecraft or payload IMU. Video can assess
  visual motion/stability, but cannot measure gravity reliably.
- Replace the included CC0 low-poly rig before a final pitch video with a
  licensed high-fidelity, articulated EVA/IVA astronaut and protocol-specific
  motion capture. Record its asset license and attribution in the submission.

NASA's current astronaut GLB is kept as a visual reference under
`assets/source/nasa_astronaut.glb`; NASA 3D-resource use remains subject to
NASA media/identifier guidance. It is not a rigged motion-capture character.
