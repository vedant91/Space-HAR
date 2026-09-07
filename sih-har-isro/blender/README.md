# BAS Payload-Rack Scene — 3D asset and dataset generator

A procedurally-built Bharatiya Antariksh Station science module, a rigged
suited crew member, and the 8-step payload protocol animated in 3D — rendered
from fixed payload cameras with **exact per-frame ground truth**.

Everything is generated from code. There are no binary `.blend` assets to
drift out of sync with the pipeline: the scene, the rig, the protocol and the
ground-truth exporter all read the same constants out of
`config/experiment_config.py`.

---

## Why this exists

The repository already had a "simulation": `simulation/renderer.py` draws a 2D
OpenCV scene, and `data_generation/synthetic_pose.py` produces pose sequences
by interpolating three wrist waypoints per step. Those two are the training
data, the test data *and* the evaluation harness. Concretely, before this
work:

| Reported metric | What it actually measured |
|---|---|
| `lstm_val_acc 0.998` | Trained on `generate_dataset(seed=7)` |
| `lstm_test_acc 0.984` | Tested on `generate_dataset(seed=99)` — **the same closed-form generator** |
| `oracle_step_acc 1.000` | Ground-truth pose injected straight into the LSTM; MediaPipe bypassed entirely |
| `hsv_red_recall 1.000` | Boxes painted in colours picked to sit inside the detector's own thresholds, drawn last so nothing occludes them |

None of those numbers involve a camera, an image, or a pose estimator. The
first honest measurement — this scene, rendered, run through the real
pipeline — put the shipped LSTM at **0.000** step accuracy.

This generator exists so the project has data whose *labels come from a
different process than the thing being scored*.

---

## Quick start

```bash
# Build the scene and render one preview per payload camera
blender --background --factory-startup --python blender/build_scene.py -- \
    --preview build/preview --seed 3

# Numeric sanity checks (IK reach, solver convergence, framing, visibility)
blender --background --factory-startup --python blender/validate_scene.py -- --stride 6

# Render the dataset (resumable; see the supervisor script below)
blender --background --factory-startup --python blender/render_dataset.py -- \
    --out dataset/blender --plan study --width 900 --height 506 --samples 16

# On a machine with a flaky GPU driver, run it supervised instead:
tools/render_dataset_resilient.sh dataset/blender study 900 506 16
```

---

## Layout

| File | Role |
|---|---|
| `bas_har/common.py` | Mesh/bmesh helpers, bevel + smooth shading, modifier baking, action-API compatibility |
| `bas_har/materials.py` | PBR library. Owns the payload colour contract (see below) |
| `bas_har/module.py` | Module shell, four rack bays, the experiment rack, payload, lighting |
| `bas_har/astronaut.py` | Segmented pressure suit + armature + arm IK + neutral body posture |
| `bas_har/protocol.py` | The 8 steps as 3D hand trajectories; error sequences |
| `bas_har/groundtruth.py` | MediaPipe-33 landmark export, occlusion-aware visibility, true payload boxes |
| `bas_har/render.py` | EEVEE configuration and the fixed payload-camera rig |
| `build_scene.py` | Entry point: assemble scene, preview, save `.blend` |
| `validate_scene.py` | Entry point: numeric checks |
| `render_dataset.py` | Entry point: render takes + ground truth |

---

## Coordinate convention

```
+X   along the module axis (down the tunnel)
+Y   into the experiment rack face
+Z   the rack's own long axis
```

The rack face is the plane `y = 0`. Structure lives behind it (`y > 0`);
everything the crew touches lives in front of it (`y < 0`). The module axis
runs through `(0, -1.03, 1.0065)`, and all four rack bays are the *same* local
build rotated about that axis — so "there is no floor" is true structurally,
not just asserted in a comment.

The crew member's local frame has `+Y` = facing direction and `+X` = their own
**right**. MediaPipe landmark names are anatomical, so `groundtruth.py` maps
`.L` bones (on `-X`) to MediaPipe's *left* indices. Getting this backwards
silently mirrors every exported label and nothing downstream would catch it.

---

## The payload colour contract

`pipeline/hsv_detector.py` thresholds OpenCV HSV. The three payload colours
must land inside those exact bands **after** the render's view transform:

| Object | Rendered OpenCV HSV | Config band |
|---|---|---|
| `red_box` | H 0, S 239, V 219 | H ≤ 10, S ≥ 120, V ≥ 70 |
| `yellow_box` | H 24, S 239, V 240 | H 20–35, S ≥ 100, V ≥ 100 |
| `main_box` | S 3, V 232 | S ≤ 30, V ≥ 200 |

**Dataset frames are rendered with the `Standard` view transform, not
Blender's default AgX.** AgX is a filmic tone map that deliberately
desaturates saturated colour; pure red comes out of it around S≈0.55 and falls
*outside* the detector's S ≥ 120 floor at the highlights. Rendering the
dataset through AgX would have silently halved HSV recall and looked like a
detector bug. `materials.verify_payload_hsv()` asserts the contract and
`render_dataset.py` records the result in every take's `meta.json`.

AgX is still used for showcase stills (`--showcase`), where fidelity to the
detector does not matter and filmic highlight roll-off looks better.

### Distractors

Real ISS handrails are a pale anodised gold that sits very close to
`HSV_YELLOW`. `--distractors` controls this deliberately:

- `none` — nothing in frame near the payload bands. Best case.
- `moderate` *(default)* — realistic warm cable ties and caution placards. A
  detector that only works in an empty world is not evidence of anything.
- `harsh` — ISS-accurate gold handrails. Use this to quantify how much of the
  reported yellow recall is an artifact of a clean scene.

---

## Ground truth

Per camera, per take:

```
take_XXXX/<camera>/pose_2d.npy      (T, 132)   MediaPipe convention, x/y/z/visibility
take_XXXX/<camera>/pose_world.npy   (T, 33, 3) metric world-frame joints
take_XXXX/<camera>/labels.npy       (T,)       step id per frame
take_XXXX/<camera>/payload.json                true pixel bbox + visible fraction
```

`pose_2d` uses MediaPipe's own convention: `x, y` normalised with origin at
top-left, `z` relative to the hip midpoint in roughly `x` units.

**Visibility is ray-cast, not assumed.** The subtlety is that a pose landmark
is a joint *centre*, which on a pressure suit sits 5–9 cm inside the shell — a
naive "did anything block the ray" test reports every landmark as occluded,
because the suit occludes its own skeleton. The rule used instead: find the
first hit; if it is within 16 cm of the landmark we are looking at that limb's
own surface and it counts as visible; a hit substantially in front is a real
occluder. Sanity check with the camera on the crew's right:

```
R_wrist    clear=1.00      L_wrist    clear=0.43   (visible only when raised)
R_shoulder clear=1.00      L_shoulder clear=0.00   (behind the torso)
```

Verify any take visually:

```bash
python tools/overlay_pose.py --take dataset/blender/take_0000 --frames 10,70,120
```

---

## The crew member

A 1.71 m figure on a 19-bone armature, built as **rigid suit segments** rather
than a smooth-skinned body — which is what a real pressure garment is: hard
upper torso, hard helmet, soft limb sections separated by convolute
(accordion) rings at every joint. Each segment is weighted 100% to one bone,
so there is no weight painting to go wrong and the joints stay watertight at
any bend angle.

Arms are driven by two-bone IK with pole targets, so a protocol step specifies
*where the hand has to be on the payload* and the arm configuration follows.
`validate_scene.py` confirms the solver actually gets there:

```
arm reach 0.538 m   max reach ratio 0.95   reach violations 0
IK convergence error ≤ 0.1 mm   landmarks off-screen: 0
```

The legs use the **neutral body posture** (NASA-STD-3001): hips flexed ~42°,
knees ~38°. A relaxed human in freefall does not hang straight, and legs
hanging vertically would put a gravity pose in every frame of a dataset whose
entire premise is that there is no gravity.

`--roll` rotates the whole crew member about the rack normal. This is the
microgravity attitude axis and the reason `pipeline/rack_frame.py` exists; it
is a first-class argument rather than something a caller has to construct.

---

## Render plans

| Plan | Takes | Purpose |
|---|---|---|
| `smoke` | 1 | 160 frames, one camera — pipeline check |
| `orientation` | 8 | Orientation sweep only |
| `study` | 12 | The plan actually rendered here: 6 orientations, 2 lighting, 2 clutter, 2 protocol-error takes |
| `core` | 18 | Adds a second camera and held-out viewpoints |

Train/test are split **by take**, so a held-out orientation is genuinely
unseen — which is what the orientation-agnostic requirement is actually about.

### Known fidelity gap: step duration

`study` renders 32 frames per step. At the pipeline's nominal 30 fps that is
**~1.07 s per step, against the 4–6 s in `EXPERIMENT_STEPS[*].duration_hint_sec`**.
The motion is therefore roughly 4× faster than a real crew member's, so
landmark *velocities* in this dataset are not to scale even though positions
and geometry are.

This is a deliberate trade against the available hardware, not an oversight —
matching the real 4–6 s would mean 120–180 frames per step, i.e. ~1,400 frames
per take and roughly 2.5 hours of render each on this machine. It is recorded
here because it bounds what the resulting accuracy number means.

Two ways to close it, in order of preference:

1. **Render longer steps** on a machine with a CUDA GPU: raise
   `frames_per_step` to 120–150 in the plan.
2. **Sample the live stream to match.** Arguably the better engineering fix
   regardless: a 30-frame window at 30 fps only spans 1 s, which is too short
   to contain a 5 s action in the first place. Feeding the buffer at ~10 fps
   gives the same 30-frame LSTM contract a 3 s temporal span, which fits the
   protocol's actual step durations far better. That is a change to
   `PROCESS_EVERY_N_FRAMES` at inference, not to the model.

---

## Performance and stability notes

This was developed on an Intel i7-8665U with **UHD 620 integrated graphics and
no CUDA GPU**, despite the config file targeting an RTX 3050. Consequences:

- **EEVEE, not Cycles.** Cycles on 8 CPU threads at 1.9 GHz is not viable for
  thousands of frames. EEVEE Next at 16 samples gives ~4 s/frame at 900×506.
- **Screen-space raytracing is off for dataset renders** (`quality="dataset"`)
  and on only for showcase stills. It is the heaviest EEVEE feature and the
  most likely to destabilise a weak GPU driver over a long unattended run.
- **The Intel driver crashes.** Sustained EEVEE work reliably produces an
  `EXCEPTION_ACCESS_VIOLATION` inside `ig9icd64.dll` after a few hundred
  frames. This is a driver fault — the same frames render fine on relaunch.
  `render_dataset.py` is therefore resumable at both take and frame
  granularity, and `tools/render_dataset_resilient.sh` restarts it until the
  run completes, aborting if an attempt makes no forward progress so a real
  failure is not hidden by a retry loop.

On a machine with a CUDA GPU, raise `--samples`, enable raytracing, and
consider Cycles for the showcase renders.
