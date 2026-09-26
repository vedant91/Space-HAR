# Handoff: YOLO + MediaPipe + rack geometry

**Repo:** `https://github.com/vedant91/Space-HAR`, branch `The-final`.
**Read this whole file before touching code.** It's written so you don't need the
other half of the team's conversation history — everything you need to know is here
or pointed to by exact file path.

## The project, in one paragraph

SIH 2026, problem statement 26174. An astronaut on the Bharatiya Antariksh Station
(BAS) does a fixed 8-step experiment (open a box, pick/examine/place a red box,
pick/examine/place a yellow box) in front of a camera fixed to the payload rack.
The system must recognise which step is happening, catch skipped/out-of-order
steps, speak an alert, log everything, stream video, and run **fully offline** on
a laptop GPU (RTX 3050, 6 GB). No gravity reference exists — the astronaut can be
at any orientation relative to the rack, which is why the architecture measures
everything relative to the rack, not "up/down".

Full architecture (read this first, it has diagrams): `docs/architecture.html`
Full 24-hour plan: `docs/ultimate-plan.html`
One-scenario walkthrough of the whole system end to end: `docs/scenario.html`

## The split

Two people, two halves, one architecture. **You own the 2D perception side.
The other half owns 3D HMR/ROMP.** Both halves plug into the same shared files
along clean interfaces described below — stick to those interfaces and the merge
should be close to painless.

**You own:**
1. YOLOv8n box detector (replaces the current colour-threshold detector)
2. MediaPipe hands/pose robustness (mostly already built — see below)
3. ArUco rack marker → rack transform (new)
4. The Qt GUI import fix
5. Real webcam data collection + labelling

**The other half owns:** ROMP/HMR crop+rotate+adaptive-schedule fix, person
tracking for the floating astronaut, the 2D/3D pose fusion gate, and the 3D
dashboard panel. You do not need to touch `pipeline/hmr_backend.py` or the
`.venv-hmr` environment at all.

## 1. YOLO detector — replace the colour detector

**The file you're replacing:** `pipeline/hsv_detector.py`. Read it first — it's
short. The class `HSVBoxDetector` is what you're building a drop-in replacement
for.

**The exact interface you must match** (so nothing downstream changes):

```python
@dataclass
class Detection:
    label: str                              # "red_box" | "yellow_box" | "main_box"
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2
    confidence: float                        # 0-1
    centroid: Tuple[int, int]
    area: int
    mask: Optional[np.ndarray] = None
    rect: Optional[Tuple[float, float, float]] = None  # ((cx,cy),(w,h),angle_deg)

class YourDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...
    def draw(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray: ...
```

`detect()` and `draw()` are called from `pipeline/har_pipeline.py` (grep for
`hsv_detector.detect` and `hsv_detector.draw` — there are ~4 call sites) and from
`tools/build_sequences.py`. Match the signature exactly and you can swap the
detector in behind a config flag without touching either of those files' logic.

**New classes to add** (this is the actual improvement over the colour detector):
`main_box_open`, `main_box_closed`, `red_box`, `yellow_box`. Open/closed as
separate classes (not a separate state machine) is what tells the system the lid
state directly from one detection, which is a real accuracy win — see
`docs/ultimate-plan.html` section 3.

**IMPORTANT — do not fine-tune away the `person` class.** The HMR half of this
project needs YOLO to also detect a generic `person` box, to crop before running
ROMP. If you fine-tune YOLOv8n on only your 4 box classes, standard fine-tuning
will overwrite its ability to detect `person` (catastrophic forgetting on the
COCO classes). **Fix: include `person` as a 5th class in your training set**
(no extra images needed — every one of your training frames already has a
person in it; label the person box too), or run two separate YOLO instances
(cheap, stock-COCO one for `person`, your fine-tuned one for boxes). Pick
whichever is less work, but confirm which one you did with the HMR side before
they build against it — this is the one dependency that will silently break the
other half's work if it's missed.

**Training data:**
- Free labels: `dataset/blender_core/take_*/payload_a/payload.json` has the
  ground-truth pixel bbox + visibility for all 3 boxes on every one of 22,349
  rendered frames. Write `tools/export_yolo_labels.py` to convert this to YOLO
  txt format (one label file per frame, class + normalized cx,cy,w,h). This is
  a ~30-line script — read `payload.json`'s structure directly
  (`json.load(open(...))`, it's a list of per-frame dicts keyed by object name)
  before writing it.
- Real data: record 10-15 webcam takes with real red/yellow boxes and a hinged
  main box (`tools/record_take.py` already exists for this). Auto-label ~300
  real frames with your synthetic-trained YOLO first, then hand-correct in
  CVAT/Roboflow — much faster than labelling from zero.
- Train: `pip install ultralytics`, then standard `yolo detect train
  model=yolov8n.pt data=<your data.yaml> epochs=40 imgsz=640 device=0`.

**Target:** yellow-box recall clearly above the current ~0.15 (the colour
detector's number — it's the reason the whole model is weak right now).

## 2. MediaPipe — mostly already built, make it robust

`pipeline/pose_backend.py` (class `PoseBackend`) already runs MediaPipe Hands +
Pose and outputs a fixed 132-dim vector: 33 landmarks × (x, y, z, visibility),
normalized to [0,1] image coordinates. This is a stable, already-working
contract — **do not change its output shape** (`NUM_LANDMARKS = 33`,
`FEATURE_DIM` in that file). Downstream code (`pipeline/features.py`, the HMR
fusion gate) depends on exactly this shape.

`pipeline/upright_pose.py` already rotates the frame to compensate for
in-plane camera/crew roll before MediaPipe runs. Your job here is verification
and robustness on **real** webcam footage, not rebuilding MediaPipe itself:
- Check hand-tracking reliability when a hand is partially inside the open main
  box (occlusion is common in this experiment).
- Check behaviour under the lighting conditions you'll actually demo with
  (dim room, one lamp, one lamp switched off mid-take).
- Confirm the roll correction actually improves detection on a real person
  leaning/rolling relative to the camera, the same way it's proven on the
  Blender renders.

## 3. ArUco rack marker → rack transform (new work)

**Why this matters to both halves:** the whole "measure everything relative to
the rack, not gravity" idea depends on knowing where the rack is. Right now
`pipeline/rack_frame.py`'s `RackFrameNormalizer` estimates rack angle from box
geometry (`pick_rack_rect`) — a reasonable fallback, but a printed marker is
more reliable and is what the architecture calls for.

**What to build:** print an ArUco marker, fix it to a corner of the mock rack,
and use OpenCV's built-in `cv2.aruco` (no new dependency — already ships with
`opencv-contrib-python`) + `cv2.solvePnP` to compute the camera-to-rack
transform. Expose it as a small, clearly-typed function, e.g.:

```python
def estimate_rack_pose(frame_bgr: np.ndarray) -> Optional[RackPose]:
    """Returns None if the marker isn't visible this frame (use the last
    known pose, or fall back to RackFrameNormalizer's geometry method)."""
```

**This is the one function the HMR half directly needs** — they use it to
project ROMP's 3D joints into rack space (see `docs/architecture.html` section
1, the "Rack pose (solvePnP)" box). Agree on the exact return type (rotation +
translation, or a 4×4 matrix — either is fine) with them before you both build
against it. `pipeline/rack_frame.py`'s existing `to_rack_xy()` method is the
current 2D equivalent — read it for the coordinate convention already in use
(rack-plane x/y) so your 3D version is consistent with it.

## 4. GUI fix

`gui/qt_dashboard.py` currently fails to import: `PyQt6` isn't installed.
`pip install PyQt6` and confirm `python main.py` (mode: `pipeline`) launches
the dashboard. Add `PyQt6` to `requirements.txt`. Low effort, blocks the whole
"GUI for monitoring" requirement until it's done.

## Files you own (edit freely)

- `pipeline/hsv_detector.py` → new `pipeline/yolo_detector.py` (keep the old
  one; it's the fallback)
- `pipeline/pose_backend.py`, `pipeline/upright_pose.py` (robustness only,
  keep the output contract)
- `pipeline/rack_frame.py` (add the ArUco method; don't remove the existing
  geometry fallback)
- `tools/export_yolo_labels.py` (new)
- `gui/qt_dashboard.py`, `requirements.txt`
- Your own YOLO training scripts/configs

## Files that are shared — coordinate before editing

- **`pipeline/features.py`** — the single shared feature-building contract
  used by training and live inference. Both halves may need to touch this
  (you: swapping the detector's output into the existing 11-dim box features;
  them: adding a `fused` pose source for HMR). **Edit additively only** — the
  existing `pose132` / `pose132+obj11` layouts must keep working unchanged.
  There's a full test suite (`tests/`, run `pytest`) that will catch a broken
  contract; keep it green.
- **`pipeline/har_pipeline.py`, `tools/build_sequences.py`** — both halves
  call into these. Prefer adding a config flag (there's already a pattern for
  this — grep `HMR_BACKEND`, `--pose-source` in `tools/build_sequences.py`)
  over changing existing branches. Small, frequent commits and a quick message
  to each other before touching either file avoids merge pain.
- **`config/experiment_config.py`** — add your own new constants (YOLO model
  path, confidence thresholds, ArUco marker ID/size), don't reorder or remove
  existing ones.

## Before you start

1. `git pull` on `The-final` and confirm you actually have the HMR work
   (`pipeline/hmr_backend.py` should have a `romp_to_feature_vector` function
   and a `check_pose_source` guard — if it doesn't, the branch hasn't been
   pushed yet, check with the other half first).
2. `pip install -r requirements.txt` and run `pytest` — should be green before
   you change anything.
3. `python main.py status` shows what's currently available/missing.

## What "done" looks like for your half

- YOLO detector recall clearly beats the colour detector on yellow box, with
  the `person` class still working (confirm with a quick `results.boxes.cls`
  check on a person-only test image).
- MediaPipe confirmed working on real webcam footage in the actual demo
  lighting.
- ArUco rack pose returns a stable transform when the marker is visible.
- `python main.py --mode pipeline` launches with the Qt GUI, no crash.
- `pytest` still green.
