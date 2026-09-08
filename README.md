# Space-HAR

**AI Human Activity Recognition for on-board BAS experiments.**

An on-board assistant that watches a fixed payload camera, recognises which
step of a scientific protocol the crew member is performing, tells them what
comes next, and raises a voice alert when a step is skipped or performed out
of order — entirely offline, with no link to the ground.

The reference protocol is an 8-step red/yellow payload sort at a payload rack.

```
camera ─┬─► HSV colour detector ────► box positions ─┐
        │                                            ├─► state machine ─┬─► voice alert
        └─► MediaPipe pose ─► 30-frame buffer ─► LSTM ┘                 ├─► timestamped log (txt + JSONL)
                                                                        ├─► Qt dashboard
                                                                        └─► local recording + UDP stream
```

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # or source .venv/bin/activate
pip install -r sih-har-isro/requirements.txt

cd sih-har-isro
python tools/fetch_pose_model.py                   # one-time, then fully offline
python main.py --mode status                       # verify the install
python main.py                                     # run the pipeline on camera 0
```

`--mode status` is the thing to run first. It reports which pose backend is
actually usable, not just whether MediaPipe imports — those are different
questions since MediaPipe 1.0 (see below).

| Command | What it does |
|---|---|
| `python main.py` | Real-time pipeline with the Qt dashboard |
| `python main.py --headless` | No GUI, terminal only |
| `python main.py --video clip.mp4` | Run against a file instead of a camera |
| `python main.py --stream --stream-host 10.0.0.5 --stream-port 5000` | Also push H.264 over UDP |
| `python main.py --mode tuner` | Interactive HSV calibration for your lighting |
| `python main.py --mode status` | Model, data and backend availability |

---

## How accuracy is measured

**Read this before quoting any number from this repository.**

`logs/e2e_report.md` reports eleven passing gates including
`lstm_test_acc 0.984` and `oracle_step_acc 1.000`. Those numbers are
**self-referential and should not be cited**:

- Training and test data both come from `data_generation/synthetic_pose.py`,
  the same closed-form generator, differing only by random seed. The metric
  measures the LSTM's ability to invert a 24-waypoint interpolator.
- `oracle_step_acc` comes from `run_pipeline_on_clip(inject_pose=True)`, which
  feeds ground-truth pose straight into the classifier. MediaPipe is bypassed,
  so camera→pose was never measured.
- `hsv_*_recall` was measured on frames whose boxes `simulation/renderer.py`
  paints in colours chosen to sit inside the detector's own thresholds, drawn
  last so nothing could occlude them.

The supported way to produce an accuracy claim is
[`tools/retrain_and_evaluate.sh`](sih-har-isro/tools/retrain_and_evaluate.sh),
which scores the pipeline against photoreal rendered video whose ground truth
comes from a **different process than the thing being scored**:

```bash
# 1. Render the dataset (see blender/README.md; ~3 h on CPU-only hardware)
tools/render_dataset_resilient.sh dataset/blender study

# 2. Rebuild sequences, retrain, score, and table against the baseline
tools/retrain_and_evaluate.sh
```

`tools/evaluate_real.py` reports `step_acc_real` (camera→MediaPipe→LSTM),
`step_acc_oracle` (ground-truth pose→LSTM), and the gap between them as
`perception_cost` — which tells you whether to fix the classifier or the pose
stage.

### Baseline

The shipped `models/lstm_classifier.pt`, measured this way:

| | Reported | Measured on rendered video |
|---|---:|---:|
| step accuracy (real) | 0.984 | **0.000** |
| step accuracy (oracle pose) | 1.000 | **0.000** |
| HSV red recall | 1.000 | 0.298 |
| HSV yellow recall | 1.000 | 0.130 |

`oracle = 0.000` is the decisive row: handed *perfect* pose, the model still
classifies nothing correctly. It did not learn the activity — it learned one
generator's coordinate patterns.

---

## Repository layout

```
sih-har-isro/
  main.py                  Entry point for every mode
  config/                  Protocol definition, thresholds, HSV bands, gates
  pipeline/
    har_pipeline.py        Real-time orchestration
    pose_backend.py        MediaPipe Tasks API (+ legacy fallback)
    hsv_detector.py        Colour-segmentation object detection (no YOLO)
    state_machine.py       Sequence validation with hold-and-recover
    rack_frame.py          Orientation-agnostic rack-anchored pose normaliser
    hmr_backend.py         Optional 3D human mesh recovery
    voice_alert.py         Offline TTS (Piper, pyttsx3 fallback)
    logger.py              Timestamped txt + JSONL experiment log
    stream_sender.py       ffmpeg → UDP video push
  train/                   LSTM and CNN training
  simulation/              2D synthetic harness (legacy; see caveats above)
  blender/                 3D scene + dataset generator — see blender/README.md
  tools/                   Evaluation, sequence building, model fetch, overlays
```

---

## Requirements notes

**MediaPipe 1.0 removed the `mp.solutions` API.** This project originally
called `mp.solutions.pose.Pose` directly and so failed at import on any
current install. Pose now goes through `pipeline/pose_backend.py`, which
prefers the Tasks API (`PoseLandmarker`) and falls back to the legacy
Solutions API on older versions, behind one unchanged 132-dim contract.

The Tasks API needs a `.task` bundle on disk. `pose_landmarker_lite` and
`_full` ship via Git LFS so a fresh clone is offline-capable;
`tools/fetch_pose_model.py --variant heavy` fetches the larger one on demand.

**Optional, off by default** (each documented in `config/experiment_config.py`):
`ENABLE_STREAMING`, `CNN_ENSEMBLE_ENABLED`, `RACK_FRAME_NORMALIZE`,
`HMR_BACKEND`.

---

## Known issues

- **`.runtime/site-packages` is ~300 MB of checked-in torch/numpy/sympy**,
  which is why a clone is ~330 MB when the source is under 1 MB. It is now
  git-ignored, but reclaiming the space needs a history rewrite that
  invalidates every existing clone — the repository owner's call.
- **`Astra` is a submodule gitlink with no `.gitmodules`**, so
  `git clone --recursive` fails with
  `no submodule mapping found in .gitmodules for path 'Astra'`. Left in place
  pending intent: either add the mapping or `git rm --cached Astra`.
- **`dataset/synthetic_gemini/`** contains 1,027 JPEGs that are flat colour
  cards with the step name printed on them — no astronaut, no boxes. They are
  placeholders from a mock generator, not training data.
- **`end_to_end_loop.py`'s auto-fix loop relaxes its own gates**: a latency
  failure increases `warmup_frames` (excluding frames from the *measurement*),
  and a state-machine failure lowers `STEP_CONFIRM_FRAMES` — the safety
  debounce — and persists it to `config/tuned_overrides.json` so the relaxed
  value ships. Left unchanged because gating policy is a project decision, but
  it is why the e2e report always passes.

---

## Problem statement coverage

| Requirement | Where |
|---|---|
| Continuously process local video to track the sequence | `pipeline/har_pipeline.py` |
| Suggest the next step at start and after each step | `voice_alert.alert_step_next`, called from the state machine callbacks |
| Voice alert on skipped or out-of-sequence steps | `pipeline/voice_alert.py`, `pipeline/state_machine.py` |
| Timestamped structured log with outcomes | `pipeline/logger.py` (txt + JSONL, three-tier outcome) |
| Stream to a specific IP **and** store locally | `pipeline/stream_sender.py` + `cv2.VideoWriter` |
| GUI for monitoring | `gui/qt_dashboard.py` |
| Runs offline on a standalone system | No network at inference; pose weights ship with the repo |
| *Optional:* orientation-agnostic 3D tracking | `pipeline/rack_frame.py` (validated by the 0–180° render sweep), `pipeline/hmr_backend.py` |
