# Space-HAR — Astronaut Activity Recognition for the BAS Protocol

**Team Traviators** · Smart India Hackathon 2026

| Field | Detail |
|---|---|
| Problem Statement ID | **26174** |
| Problem Statement Title | AI Human Activity Recognition for On-board BAS Experiments |
| Organization | Indian Space Research Organisation (ISRO) |
| Theme | Space Technology |
| Category | Software |
| Team Name | Traviators |

A single fixed camera watches a crew member run an 8-step scientific
protocol on the Bharatiya Antariksh Station. The system recognises which
step is being performed, confirms it, tells the crew what to do next, and
raises a voice alert the moment a step is skipped or done out of order —
**entirely offline**, on one laptop GPU, with no link to the ground. That
matters here specifically because orbital communication delay runs from
1.3 seconds to 24 minutes: real-time verification from mission control is
not physically possible, so the verification has to happen on board.

```
CAPTURE               VISION (3 models, run together)              TEMPORAL MODEL            PROTOCOL ENGINE
──────────            ───────────────────────────────               ──────────────            ────────────────
fixed camera   ──┬──► YOLOv8n            → red/yellow/main box  ──┐
100 ms clock      │   MediaPipe (2D)      → 33-landmark skeleton  ├─► fuse → 143-d features ─► BiLSTM 3×256   ─► confirm / advance /
upright + rack    │                                               │    → 3 s / 30-sample        + attention       skip-check / HOLD
frame             └──► HMR3DNet (ours)    → 3D pose, any roll   ──┘    sliding window        → entropy gate       + recover
                                                                                                                ─► voice · dashboard ·
                                                                                                                   JSONL/TXT log ·
                                                                                                                   UDP stream · MP4
```

Full diagram: [`sih-har-isro/docs/sih-architecture/slide1.png`](sih-har-isro/docs/sih-architecture/slide1.png).
Literature grounding and prior-art comparison: [`sih-har-isro/docs/sih-architecture/research.png`](sih-har-isro/docs/sih-architecture/research.png).

---

## What makes this different from "run YOLO + MediaPipe on a video"

1. **A perception model trained specifically for a suited body in microgravity.**
   Pretrained 3D human-pose models are trained on clothed people standing
   upright on Earth. Measured directly on this project's own footage, a
   pretrained model tracks **0.00** of the body at 60° of roll — not
   degraded, failed. There is no public dataset of suited astronauts at
   arbitrary orientations to fine-tune from, so we built one: a Blender
   digital twin of the rack, boxes and suited crew member rendered at every
   roll, pitch and lighting condition, with exact 3D ground truth baked in
   by the render itself (33 takes, 31,891 frames). Our model, **HMR3DNet**,
   trained on that data, goes from 0.00 to **0.77** tracked joints at 60°
   roll, and 0.60 to **0.91** upside down — see the honest before/after table
   below, including where it still doesn't work.
2. **Orientation-agnostic by construction.** The frame is canonicalised
   upright and every joint is expressed relative to the payload rack, not
   the camera's floor — so a crew member floating at any angle produces the
   same features a ground-standing one would.
3. **A model-contract guard, not a hope.** Every checkpoint records the
   exact feature layout, sampling clock and detector it was trained on.
   `pipeline/har_pipeline.py` refuses to serve a model whose recorded
   contract doesn't match the runtime, instead of silently producing
   wrong predictions on mismatched features — this is how a real
   train/serve mismatch was actually caught during development.
4. **An entropy-gated decision engine, not a raw softmax.** A step is only
   confirmed after 5 consecutive samples agree, with confidence ≥ 0.65 and
   prediction entropy ≤ 1.4 nats. The system is built to stay silent when
   unsure rather than announce a wrong step with false confidence.
5. **Honest evaluation, split by take, never by frame.** Every accuracy
   number below is measured on protocol runs the reporting model never
   trained on. Adjacent frames of the same take are near-duplicates; a
   frame-level split would leak and inflate every number in this file.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r sih-har-isro/requirements.txt

cd sih-har-isro
python tools/fetch_pose_model.py                    # one-time, then fully offline
python main.py --mode status                        # verify weights + backends are present
python main.py --mode pipeline --camera 0            # run live
```

| Command | What it does |
|---|---|
| `python main.py --mode pipeline --camera 0` | Real-time pipeline on a webcam, with the Qt dashboard |
| `python main.py --mode pipeline --video clip.mp4 --headless` | Run against a file, no GUI |
| `python main.py --mode pipeline --camera 0 --viz-3d hmr` | Drive the monitor's 3D panel from HMR3DNet instead of MediaPipe's own 3D |
| `python main.py --mode pipeline --stream --stream-host <ip> --stream-port 5000` | Also push H.264 over UDP to a ground station |
| `python main.py --mode status` | Reports which pose backend, detector and model checkpoints actually load |
| `python tools/render_har_video.py --video clip.mp4 --out monitor.mp4 --viz-3d hmr` | Render the full monitor view (skeleton, 3D panel, checklist) for a video file, offline |

---

## Measured results

Every number below is on takes the reporting model was never trained on, held out
by take (whole protocol run), not by frame. Full test suite: 56 automated
checks — `cd sih-har-isro && pytest`.

**Object detector — YOLOv8n, fine-tuned on the exact rack, boxes and suit**

| | mAP@50 |
|---|---:|
| Synthetic-only baseline, on synthetic validation | 0.933 |
| Synthetic-only baseline, on an unseen real clip | 0.355 |
| Fine-tuned on real + synthetic, on an unseen real clip (held out entirely) | **0.798** |

**3D pose — HMR3DNet vs. a pretrained human-mesh-recovery model, same crop**

Tracked = median upper-body joint error under 0.5× torso length, against Blender's
exact ground truth, on takes never seen during training.

| Orientation | Pretrained model | HMR3DNet (ours) |
|---|---:|---:|
| 30° roll | 0.00 – 0.04 | **0.80** |
| 60° roll | 0.00 | **0.77** |
| 180° roll (inverted) | 0.60 | **0.91** |
| 90° roll (edge-on to one camera) | 0.00 | 0.07 — still an open limit |

At exactly 90°, half the body is self-occluded from a single fixed camera. That's
a genuine information limit of one viewpoint, not something more training data
alone fixes — a second camera angle is the honest next step, not claimed as solved.

**Step classifier — BiLSTM, 3 layers × 256, attention, 143-d features**

| | Step accuracy |
|---|---:|
| Reported validation accuracy (own split) | 0.754 |
| Apples-to-apples: same held-out takes, before this session's detector retrain | 0.680 |
| Apples-to-apples: same held-out takes, after | **0.707** |

**Live throughput:** 14–16 FPS end-to-end (capture → perception → fusion →
classification → decision → render) on a single RTX 3050 laptop GPU, fully offline.

---

## Repository layout

```
sih-har-isro/
  main.py                    Entry point for every mode (pipeline / train / status / ...)
  config/experiment_config.py  Protocol steps, thresholds, sampling clock, feature layout
  pipeline/
    har_pipeline.py          Real-time orchestration: capture → perceive → fuse → classify → decide
    yolo_detector.py         YOLOv8n red/yellow/main-box detector
    pose_backend.py          MediaPipe Tasks API pose (Tasks API + legacy fallback)
    upright_pose.py          Rotates the frame upright before pose runs (crew at any roll)
    rack_frame.py            Joints expressed relative to the payload rack, not the camera
    hmr_backend.py           HMR3DNet: our own 3D pose model for suited, floating crew
    pose_viz.py              Display-only: stabilised skeleton, rigid-bone 3D view, monitor panel
    features.py              Shared feature-building code — used by BOTH training and live inference
    state_machine.py         Confirm / advance / skip-check / HOLD-and-recover
    voice_alert.py           Offline TTS (Piper, pyttsx3 fallback)
    logger.py                Timestamped JSONL + TXT experiment log
    stream_sender.py         ffmpeg → UDP video push
  train/
    train_lstm.py            Step classifier, held out by take
    train_hmr3d.py            HMR3DNet, supervised by Blender's own 3D ground truth
    hmr3d_model.py           MobileNetV3-Small backbone + regression head
  tools/
    build_sequences.py       Camera frames → training windows (same code path the live loop runs)
    promote_model.py         Checks a checkpoint's contract before it can be served live
    evaluate_real.py         Honest accuracy scoring on real/held-out footage
    render_har_video.py      Renders the full live monitor view for an offline video file
  blender/                   Digital-twin scene + dataset generator
  gui/qt_dashboard.py        PyQt6 monitoring GUI
  docs/sih-architecture/     Architecture and research-reference diagrams (source + rendered PNGs)
  tests/                     56 pytest checks: contract guard, feature layout, state machine, ...
```

---

## Problem statement coverage

| Requirement | Where |
|---|---|
| Train an AI model that recognises and validates a pre-defined experiment sequence | `train/train_lstm.py`, `train/train_hmr3d.py`, `train/train_yolo_boxes.py` |
| Standalone / edge: process locally, no raw video to ground | `pipeline/har_pipeline.py` — inference is fully offline; only `--stream` optionally pushes an annotated feed |
| Dataset generation for detection, pose and hand-object interaction | `blender/` digital twin (33 takes, 31,891 frames), `tools/build_sequences.py` |
| Continuously process local video, suggest the next step | `pipeline/state_machine.py`, `pipeline/voice_alert.alert_step_next` |
| Voice alert on a skipped or out-of-sequence step | `pipeline/voice_alert.py` + `pipeline/state_machine.py`'s recovery path |
| Timestamped, structured, lightweight log of steps and outcomes | `pipeline/logger.py` (JSONL + TXT) |
| Stream to a specific IP **and** store locally | `pipeline/stream_sender.py` (UDP) + `cv2.VideoWriter` (local MP4), both running together |
| GUI for monitoring | `gui/qt_dashboard.py`, plus the render-side monitor in `pipeline/pose_viz.py` |
| Deliverable: trained model on an offline standalone system | `models/*.pt` / `*.onnx`, served through `tools/promote_model.py`'s contract check |
| *Optional:* orientation-agnostic 3D tracking relative to the rack, not the floor | `pipeline/rack_frame.py`, `pipeline/upright_pose.py`, `pipeline/hmr_backend.py` |

---

## Known limitations

Stated plainly, because every number above is reported the same way:

- **Exactly 90° roll is still an open problem** (0.07 tracked, see table above) — a
  single-camera self-occlusion limit, not yet closed by more training data.
- **A camera mounted directly overhead** degrades MediaPipe's 2D skeleton — it is
  trained on front/side viewpoints. Object detection and step classification are
  largely unaffected; the skeleton overlay is the part that suffers.
- **A clip that starts mid-protocol** correctly reports step 1 as skipped — that's
  the state machine working as specified, not a bug, but it currently repeats the
  warning every few seconds rather than once per missed step.
- **`Astra/`** is a separate, unrelated prototype tracked as a bare gitlink with no
  `.gitmodules` entry — not part of this submission; left untouched.

---

## Tech stack

Python · PyTorch · Ultralytics YOLOv8 · MediaPipe (BlazePose) · OpenCV · ONNX Runtime ·
PyQt6 · NumPy · FFmpeg · Blender (digital-twin data generation) · pytest (56 tests) ·
CUDA (RTX 3050, real-time inference)

---

## Team

| Name | Role |
|---|---|
| _fill in_ | _fill in_ |

Demo video: `<link>` · Presentation: `<link>`
