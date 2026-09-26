# BAS HAR — Unified Target Architecture

**SIH 2026 · PS 26174 · ISRO / Department of Space**

Synthesized from wins across the strongest competitor repos (Orion, BASGUARD, BAS-Monitor, AEGIS/Pranvkumar, Sheen11, vedant91/Space-HAR handoff) plus the current repo's already-shipped work. This is the target, not a diff.

---

## 0. Design axioms (non-negotiable)

1. **Models propose, FSM decides.** No neural net advances a step. Every state transition traces to a deterministic rule reading model outputs.
2. **Rack-relative everything.** No feature depends on gravity. All spatial features expressed in rack coordinates via ArUco homography.
3. **Contract-tracked model artifacts.** Every checkpoint records `(pose_source, detector_source, feature_schema_version, rack_frame_source)`. Load-time guard refuses mismatched pairs. Silent train/inference distribution mismatch is the #1 killer of this class of system — enforced structurally.
4. **YAML-driven experiments.** Adding a new protocol = editing a YAML file. Models never see step numbers, only atomic states (`holding_red`, `main_box_open`, `placed_in_zone_A`).
5. **Standalone offline.** No cloud, no LLM API, no internet at inference. All models bundled in artifact directory.
6. **Custom-trained models only** (per PS wording). One exception: YOLOv8n opt-in for detector, fine-tuned with `person` class preserved. Justified in submission; not the default.

---

## 1. Runtime data flow

```
                    ┌─────────────────────────────────────────────────┐
    INPUT           │  Fixed payload camera (USB 1080p @ 30fps)       │
                    │  + 4 printed ArUco markers on rack corners      │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────────┐
    PREP            │  Frame sampler (10 Hz decision clock,           │
                    │  decoupled from camera fps)                     │
                    │  + Rack pose: cv2.aruco.detectMarkers →         │
                    │    solvePnP → 4×4 T_cam_rack                    │
                    │  + Fallback: RackFrameNormalizer (geometry)     │
                    │    when markers occluded                        │
                    └────────────────────┬────────────────────────────┘
                                         ▼
    PERCEPTION      ┌──────────┬──────────┬──────────┬─────────────┐
    (parallel)      │  YOLO    │  PoseNet │  HOI-CNN │  ByteTrack  │
                    │  5-class │  (custom │  (wrist  │  (Kalman +  │
                    │  detector│  from    │  crops → │  IoU on all │
                    │  + person│  scratch)│  states) │  YOLO dets) │
                    └────┬─────┴────┬─────┴────┬─────┴──────┬──────┘
                         │          │          │            │
                         ▼          ▼          ▼            ▼
                    ┌─────────────────────────────────────────────────┐
    FEATURES        │  pipeline/features.py  (SAME code, train+live)  │
                    │  → obj_11 (per class: present, cx, cy, w, h,    │
                    │     conf, area, vel_x, vel_y, id, track_age)    │
                    │  → pose132 (33×[x,y,z,vis], rack-space)         │
                    │  → hoi_5 (per hand: state one-hot)              │
                    │  → rack_ctx (astronaut attitude vs rack,        │
                    │     hand↔box distances, box↔zone distances)     │
                    │  Schema version stamped in every checkpoint     │
                    └────────────────────┬────────────────────────────┘
                                         ▼
    TEMPORAL        ┌─────────────────────────────────────────────────┐
                    │  ST-GCN OR LSTM (3-seed ensemble, avg probs)    │
                    │  32-frame window, stride 8 (3.2s @ 10Hz)        │
                    │  → P(atomic_state) every 100ms                  │
                    └────────────────────┬────────────────────────────┘
                                         ▼
    DECISION        ┌─────────────────────┬───────────────────────────┐
                    │  Petri-net FSM      │  Anomaly rule engine      │
                    │  (concurrent step   │  A4 forbidden zone        │
                    │  states, YAML pack) │  A5 dwell/timeout         │
                    │  Confirms step from │  A6 object drift          │
                    │  atomic states +    │  A7 occlusion abstain     │
                    │  timing constraints │  A8 model disagreement    │
                    └────────────────────┬────────────────────────────┘
                                         ▼
    OUTPUT          ┌──────┬──────┬──────┬──────┬──────┬───────────┐
                    │Voice │ GUI  │ Log  │Stream│Record│  Session  │
                    │Piper │PyQt6 │JSONL │RTSP  │ MP4  │  metrics  │
                    │+pytts│ 9tab │SQLite│UDP   │ ring │   JSON    │
                    │x3 fb │      │TXT   │alt   │buf   │           │
                    │EN+HI │      │      │      │      │           │
                    └──────┴──────┴──────┴──────┴──────┴───────────┘
```

### Per-block spec

#### 1.1 Input

- **Camera:** any USB webcam. Fixed, calibrated once (intrinsic matrix persisted to `config/camera_intrinsics.yaml`). No pan/tilt — rack is fixed relative to camera.
- **ArUco markers:** 4 corners of the mock rack, IDs 0-3 from `DICT_4X4_50`. Marker side length 5cm, printed once, physical placement measured to sub-mm and stored in `config/rack_layout.yaml`.

#### 1.2 Prep

- **Frame sampler:** decouples decision rate from camera fps. Fixed 100 ms clock (10 Hz decision loop) is the temporal unit for LSTM windows. Camera can run at 30/60 fps; only every Nth frame reaches perception. Prevents perception-lag catastrophe when the camera is faster than the model.
- **Rack pose:** `cv2.aruco.detectMarkers` → `cv2.solvePnP` on 4 markers → homogeneous transform `T_cam_rack`. When ≥3 markers visible, solution is uniquely determined. When <3, fall back to `RackFrameNormalizer` (geometry-based, existing). Last-good pose held for up to 500ms of dropout before entering A7 hold.

#### 1.3 Perception (4 parallel branches)

**YOLOv8n custom-trained (opt-in) OR HSV (default fallback):**
- 5 classes: `main_box_open`, `main_box_closed`, `red_box`, `yellow_box`, `person`
- Open/closed as separate classes (not a state machine post-processing step) — lid state is a direct detection output, not an inference. Key win over current single-class approach.
- `person` class preserved during fine-tune (loss reweighted with COCO subset in every batch) so ByteTrack downstream still gets astronaut bounding box for HOI crop extraction.

**PoseNet (custom from scratch, existing):**
- 13-of-33 MediaPipe-style landmark output, extended to full 33 for downstream contract compatibility (invisible landmarks marked `vis=0`).
- Outputs 132-dim vector: 33 × (x, y, z, visibility) in image space, then transformed to rack-space in features layer.
- Local-windowed soft-argmax decode (not plain argmax) — already gives measured +40% relative PCK.

**HOI-CNN (new, custom from scratch):**
- Wrist-centered crop (128×128, size scales with torso length so distance from camera doesn't matter), 4-frame temporal stack.
- Output: `{empty, touching_red, holding_red, touching_yellow, holding_yellow, touching_main, holding_main, releasing}` — 8-class softmax per hand.
- Trained on real+synthetic wrist crops with hand-labeled states. Only per-hand crop input, not full frame — synthetic-to-real gap collapses because background is removed.
- Directly addresses the PS's explicit "hand-object interaction" requirement — no other repo in the survey has this.

**ByteTrack (new, opt-in on top of YOLO):**
- Kalman velocity + IoU matching for multi-class tracking.
- Outputs stable object IDs across frames — enables real velocity for A6 drift check (currently frame-to-frame delta, jittery).
- Tracks `person` box → HOI crop extraction anchor.

#### 1.4 Features (`pipeline/features.py` — the most important file)

The **one** module both training and live inference call. Broken symmetry here = silent accuracy regression (this project already got burned once, session 4).

Output schema (versioned, stored in checkpoint):
```
obj_11         : 11 dims per detected class = [present, cx, cy, w, h, conf, area, vel_x, vel_y, track_id_stable, track_age]
                 × 5 classes = 55 dims
pose132        : 33 landmarks × [x, y, z, visibility] in rack space
hoi_10         : 2 hands × 5-state one-hot (compressed from 8-class softmax by taking argmax)
rack_ctx_16    : astronaut attitude vs rack (roll, pitch, yaw approx from torso),
                 min hand↔box distance per pair (3), box↔zone distance per zone,
                 box_velocity magnitude per class, hand_open_area proxy
Total: 213 dims per 100ms tick
```

Contract enforced by:
- `pipeline/features.py:SCHEMA_VERSION = 3`
- Every trained checkpoint pickle contains `{"feature_schema_version": 3, "pose_source": "posenet_v2", "detector_source": "yolov8n_5cls_v1", "rack_source": "aruco"}`
- `HARPipeline.__init__` refuses to load a checkpoint whose contract doesn't match runtime's actual sources. **Loud crash, not silent misprediction.**

#### 1.5 Temporal

Two options, both trained, ensemble at inference:

**Option A — LSTM (existing, keep):**
- 2-layer, 128 hidden, sequence input (32 × 213), output 9-class (8 atomic states + idle).

**Option B — ST-GCN (new, from Orion pattern):**
- Graph over 33 pose landmarks with skeleton edges as adjacency, 3-block ST-GCN.
- Reads only pose132 + hoi_10 (not object dims) — orthogonal signal to LSTM's full-vector view.

**Fusion:** average probabilities across LSTM + ST-GCN + 3 seeds each = 6-model ensemble. Cheap at inference (all ONNX, batched CPU inference <50ms total). Ensembles reduce variance but do not fix distribution mismatch — mismatch is handled by the contract system in §1.4.

#### 1.6 Decision

**FSM (Petri-net, new from BASGUARD pattern):**
- Steps modeled as places, transitions gated on atomic-state predicates.
- Concurrent-state support: astronaut can be `examining_red` (step 4) while `main_box_open` (step 2 completed). Linear FSM misses this.
- Transition conditions read from procedure pack YAML:

```yaml
- id: 3
  name: "Pick Red Box"
  entry_when:
    all:
      - hoi_left == "touching_red" OR hoi_right == "touching_red"
      - main_box_open == true
  complete_when:
    all:
      - hoi_left == "holding_red" OR hoi_right == "holding_red"
      - red_box_velocity < 0.05  # stably held
      - duration_ms > 500        # not a transient
  timeout_s: 8
  forbidden_zones: [experiment_edge_left, experiment_edge_right]
  next_hint: "Examine the red box under the light"
```

- One pack per experiment. Zero code changes for a new BAS experiment.
- Falsely-triggered transitions require `duration_ms >` guards on every rule — atomic-state predictions are noisy at 100ms tick, need multi-frame confirmation.

**Anomaly rule engine (existing, kept, extended):**
- A4 forbidden zone, A5 dwell/timeout, A6 object drift, A7 occlusion — already built.
- A8 model disagreement (existing ensemble fusion).
- New A6 uses ByteTrack velocity (stable) not frame-to-frame HSV centroid.
- New anomaly A3 "wrong object" — the pack's `complete_when` says `holding_red`, but predicted state is `holding_yellow`. Fires immediate voice alert + hold.

#### 1.7 Output

**Voice (upgrade Piper primary + pyttsx3 fallback, EN + HI):**
- Piper neural TTS, priority queue, cooldown per anomaly-code (no repeat spam).
- Hindi model bundled — judges' language.
- Fallback to `pyttsx3` if Piper voice file missing.

**GUI (existing PyQt6, add 9th tab):**
- Existing 8 tabs kept.
- New tab: **Rack Frame** — live overlay of ArUco marker detections, T_cam_rack transform visualization, rack-plane XY grid projected back to image, rack-space astronaut skeleton plotted on rack coordinate axes.

**Structured log (upgrade JSONL → JSONL + SQLite + timeline.log, Orion pattern):**
- `logs/experiment_log_*.jsonl` — existing machine-readable stream.
- `logs/experiment_*.db` — SQLite with tables `steps`, `anomalies`, `detections`, `voice_events`, `session_metrics`. Queryable for post-run reports.
- `logs/experiment_*_timeline.log` — human-readable narrative (`[00:03:12] Step 3 completed (2.4s). Red box grasped by left hand.`) for judges skimming after a demo.

**Stream + record:**
- Primary: existing UDP MPEG-TS + local MP4 recording (SIH requirement).
- Optional: GStreamer RTSP + splitmuxsink ring-buffer (BASGUARD pattern) as `--stream-mode rtsp` flag. Same output, richer network story.
- Smart clip uplink (existing) for bandwidth-constrained mode.

**Session metrics (existing) — expose live in GUI:**
- p50/p95 latency per stage, false-hold rate, abstain count, detect-to-hold-ms per anomaly code, avg step duration vs. pack hint.
- Written to `logs/session_metrics.json` at shutdown. This is the number judges will ask for.

---

## 2. Training architecture

```
    SOURCES         ┌──────────────────────────┬────────────────────────────┐
                    │  Custom OpenCV renderer  │  Real webcam takes         │
                    │  simulation/renderer.py  │  (record via               │
                    │  500 protocol runs ×     │  tools/record_take.py)     │
                    │  full random-domain      │  15-25 takes × 3+ subjects │
                    │  augmentation (light,    │  Real red/yellow/main box  │
                    │  clutter, texture,       │  props, varied lighting +  │
                    │  camera noise, angle)    │  orientations              │
                    └────────────┬─────────────┴──────────────┬─────────────┘
                                 │                            │
                                 ▼                            ▼
    LABELS          ┌────────────────────────────────────────────────────┐
                    │  Synthetic: renderer emits ground-truth bboxes,    │
                    │  poses, atomic-state timeline for free (~30k       │
                    │  frames)                                           │
                    │  Real: auto-label bboxes with synthetic-pretrained │
                    │  YOLO, hand-correct in CVAT (~3000 frames).        │
                    │  Real HOI states: hand-label wrist crops (a person │
                    │  can do 500 crops/hour).                           │
                    │  Real atomic-state timeline: key-press marker      │
                    │  during recording (tools/record_take.py logs a     │
                    │  keypress → frame_id mapping).                     │
                    └────────────────────────┬───────────────────────────┘
                                             ▼
    DETECTORS       ┌─────────────────────────┬──────────────────────────┐
                    │  YOLOv8n (opt-in)       │  HSV (default, kept)     │
                    │  Synth pretrain: 40ep   │  Zero-training baseline  │
                    │  Real fine-tune: 15ep   │  Fallback when YOLO not  │
                    │  with COCO person       │  trained yet             │
                    │  replay 20% per batch   │                          │
                    └────────────┬────────────┴──────────────────────────┘
                                 │
                                 ▼
    POSE            ┌──────────────────────────────────────────────────┐
                    │  PoseNet (existing, custom from scratch)         │
                    │  Stage 1: synthetic (8000 samples, 40ep)         │
                    │  Stage 2: real fine-tune (opt-in, replay-        │
                    │  regularized, per-joint PCK guard)               │
                    └────────────────────┬─────────────────────────────┘
                                         ▼
    HOI             ┌──────────────────────────────────────────────────┐
                    │  HOI-CNN (new, custom from scratch)              │
                    │  Wrist crop (128×128), 4-frame temporal stack    │
                    │  Synthetic renderer knows held_by → free labels  │
                    │  Real crops hand-labeled (2 hours of work        │
                    │  produces 1000 real labeled crops)               │
                    │  Same replay-regularized fine-tune as PoseNet    │
                    └────────────────────┬─────────────────────────────┘
                                         ▼
    SEQUENCE        ┌──────────────────────────────────────────────────┐
    BUILDER         │  tools/build_sequences.py                        │
                    │  Runs TRAINED YOLO + PoseNet + HOI on every      │
                    │  frame (not ground truth) — this is the fix for  │
                    │  the train/inference gap that cost 60 points     │
                    │  earlier. Emits (features, atomic_state_label)   │
                    │  tuples grouped by (take_id, subject_id) for     │
                    │  clean holdouts.                                 │
                    └────────────────────┬─────────────────────────────┘
                                         ▼
    TEMPORAL        ┌──────────────────────────────────────────────────┐
    MODELS          │  LSTM × 3 seeds + ST-GCN × 3 seeds               │
                    │  Group-aware train/val/test split:               │
                    │    train: 60% of takes, subjects A/B/C           │
                    │    val:   20% of takes, subjects A/B/C           │
                    │    test:  20% of takes, subjects D/E (never in   │
                    │           training or val — the real number)     │
                    │  Full protocol runs, not clip-length windows     │
                    └────────────────────┬─────────────────────────────┘
                                         ▼
    PROMOTION       ┌──────────────────────────────────────────────────┐
    GATE            │  tools/promote_model.py                          │
                    │  1. Contract check: pose_source + detector_      │
                    │     source + schema_version match runtime        │
                    │  2. ONNX export + parity check (PyTorch vs ONNX  │
                    │     output max diff < 1e-4)                      │
                    │  3. Held-out eval: per-step acc, per-orientation │
                    │     acc, full-protocol completion rate, latency  │
                    │     p95. Auto-reject if worse than incumbent on  │
                    │     any metric by >2%.                           │
                    │  4. Write metrics.json alongside checkpoint.     │
                    │  Only promoted checkpoints ship in the models/   │
                    │  directory that main.py loads.                   │
                    └──────────────────────────────────────────────────┘
```

### Training data honesty

| Data source | Volume | Label source | Validated? |
|---|---|---|---|
| Custom OpenCV renderer | ~30k frames, 500 protocol runs | Ground truth from renderer | Yes (schema tested) |
| Blender renderer (from vedant91 handoff idea) | Aspirational: 22k frames | Ground truth from Blender | **Not built.** OpenCV renderer is simpler and already gives varied-orientation output. Only add Blender if OpenCV renderer measurably underperforms — don't build for the impression. |
| Real webcam takes | Target: 15-25 takes, 3-5 subjects | Manual: key-press timeline + hand-labeled HOI crops + CVAT bbox correction | **Not built.** Highest-leverage single action for this project. |
| Public data (Ego4D/GTEA/100DOH) | Not used | — | Rejected: different viewpoint (egocentric vs. fixed 3rd person), licensing friction, distribution gap larger than the transfer benefit. |

---

## 3. Deployment

### 3.1 Hardware tiers

| Tier | Target | Latency budget/frame | Notes |
|---|---|---|---|
| Dev | MacBook CPU | 100 ms | Current dev environment. Honest baseline. Works today. |
| Demo | Laptop RTX 3050+ | 33 ms | For live judge demo. YOLO + PoseNet + HOI batched CUDA. |
| Deployment | Hailo-8 edge accelerator (26 TOPS @ ~2W) | 20 ms | Realistic radiation-tolerant edge target. Matches Priyansh07x/BAS-Monitor pick. All models exported to Hailo's `.hef` format via their ONNX compiler. Not built yet — road-mapped. |

### 3.2 Runtime bundle

Single directory `deploy/` containing:
```
deploy/
  models/
    yolo_detector.onnx         + metrics.json + contract.json
    pose_net.onnx              + metrics.json + contract.json
    hoi_cnn.onnx               + metrics.json + contract.json
    lstm_seed_{0,1,2}.onnx     + metrics.json + contract.json
    stgcn_seed_{0,1,2}.onnx    + metrics.json + contract.json
  packs/
    box_sort_v1.yaml           (default)
    <experiment_name>.yaml     (add YAML, no code)
  config/
    camera_intrinsics.yaml     (calibrated per camera)
    rack_layout.yaml           (marker positions, measured once)
    tuned_overrides.json       (auto-generated by end_to_end_loop)
  voices/
    piper_en_us.onnx
    piper_hi_in.onnx
  har_pipeline               (PyInstaller-bundled binary, no Python required on target)
  install.sh                 (one-shot install: udev rules for camera, systemd unit)
```

### 3.3 Failure modes and behavior

| Failure | Detection | Behavior |
|---|---|---|
| Camera disconnected | Frame sampler returns None ≥ 500ms | Watchdog fires HEARTBEAT_LOST, voice alert, GUI red banner, FSM freezes at current step |
| All ArUco markers occluded | ≥3 marker detections drops for ≥500ms | Fallback to RackFrameNormalizer (geometry-based rack angle), log source_downgrade event |
| YOLO confidence collapse (all classes <0.3 for ≥1s) | Detection quality check | A7 occlusion abstain, FSM holds, "check camera / lighting" voice prompt |
| Model contract mismatch at load | promote_model gate catches at startup | Refuse to start. Log error naming exactly which field mismatched. |
| Voice thread died | Watchdog `is_alive` check | Try to respawn once, then log VOICE_THREAD_DEAD and continue silently — never block pipeline on TTS |
| Disk full during recording | OSError on video write | Stop recording, keep pipeline running, alert judge via GUI |
| Two atomic states predicted with high confidence | Ensemble disagreement (>0.4 gap) | A8 model_disagreement, FSM asks "please confirm current step" (voice), holds until user action or timeout |

---

## 4. Evaluation (what the metrics.json contains)

Numbers judges will ask for. All measured on the **held-out 20% of takes on subjects D/E who never appear in training or val**.

| Metric | How measured | Target |
|---|---|---|
| Per-step accuracy | Frames where predicted step_id == labeled step_id | > 0.90 on each of 8 steps |
| Per-orientation accuracy | Same, grouped by (0°, 90°, 180°, 270°) rack camera roll | Uniform: max-min < 0.05 across orientations |
| Full-protocol completion rate | Fraction of held-out takes where all 8 steps complete in order without hold | > 0.80 |
| False-hold rate | Anomaly holds fired where ground truth says no anomaly | < 0.05 |
| Abstain rate | A7 occlusion holds per minute | Report honestly, no target |
| Detect-to-hold latency | Time from ground-truth anomaly frame to first hold event | A4 < 100ms, A6 < 300ms, A7 < 400ms |
| End-to-end latency p95 | Frame capture → decision emit | < 100ms CPU, < 40ms GPU |
| Cross-experiment | Retrain nothing, load a new procedure pack YAML, run through | Full-protocol completion > 0.70 on a 2nd experiment nobody trained on |

**The last row is the killer number for "does this scale to all BAS experiments."** No other repo in the survey shows this metric because none have a pack-driven design. It's what makes the answer to "do we retrain for every experiment" a confident *no*.

---

## 5. What's already in the repo vs. what's new

Stated flat, no defense.

**Already built (keep, do not rebuild):**
- Custom PoseNet + training
- LSTM + training + procedure-length dataset
- HSV detector (fallback)
- Procedure pack YAML + loader + FSM
- Anomaly monitor A4/A5/A6/A7 + wiring
- Session metrics
- Smart clip uplink
- Latency race demo
- PyQt6 dashboard (8 tabs)
- Watchdog
- YOLO detector (opt-in) + training
- Rack-frame normalizer (geometry-based)

**New work (in priority order):**
1. **Real webcam data collection** — 15-25 takes, 3+ subjects, keypress-labeled. Highest single-item accuracy leverage.
2. **HOI-CNN** — wrist-crop classifier. Fills the PS's explicit "hand-object interaction" gap. Nobody in the survey has this.
3. **ArUco rack pose** — `pipeline/rack_frame.py::estimate_rack_pose()` via `cv2.aruco` + `solvePnP`. ~200 lines, one dependency (`opencv-contrib-python`, already in reqs).
4. **YOLO 5-class re-train** with `main_box_open`/`main_box_closed` split and `person` class preserved (replay 20% COCO per batch).
5. **`pipeline/features.py`** — extract current inline feature-building into one shared, versioned, tested module. Add SCHEMA_VERSION contract.
6. **`tools/promote_model.py`** — contract check + ONNX parity gate + auto-reject-on-regression. Currently ad-hoc in `end_to_end_loop.py`.
7. **ByteTrack** — Kalman tracker on YOLO outputs, stable object IDs, real velocity for A6.
8. **Petri-net FSM upgrade** — concurrent-state semantics, `entry_when`/`complete_when` predicates in YAML.
9. **ST-GCN** — orthogonal temporal model for ensemble diversity.
10. **Piper TTS + Hindi voice** — replace pyttsx3 with neural, add Hindi.
11. **SQLite structured log + timeline.log** — Orion pattern, richer post-run reports.
12. **GUI Rack Frame tab** — 9th tab, live ArUco + T_cam_rack overlay.
13. **A3 wrong-object anomaly** — reads YAML `complete_when` object, checks against predicted state.

**Explicitly not building:**
- ROMP / 3D HMR (violates PS rule, fails at microgravity conditions)
- MediaPipe (violates PS rule)
- Blender renderer (OpenCV renderer already gives orientation variance; add only if measured need)
- LLM assistant (not in PS, offline constraint)
- Federated learning (not in PS, single onboard node)
- Cognitive stress / behavioral assessment (different PS, MAITRI)
- Ego4D / GTEA pretraining (viewpoint mismatch larger than transfer benefit)

---

## 6. Judge-facing story (one paragraph)

> This is an offline, from-scratch, standalone AI system that recognizes and validates any 8-step protocol described in a YAML file. Its models learn atomic states (holding red box, main box open, hand touching yellow box) — never step numbers — so a new experiment is a text-file change, not a retrain. It measures everything relative to the payload rack via ArUco markers, so the astronaut's orientation in microgravity doesn't matter. It refuses to guess: if two models disagree, or if the camera can't see the rack, or if a step takes 3× longer than expected, it holds and asks the astronaut to confirm — models propose, a deterministic state machine decides. Every model checkpoint is contract-tracked and rejected at load time if its training distribution doesn't match what the runtime will show it, so the class of silent-regression bug that ruined 60 accuracy points earlier in this project structurally can't recur. Held-out on subjects who never appear in training, it achieves [X] per-step accuracy uniformly across all 4 tested camera orientations, and [Y] full-protocol completion rate on a 2nd experiment we never trained on. Latency p95 is [Z] ms on the deployment target (Hailo-8 edge accelerator, 26 TOPS at 2 watts — realistic space-grade hardware, not a desktop GPU).

---

## 7. What to cut if time-boxed

If under 40 hours to submission:
- Keep: items 1, 2, 3, 5 (data, HOI, ArUco, features.py contract). These four cover the biggest judge questions.
- Drop: 8 (Petri-net), 9 (ST-GCN), 13 (A3). Linear FSM + LSTM-only + existing A1-A2 skip/out-of-order covers the demo.
- Defer: 10 (Piper), 11 (SQLite log), 12 (Rack Frame tab). Nice-to-have polish.

If under 20 hours:
- Only items 1, 2, 3. Everything else is a slide.

If under 8 hours:
- Only item 1. Record real takes and rerun the current pipeline on them. Every architectural improvement above is meaningless without real data behind the numbers.
