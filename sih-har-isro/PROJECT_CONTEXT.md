# Project context — ISRO BAS Human Activity Recognition

Snapshot as of **2026-09-24**. Single source for "what is this, what exists,
what is happening, what is next". Companion docs:

| Doc | Use it for |
|---|---|
| `EXPLANATION.md` | Plain-language walkthrough and glossary (written 2026-09-21; its section 7 is partly stale, this file supersedes it) |
| `docs/designs/step-classifier-generalisation.md` | The approved plan: premises, decisions, Review Outcomes, 15 implementation tasks |
| `docs/hmr-architecture.html` | Orientation-agnostic 3D HMR: rotation representations, architecture, gated plan |
| `dataset/README.md` | Every dataset: what it feeds, status, how to regenerate, what was deliberately not collected |
| `logs/RESULTS.md` | Measured results on the old 12-take dataset |
| `blender/README.md`, `docs/BLENDER_DIGITAL_TWIN.md` | The digital twin in detail |
| `.claude/agents/README.md` (repo root) | The agent team and HMR swarm: who owns which files |

---

## 1. In one paragraph

An offline edge system that watches an astronaut perform an 8-step experiment
on a payload rack through a fixed camera, recognises which step is being done,
catches skipped or out-of-order steps, speaks a warning, and logs everything
with timestamps. Built for Smart India Hackathon 2026 (ISRO problem statement,
`../AI_HAR_for_Onboard_BAS_Experiments.pdf`). **The whole system works end to
end except the part that is graded: the trained step classifier scores 0.201
on unseen performances against a 0.125 chance floor.** The current effort is a
new, physically corrected synthetic dataset plus fixes to three train/serve
mismatches, so the classifier can be retrained and measured honestly.

## 2. Status board

| Area | State |
|---|---|
| Live pipeline (camera → pose → LSTM → state machine → voice/log/stream/GUI) | **Working** |
| Orientation-agnostic pose (`upright_pose.py`) | **Working, measured:** 25% → 100% detection at 135° crew roll |
| Step classifier | **Broken:** 0.201 held-out vs 0.125 chance; the model the demo loads scores 0.170 |
| Synthetic dataset `dataset/blender_core` (23 takes, ~22k frames) | **Rendering**, launched detached. Last observed: take 0 at 648/960 frames, 0.3 s/frame, ~2 h total. Not re-checked for this snapshot |
| Generator physics + variety fixes | **Done, gated** (`check_dataset.py` OK on all 23 takes) |
| Real footage | NASA clips collected (8, 74 MB); **your own protocol recordings not yet made** |
| Model promotion + honest scoring script | **Done** (release-engineer, verified) |
| Shared feature/windowing refactor | **Partial** — pipeline-engineer stopped mid-task when the session ended (§9) |
| Test suite | **None exists yet** — test-engineer briefed, not launched |
| 3D HMR (optional challenge) | **Blocked** on SMPL registration (you) and a C++ compiler for `simple_romp` |

## 3. What the problem statement asks, and where it lives

| Requirement | Where | Status |
|---|---|---|
| Trained AI model recognising experiment steps, offline | `train/train_lstm.py`, `models/`, ONNX inference in `pipeline/har_pipeline.py` | Model at chance; being fixed |
| Sequence validation, skip / out-of-order detection | `pipeline/state_machine.py` | Working |
| Next-step prompts and voice alerts | `pipeline/voice_alert.py` | Working |
| Timestamped structured logs | `pipeline/logger.py` → `logs/*.jsonl` | Working |
| Local recording + IP video streaming | `pipeline/stream_sender.py`, `recordings/` | Working |
| GUI | `gui/`, `main.py` (PyQt6) | Working |
| **Optional:** orientation-agnostic tracking relative to the rack, not the floor | `pipeline/upright_pose.py` + `pipeline/rack_frame.py` (2D) | Roll solved and measured; pitch/yaw untested until the new render |
| **Optional:** 3D Human Mesh Recovery | `pipeline/hmr_backend.py` (adapter, off) | Blocked (§10) |

## 4. How it works

### Live path (what runs on the edge box)

```
camera frame (1280x720, 30 fps)
   |
   +--> HSV box detector ------------------+   red / yellow / container, + rack roll
   |                                       |
   +--> upright pose: rotate IMAGE by rack roll -> MediaPipe -> rotate landmarks back
                                           |
                     rack-frame normaliser (body measured against the rack)
                                           |
                          132 numbers (33 landmarks x x,y,z,visibility)
                                           |
                          rolling buffer, 30 entries
                                           |
                          LSTM (ONNX)  -> step id + confidence
                                           |
                 confidence >= 0.65, then >= 10 of last 15 must agree
                                           |
              state machine: next step / skipped / out of order
                                           |
          voice alert + JSONL log + annotated frame + IP stream + dashboard
```

There is **no 3D reconstruction** in the live path today. MediaPipe's `z` is a
weak per-landmark depth guess, not a reconstruction.

### Training path (how the model is made)

```
Blender digital twin (blender/bas_har/)  -- renders takes with exact per-frame labels
   |
tools/build_sequences.py   -- runs the SAME pose pipeline over rendered frames,
   |                          cuts windows, groups BY TAKE
tools/retrain_and_evaluate.sh
   |   stage 0  gate: tools/check_dataset.py (non-empty, holdout tags exist)
   |   stage 1  build sequences (rack-normalised)
   |   stage 2  train LSTM (split by take, so held-out = unseen performance)
   |   stage 3  score train / holdout / skip-recover separately
   |   stage 4  re-score the baseline on the same holdout
   v
tools/promote_model.py      -- the ONLY way a model reaches the live demo
```

Splitting by take is the project's integrity rule, and it is why its numbers
are low and real where comparable public repos report an undocumented 99%.

## 5. Codebase map

Line counts as of this snapshot. Status: **live** (runs in the deployed
system), **train** (dataset/model production), **demo** (showcase videos),
**legacy** (superseded, kept for history).

### `pipeline/` — the live system
| File | Lines | Role | Status |
|---|---:|---|---|
| `har_pipeline.py` | 1021 | Main loop: capture, threaded HSV + pose, buffer, LSTM, state machine, outputs. `run()` and `process_frame()` | live |
| `hsv_detector.py` | 408 | Finds red/yellow boxes and the container by hue; `minAreaRect` gives rack roll | live |
| `upright_pose.py` | 270 | The orientation fix: rotate the image upright before MediaPipe | live |
| `pose_backend.py` | 343 | MediaPipe wrapper; emits the 132-dim contract | live |
| `rack_frame.py` | 400 | Re-expresses landmarks in the rack's frame (EMA-smoothed angle/scale, latched polarity). **Has partial uncommitted edits from pipeline-engineer** | live |
| `state_machine.py` | 298 | Protocol order, 15-frame confirm buffer, skip/out-of-order callbacks | live |
| `voice_alert.py` | 230 | Spoken prompts and warnings | live |
| `logger.py` | 205 | Timestamped JSONL + text logs | live |
| `stream_sender.py` | 136 | IP video stream | live |
| `gravity_detector.py` | 535 | Stability / "g-level" classification from pose jitter; feeds the HUD | live |
| `hmr_backend.py` | 215 | Optional 3D HMR adapter (hmr2/wham), same 132-dim contract; `HMR_BACKEND = "none"` | off |

### `tools/` — dataset, training and scoring utilities
| File | Lines | Role | Status |
|---|---:|---|---|
| `build_sequences.py` | 291 | Frames → MediaPipe → windows grouped by take. Accepts real takes (labels, no pose GT). Default excludes `skip,recover,heldout_view` | train |
| `evaluate_real.py` | 506 | Scores a model on rendered takes against independent ground truth: step accuracy, oracle, pose error, HSV recall | train |
| `retrain_and_evaluate.sh` | — | The sanctioned way to produce an accuracy claim (§4). Requires a dataset argument | train |
| `promote_model.py` | 126 | Copies a checkpoint + self-contained ONNX into the live model path; refuses non-LSTM or mismatched ONNX; preserves the 0.201 baseline | train |
| `check_dataset.py` | 93 | Gate: label/frame mismatch, missing steps, empty holdout, wrist teleports > 0.04 m/frame, crew out of frame | train |
| `record_take.py` | 137 | Webcam recorder for real protocol takes, live 1–8 labelling, same layout as renders | train |
| `render_dataset_resilient.sh` | — | Relaunches Blender across driver crashes; resumable per frame | train |
| `fetch_pose_model.py` | 76 | Downloads the MediaPipe pose model | train |
| `showcase_overlay.py`, `compose_showcase.py`, `overlay_pose.py`, `frames_to_video.py` | 612 / 239 / 134 / 58 | Demo video composition and overlays | demo |

### `train/`
| File | Lines | Role |
|---|---:|---|
| `train_lstm.py` | 364 | BiLSTM + attention classifier, GroupShuffleSplit by take, ONNX export beside the checkpoint |
| `train_cnn.py` | 578 | 8-frame-stack CNN trained from scratch (no pretrained weights). Optional ensemble, off by default |

### `blender/` — the digital twin and renders
| File | Lines | Role |
|---|---:|---|
| `bas_har/protocol.py` | 412 | **The 8-step motion.** Per-take style, reach phases, reaction coupling, box drift, hand-speed cap |
| `bas_har/astronaut.py` | 554 | Rigged crew: NASA-STD-3001 neutral posture, IK hands, roll / **pitch** / yaw about the shoulder pivot |
| `bas_har/module.py` | 627 | Space module: four working walls, rack, handrails, container, trays |
| `bas_har/groundtruth.py` | 308 | Projects the rig to 2D/3D pose and true box visibility per frame |
| `bas_har/materials.py`, `render.py`, `common.py` | 308 / 195 / 359 | Materials, EEVEE setup and cameras, helpers |
| `build_scene.py` | 171 | Assembles one take's scene |
| `render_dataset.py` | 305 | Render plans (`smoke`, `orientation`, `study`, `core`); writes frames + GT + meta per take |
| `har_showcase.py`, `har_3d_showcase_v2.py`, `render_har_showcase.py`, `render_video.py`, `validate_scene.py` | 1528 / 595 / 699 / 162 / 143 | Showcase video renders and scene checks (demo) |

### `config/experiment_config.py` (288 lines)
All tunables: camera (`CAMERA_INDEX=0`, `1280x720`, `FPS=30`), `SEQUENCE_WINDOW=30`,
`SKELETON_FEATURES=132`, `STEP_CONFIDENCE_THRESHOLD=0.65`, `STEP_CONFIRM_FRAMES=15`,
`RACK_FRAME_NORMALIZE=True`, `UPRIGHT_POSE=True`, `HMR_BACKEND="none"`,
`CNN_ENSEMBLE_ENABLED=False`, model paths. **Has partial uncommitted edits from
pipeline-engineer.**

### Legacy and other
| Path | What |
|---|---|
| `simulation/` (`renderer.py`, `space_sim.py`, …) | Old 2D simulator. Painted boxes in the detector's own colours, so its 1.000 HSV recall was tautological |
| `data_generation/synthetic_pose.py` | Old closed-form pose generator; its 0.98 test accuracy measured nothing |
| `data_generation/mediapipe_labeler.py`, `augmentation.py` | Per-step clip labeller and augmentation (pre-take-grouping path) |
| `data_generation/gemini_video_gen.py` | Gemini video generation; the only user of `google-genai` in `requirements.txt` |
| `end_to_end_loop.py` | Retired loop; carries a deprecation header — its `lstm_test_acc` must not be quoted |
| `main.py` (318) | Entry point for the live system and GUI |
| `gui/`, `recordings/`, `assets/`, `build/` | Dashboard code, local recordings, assets, build output |
| `../Astra/` | Separate prototype that calls Gemini in 7 files; cannot ship in an offline submission as it stands |

## 6. Data

| Folder | Content | Status |
|---|---|---|
| `dataset/blender_core/` | 23 takes: roll ×8, yaw ×3, pitch ×3, lighting ×4, clutter ×2, skip, recover, unseen camera. 120 frames/step. Per take: frames, `pose_2d.npy`, `pose_world.npy`, `labels.npy`, `payload.json`, `meta.json` | Rendering |
| `dataset/nasa_iss/` | 8 NASA public-domain ISS clips + `manifest.json` | Done |
| `dataset/real/` | Your webcam takes via `record_take.py` | Not started |
| `dataset/seq_final_rack*`, `seq_final_raw*` | Windows from the old 12-take render (1,378 train / 576 holdout) | Frozen — the frames they came from **no longer exist** |
| `dataset/annotated/`, `skeleton_sequences*/`, `space_sim/`, `synthetic_gemini/` | Old simulator / generator output | Retired |

Holdout (never trained on): `orient_180`, `orient_-60`, `yaw_90`, `pitch_60`.
Excluded from training: `skip`, `recover`, `heldout_view`.

## 7. Models (`models/`)

| File | What |
|---|---|
| `lstm_classifier.pt` / `.onnx` | **What the live demo loads.** The shipped model, 0.170 on unseen takes |
| `lstm_final_rack.pt` / `.onnx` | Best retrained model, **0.201 held-out** — the published baseline |
| `lstm_final_raw.pt`, `lstm_rack.pt`, `lstm_raw.pt`, `lstm_classifier_real.pt` | Other experiments from the old dataset |
| `activity_cnn.pt` / `.onnx` | The frame-stack CNN (ensemble, off) |

Nothing has been promoted since the new tooling landed. `baseline_rack_0201.pt`
is created by the first promotion.

## 8. What we know from measurement

| Measurement | Value | Meaning |
|---|---|---|
| Chance | 0.125 | 8 steps |
| Shipped model, unseen takes | 0.170 | At chance |
| Best retrained, unseen takes | **0.201** | Barely above chance |
| Same model, its training takes | 0.561–0.753 | Memorised the performances |
| `step_acc_real` (pooled) | 0.4032 | **Never quote** — mixes train and holdout |
| `step_acc_oracle` | 0.1448 | Worse on *perfect* pose than on MediaPipe pose |
| `perception_cost` | −0.2584 | Evidence the model keys on pose-estimator artefacts |
| Pose detection at 135° roll, MediaPipe alone → with `upright_pose` | 25% → 100% | The orientation fix works |
| HSV recall red / yellow / container | 0.324 / 0.118 / 0.885 (precision 1.000) | Small boxes mostly missed |
| Render speed on RTX 3050 (EEVEE) | 0.26–0.5 s/frame | Full dataset ~2 h, not ~21 h |

## 9. What happened in the last sessions

**Defects found and fixed**
- Hands **teleported on the first frame of every step** in the generator — a
  one-frame spike marking every label boundary for free. Fixed with
  distance-scaled reach phases; the gate now fails any wrist step > 4 cm/frame.
- Every take replayed **one scripted trajectory**. Each take now has its own
  style: lift height, pull distance, inspect point, timing, one- or two-handed.
- **Pitch did not exist** (`astronaut.py` hardcoded it to 0); yaw was plumbed
  but never set. Both now in the render plan.
- **Newton's third law** (torso counter-rotates against arm motion) and
  **first law** (a box abandoned mid-air keeps drifting) added.
- The `core` plan had **no holdout tags** (empty test set) and the scripts asked
  for `orient_-45`, which it did not contain. Fixed.
- Steps were too short for the planned 3-second window; now 120 frames/step
  (shortest after jitter ≈ 99 frames).
- `retrain_and_evaluate.sh` pointed at a nonexistent `.venv312` and defaulted
  to a dataset that no longer exists. Fixed.

**Found, not yet fixed (assigned to pipeline-engineer)**
- Seven separate places cut windows independently, with two different purity
  filters (0.75 vs 0.6).
- A 30-frame window at 30 fps is 1 s; protocol steps take 4–6 s.
- The live loop runs ~14 fps (70 ms pose), so buffer entries are ~70 ms apart
  vs 33 ms in training — a timing skew; resampling against timestamps is the fix.
- No input-dimension check when the live pipeline loads a model.
- (Rack-normalisation mismatch: fixed on the script side by release-engineer.)

**Team** (`.claude/agents/` at the repo root)

| Agent | State |
|---|---|
| release-engineer | **Done, verified** — `promote_model.py`, required dataset arg, three-way table, baseline re-score, deprecation header |
| hmr-env | **Done** — `D:\SIH-2026\.venv-hmr` with CUDA torch works; `simple_romp` blocked on a missing MSVC compiler |
| pipeline-engineer | **Stopped mid-task** when the session ended. Partial edits in `config/experiment_config.py` and `pipeline/rack_frame.py`; `pipeline/features.py` not yet created. Must be resumed and its edits reviewed before anything trains |
| test-engineer | Briefed, not launched (waits on pipeline-engineer) |
| ml-engineer | Briefed, not launched (waits on the render) |

## 10. Blocked, and what unblocks it

| Blocker | Unblock | Who |
|---|---|---|
| Any 3D HMR backend | Register at smpl.is.tue.mpg.de, download `SMPL_NEUTRAL.pkl` v1.1.0 | **You** |
| `simple_romp` build | Install "Build Tools for Visual Studio" (C++ workload, ~6–7 GB). Deferred until the pitch/yaw measurement shows 2D actually breaks | You decide |
| Real-domain evaluation | Record takes: `python tools/record_take.py --out dataset/real --tag real_nominal`. Target 10 nominal, 3 skip, 2 recover, two people | **You** |
| Retraining | pipeline-engineer's refactor must land; the render must finish and pass the gate | Team |

Open decisions for you: delete `dataset/space_sim/sim_report.json` (stale
0.9948 claim, tracked in git)? Which model to promote into the demo (after new
held-out numbers exist)?

## 11. Decisions on record

1. Fix generator physics **before** rendering, render once.
2. Per-take trajectory variety instead of more renders of one motion.
3. One shared feature builder and one shared windowing function (7 consumers).
4. Sample at 10 fps (`SAMPLE_EVERY = 3`) where frames enter the buffer, in both
   `run()` and `process_frame()`; `PROCESS_EVERY_N_FRAMES` stays 1.
5. Pin the live model path; promote explicitly (not "newest .onnx").
6. Object channels (11 dims) with absence encoded as a constant **after**
   rack normalisation.
7. The retrain script requires a dataset argument.
8. Smoke render + gate before any full render.
9. pytest from zero, all 24 identified paths, sequence-level parity test first.
10. Window purity 0.75 everywhere; re-measure the baseline under it.
11. Oracle experiment first after the render (settles the artefact question).
12. Transfer learning only if held-out is still below 0.40.
13. Every rotation fed to a network uses the 6D continuous representation
    (HMR work).
14. Keep the take-grouped honest evaluation protocol unchanged.

Target: held-out step accuracy **≥ 0.40** (stretch 0.60).

## 12. How to run things

```bash
# live system
python main.py

# validate the generator without rendering (~5 min)
blender --background --factory-startup --python blender/render_dataset.py -- \
    --out .tmp/gtcheck --plan core --gt-only
python tools/check_dataset.py .tmp/gtcheck

# render the dataset (resumable)
BLENDER='D:/SIH-2026/tools/blender-5.2.0/blender-5.2.0-windows-x64/blender.exe' \
    tools/render_dataset_resilient.sh dataset/blender_core core 900 506 16

# retrain and score honestly
DRY_RUN=1 tools/retrain_and_evaluate.sh dataset/blender_core   # gate only
tools/retrain_and_evaluate.sh dataset/blender_core

# put a model into the live demo
python tools/promote_model.py models/<checkpoint>.pt

# record a real take
python tools/record_take.py --out dataset/real --tag real_nominal
```

## 13. Environment

- Windows 11, RTX 3050 6 GB laptop GPU. System Python 3.12.6 with torch
  2.5.1+cu121, mediapipe 0.10.14, opencv 5.0.0 — **no venv** for the main
  pipeline. HMR experiments only in `D:\SIH-2026\.venv-hmr`.
- Blender 5.2 portable at `D:\SIH-2026\tools\blender-5.2.0\...`; EEVEE renders
  on the NVIDIA GPU in background mode.
- Disk: C: ~34 GB free, D: ~137 GB free (last measured). Keep `TEMP`/`TMP` on
  `D:\SIH-2026\.tmp` for long jobs.
- The working tree holds a lot of uncommitted and staged work. A plain
  `git commit` would sweep in staged models and datasets; commit by path.
