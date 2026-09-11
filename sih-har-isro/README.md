# ISRO SIH26175 — AI Human Activity Recognition for On-board BAS Experiments

Real-time monitoring of an astronaut performing a red/yellow box-sorting protocol
(8 steps, see `config/experiment_config.py:EXPERIMENT_STEPS`), with sequence
validation, voice guidance, structured logging, and a detailed live dashboard.

## Current results (this branch, trained on synthetic + real "gravitational mimic" data)

Two different numbers matter here, and conflating them is how ML systems end up looking
better on paper than they work in practice — so both are reported:

| Metric | Value | Gate |
|---|---:|---:|
| HARPoseNet synthetic PCK@0.10·torso | 0.665 (Stage 1) → 0.649 (+Stage 2 real fine-tune) | ≥0.55 |
| LSTM held-out window accuracy (PoseNet's own predictions, not ground truth) | 0.938 | — |
| LSTM val accuracy (`main.py --mode train`'s production model) | 0.843 | — |
| CNN val accuracy (combined synthetic+real frames) | 0.757 | — |
| Oracle step accuracy (clean ground-truth pose → LSTM → FSM) | 0.108 (see note below) | ≥0.85 |
| **Real (non-oracle) step-window accuracy — actual camera→PoseNet→LSTM path** | **0.45** | — |
| **Full 8-step protocol completion, non-oracle, one continuous take** | **Does not complete** (holds at step 2-3) | — |
| HSV red/yellow/main box recall & precision | 1.000 / 1.000 | ≥0.85 |
| Mean / p95 per-frame latency (CPU) | 25.4ms / 27.2ms (~39 FPS) | ≤80ms / ≤130ms |
| Sequence FSM logic itself (complete/skip/recovery, scripted predictions) | all pass | — |

**Two different LSTM training paths exist in this repo — know which one you're running.**
`main.py --mode train` (recommended for the actually-deployed model) trains the LSTM on
PoseNet's own realistic predictions, per the finding below (0.843 val acc, honest).
`main.py --mode e2e` / `end_to_end_loop.py` trains its own LSTM internally, fast, on pure
ground-truth pose vectors (by design — it's iterating a hyperparameter search many times
and needs each iteration to be cheap; see its own `E2E_LSTM_MIN_VAL_ACC=0.90` gate, which
targets *that* ground-truth task, not the realistic one). **Both write to the same
`models/lstm_classifier.pt`** — running `--mode e2e` after `--mode train` will overwrite
the honestly-trained model with the easier-task one. Run `--mode e2e` to sanity-check FSM/
pipeline-integration logic and latency; run `--mode train` last if you want the model that
actually ships.

**Read this part — it's the most important finding in this codebase's history.** The
first version of this branch trained the LSTM almost entirely on
`data_generation/synthetic_pose.py`'s exact ground-truth pose vectors, which is what
PoseNet is trained *against* but is **not** what the LSTM sees at actual deployment —
real inference feeds it PoseNet's own, imperfect predictions. That version reported a
misleadingly perfect 0.985-1.0 val/test accuracy and passed every gate, while the actual
camera→PoseNet→LSTM path scored **0.125** step-window accuracy (~random) and got stuck
after step 1 in a full-protocol test. `data_generation/build_posenet_realistic_dataset.py`
fixes this by training the LSTM on PoseNet's own predictions on freshly-rendered frames
instead (free, exact labels — the renderer drove them) — real accuracy improved to 0.45,
a genuine ~3.5x gain, but still well short of reliable. **This is disclosed, not hidden:**
the gap traces to PoseNet's per-joint accuracy being uneven (nose/hips/knees/ankles score
0.70-0.99 PCK — they barely move, an easy target; elbows/wrists — the joints that actually
drive step classification, see `_STEP_WAYPOINTS` in `synthetic_pose.py` — score only
0.22-0.40 at this heatmap resolution and CPU training budget), which the temporal LSTM
partially but not fully compensates for. Closing this further needs more PoseNet training
data/time, a higher heatmap resolution, or more real footage than the four clips available
this session — legitimate follow-up work, tracked as a known limitation rather than
papered over with an oracle-only number. The state machine's strict hold-and-correct
policy (never guess through an anomaly) is doing exactly what the problem statement asks
of it here — it's *why* the system reports "stuck," not "wrong."

The CNN's 0.757 (vs. a pure-synthetic run's ~0.96 in earlier project history) reflects a
genuinely harder, more realistic combined dataset (real, imperfectly-pseudo-labeled frames
mixed with clean synthetic ones), not a regression — see `dataset/annotated_real` and
`dataset/real_pseudo/autolabel_report.json` for exactly what was added.

Full gate table: `logs/final_space_sim_report.json` + `simulation/gates.py`. Against the
production model, **11 of 13 gates pass**; the 2 that don't (`lstm_val_acc` 0.84 < 0.90,
`oracle_step_acc` 0.11 < 0.85) fail for a specific, understood reason, not a mystery: those
two gates were calibrated for a ground-truth-fed LSTM (`end_to_end_loop.py`'s own internal
training/eval), and this LSTM was deliberately trained on noisier, realistic PoseNet
features instead — feeding it clean ground truth (what the oracle check does) is now
itself out-of-distribution for it, the mirror image of the original bug. The metric that
actually matters for this architecture — real camera→PoseNet→LSTM accuracy — is the 0.45
reported above, evaluated the honest way.

## No open-source / pretrained model anywhere

Every model in this system is trained from scratch on this project's own data —
no YOLO, no MediaPipe, no other third-party pretrained network, ever, at
inference time:

| Capability          | Approach                                            | Pretrained weights? |
|----------------------|-----------------------------------------------------|:---:|
| Object detection (boxes, hand) | Classical CV — HSV color segmentation + skin-tone/motion (`pipeline/hsv_detector.py`) | No — zero training needed |
| Pose / movement tracking | **HARPoseNet** — a ~1.2M-param CNN trained from scratch (`pipeline/pose_net.py`, `train/train_posenet.py`) | No — replaces MediaPipe entirely |
| Frame-level activity  | **HARActivityCNN** — custom ResNet-lite, trained from scratch (`train/train_cnn.py`) | No |
| Temporal step classification | **HARLSTMClassifier** — BiLSTM + attention, trained from scratch (`train/train_lstm.py`) | No |

## Architecture

```
Camera/video frame
      │
      ├─► HSVBoxDetector ──────► box + hand detections (classical CV)
      │
      └─► HARPoseNet (ONNX) ───► 132-dim pose feature (13 tracked joints)
                                        │
                          (optional) RackFrameNormalizer (orientation-agnostic)
                                        │
                              sliding window (30 frames)
                                        │
                      ┌─────── HARLSTMClassifier ───────┐
                      │                                  │
              (optional CNN ensemble, disagreement→"uncertain, please confirm")
                                        │
                            ExperimentStateMachine
                          (hold-and-correct, never guesses through a skip)
                                        │
                     Voice alert + structured log + GUI (6-tab PyQt6 dashboard)
```

### Why HARPoseNet can be trained from scratch with zero manual labeling

`simulation/renderer.py` procedurally draws the astronaut, rack, and boxes —
which means it **knows the exact pixel location of every joint it draws**.
`train/train_posenet.py` uses that as free, exact ground truth to train a
heatmap-regression CNN (Stage 1). The real "gravitational mimic" footage in
`../new video data/` has no such ground truth, so it's bridged with **classical-
CV pseudo-labels** instead of a pretrained model:

- **Pose**: skin-tone + motion tracking finds the two hands per frame
  (`data_generation/real_video_autolabel.py:hands_to_wrist_pose`) — only the
  wrist joints are ever pseudo-labeled (deliberately; see that module's
  docstring for why guessing shoulders/hips would be unsafe).
- **Step label**: a rule engine reads HSV box-visibility/position/hand-proximity
  per frame and maps it onto the 8-step protocol
  (`RuleBasedStepLabeler`), the same `required_objects` logic
  `pipeline/state_machine.py` already encodes, run in reverse.

PoseNet is then fine-tuned (Stage 2) on those pseudo-labels with a **masked
loss** — only joints the pseudo-labeler actually found contribute gradient.

## Directory guide

```
config/
  experiment_config.py           Single source of truth for every constant/threshold
  procedure_pack.py               Procedure-pack YAML loader (fails loud, no silent fallback)
packs/
  box_sort_v1.yaml                 Default pack — this project's protocol, exported 1:1
  toy_3step.yaml                   Minimal second pack proving packs swap freely
pipeline/
  hsv_detector.py                Classical-CV box + hand detector
  pose_net.py                    HARPoseNet runtime wrapper (ONNX/PyTorch)
  rack_frame.py                  Orientation-agnostic (microgravity) pose normalization
  state_machine.py               Protocol sequence validation, hold-and-correct
  earth_delay_channel.py          Latency-race demo channel (display-only, never feeds the FSM)
  har_pipeline.py                Wires it all together, real-time loop
  hmr_backend.py                 Retired 3D-mesh stage (would need a pretrained model — inert shim)
train/
  train_posenet.py               HARPoseNet: synthetic pretrain + real fine-tune
  train_cnn.py                   HARActivityCNN, from scratch
  train_lstm.py                  HARLSTMClassifier, from scratch
data_generation/
  synthetic_pose.py              Procedural pose sequences (exact ground truth — PoseNet's own training target)
  build_posenet_realistic_dataset.py  LSTM's REAL training data: PoseNet's own predictions on rendered frames
  real_video_autolabel.py        Pseudo-labels the real "gravitational mimic" videos
  build_real_dataset.py          Turns pseudo-labels into LSTM/CNN training data
simulation/
  renderer.py, space_sim.py      Synthetic ISS-module renderer + full pipeline test harness
  race_demo.py                    Latency-race demo driver (scripted fault, --mode race)
gui/qt_dashboard.py              7-tab PyQt6 dashboard (see below)
main.py                          Entry point — see `--mode` below
end_to_end_loop.py               Auto-fix loop: data → train → sim → diagnose → retry
```

## Running it

```bash
cd sih-har-isro
python main.py --mode status                 # what's trained, what's missing
python main.py --mode train                   # PoseNet -> real fine-tune -> CNN -> LSTM
python main.py --mode pipeline --video some.mp4   # run the live pipeline + dashboard
python main.py --mode e2e                     # full auto-fix loop until gates pass
```

Other useful modes: `posenet` (pose model only, `--finetune-real` to also run
Stage 2), `autolabel` (pseudo-label the real videos only), `tuner` (interactive
HSV calibration), `sim` (space-sim latency/accuracy report only), `race` (see below).
Any mode accepts `--pack path/to/pack.yaml` to swap the experiment protocol.

## Latency-race demo — the core thesis, made undeniable

```bash
python main.py --mode race --earth-delay 4          # GUI, live
python main.py --mode race --earth-delay 8 --headless   # prints + saves JSON
```

Runs the same deterministic scripted fault (skip step 2, then correct it) through
the real pipeline twice at once: the **local** channel fires the instant an
anomaly happens (as it always does); a simulated **Earth** channel replays the
same event only after the configured delay (2/4/8s — the "practical ops/video"
range, not raw speed-of-light — see `pipeline/earth_delay_channel.py`'s
docstring). The GUI's **Latency Race** tab shows both feeds side by side with a
live "LOCAL WINS by Ns" banner. `--headless` prints a table and writes
`dataset/race_demo/race_report.json`. The Earth channel is display/logging-only
by construction — it is fed a *copy* of an already-fired local alert and can
never reach `ExperimentStateMachine`, so this demo cannot change what the
onboard system actually does.

## Procedure packs — swap the experiment without touching code

`config/experiment_config.py`'s `EXPERIMENT_STEPS` loads from a YAML pack
(`config/procedure_pack.py`), not a hardcoded table. `packs/box_sort_v1.yaml` is
the default (this project's protocol, exported 1:1); `packs/toy_3step.yaml` is a
minimal second pack proving the swap needs zero FSM/pipeline code changes:

```bash
python main.py --pack packs/toy_3step.yaml --mode pipeline --video some.mp4
```

Set via the `HAR_PROCEDURE_PACK` env var (which `--pack` sets before any
config-dependent module imports) — read once at `config/experiment_config.py`'s
own import time. A missing or malformed pack raises immediately at startup
(`ProcedurePackError`, no silent fallback to the wrong protocol) — see
`config/procedure_pack.py`'s validation (contiguous 1..N step ids, required
fields, non-empty steps list).

## Typed anomalies — A4 forbidden zone, A5 dwell/timeout, A7 occlusion

`pipeline/anomaly_monitor.py`'s `AnomalyMonitor` runs three checks every frame
(A1 skipped-step and A2 out-of-sequence already existed via
`state_machine.py`'s own logic; A8 model-disagreement already existed via the
CNN/LSTM ensemble — see `BAS_Onboard_Edge_Vision_LLM_Context.md` section 7 for
the full taxonomy; A3/A6/A9 are future work):

- **A4 forbidden zone** — a hand centroid enters a per-step rectangle from the
  active pack's `forbidden_zones` (normalized `[x1,y1,x2,y2]`). Edge-triggered
  HOLD on entry, clears when every hand leaves every active zone.
- **A5 dwell/timeout** — a step stays "current" too long with no FSM progress.
  Soft voice/log prompt at 70% of the pack's `timeout_s`; a hard HOLD at 100%.
  Clears when the FSM actually advances past that step.
- **A7 occlusion/abstain** — 10 consecutive frames with zero detections at all
  (covered lens, rack out of frame) HOLDs rather than letting a stale
  prediction stream sneak a wrong step through. Clears the instant detections
  resume.

Same authority rule as everywhere else in this project: **models/heuristics
propose, the FSM decides.** The monitor never advances or confirms a step — it
only calls `ExperimentStateMachine.force_hold(code, message)` /
`clear_hold(code)`. While `external_holds` is non-empty, `feed_prediction()`
no-ops entirely (inference keeps running every frame; nothing acts on it).
Multiple codes can hold at once (e.g. A5 + A7 simultaneously — verified in
testing); the FSM only unblocks once every held code clears. Each event is
voice-alerted (soft prompts are non-priority; holds/abstains interrupt) and
written to the structured JSONL log via `ExperimentLogger.log_anomaly()`
(`anomaly_code`, `severity`, `message`, plus per-type extras). The dashboard's
**Live Monitor** tab shows a sticky red/amber banner for as long as any typed
anomaly is active.

Measured: A4 detect-to-hold latency 0.32ms (well under any real-time budget).
A5/A7 verified end-to-end through the full `HARPipeline`, including
simultaneous multi-code holds and independent per-code clearing.

## Smart clip uplink — bandwidth thesis, made concrete

```bash
python main.py --uplink-mode clip   --mode pipeline --video some.mp4   # clips only, no stream
python main.py --uplink-mode both   --mode pipeline --video some.mp4   # stream + clips, compare
```

`pipeline/clip_uplink.py`'s `ClipUplinkManager` keeps a rolling `CLIP_PRE_ROLL_S`
(default 3s) ring buffer of raw frames at near-zero cost, and on any
anomaly-ish event (A4/A5/A7, step_skipped, out_of_sequence, uncertain — the
same choke point the latency-race demo already fires on) writes a clip
containing that pre-roll plus `CLIP_POST_ROLL_S` (default 3s) more frames to
`dataset/clips/`. Three modes, set via `--uplink-mode` / `HAR_UPLINK_MODE` /
`config.UPLINK_MODE` (default **`stream`**, today's behavior unchanged — zero
cost, `ClipUplinkManager` isn't even constructed):

- **`stream`** — continuous full IP stream (SIH requirement), no clips.
- **`clip`** — no continuous stream; only event-triggered clips. Full local
  recording is untouched either way (never lose the raw video for review).
- **`both`** — stream + clips, so a run can report the actual bandwidth
  difference on itself rather than a theoretical estimate.

`ClipUplinkManager.bandwidth_summary()` is the actual number behind the
brief's "don't ship 25 Mbps continuously, ship the 4 seconds that matter"
argument: raw-frame bytes a continuous stream would have sent this session vs.
bytes the clip files actually came to. The dashboard's **Clip Uplink** tab
shows the live savings percentage and every clip saved so far; each clip is
also written to the structured JSONL log via `ExperimentLogger.log_clip_saved()`.

Verified end-to-end through a real `HARPipeline`: `clip` mode disables the
continuous stream, a triggered clip flushes to a valid non-empty `.mp4` with
the expected pre-roll+post-roll frame count, re-triggering while a clip is
already pending is a no-op (matches `AnomalyMonitor`'s own "don't flood"
edge-triggering), and `close()` flushes any still-pending clip at shutdown.

## The dashboard (`gui/qt_dashboard.py`)

Eight tabs, all live:

1. **Live Monitor** — video feed, step checklist, alerts, recording/stream status
2. **Pipeline Internals** — per-stage latency bar chart, rolling FPS line chart,
   model backend info (ONNX vs PyTorch, threaded inference, rack-frame on/off)
3. **Detections** — live table of every HSV/hand detection this frame
4. **Latency Race** — onboard vs. simulated delayed-Earth alert feeds, side by side
5. **Clip Uplink** — active uplink mode, live bandwidth-savings %, every
   anomaly-triggered clip saved so far
6. **Model & Dataset** — checkpoint metrics (val accuracy/PCK), dataset sizes
   (synthetic vs real), active procedure pack, the real pseudo-label quality
   report, one click "Refresh"
7. **Training Console** — launches `train` / `posenet` / `autolabel` / `e2e` as a
   live subprocess with streamed output, right from the GUI
8. **Logs** — tails the newest structured experiment log

## Known limitations (stated, not hidden)

- **Real end-to-end accuracy (0.45 step-window, full-protocol run does not complete)
  is the main open problem**, not the pipeline/GUI/training infra around it — see
  "Current results" above for the full diagnosis (PoseNet's elbow/wrist PCK is the
  bottleneck) and what would actually move it (more PoseNet training budget/data,
  higher heatmap resolution, more real footage).
- The real "gravitational mimic" footage is a handful of short clips; pseudo-
  labels are heuristic, not ground truth — `frac_unknown` /
  `wrist_detection_rate` per video are reported in the Model & Dataset tab and
  `dataset/real_pseudo/autolabel_report.json` so this can be spot-checked.
  Some steps (1, 2 — closed-lid approach/open) have little or no real coverage
  in these particular clips; synthetic data remains the reliable backbone for
  full 8-class coverage.
- HARPoseNet only tracks 13 of the 33 MediaPipe-style landmark slots (the ones
  that actually drive step classification); the rest stay zero/invisible by
  design rather than guessed.
- `RACK_FRAME_NORMALIZE` and `CNN_ENSEMBLE_ENABLED` remain off by default —
  flipping them needs a matching retrain (see `config/experiment_config.py`'s
  comments on each).
