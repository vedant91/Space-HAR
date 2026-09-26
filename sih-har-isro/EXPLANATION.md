# What this project is, end to end

Written 2026-09-21. Plain-language companion to
`docs/designs/step-classifier-generalisation.md`. That document is the plan.
This one explains what exists, how it fits together, and what is actually true.

---

## 1. The problem in one paragraph

An astronaut on a space station performs a scripted experiment on a payload
rack: eight steps, in order. A fixed camera watches. Our system has to work out
which step they are on, notice when a step is skipped or done out of order, warn
them by voice, and log everything with timestamps. It must run offline on a
small box next to the camera, with no internet and no sending video to the
ground. The judged deliverable is a **trained AI model that does this**.

The hard part is not the alerts or the logging. Those work. The hard part is the
model, because in microgravity there is no floor, so an astronaut can be
sideways, upside down, or drifting, and ordinary pose software assumes people
stand upright on the ground.

---

## 2. The eight steps

```
  1  approach and stabilise at the rack
  2  open the container lid
  3  remove the red box
  4  inspect / manipulate the red box
  5  place the red box in the left restraint tray
  6  remove the yellow box
  7  inspect / manipulate the yellow box
  8  place the yellow box in the right restraint tray
```

Eight steps means random guessing gets **12.5%** right. Remember that number.
Every accuracy figure in this project is meaningless without it.

---

## 3. The live system: camera to voice alert

This is what runs on the edge box during an experiment. All of it is built and
working today.

```
                         FIXED PAYLOAD CAMERA
                                  |
                                  v
                          one video frame
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
     HSV BOX DETECTOR                         UPRIGHT POSE
     pipeline/hsv_detector.py                 pipeline/upright_pose.py
     finds the red box, yellow box,           1. read the rack tilt from
     and the white container by                  the box detector
     colour                                   2. rotate the whole IMAGE so
              |                                  the rack is level
              |                                3. run MediaPipe on the
              |                                   straightened image
              |                                4. rotate the 33 body points
              |                                   back to original coords
              |                                       |
              +-------------------+-------------------+
                                  |
                                  v
                       RACK-FRAME NORMALISER
                       pipeline/rack_frame.py
                       re-express body points relative to the
                       RACK, not the floor: "hand near the
                       container" instead of "hand 1.2 m off
                       the ground"
                                  |
                                  v
                       132 numbers per frame
                       (33 body points x 4 values:
                        x, y, depth, confidence)
                                  |
                                  v
                       SLIDING BUFFER  (30 frames)
                       pipeline/har_pipeline.py
                       keeps the last 30 frames, because a
                       single frame cannot tell you whether a
                       hand is arriving or leaving
                                  |
                                  v
                       LSTM CLASSIFIER
                       models/*.onnx
                       reads the 30-frame sequence, outputs
                       "this looks like step 4, 71% sure"
                                  |
                                  v
                       STATE MACHINE
                       pipeline/state_machine.py
                       knows the protocol order. Needs the same
                       answer 15 frames running before it
                       believes a step changed
                                  |
                +-----------------+-----------------+
                |                 |                 |
                v                 v                 v
          correct order     step skipped      out of order
                |                 |                 |
                v                 v                 v
          prompt next        VOICE ALERT      VOICE ALERT
          step               "complete        "wrong order"
                             step 3"
                |                 |                 |
                +--------+--------+-----------------+
                         v
         TIMESTAMPED LOG + LOCAL RECORDING + IP STREAM + Qt DASHBOARD
         pipeline/logger.py  pipeline/stream_sender.py  main.py
```

**Everything in that diagram works.** The voice alerts fire, the state machine
catches skipped steps, the dashboard draws, the logs are written. The one box
that does not work properly is the LSTM classifier.

---

## 4. The training system: how the model gets made

This is the other half, and it is where the problem lives.

```
   BLENDER DIGITAL TWIN                 blender/bas_har/
   a 3D model of the space module,
   the rack, the boxes, and an
   astronaut with a skeleton
            |
            |  render 18 takes (complete run-throughs),
            |  varying crew orientation, lighting, clutter
            v
   RENDERED FRAMES + PERFECT LABELS      dataset/blender_core/
   we know exactly which step each
   frame belongs to, because we
   scripted it
            |
            v
   tools/build_sequences.py
   run the SAME pose pipeline the live
   system uses over every rendered frame,
   then cut it into 30-frame windows
            |
            v
   X_sequences.npy  +  y_labels.npy  +  groups.npy
            |
            v
   train/train_lstm.py
   train the classifier, splitting BY TAKE
   so a whole performance is either training
   data or test data, never both
            |
            v
   models/lstm_classifier_real.pt  ->  .onnx
            |
            v
   tools/evaluate_real.py
   score it against takes it has never seen
```

### Why we train on renders instead of real video

We need thousands of frames where we know the exact step, filmed at astronaut
orientations nobody can reproduce on Earth. Rendering gives perfect labels for
free. The risk is that the model learns the renderer instead of the activity,
which is exactly what appears to have happened.

### The one rule that makes our numbers honest

`groups.npy` holds the **take number**. Training splits by take, so the model is
tested on a complete performance it has never seen. Most public projects in this
space cut windows from a few videos and shuffle them randomly, which puts two
windows a fraction of a second apart on opposite sides of the split. They report
99%. It means nothing. Our protocol is why our number is low and real.

---

## 5. The honest numbers

Chance is 0.125.

| What | Number | Meaning |
|---|---|---|
| Shipped model, unseen takes | **0.170** | basically chance |
| Best retrained model, unseen takes | **0.201** | barely above chance |
| Same model, takes it trained on | 0.561 - 0.753 | it memorised those |
| Pose detection at 135 deg roll, MediaPipe alone | 25% | it usually sees nobody |
| Same, with `upright_pose.py` | **100%** | our fix works |
| Red box detection recall | 0.324 | misses two thirds |
| Yellow box detection recall | **0.118** | misses seven of eight |
| Box detection precision | 1.000 | never fires falsely |

**What works:** the orientation fix. Rotating the image by the rack's own tilt
took pose detection from 25% to 100% at extreme crew angles. That is a genuine
result and no comparable public project publishes anything like it.

**What does not:** the step classifier. 0.201 against a 0.125 floor.

---

## 6. Why the model is broken: three suspects

### Suspect 1: it learned the pose estimator's quirks, not the body

This is the finding that changed our plan tonight. From
`logs/real_eval_retrained.json`:

```
  step_acc_real    = 0.4032     scored using MediaPipe's noisy pose
  step_acc_oracle  = 0.1448     scored using PERFECT ground-truth pose
  perception_cost  = -0.2584
```

Read that twice. The model does **worse** when you hand it perfect information
about where the body is. If it had learned "step 4 looks like this arm
configuration", clean data would help. Getting worse means it is leaning on
something else: MediaPipe's dropout patterns, its confidence values, its jitter.
Those are fingerprints of the pose estimator, they vary per take, and that is a
perfect memorisation shortcut.

Caveat, stated honestly: the model was trained on MediaPipe output, so testing
it on ground-truth pose is an out-of-distribution test and some drop is
expected. A drop all the way to chance is still a strong signal. The experiment
that settles it is cheap: train on ground-truth pose, test on ground-truth pose.

### Suspect 2: there is only one performance in the entire dataset

Every take replays the **same scripted hand trajectory**. Per-take variation is
only: crew roll angle, lighting, background clutter, a speed multiplier between
0.88 and 1.14, a 12 mm random offset, and some drift phases.

```
   what we have                    what we need
   ------------                    ------------
   one motion, 18 camera           18 different performances
   angles and speeds               of the same task

   [==== step 4 ====]              [== step 4 ==]
   [==== step 4 ====]              [===== step 4 =====]
   [==== step 4 ====]              [== step 4 ==]   (regrip)
        ...                        [=== step 4 ===]  (other hand)
```

A model cannot learn what makes step 4 *step 4* when only one example of step 4
exists in the world. This is why the plan now varies the hand waypoints per take
before rendering. It costs code, not GPU hours.

### Suspect 3: the window is too short

`SEQUENCE_WINDOW = 30` frames at `FPS = 30` is **one second**. A real protocol
step takes **four to six seconds**. The model is being asked to name an action
from a one-second glimpse of it. The fix is to sample at 10 frames per second so
those same 30 slots cover three seconds, and to render longer steps so a step
actually fills the window.

---

## 7. Things that are quietly wrong (found tonight, being fixed)

| Problem | Where | Effect |
|---|---|---|
| The rendered dataset no longer exists | `dataset/blender/` | every published number is currently unreproducible; we must re-render before anything |
| Training and serving disagree on normalisation | `retrain_and_evaluate.sh` never passes `--rack-normalize` | trains one way, runs another |
| Retrained models never reach the live system | exporter avoids `LSTM_ONNX_PATH` on purpose | the demo runs the old broken model |
| The new render plan has no holdout tags | `render_dataset.py:118` | the test set would come out **empty** |
| Holdout asks for `orient_-45`, which `core` does not contain | `retrain_and_evaluate.sh:32` | test set of **one take** |
| Held-out camera takes leak into training | `build_sequences.py:52` | the unseen-camera test is contaminated |
| Two different window filters, 0.75 and 0.6 | `build_sequences.py:123` vs `evaluate_real.py:307` | training and scoring disagree about what a window is |
| Released boxes freeze in mid-air | `protocol.py:287-300` | physically wrong for microgravity |
| Body drift ignores the arms | `protocol.py:222-245` | no Newton's third law, and a per-take signature to memorise |
| Zero tests in the entire project | everywhere | nothing catches any of the above |

---

## 8. What we decided, and why

Thirteen decisions from tonight's design session and engineering review:

1. **Fix the generator physics before rendering**, not after. Takes rendered
   before and after a generator change are different datasets and cannot be
   pooled.
2. **Add trajectory variety** to the hand waypoints so each take is a different
   performance. Attacks suspect 2 with code instead of GPU hours.
3. **One shared feature function** (`pipeline/features.py`) used by both training
   and the live pipeline, so they cannot drift apart.
4. **One shared windowing function** across all seven places that currently cut
   windows independently.
5. **Sample at 10 fps** on both sides, applied where frames enter the buffer, so
   the 30-frame window covers three seconds.
6. **Pin the model path** and copy the chosen model into place explicitly, rather
   than loading whichever `.onnx` is newest. `models/` contains a CNN too.
7. **Add object channels** (box positions, hand-to-box distances) with absence
   encoded as a constant applied *after* normalisation.
8. **Require the dataset argument** on the retrain script, so it can never
   silently score the wrong data.
9. **Smoke-render with assertions** before committing hours to a full render.
10. **Add pytest** and cover all 24 identified paths, including six regressions.
11. **Standardise the window purity filter at 0.75** and re-measure the baseline.
12. **Transfer learning** via self-supervised pretraining, gated: only if
    held-out accuracy is still under 0.40 after everything else.
13. **Keep the honest evaluation protocol** exactly as it is. It is the
    differentiator.

---

## 9. What happens next, in order

```
  NOW   free disk space, move TEMP off the full C: drive
   |
   v    fix the generator: trajectory variety, reaction-coupled drift,
   |    release momentum, holdout tags, frames_per_step 120, drop the
   |    unused second camera
   v
  SMOKE render one take, assert labels line up, measure the time
   |
   v
  FULL render, roughly 19k frames, unattended
   |
   v    while it renders: shared feature function, shared windowing,
   |    10 fps sampling, pytest, the six regression tests
   v
  TRAIN and score on takes it has never seen
   |
   v
  THE EXPERIMENT: train on ground-truth pose, test on ground-truth pose.
   |             If it scores well, the problem is the pose features.
   |             If it scores badly, the problem is the generator.
   v
  DECIDE whether transfer learning is needed, based on the number
```

---

## 10. Where each piece lives

| What it does | File |
|---|---|
| Find coloured boxes | `pipeline/hsv_detector.py` |
| Pose that works at any crew angle | `pipeline/upright_pose.py` |
| Measure the body against the rack | `pipeline/rack_frame.py` |
| Optional 3D body reconstruction | `pipeline/hmr_backend.py` (off) |
| Run everything, per frame | `pipeline/har_pipeline.py` |
| Protocol order, skip detection | `pipeline/state_machine.py` |
| Voice alerts | `pipeline/voice_alert.py` |
| Timestamped logs | `pipeline/logger.py` |
| IP streaming | `pipeline/stream_sender.py` |
| The 3D space module and astronaut | `blender/bas_har/` |
| Script the 8-step performance | `blender/bas_har/protocol.py` |
| Render a dataset | `blender/render_dataset.py` |
| Rendered frames to training windows | `tools/build_sequences.py` |
| Train the classifier | `train/train_lstm.py` |
| Score it honestly | `tools/evaluate_real.py` |
| One-command retrain and score | `tools/retrain_and_evaluate.sh` |
| All tunable constants | `config/experiment_config.py` |
| Measured results, the honest record | `logs/RESULTS.md` |

---

## 11. Glossary

- **Take** — one complete run-through of all eight steps, rendered as a video.
- **Window** — 30 consecutive frames. The model's unit of input.
- **Held-out** — data the model never saw during training. The only honest score.
- **Chance** — 0.125 here. Eight steps, random guess.
- **MediaPipe** — Google's pose library. Gives 33 body points per frame.
- **LSTM** — a network that reads a sequence and remembers what came earlier.
- **ONNX** — a portable model file format, fast on CPU, used by the live system.
- **HSV** — a colour space. The box detector finds boxes by hue.
- **Rack frame** — measuring the body relative to the equipment rack rather than
  the floor, because in microgravity there is no floor.
- **HMR** — Human Mesh Recovery, 3D body reconstruction. Optional, already
  stubbed in `pipeline/hmr_backend.py`, not currently switched on.
- **Train/serve skew** — when training data is built differently from live data.
  The bug class this project keeps hitting.
