# Measured results

All figures below come from `tools/evaluate_real.py` run against
`dataset/blender` — 12 takes, 3,144 rendered frames, ground truth produced by
projecting the Blender rig rather than by the process being scored.

Reproduce with:

```bash
tools/render_dataset_resilient.sh dataset/blender study   # ~3 h, CPU-only
tools/retrain_and_evaluate.sh
```

Chance accuracy is **0.125** (8 protocol steps).

---

## 1. Step classification

Scored on 2,376 dominant-label windows per model, pooled as
`sum(correct) / sum(windows)`.

| Model | Train takes | **Held-out takes** | Error sequences |
|---|---:|---:|---:|
| Shipped (`synthetic_pose`-trained) | 0.124 | 0.170 | 0.140 |
| Retrained, raw image coords | **0.753** | 0.165 | 0.256 |
| Retrained, rack-normalised | 0.561 | **0.201** | 0.164 |

Held-out takes are `orient_180`, `orient_-45`, `light_single_fail` — a crew
orientation and a lighting failure the model never saw. Error sequences
(`skip`, `recover`) were excluded from training entirely.

**Read this honestly:**

- The **shipped model is at chance on every split.** Its reported
  `lstm_test_acc` of 0.984 does not survive contact with an image. This is the
  single most important result here, and it is not a close call.
- The retrained models **fit the takes they saw** (0.75 raw) and **do not
  generalise** (0.17). Training loss collapsing while validation loss climbs
  says the same thing. With seven training takes the model is memorising
  performances, not learning the activity.
- Rack normalisation trades train fit for generalisation exactly as an
  invariance should: lower on train (0.561 vs 0.753), higher on held-out
  (0.201 vs 0.165). The direction is right; the magnitude is small because
  there is not enough data for the difference to matter yet.
- **The binding constraint is dataset size, not architecture.** Nothing here
  justifies a bigger model or a different classifier. It justifies more render
  time.

Anyone quoting a single number from this project should quote the **held-out**
column, and should say how many takes it was trained on.

---

## 2. Pose estimation under crew orientation

The problem statement predicts that ground-referenced pose models fail in
microgravity. They do, and the failure mode is worse than degraded accuracy —
the detector stops returning anything at all.

Detection rate, 20 mid-protocol frames per take:

| Crew roll | MediaPipe alone | With `upright_pose` |
|---|---:|---:|
| 0° | 19/20 (95%) | 19/20 (95%) |
| 90° | 7/20 (35%) | **20/20 (100%)** |
| 135° | 5/20 (25%) | **20/20 (100%)** |
| 180° | 12/20 (60%) | 19/20 (95%) |

Across whole takes with `upright_pose` enabled, detection is **99.3–100% at
every orientation**.

`pipeline/rack_frame.py` cannot deliver this on its own — it normalises the
landmarks MediaPipe returns, and at 90° there were none to normalise 65% of
the time. `pipeline/upright_pose.py` rotates the *frame* upright first, using
the payload rack's own roll from the HSV main-box rect, then maps landmarks
back to image coordinates.

Landmark accuracy, on frames where the detector fires and the ground truth
says the landmark is visible:

| | |
|---|---:|
| Key-joint mean error | 100.97 px (900 px frame) |
| PCK@5% of frame width | 0.191 |

Part of this is definitional rather than error: MediaPipe localises the suit
surface, the ground truth is the joint centre inside it, and the pressure suit
is ~7 cm of padding. A *systematic* offset is largely harmless for activity
recognition — the sequence model learns it — which is why sequences are built
from MediaPipe output rather than from ground-truth pose. The detection-rate
collapse was the real defect.

---

## 3. HSV object detection

Measured against true pixel boxes and ray-cast occlusion, not against the
renderer's own assertions.

| Object | Reported in `e2e_report.md` | Measured | Precision |
|---|---:|---:|---:|
| `red_box` | 1.000 | **0.324** | 1.000 |
| `yellow_box` | 1.000 | **0.118** | 1.000 |
| `main_box` | — | 0.885 | 1.000 |

Precision is perfect — the detector never fires on something that is not
there. Recall is poor: it misses most genuinely visible instances of the small
coloured boxes, which are shaded, partly occluded and only a few hundred
pixels in a realistic frame. The large white container survives.

The original 1.000 was tautological: `simulation/renderer.py` paints the boxes
in colours chosen to sit inside the detector's own thresholds
(`RED_BGR = (12, 12, 230)`, commented as such) and draws them last so nothing
can occlude them.

---

## 4. What to do next, in order

1. **More takes.** This is the whole answer to the generalisation gap. The
   `core` plan (18 takes, two cameras) exists and needs a machine with a CUDA
   GPU, or roughly a day on CPU.
2. **Longer steps.** Rendered steps run ~1.07 s against the protocol's 4–6 s,
   so landmark velocities are ~4× fast. Either render 120–150 frames per step,
   or sample the live stream at ~10 fps so the 30-frame window spans 3 s —
   the latter is better engineering regardless, since a 1 s window cannot
   contain a 5 s action.
3. **Feed the detector's output into the classifier.** The step is largely
   determined by *which box is where* — red in the container, in a hand, or in
   the left restraint zone. HSV gives that at precision 1.000. Concatenating
   box centroids and visibility onto the 132-dim pose vector is cheap and
   attacks the part of the problem pose is worst at.
4. **Improve HSV recall** before trusting the object channel: per-frame
   adaptive thresholds, or a small trained detector, now that there is
   labelled data to train one on.
