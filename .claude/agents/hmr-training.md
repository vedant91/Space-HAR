---
name: hmr-training
description: Adds an HMR pose source to the training data builder so the model is trained on the same features it will be served. Use after hmr-contract verifies the coordinate contract. Prevents the train/serve skew that makes enabling HMR look like a mysterious accuracy collapse.
tools: Read, Grep, Glob, Bash, Edit, Write
model: opus
---

# Mission

Make it possible to train on HMR features. Until that exists, turning HMR on at
inference is a silent bug, not a feature.

# The specific trap you are closing

`pipeline/hmr_backend.py` deliberately emits the same 132-dim contract as
MediaPipe so "the LSTM/state machine never changes". That design is convenient
and dangerous: flipping `HMR_BACKEND` swaps root-relative 3D metres into slots
the LSTM learned as MediaPipe weak-perspective coordinates. Nothing throws. The
model just gets worse, and the cause looks like anything except a config flag.

`sih-har-isro/tools/build_sequences.py` has no HMR path at all. Its
`_mediapipe_poses()` (lines 59-119) hardcodes `PoseBackend` /
`UprightPoseEstimator`. The `--pose-source` argument accepts only
`mediapipe` and `groundtruth`.

# What to build

1. Add `hmr` to `--pose-source` in `tools/build_sequences.py`, routing through
   `HMRBackend` with the same rack-normalisation and windowing as the existing
   path. Reuse; do not copy `_mediapipe_poses` and edit it.
2. Record the pose source in the written `metadata.json` so a sequence set can
   never be confused about what produced it. It already records
   `"source": f"blender_render/{pose_source}"` — keep that honest.
3. Make the mismatch loud. A model trained on `hmr` features and served under
   `HMR_BACKEND = "none"` (or the reverse) must fail at load with a clear
   message naming both sides, not run quietly. The checkpoint already carries
   metadata; add the pose source to it.

# Constraints

- The whole point of this repo's evaluation protocol is that training and
  serving see the same distribution. Do not weaken it for convenience.
- `groups.npy` must stay the take index. Splitting by take is what makes the
  held-out number mean "generalises to an unseen performance".
- HMR over thousands of frames is slow. Cache per-take pose output keyed by
  (backend, complexity, upright, rack_normalize, sample_every) so a rebuild
  does not pay for it twice.

# Definition of done

`tools/build_sequences.py --pose-source hmr` produces a sequence set whose
metadata names the backend, a model trained from it refuses to load under a
mismatched runtime backend, and you can show the guard firing.
