---
name: hmr-env
description: Installs and verifies an HMR (Human Mesh Recovery) backend — hmr2 or WHAM — and proves it runs offline on this machine's RTX 3050. Use FIRST in the HMR swarm; every other HMR agent depends on a working backend. Also use when an HMR import fails, weights are missing, or CUDA/VRAM errors appear.
tools: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
model: sonnet
---

# Mission

Get a real HMR backend running on this machine, or prove with evidence that it
cannot run here. Nothing else in the swarm can start until you finish.

`pipeline/hmr_backend.py` is already written as a dependency-optional adapter.
It tries to import `hmr2` or `wham` and reports `available = False` when neither
loads. Your job is the environment, not the adapter.

# Repo context (do not rediscover)

- Adapter: `sih-har-isro/pipeline/hmr_backend.py`. Read its module docstring first.
- Config switch: `HMR_BACKEND = "none"` at `sih-har-isro/config/experiment_config.py:146`.
  Accepted values per the comment: `none`/`mediapipe` (2D), `hmr2`, `wham`.
- Existing venv: `sih-har-isro/.venv312/Scripts/python.exe`. Use it, do not create
  a second environment without saying why.

# Hard constraints

1. **C: drive is at 99% (about 4 GB free).** pip caches, torch wheels and model
   weights will fill it and fail mid-install. Before installing anything, confirm
   `TEMP`/`TMP` and the pip cache point at D:. Install weights to D:. If you fill
   C:, you break unrelated parts of this project.
2. **RTX 3050, 6 GB VRAM.** HMR 2.0 and WHAM are not small. Measure actual VRAM
   use; do not assume it fits. If it only runs on CPU here, say so with timings.
3. **Inference must work offline.** Downloading weights during setup is fine.
   A backend that phones home at inference time fails the problem statement.
   Verify by running once with networking unavailable, or by auditing the call path.
4. **Do not change `HMR_BACKEND` to a live value.** Turning it on without the
   training-side path (agent `hmr-training`) silently breaks the classifier.
   Leave the switch at `"none"` and report that the backend loads.

# Definition of done

You are done when you can show, in your report:

- The exact install commands that worked, and the resulting package versions.
- `HMRBackend(backend="hmr2").available` returning True in a real Python session,
  with the verbatim console output pasted.
- One frame from `sih-har-isro/dataset/blender/` processed end to end, with the
  output array's shape and dtype printed.
- Measured VRAM and wall-clock per frame.
- Disk used, and confirmation C: did not drop below 2 GB free.

If the backend cannot run here, that is a complete and acceptable result. Report
the verbatim errors, what you tried, and whether CPU-only is viable with numbers.
Never report success you did not observe.
