# Core engineering team

Implements `docs/designs/step-classifier-generalisation.md` (Review Outcomes +
Implementation Tasks). Agents share one working tree — git worktrees would
branch from HEAD and miss uncommitted work — so each owns a DISJOINT file set
and sends anything outside it back to the lead as a request.

| Agent | Owns | Tasks | Starts after |
|---|---|---|---|
| lead (main session) | `blender/`, `dataset/`, `tools/check_dataset.py`, `tools/record_take.py`, render supervisor | Data collection, render, coordination, promotion decisions | — |
| `pipeline-engineer` | `pipeline/`, `config/`, `tools/build_sequences.py`, `tools/evaluate_real.py`, `train/train_lstm.py`, 3 legacy windowing consumers | T6, T7, T13, load-time dim assert, purity 0.75, pose cache | — |
| `release-engineer` | `tools/retrain_and_evaluate.sh`, `tools/promote_model.py`, `end_to_end_loop.py` | Promote step, required dataset arg, three-way eval table, stale claims | — |
| `test-engineer` | `tests/`, `pytest.ini` | pytest from zero: sequence parity first, 6 regressions, 24 paths | pipeline + release |
| `ml-engineer` | `logs/`, new `models/` names, `train/experiments/` | Oracle experiment, baseline re-score, results table, gated pretraining | render gated |

---

# HMR swarm

Six agents for adding Human Mesh Recovery (3D body reconstruction) to the
BAS HAR pipeline. HMR is the problem statement's **optional advanced**
component, not the critical path. Read `sih-har-isro/docs/designs/step-classifier-generalisation.md`
section "On HMR" before launching any of these.

## Launch order (dependencies are real, not advisory)

```
  hmr-env          install a backend, prove it runs offline on the 3050
      |
      v
  hmr-contract     verify SMPL->MediaPipe mapping + coordinate space
      |
      +-------------------+-------------------+
      v                   v                   v
  hmr-training       hmr-perf            (blocked until
  HMR pose source    latency/VRAM         contract is signed off)
  in build_sequences  cadence
      |                   |
      +---------+---------+
                v
            hmr-eval       does HMR beat 2D upright-pose? may say NO
                |
                v
        hmr-integration    wire it, fallback, GUI  (skip if eval says no)
```

`hmr-env` and `hmr-contract` are strictly sequential. `hmr-training` and
`hmr-perf` can run in parallel once the contract is signed. `hmr-eval` gates
`hmr-integration`.

## Launching

```
Agent(subagent_type: "hmr-env", prompt: "...")
```

Run them one or two at a time. Parallel fan-out has hit session rate limits on
this account.

## The two rules that govern all six

1. **Never enable `HMR_BACKEND` without the matching training path.** The
   adapter emits the same 132-dim contract as MediaPipe, so flipping the flag
   swaps root-relative metres into slots the model learned as image
   coordinates. Nothing throws; the model just gets quietly worse.
2. **HMR did not solve orientation — `upright_pose.py` did.** At 135 degrees of
   crew roll MediaPipe returned nothing on 15 of 20 frames; a mesh regressor
   cannot recover a body from a detection that never fired. Rotating the frame
   by the rack's own roll took detection from 25% to 100%, in 2D. HMR's open
   claim is depth for hand-object reasoning. Test that claim, not the settled one.

## Honest outcome

`hmr-eval` is allowed to conclude that HMR costs latency and buys nothing
measurable on this dataset. That is a result. Ship the negative finding rather
than wiring an unproven layer into a system that currently works.
