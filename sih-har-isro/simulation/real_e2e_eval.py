"""
Real (non-oracle) end-to-end evaluation: camera frame -> PoseNet -> LSTM -> FSM.

The honest accuracy number for this architecture. Unlike simulation/space_sim.py's
gate run (which injects ground-truth pose), this feeds only rendered pixels
through the real pipeline, so PoseNet's own errors are included.

Reports step-window accuracy (single-step 30-frame windows only, same scoring as
space_sim.run_pipeline_on_clip), a truth-vs-predicted confusion matrix, FSM
completion, and per-frame latency.

Usage:
    python simulation/real_e2e_eval.py                 # seeds 5,6,7
    python simulation/real_e2e_eval.py --seeds 5 --frames-per-step 72
    python simulation/real_e2e_eval.py --random-orientation   # stricter: orientation independent of step
    HAR_MODEL_DIR=/tmp/exp_models python simulation/real_e2e_eval.py   # A/B a checkpoint set
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import SEQUENCE_WINDOW  # noqa: E402
from simulation.renderer import render_protocol  # noqa: E402


def evaluate_seed(seed: int, frames_per_step: int = 72, warmup: int = 20,
                  random_orientation: bool = False) -> dict:
    from pipeline.har_pipeline import HARPipeline

    # Default render_protocol uses a fixed step->orientation schedule. A random one
    # (independent of step) is the stricter test: no orientation cue can stand in for step.
    oris = np.random.default_rng(1000 + seed).integers(0, 4, size=8).tolist() if random_orientation else None
    frames, metas = render_protocol(frames_per_step=frames_per_step, seed=seed, orientations=oris)
    pipe = HARPipeline(source=0, headless=True, enable_voice=False,
                       enable_recording=False, use_threaded=False)
    pipe.reset_runtime()

    conf = np.zeros((9, 9), dtype=int)   # [truth step 1..8][pred 0..8]
    window, lat, correct, total = [], [], 0, 0
    for i, (frame, meta) in enumerate(zip(frames, metas)):
        r = pipe.process_frame(frame, injected_skel=None, annotate=False)
        if i >= warmup:
            lat.append(r["timings_ms"]["total"])
        window.append(int(meta["step_id"]))
        window = window[-SEQUENCE_WINDOW:]
        if len(window) == SEQUENCE_WINDOW and len(set(window)) == 1 and r["pred_step"] > 0:
            conf[window[-1], r["pred_step"]] += 1
            total += 1
            correct += int(r["pred_step"] == window[-1])
    sm = pipe.state_machine.get_status_summary()
    done = [s["id"] for s in sm["steps"] if s["status"] == "COMPLETED"]
    pipe.close()
    return {
        "seed": seed, "step_window_acc": correct / max(total, 1), "n_windows": total,
        "completed_steps": done, "experiment_complete": bool(sm["experiment_complete"]),
        "mean_latency_ms": float(np.mean(lat)), "confusion": conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--frames-per-step", type=int, default=72)
    ap.add_argument("--random-orientation", action="store_true",
                    help="draw each step's body orientation independently (stricter than the fixed schedule)")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)   # per-frame pipeline logging drowns the report

    results = [evaluate_seed(s, args.frames_per_step, random_orientation=args.random_orientation)
               for s in args.seeds]
    total_conf = sum(r["confusion"] for r in results)
    for r in results:
        print(f"seed {r['seed']}: step-window acc={r['step_window_acc']:.3f} (n={r['n_windows']}) "
              f"completed={r['completed_steps']} full_protocol={r['experiment_complete']} "
              f"latency={r['mean_latency_ms']:.1f}ms")
    accs = [r["step_window_acc"] for r in results]
    print(f"\nMEAN step-window acc over {len(accs)} seeds: {np.mean(accs):.3f}  "
          f"(min {min(accs):.3f}, max {max(accs):.3f}); "
          f"full-protocol completions: {sum(r['experiment_complete'] for r in results)}/{len(results)}")
    print("\nConfusion (rows = true step, cols = predicted 1..8), pooled over seeds:")
    print("      " + " ".join(f"p{c}".rjust(4) for c in range(1, 9)))
    for t in range(1, 9):
        row = total_conf[t]
        n = max(row.sum(), 1)
        print(f"t{t}:   " + " ".join(f"{row[c]:4d}" for c in range(1, 9)) + f"   acc={row[t]/n:.2f}")


if __name__ == "__main__":
    main()
