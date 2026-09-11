"""
Rack-Frame / Orientation-Agnostic Demo (Brief §7 SIH-optional HMR bullet, G7)
================================================================================
SIH's optional PS bullet asks for rack-relative (not floor-relative) tracking
— there is no fixed "up" in microgravity, an astronaut can work sideways or
inverted relative to the rack. Per the audit doc's own guidance ("prefer:
enable/demo rack-frame normalization already conceived in architecture...
only pursue mesh HMR if judges/PS reviewers insist"), this demo measures
what this project's *existing* two mechanisms actually do for that story,
instead of guessing:

  1. Training-time orientation augmentation — `generate_sequence()`'s
     `orientation` parameter (0/1/2/3 = 0/90/180/270 degree whole-scene
     rotation of astronaut+rack together) is already baked into the default
     synthetic training set (`include_orientations=True` in
     `data_generation/synthetic_pose.py`). If this alone gives orientation-
     robust classification, the "no fixed up" story is already true today,
     just never measured or documented as such.
  2. `pipeline/rack_frame.py`'s `RackFrameNormalizer` — an explicit runtime
     correction (rotate the skeleton into rack-relative coordinates using
     the HSV-detected rack rectangle's angle) that config.RACK_FRAME_NORMALIZE
     gates, off by default. Flipping it on changes the LSTM's input feature
     distribution to something the current checkpoint was never trained on
     — this demo measures whether that hurts, so the "retrain if needed"
     line in the build plan is a measured decision, not a guess.

Renders every step at all 4 orientations (`simulation/renderer.py`'s
`render_step_clip`, which already supports this — this demo doesn't add new
rendering capability, it's the first thing to actually *use* the parameter
for an accuracy measurement), runs each render through the REAL pipeline —
real PoseNet inference on the rendered pixels, not oracle ground-truth pose
injection. Oracle injection (as the latency-race demo uses, correctly, for
its own timing-only purpose) would be actively misleading here: the current
LSTM checkpoint was retrained on PoseNet's own noisy predictions rather than
ground truth (see README's "train/inference distribution mismatch" section)
precisely so it matches real deployment input — feeding it oracle vectors
instead puts it off its own training distribution and produces confidently
wrong predictions that have nothing to do with orientation. Runs each
render once with `rack_frame_normalize=False` and once `=True`, and reports
real (non-oracle) step-classification accuracy per orientation per mode.

Usage:
    python simulation/rack_frame_demo.py
    python simulation/rack_frame_demo.py --frames-per-step 90
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ORIENTATION_NAMES = {0: "0° (upright)", 1: "90°", 2: "180° (inverted)", 3: "270°"}


def _run_one(step_id: int, orientation: int, rack_frame_normalize: bool,
            frames_per_step: int, seed: int) -> Dict:
    from simulation.renderer import render_step_clip
    from pipeline.har_pipeline import HARPipeline
    from config.experiment_config import SEQUENCE_WINDOW

    frames, metas = render_step_clip(step_id, n_frames=frames_per_step, seed=seed,
                                     orientation=orientation)

    pipeline = HARPipeline(
        source=0, headless=True, enable_voice=False, enable_recording=False,
        enable_streaming=False, use_threaded=False,
        rack_frame_normalize=rack_frame_normalize,
    )
    pipeline.reset_runtime()

    correct, voted = 0, 0
    for frame, meta in zip(frames, metas):
        # No injected_skel: real PoseNet inference on the rendered pixels,
        # matching actual deployment — see the module docstring for why
        # oracle pose injection would be misleading for this measurement.
        result = pipeline.process_frame(frame, annotate=False)
        if len(pipeline.skeleton_buffer) >= SEQUENCE_WINDOW:
            voted += 1
            if result["pred_step"] == step_id:
                correct += 1
    pipeline.close()

    return {
        "step_id": step_id, "orientation": orientation,
        "rack_frame_normalize": rack_frame_normalize,
        "windows_voted": voted, "windows_correct": correct,
        "accuracy": round(correct / voted, 3) if voted else None,
    }


def run_rack_frame_demo(frames_per_step: int = 72, seed: int = 7,
                        out_dir: str = "dataset/rack_frame_demo") -> Dict:
    from config.experiment_config import EXPERIMENT_STEPS

    step_ids = [s["id"] for s in EXPERIMENT_STEPS]
    orientations = (0, 1, 2, 3)
    results: List[Dict] = []

    t0 = time.perf_counter()
    for mode in (False, True):
        for ori in orientations:
            for step_id in step_ids:
                r = _run_one(step_id, ori, mode, frames_per_step, seed)
                results.append(r)
                logger.info("step=%d orientation=%s rack_frame_normalize=%s -> accuracy=%s",
                           step_id, ORIENTATION_NAMES[ori], mode, r["accuracy"])

    def _agg(mode: bool) -> Dict:
        by_ori = {}
        for ori in orientations:
            rows = [r for r in results if r["rack_frame_normalize"] == mode and r["orientation"] == ori]
            correct = sum(r["windows_correct"] for r in rows)
            voted = sum(r["windows_voted"] for r in rows)
            by_ori[ORIENTATION_NAMES[ori]] = round(correct / voted, 3) if voted else None
        all_rows = [r for r in results if r["rack_frame_normalize"] == mode]
        correct = sum(r["windows_correct"] for r in all_rows)
        voted = sum(r["windows_voted"] for r in all_rows)
        return {"by_orientation": by_ori,
               "overall": round(correct / voted, 3) if voted else None}

    report = {
        "frames_per_step": frames_per_step,
        "seed": seed,
        "steps_tested": step_ids,
        "orientations_tested": [ORIENTATION_NAMES[o] for o in orientations],
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "raw_skeleton_mode": _agg(False),   # today's default: RACK_FRAME_NORMALIZE off
        "rack_normalized_mode": _agg(True),  # explicit runtime correction, current checkpoint not retrained for it
        "per_run": results,
        "narrative": None,
    }
    raw_ok = report["raw_skeleton_mode"]["overall"]
    norm_ok = report["rack_normalized_mode"]["overall"]
    if raw_ok is not None and norm_ok is not None:
        if raw_ok >= norm_ok:
            report["narrative"] = (
                f"Raw-skeleton mode (today's default, no retrain) already classifies "
                f"{raw_ok*100:.0f}% of real (non-oracle, PoseNet-inferred) windows correctly across "
                f"all 4 orientations, thanks "
                f"to the orientation augmentation already baked into the synthetic training set "
                f"(generate_sequence()'s orientation param, include_orientations=True). "
                f"Rack-frame normalization scored {norm_ok*100:.0f}% — WORSE, because the current "
                f"LSTM checkpoint was never trained on rack-normalized features (a real distribution "
                f"shift, not a bug). Recommendation: keep RACK_FRAME_NORMALIZE=False as the shipped "
                f"default; the normalizer path is real and available, but needs a matched retrain "
                f"(train/train_lstm.py with the flag on first) before it's a net improvement — "
                f"exactly the 'retrain if needed' the build plan flagged."
            )
        else:
            report["narrative"] = (
                f"Rack-frame normalization scored {norm_ok*100:.0f}% vs. raw-skeleton mode's "
                f"{raw_ok*100:.0f}% across all 4 orientations — normalization helps even without a "
                f"retrain. Recommendation: consider flipping RACK_FRAME_NORMALIZE=True as the "
                f"shipped default."
            )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "rack_frame_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Rack-frame demo report written to %s", out_path)
    return report


def print_report(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("RACK-FRAME / ORIENTATION-AGNOSTIC DEMO")
    print("=" * 70)
    print(f"{len(report['steps_tested'])} steps x {len(report['orientations_tested'])} "
         f"orientations, {report['frames_per_step']} frames/step (real PoseNet inference, "
         f"not oracle pose)")
    print("\nRaw-skeleton mode (today's default):")
    for ori, acc in report["raw_skeleton_mode"]["by_orientation"].items():
        print(f"    {ori:20s} accuracy={acc}")
    print(f"    {'OVERALL':20s} accuracy={report['raw_skeleton_mode']['overall']}")
    print("\nRack-normalized mode (not retrained for):")
    for ori, acc in report["rack_normalized_mode"]["by_orientation"].items():
        print(f"    {ori:20s} accuracy={acc}")
    print(f"    {'OVERALL':20s} accuracy={report['rack_normalized_mode']['overall']}")
    print("\n" + "-" * 70)
    print(report["narrative"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rack-frame / orientation-agnostic demo")
    parser.add_argument("--frames-per-step", type=int, default=72)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="dataset/rack_frame_demo")
    args = parser.parse_args()

    report = run_rack_frame_demo(frames_per_step=args.frames_per_step, seed=args.seed,
                                 out_dir=args.out_dir)
    print_report(report)
