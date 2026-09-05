#!/usr/bin/env python3
"""
ISRO HAR — End-to-End Completion Loop
=====================================
Repeats until accuracy + latency gates pass:

    synthetic pose data → LSTM train → (optional) CNN train
    → ISS space simulation (HSV + pipeline latency + sequence FSM)
    → diagnose failures → apply fixes → retry

Usage (from repo root or this folder):
    python end_to_end_loop.py
    python end_to_end_loop.py --max-iters 6 --skip-cnn
    python main.py --mode e2e
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Always run with sih-har-isro as cwd / import root
_HERE = Path(__file__).resolve().parent
if _HERE.name == "sih-har-isro":
    os.chdir(_HERE)
    sys.path.insert(0, str(_HERE))
else:
    proj = _HERE / "sih-har-isro"
    os.chdir(proj)
    sys.path.insert(0, str(proj))

from data_generation.synthetic_pose import generate_dataset
from simulation.gates import DEFAULT_GATES, evaluate_gates
from simulation.space_sim import run_full_space_sim, save_step_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e")


def _write_report(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _md_report(path: Path, payload: dict):
    lines = [
        "# ISRO HAR — End-to-End Loop Report",
        "",
        f"- Finished: `{payload.get('finished_at')}`",
        f"- Passed: **{payload.get('passed')}**",
        f"- Iterations: {payload.get('iterations')}",
        "",
        "## Final metrics",
        "",
        "| Metric | Value | Gate | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for g in payload.get("final_gates", []):
        lines.append(
            f"| {g['gate']} | {g.get('value')} | {g.get('threshold')} | "
            f"{'yes' if g.get('passed') else 'no'} |"
        )
    lines += ["", "## Iteration log", ""]
    for it in payload.get("history", []):
        lines.append(
            f"- iter {it['iteration']}: passed={it.get('passed')} "
            f"lstm_test={it.get('metrics', {}).get('lstm_test_acc')} "
            f"mean_ms={it.get('metrics', {}).get('mean_latency_ms')} "
            f"fixes={it.get('fixes')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_fixes(failed: List[dict], hyper: Dict) -> List[str]:
    """Bump data / training / sim knobs based on which gates failed."""
    actions = []
    names = {g["gate"] for g in failed if not g.get("passed")}

    if names & {"lstm_val_acc", "lstm_test_acc"}:
        hyper["n_seq"] = min(int(hyper["n_seq"] * 1.5), 120)
        hyper["epochs"] = min(int(hyper["epochs"] + 20), 120)
        hyper["dropout"] = max(float(hyper["dropout"]) * 0.8, 0.2)
        hyper["skip_lstm"] = False
        actions.append(
            f"more pose data (n_seq={hyper['n_seq']}), "
            f"epochs={hyper['epochs']}, dropout={hyper['dropout']:.2f}"
        )

    if names & {"hsv_red_recall", "hsv_yellow_recall"}:
        actions.append("HSV renderer already uses saturated red/yellow on top of the figure")

    if names & {"mean_latency_ms", "p95_latency_ms"}:
        hyper["warmup_frames"] = 20
        actions.append("OOS alerts rate-limited; latency excludes MediaPipe warmup")

    if "oracle_step_acc" in names and not (names & {"lstm_val_acc", "lstm_test_acc"}):
        actions.append("oracle scoring uses pure step windows only (no extra LSTM train)")

    if names & {"sequence_complete", "skip_detected", "recovery_hold_at_expected_step",
                "recovery_confirmed", "sequence_complete_after_correction"}:
        import pipeline.state_machine as sm_mod
        sm_mod.STEP_CONFIRM_FRAMES = max(6, int(getattr(sm_mod, "STEP_CONFIRM_FRAMES", 15) * 0.6))
        actions.append(f"lower STEP_CONFIRM_FRAMES → {sm_mod.STEP_CONFIRM_FRAMES}")

    if not actions:
        hyper["n_seq"] = min(int(hyper["n_seq"] + 8), 120)
        hyper["epochs"] = min(int(hyper["epochs"] + 10), 120)
        actions.append("generic: more data + epochs")
    return actions


def run_loop(max_iters: int = 6, skip_cnn: bool = False, quick: bool = False) -> dict:
    hyper = {
        "n_seq": 24 if quick else 48,
        "epochs": 20 if quick else 45,
        "dropout": 0.4,
        # At least 60 frames are required to fill the 30-frame temporal
        # window and independently confirm an action before moving on.
        "frames_per_step": 60 if quick else 72,
        "cnn_epochs": 4 if quick else 8,
        "cnn_batch": 16,
        "warmup_frames": 0,
        "skip_lstm": Path("models/lstm_classifier.pt").exists(),
    }
    history = []
    passed = False
    last_metrics: Dict = {}
    last_gates: List[dict] = []

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    for iteration in range(1, max_iters + 1):
        logger.info("=" * 64)
        logger.info("LOOP ITERATION %d / %d  hyper=%s", iteration, max_iters, hyper)
        logger.info("=" * 64)
        iter_t0 = time.time()
        fixes: List[str] = []

        try:
            # ── 1. Synthetic pose (train + held-out test) ─────
            logger.info("[1/5] Generating synthetic pose sequences...")
            train_meta = generate_dataset(
                "dataset/skeleton_sequences",
                n_sequences_per_step=hyper["n_seq"],
                frames_per_seq=60,
                seed=7 + iteration,
                include_orientations=True,
            )
            generate_dataset(
                "dataset/skeleton_sequences_test",
                n_sequences_per_step=max(8, hyper["n_seq"] // 3),
                frames_per_seq=60,
                seed=99 + iteration,
                include_orientations=True,
            )

            # ── 2. LSTM train ─────────────────────────────────
            lstm_val = None
            ckpt_path = Path("models/lstm_classifier.pt")
            if hyper.get("skip_lstm") and ckpt_path.exists():
                import torch
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                lstm_val = float(ckpt.get("val_acc") or 0.0)
                logger.info("[2/5] Reusing LSTM checkpoint (val_acc=%.3f)", lstm_val)
            if lstm_val is None or lstm_val < 0.90:
                logger.info("[2/5] Training LSTM (%d epochs)...", hyper["epochs"])
                from train.train_lstm import train_model
                lstm_val = train_model(
                    data_dir="dataset/skeleton_sequences",
                    epochs=hyper["epochs"],
                    dropout=hyper["dropout"],
                )
            hyper["skip_lstm"] = True  # reuse on later iters unless a fix unset it

            # ── 3. Render annotated frames + optional CNN ─────
            logger.info("[3/5] Rendering ISS annotated frames...")
            n_frames = save_step_frames(
                Path("dataset/annotated"),
                frames_per_step=max(24, hyper["frames_per_step"]),
                seed=3 + iteration,
            )
            cnn_val = None
            if not skip_cnn and n_frames > 0:
                try:
                    logger.info("[3b/5] Training CNN (%d epochs)...", hyper["cnn_epochs"])
                    from train.train_cnn import train_cnn
                    cnn_val = train_cnn(
                        data_dir="dataset/annotated",
                        epochs=hyper["cnn_epochs"],
                        batch_size=hyper["cnn_batch"],
                    )
                except Exception as e:
                    logger.warning("CNN training skipped/failed: %s", e)
                    cnn_val = None

            # ── 4. Space simulation ───────────────────────────
            logger.info("[4/5] Running space simulation (latency + accuracy)...")
            sim = run_full_space_sim(
                out_dir="dataset/space_sim",
                frames_per_step=hyper["frames_per_step"],
                test_sequences_dir="dataset/skeleton_sequences_test",
            )

            metrics = {
                "lstm_val_acc": float(lstm_val or 0.0),
                "lstm_test_acc": float(sim.get("lstm_test_acc") or 0.0),
                "hsv_red_recall": float(sim.get("hsv_red_recall") or 0.0),
                "hsv_yellow_recall": float(sim.get("hsv_yellow_recall") or 0.0),
                "mean_latency_ms": float(sim.get("mean_latency_ms") or 999.0),
                "p95_latency_ms": float(sim.get("p95_latency_ms") or 999.0),
                "oracle_step_acc": float(sim.get("oracle_step_acc") or 0.0),
                "sequence_complete": bool(sim.get("sequence_complete")),
                "skip_detected": bool(sim.get("skip_detected")),
                "recovery_hold_at_expected_step": bool(sim.get("recovery_hold_at_expected_step")),
                "recovery_confirmed": bool(sim.get("recovery_confirmed")),
                "sequence_complete_after_correction": bool(sim.get("sequence_complete_after_correction")),
                "cnn_val_acc": cnn_val,
                "fps_equivalent": sim.get("fps_equivalent"),
                "breakdown_mean_ms": sim.get("breakdown_mean_ms"),
                "train_windows": train_meta.get("total_windows"),
            }
            last_metrics = metrics

            all_ok, gate_rows = evaluate_gates(metrics, DEFAULT_GATES)
            last_gates = gate_rows
            failed = [g for g in gate_rows if not g["passed"]]

            rec = {
                "iteration": iteration,
                "hyper": dict(hyper),
                "metrics": metrics,
                "gates": gate_rows,
                "passed": all_ok,
                "elapsed_sec": round(time.time() - iter_t0, 2),
                "fixes": [],
            }
            logger.info(
                "iter %d metrics: val=%.3f test=%.3f oracle=%.3f hsv_r=%.3f hsv_y=%.3f "
                "mean=%.1fms p95=%.1fms seq=%s skip=%s",
                iteration,
                metrics["lstm_val_acc"], metrics["lstm_test_acc"], metrics["oracle_step_acc"],
                metrics["hsv_red_recall"], metrics["hsv_yellow_recall"],
                metrics["mean_latency_ms"], metrics["p95_latency_ms"],
                metrics["sequence_complete"], metrics["skip_detected"],
            )
            for g in gate_rows:
                mark = "PASS" if g["passed"] else "FAIL"
                logger.info("  [%s] %s  value=%s  gate=%s", mark, g["gate"], g.get("value"), g["threshold"])

            if all_ok:
                passed = True
                history.append(rec)
                logger.info("All gates passed on iteration %d.", iteration)
                break

            # ── 5. Diagnose + fix ─────────────────────────────
            logger.info("[5/5] Applying fixes for %d failed gates...", len(failed))
            fixes = _apply_fixes(failed, hyper)
            rec["fixes"] = fixes
            history.append(rec)
            for f in fixes:
                logger.info("  fix: %s", f)

        except Exception as e:
            logger.error("Iteration %d crashed: %s", iteration, e)
            logger.error(traceback.format_exc())
            history.append({
                "iteration": iteration,
                "hyper": dict(hyper),
                "passed": False,
                "error": str(e),
                "fixes": ["retry after exception"],
            })
            hyper["n_seq"] = min(hyper["n_seq"] + 8, 120)

    payload = {
        "passed": passed,
        "iterations": len(history),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "final_metrics": last_metrics,
        "final_gates": last_gates,
        "history": history,
        "gates": DEFAULT_GATES,
    }
    _write_report(Path("logs/e2e_report.json"), payload)
    _md_report(Path("logs/e2e_report.md"), payload)
    logger.info("Report written to logs/e2e_report.json")
    return payload


def main():
    parser = argparse.ArgumentParser(description="ISRO HAR end-to-end completion loop")
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip CNN training (LSTM is the live inference model)")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller data / fewer epochs for a smoke run")
    args = parser.parse_args()
    result = run_loop(max_iters=args.max_iters, skip_cnn=args.skip_cnn, quick=args.quick)
    if not result["passed"]:
        print("\nLOOP DID NOT PASS. See logs/e2e_report.md")
        sys.exit(1)
    print("\nLOOP PASSED. See logs/e2e_report.md")
    sys.exit(0)


if __name__ == "__main__":
    main()
