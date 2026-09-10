"""
Latency-Race Demo — "onboard catches it now vs. Earth finds out later"
=========================================================================
Runs the SAME deterministic scripted fault (skip step 2, then correct it —
simulation/space_sim.py's existing recovery scenario, oracle pose injected
so the fault and its detection are exactly repeatable run to run) through
the real pipeline with the latency-race channel enabled, and reports how
long the "Earth" channel would have taken to find out about each anomaly
the onboard system already handled immediately.

This does NOT model a real network to any ground station — see
pipeline/earth_delay_channel.py's docstring for why that's a deliberate
non-goal, and for the guarantee that the delayed channel never feeds back
into ExperimentStateMachine.

Usage:
    python simulation/race_demo.py                      # default 4s delay
    python simulation/race_demo.py --delay 2             # 2/4/8s per the pitch narrative
    python simulation/race_demo.py --delay 8 --gui       # also show it live in the dashboard
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_race_demo(delay_s: float = 4.0, out_dir: str = "dataset/race_demo",
                  frames_per_step: int = 72, seed: int = 5,
                  gui_queue=None) -> Dict:
    from simulation.space_sim import generate_sim_assets
    from pipeline.har_pipeline import HARPipeline

    logger.info("Rendering the scripted fault scenario (skip step 2, then correct)...")
    assets = generate_sim_assets(out_dir, frames_per_step=frames_per_step, seed=seed)
    frames, metas = assets["recovery_frames"], assets["recovery_metas"]

    pipeline = HARPipeline(
        source=0, headless=gui_queue is None, gui_queue=gui_queue,
        enable_voice=False, enable_recording=False, use_threaded=False,
        enable_race=True, earth_delay_s=delay_s,
    )
    pipeline.reset_runtime()

    logger.info("Running %d frames through the real pipeline (oracle pose injected — this "
               "is a fault-timing demo, not a pose-accuracy test)...", len(frames))
    t0 = time.perf_counter()
    for frame, meta in zip(frames, metas):
        pipeline.process_frame(frame, injected_skel=meta["pose"], annotate=gui_queue is not None)
        if gui_queue is not None:
            pipeline._push_gui_status()

    logger.info("Waiting for the Earth channel's delayed deliveries (%.0fs)...", delay_s)
    pipeline.earth_channel.flush()
    events = pipeline.earth_channel.summary()
    voice_history = list(pipeline.voice.history)
    pipeline.close()

    report = {
        "delay_s": delay_s,
        "fault_scenario": "skip_step_2_then_correct",
        "n_frames": len(frames),
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "events": events,
        "n_events": len(events),
        "local_always_first": all(e["delay_s"] > 0 for e in events) if events else None,
        "voice_history": voice_history,
        "narrative": (
            f"Onboard alerted on every anomaly the instant it happened. If the same alert "
            f"had to reach a human on Earth and come back, the crew would have learned about "
            f"it {delay_s:.0f}s later — long enough, on a real timeline, for an irreversible "
            f"step to already be committed."
        ),
    }
    out_path = Path(out_dir) / "race_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Race report written to %s", out_path)
    return report


def print_report(report: Dict) -> None:
    print("\n" + "=" * 70)
    print(f"LATENCY RACE — Earth delay = {report['delay_s']:.0f}s "
         f"(fault: {report['fault_scenario']})")
    print("=" * 70)
    if not report["events"]:
        print("No anomaly events fired during this run.")
    for i, e in enumerate(report["events"], 1):
        msg = e["payload"].get("message", e["kind"])
        print(f"\n[{i}] {msg}")
        print(f"    LOCAL   alert fired at t=+{e['local_fire_time'] - report['events'][0]['local_fire_time']:.2f}s (immediate)")
        print(f"    EARTH   would have found out {e['delay_s']:.1f}s later "
             f"({'delivered' if e['delivered'] else 'PENDING'})")
    print("\n" + "-" * 70)
    print(report["narrative"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Onboard vs. delayed-Earth alert race demo")
    parser.add_argument("--delay", type=float, default=4.0, choices=[2.0, 4.0, 8.0],
                        help="Simulated Earth round-trip delay in seconds (2/4/8 per the pitch)")
    parser.add_argument("--out-dir", type=str, default="dataset/race_demo")
    parser.add_argument("--gui", action="store_true", help="Show the race live in the Qt dashboard")
    args = parser.parse_args()

    if args.gui:
        import queue
        import threading
        gui_queue = queue.Queue(maxsize=60)
        t = threading.Thread(target=run_race_demo, kwargs=dict(
            delay_s=args.delay, out_dir=args.out_dir, gui_queue=gui_queue), daemon=True)
        t.start()
        from gui.qt_dashboard import launch_qt_dashboard
        launch_qt_dashboard(gui_queue)
        t.join(timeout=1.0)
    else:
        report = run_race_demo(delay_s=args.delay, out_dir=args.out_dir)
        print_report(report)
