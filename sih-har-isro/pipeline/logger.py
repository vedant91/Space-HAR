"""
Structured Experiment Log Writer
==================================
Generates timestamped, human-readable + machine-parseable logs
of experiment steps, matching space mission data logging standards.

Log format:
    [HH:MM:SS.mmm] [STATUS]  Step N | {name} | confidence={val} | duration={val}s
"""

import os
import time
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from config.experiment_config import LOG_DIR, LOG_FILENAME_FORMAT, EXPERIMENT_STEPS


class ExperimentLogger:
    """
    Writes a structured, timestamped log file of experiment execution.
    Each log entry is both human-readable and parseable as JSON Lines.
    """

    def __init__(self, session_id: Optional[str] = None, log_dir: str = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # LOG_FILENAME_FORMAT has only second resolution, so two loggers built
        # in the same wall-clock second (e.g. simulation/space_sim.py building
        # one HARPipeline per clip) would otherwise collide on the same path.
        # A short per-instance suffix makes every session's filename unique.
        disambiguator = uuid.uuid4().hex[:6]
        self.session_id = session_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{disambiguator}"
        self.start_time = time.time()

        # Human-readable log
        ts_base = datetime.now().strftime(LOG_FILENAME_FORMAT)
        ts = f"{ts_base[:-4]}_{disambiguator}{ts_base[-4:]}"  # insert before ".txt"
        self.txt_path = self.log_dir / ts
        # Machine-readable JSONL log
        self.jsonl_path = self.log_dir / ts.replace(".txt", ".jsonl")

        self._init_log()
        logger.info("Experiment logger initialized: %s", self.txt_path)

    def _init_log(self):
        # Never clobber an existing file's header — the disambiguated filename
        # already makes a same-second collision astronomically unlikely, but a
        # residual collision (or a caller passing a fixed session_id) must not
        # silently truncate a prior session's log.
        if self.txt_path.exists():
            return
        header = (
            "=" * 70 + "\n"
            f"ISRO HAR EXPERIMENT LOG\n"
            f"Session ID : {self.session_id}\n"
            f"Start Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Total Steps: {len(EXPERIMENT_STEPS)}\n"
            "=" * 70 + "\n\n"
        )
        self.txt_path.write_text(header, encoding="utf-8")
        self.jsonl_path.touch()

    def _elapsed(self) -> str:
        e = time.time() - self.start_time
        h = int(e // 3600)
        m = int((e % 3600) // 60)
        s = e % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def _write(self, text_line: str, json_entry: dict):
        """Append to both log files."""
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(text_line + "\n")
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_entry) + "\n")

    def log_step_start(self, step_id: int, step_name: str, confidence: float):
        ts = self._elapsed()
        text = f"[{ts}] [START]     Step {step_id:02d} | {step_name:<30} | conf={confidence:.3f}"
        entry = {
            "event": "step_start", "timestamp": ts, "wall_time": datetime.now().isoformat(),
            "step_id": step_id, "step_name": step_name, "confidence": round(confidence, 3)
        }
        self._write(text, entry)
        logger.info("LOG: %s", text)

    def log_step_complete(self, step_id: int, step_name: str,
                          confidence: float, duration_sec: float,
                          recovered: bool = False):
        ts = self._elapsed()
        status = "RECOVERED" if recovered else "OK"
        text = (f"[{ts}] [COMPLETE]  Step {step_id:02d} | {step_name:<30} | "
                f"conf={confidence:.3f} | dur={duration_sec:.1f}s | STATUS={status}")
        entry = {
            "event": "step_complete", "timestamp": ts, "wall_time": datetime.now().isoformat(),
            "step_id": step_id, "step_name": step_name, "confidence": round(confidence, 3),
            "duration_sec": round(duration_sec, 2), "status": status, "recovered": recovered,
        }
        self._write(text, entry)
        logger.info("LOG: %s", text)

    def log_uncertain(self, step_id: int, lstm_step: int, cnn_step: int,
                      lstm_conf: float, cnn_conf: float):
        """LSTM/CNN ensemble disagreement — the pipeline chose to ask rather
        than guess, per the PS's own confidence principle."""
        ts = self._elapsed()
        text = (f"[{ts}] [UNCERTAIN] Expected Step {step_id:02d} | "
                f"lstm={lstm_step}({lstm_conf:.2f}) cnn={cnn_step}({cnn_conf:.2f}) | "
                f"STATUS=NEEDS_CONFIRMATION")
        entry = {
            "event": "uncertain", "timestamp": ts, "wall_time": datetime.now().isoformat(),
            "expected_step_id": step_id, "lstm_step": lstm_step, "lstm_confidence": round(lstm_conf, 3),
            "cnn_step": cnn_step, "cnn_confidence": round(cnn_conf, 3), "status": "NEEDS_CONFIRMATION",
        }
        self._write(text, entry)
        logger.warning("LOG: %s", text)

    def log_step_skipped(self, expected_step_id: int, expected_name: str,
                         observed_step_id: int):
        ts = self._elapsed()
        text = (f"[{ts}] [HOLD ⚠️ ]  Step {expected_step_id:02d} | {expected_name:<30} | "
                f"STATUS=RECOVERY_REQUIRED → observed_step={observed_step_id}")
        entry = {
            "event": "step_skipped", "timestamp": ts, "wall_time": datetime.now().isoformat(),
            "expected_step_id": expected_step_id, "expected_name": expected_name,
            "observed_step_id": observed_step_id, "status": "RECOVERY_REQUIRED"
        }
        self._write(text, entry)
        logger.warning("LOG: %s", text)

    def log_out_of_sequence(self, expected_step_id: int, observed_step_id: int):
        ts = self._elapsed()
        text = (f"[{ts}] [OOS  ⚠️ ]  Expected Step {expected_step_id:02d} | "
                f"Observed Step {observed_step_id:02d} | STATUS=OUT_OF_SEQUENCE")
        entry = {
            "event": "out_of_sequence", "timestamp": ts, "wall_time": datetime.now().isoformat(),
            "expected_step_id": expected_step_id, "observed_step_id": observed_step_id,
            "status": "OUT_OF_SEQUENCE"
        }
        self._write(text, entry)
        logger.warning("LOG: %s", text)

    def log_alert(self, alert_text: str):
        ts = self._elapsed()
        text = f"[{ts}] [ALERT]     {alert_text}"
        entry = {"event": "alert", "timestamp": ts, "wall_time": datetime.now().isoformat(),
                 "message": alert_text}
        self._write(text, entry)

    def log_experiment_start(self):
        ts = self._elapsed()
        text = f"[{ts}] [STARTED]   Experiment execution begun."
        entry = {"event": "experiment_start", "timestamp": ts,
                 "wall_time": datetime.now().isoformat()}
        self._write(text, entry)

    def log_experiment_complete(self, total_duration_sec: float,
                                steps_completed: int, steps_skipped: int,
                                recovery_events: int = 0):
        ts = self._elapsed()
        # Three-tier outcome: a run that needed a hold+correction mid-way is
        # real information the PS asks for ("outcomes/status") — it must not
        # be indistinguishable from a clean run just because every step was
        # eventually completed. steps_skipped stays for a genuinely abandoned
        # step (the state machine currently always holds for correction
        # instead, so this is 0 in practice today, but a future policy change
        # could set it — keep the field meaningful either way.)
        if steps_skipped > 0:
            outcome = "PARTIAL"
        elif recovery_events > 0:
            outcome = "RECOVERED"
        else:
            outcome = "SUCCESS"
        text = (
            "\n" + "=" * 70 + "\n"
            f"[{ts}] [COMPLETE]  EXPERIMENT FINISHED\n"
            f"           Total Duration : {total_duration_sec:.1f}s\n"
            f"           Steps Completed: {steps_completed}/{len(EXPERIMENT_STEPS)}\n"
            f"           Steps Skipped  : {steps_skipped}\n"
            f"           Recovery Events: {recovery_events}\n"
            f"           Outcome        : {outcome}\n"
            "=" * 70
        )
        entry = {
            "event": "experiment_complete", "timestamp": ts,
            "wall_time": datetime.now().isoformat(),
            "total_duration_sec": round(total_duration_sec, 2),
            "steps_completed": steps_completed,
            "steps_skipped": steps_skipped,
            "recovery_events": recovery_events,
            "outcome": outcome,
        }
        self._write(text, entry)
        logger.info("Experiment log saved: %s", self.txt_path)

    def get_log_path(self) -> str:
        return str(self.txt_path)
