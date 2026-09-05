"""
State Machine for Experiment Sequence Validation
=================================================
Tracks current experiment step, validates ordering,
detects skips, and triggers voice alerts.
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Callable, List
from dataclasses import dataclass, field

from config.experiment_config import EXPERIMENT_STEPS, STEP_CONFIRM_FRAMES, STEP_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    SKIPPED = auto()
    ERROR = auto()


@dataclass
class StepRecord:
    step_id: int
    name: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 0.0
    notes: str = ""
    recovered: bool = False

    @property
    def duration_sec(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None


class ExperimentStateMachine:
    """
    Finite State Machine that validates astronaut activity against
    the expected experiment protocol sequence.

    Emits events:
      - on_step_completed(step_record)
      - on_step_skipped(expected_id, observed_id)
      - on_out_of_sequence(expected_id, observed_id)
      - on_experiment_complete()
    """

    def __init__(self):
        self.steps = EXPERIMENT_STEPS
        self.total_steps = len(self.steps)

        # Step records for logging
        self.step_records: List[StepRecord] = [
            StepRecord(step_id=s["id"], name=s["name"])
            for s in self.steps
        ]

        # Current state
        self.current_step_idx: int = 0       # 0-indexed
        self.experiment_started: bool = False
        self.experiment_complete: bool = False
        self.experiment_start_time: Optional[float] = None

        # Confirmation buffer (debounce noisy predictions)
        self._confirm_buffer: List[int] = []
        self._confirm_threshold = STEP_CONFIRM_FRAMES
        self._last_oos_key = None  # rate-limit out-of-sequence alerts
        # A future action is never permission to silently advance a mission
        # protocol.  Hold at the missing action until it is actually observed.
        self.recovery_required: bool = False
        self.recovery_expected_id: Optional[int] = None
        self.recovery_observed_id: Optional[int] = None

        # Callbacks
        self.on_step_completed: Optional[Callable] = None
        self.on_step_skipped: Optional[Callable] = None
        self.on_out_of_sequence: Optional[Callable] = None
        self.on_step_recovered: Optional[Callable] = None
        self.on_experiment_complete: Optional[Callable] = None
        self.on_step_started: Optional[Callable] = None

        logger.info("State machine initialized with %d steps.", self.total_steps)

    @property
    def current_step(self) -> Optional[dict]:
        if self.current_step_idx < self.total_steps:
            return self.steps[self.current_step_idx]
        return None

    @property
    def next_step(self) -> Optional[dict]:
        nxt = self.current_step_idx + 1
        if nxt < self.total_steps:
            return self.steps[nxt]
        return None

    @property
    def expected_step_id(self) -> int:
        if self.current_step:
            return self.current_step["id"]
        return -1

    def feed_prediction(self, predicted_step_id: int, confidence: float):
        """
        Feed a raw model prediction (step id, confidence).
        Uses a confirmation buffer to debounce noisy predictions.
        """
        if self.experiment_complete:
            return

        if confidence < STEP_CONFIDENCE_THRESHOLD:
            return  # Ignore low-confidence predictions

        if predicted_step_id <= 0:
            return  # Idle/unknown

        # Start experiment on first confident prediction
        if not self.experiment_started:
            self.experiment_started = True
            self.experiment_start_time = time.time()
            logger.info("Experiment started.")

        # Confirmation buffer
        self._confirm_buffer.append(predicted_step_id)
        if len(self._confirm_buffer) > self._confirm_threshold:
            self._confirm_buffer.pop(0)

        # Only act when buffer is stable (majority vote)
        if len(self._confirm_buffer) >= self._confirm_threshold:
            majority = max(set(self._confirm_buffer), key=self._confirm_buffer.count)
            majority_count = self._confirm_buffer.count(majority)

            if majority_count >= int(self._confirm_threshold * 0.7):
                self._process_stable_prediction(majority, confidence)

    def _process_stable_prediction(self, stable_step_id: int, confidence: float):
        """Called when a prediction has been stable for N frames."""
        expected = self.expected_step_id

        if stable_step_id == expected:
            self._confirm_step(stable_step_id, confidence)

        elif stable_step_id > expected:
            # Do not mark a future step complete.  In a safety-critical
            # experiment a skip is an interruption that must be corrected,
            # not a shortcut through the protocol.
            key = (expected, stable_step_id)
            if key == self._last_oos_key:
                return
            self._last_oos_key = key
            self.recovery_required = True
            self.recovery_expected_id = expected
            self.recovery_observed_id = stable_step_id
            rec = self._get_record(expected)
            if rec:
                rec.notes = f"Recovery required: observed step {stable_step_id} before this step"
            logger.warning("SEQUENCE HOLD: Expected %d, got %d. Awaiting correction.",
                           expected, stable_step_id)
            if self.on_step_skipped:
                self.on_step_skipped(expected, stable_step_id)
            self._confirm_buffer.clear()

        elif stable_step_id < expected:
            # Regression: astronaut doing an already-completed step again.
            # The final pose of a just-completed action often persists for a
            # few frames while the astronaut moves to the next step.  Ignore
            # that immediate one-step tail; more distant regressions are still
            # safety events and are reported below.
            if stable_step_id == expected - 1:
                self._confirm_buffer.clear()
                return
            # Fire once per (expected, observed) pair — repeating it every
            # frame floods logs and adds I/O latency in the hot loop.
            key = (expected, stable_step_id)
            if key == self._last_oos_key:
                return
            self._last_oos_key = key
            logger.warning("OUT-OF-SEQUENCE: Expected %d, got already-done %d",
                           expected, stable_step_id)
            if self.on_out_of_sequence:
                self.on_out_of_sequence(expected, stable_step_id)

    def _confirm_step(self, step_id: int, confidence: float):
        """Mark a step as in-progress or completed."""
        # Find record
        rec = self._get_record(step_id)
        if rec is None:
            return

        if rec.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
            return  # Already processed

        now = time.time()

        if rec.status == StepStatus.PENDING:
            rec.status = StepStatus.IN_PROGRESS
            rec.start_time = now
            rec.confidence = confidence
            if self.on_step_started:
                self.on_step_started(rec)
            logger.info("Step %d '%s' IN PROGRESS (conf=%.2f)", step_id, rec.name, confidence)

        elif rec.status == StepStatus.IN_PROGRESS:
            # Transition to completed
            rec.status = StepStatus.COMPLETED
            rec.end_time = now
            rec.confidence = confidence

            was_recovery = self.recovery_required and self.recovery_expected_id == step_id
            if was_recovery:
                rec.recovered = True
                rec.notes = "Corrected after sequence hold"

            # Advance state machine pointer
            while (self.current_step_idx < self.total_steps and
                   self.step_records[self.current_step_idx].status in
                   (StepStatus.COMPLETED, StepStatus.SKIPPED)):
                self.current_step_idx += 1

            self._last_oos_key = None
            if was_recovery:
                observed = self.recovery_observed_id
                self.recovery_required = False
                self.recovery_expected_id = None
                self.recovery_observed_id = None
                if self.on_step_recovered:
                    self.on_step_recovered(rec, observed)
            if self.on_step_completed:
                self.on_step_completed(rec)

            logger.info("Step %d '%s' COMPLETED in %.1fs", step_id, rec.name, rec.duration_sec or 0)

            # Check experiment completion
            if self.current_step_idx >= self.total_steps:
                self.experiment_complete = True
                if self.on_experiment_complete:
                    self.on_experiment_complete()
                logger.info("🎉 Experiment COMPLETE!")

        # Clear buffer after confirmed transition
        self._confirm_buffer.clear()

    def _get_record(self, step_id: int) -> Optional[StepRecord]:
        for rec in self.step_records:
            if rec.step_id == step_id:
                return rec
        return None

    def get_status_summary(self) -> dict:
        """Return current state for GUI display."""
        return {
            "experiment_started": self.experiment_started,
            "experiment_complete": self.experiment_complete,
            "current_step": self.current_step,
            "next_step": self.next_step,
            "recovery": {
                "active": self.recovery_required,
                "expected_step_id": self.recovery_expected_id,
                "observed_step_id": self.recovery_observed_id,
            },
            "elapsed_sec": round(time.time() - self.experiment_start_time, 1)
                           if self.experiment_start_time else 0,
            "steps": [
                {
                    "id": r.step_id,
                    "name": r.name,
                    "status": r.status.name,
                    "duration": r.duration_sec,
                    "confidence": round(r.confidence, 2),
                    "notes": r.notes,
                    "recovered": r.recovered,
                }
                for r in self.step_records
            ],
        }

    def reset(self):
        """Reset machine for a new experiment run."""
        self.__init__()
        logger.info("State machine reset.")
