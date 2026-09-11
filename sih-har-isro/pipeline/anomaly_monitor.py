"""
Typed Anomaly Monitor — A4 (forbidden zone), A5 (dwell/timeout), A7 (occlusion)
==================================================================================
Concrete implementations of three items from the research brief's anomaly
taxonomy (BAS_Onboard_Edge_Vision_LLM_Context.md section 7). A1 (wrong step)
and A2 (skipped step) already exist via pipeline/state_machine.py's own
skip/out-of-sequence detection; A8 (model disagreement) already exists via
har_pipeline.py's CNN/LSTM ensemble fusion. A3/A6/A9 are future work.

Design rule (unchanged from the rest of this project): **models/heuristics
propose, the FSM decides.** This monitor never advances or confirms a step —
it only ever asks ExperimentStateMachine to hold (force_hold/clear_hold),
same authority boundary as the FSM's own recovery-hold logic, just for
reasons outside the FSM's own sequence bookkeeping.

  A4 forbidden-zone HOLD — a hand/tool centroid enters a per-step rectangular
    zone (normalized [0,1] coords, from the active procedure pack's
    `forbidden_zones`). Edge-triggered: fires once on entry, clears once
    every tracked hand has left every active zone.

  A5 dwell/timeout — a step has been "current" for too long with no FSM
    progress. Soft prompt (voice/log only) at 70% of the step's `timeout_s`
    (from the pack); a hard HOLD at 100%. Per-step, so it resets cleanly
    when the FSM actually advances.

  A7 occlusion/abstain — N consecutive frames with zero HSV detections at
    all (no boxes, no hand) is treated as "the camera can't see the rack" —
    covered lens, rack out of frame, etc. — and holds rather than letting a
    stale/noisy prediction stream sneak a wrong step through.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ForbiddenZone:
    name: str
    rect: Tuple[float, float, float, float]  # (x1, y1, x2, y2), normalized [0,1]

    def contains(self, x_norm: float, y_norm: float) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x_norm <= x2 and y1 <= y_norm <= y2


class AnomalyMonitor:
    """
    One instance per pipeline run (reset() between trials — mirrors
    RackFrameNormalizer's own per-session reset convention).

    Usage per frame (see pipeline/har_pipeline.py):
        events = monitor.step(detections, current_step, step_status_name)
        for code, active, severity, message, extra in events:
            ... sync with state_machine.force_hold/clear_hold, log, voice, gui ...
    """

    OCCLUSION_STREAK_FRAMES = 10  # ~0.3-0.4s at 24-30fps

    def __init__(self, frame_width: int, frame_height: int):
        self.fw, self.fh = frame_width, frame_height
        self._occlusion_streak = 0
        self._occlusion_active = False
        self._zone_active = False
        self._step_timer_start: Dict[int, float] = {}
        self._step_soft_prompted: Set[int] = set()
        self._step_escalated: Set[int] = set()

    def reset(self) -> None:
        self._occlusion_streak = 0
        self._occlusion_active = False
        self._zone_active = False
        self._step_timer_start.clear()
        self._step_soft_prompted.clear()
        self._step_escalated.clear()

    def clear_step_timer(self, step_id: int) -> None:
        """Call when a step completes, so a repeated run (or the same pack
        reused across trials) starts that step's dwell timer fresh."""
        self._step_timer_start.pop(step_id, None)
        self._step_soft_prompted.discard(step_id)
        self._step_escalated.discard(step_id)

    # ── A4 — forbidden zone ──────────────────────────────────────────────

    def _check_forbidden_zone(self, detections: List, step: Optional[dict]
                              ) -> Optional[Tuple[str, bool, str, str, dict]]:
        zones_raw = (step or {}).get("forbidden_zones") or []
        zones = [ForbiddenZone(z["name"], tuple(z["rect"])) for z in zones_raw]
        if not zones:
            if self._zone_active:
                self._zone_active = False
                return ("A4", False, "cleared", "Forbidden zone clear", {})
            return None

        hand_points = [(d.centroid[0] / self.fw, d.centroid[1] / self.fh)
                       for d in detections if getattr(d, "label", None) == "hand"]
        hit = None
        for x, y in hand_points:
            for z in zones:
                if z.contains(x, y):
                    hit = z
                    break
            if hit:
                break

        if hit is not None and not self._zone_active:
            self._zone_active = True
            return ("A4", True, "hold", f"Hand entered forbidden zone '{hit.name}'",
                    {"zone": hit.name})
        if hit is None and self._zone_active:
            self._zone_active = False
            return ("A4", False, "cleared", "Hand left the forbidden zone", {})
        return None

    # ── A5 — dwell / timeout ─────────────────────────────────────────────

    def _check_dwell(self, step: Optional[dict], step_status: Optional[str]
                     ) -> Optional[Tuple[str, bool, str, str, dict]]:
        if step is None:
            return None
        step_id = step["id"]
        if step_status == "COMPLETED":
            self.clear_step_timer(step_id)
            return None

        now = time.perf_counter()
        started = self._step_timer_start.setdefault(step_id, now)
        elapsed = now - started
        timeout = float(step.get("timeout_s") or step.get("duration_hint_sec") or 30)

        if elapsed >= timeout and step_id not in self._step_escalated:
            self._step_escalated.add(step_id)
            return ("A5", True, "hold",
                    f"Step {step_id} exceeded its {timeout:.0f}s window with no progress",
                    {"step_id": step_id, "elapsed_s": round(elapsed, 1), "timeout_s": timeout})
        if elapsed >= 0.7 * timeout and step_id not in self._step_soft_prompted:
            self._step_soft_prompted.add(step_id)
            return ("A5", False, "soft",
                    f"Step {step_id} is at 70% of its {timeout:.0f}s window — still there?",
                    {"step_id": step_id, "elapsed_s": round(elapsed, 1), "timeout_s": timeout})
        return None

    def clear_dwell_hold(self, step_id: int) -> Optional[Tuple[str, bool, str, str, dict]]:
        """Call once the FSM actually advances past a step that had escalated
        to a hard A5 hold, so the hold releases."""
        if step_id in self._step_escalated:
            self.clear_step_timer(step_id)
            return ("A5", False, "cleared", f"Step {step_id} progressed — dwell hold cleared", {})
        return None

    # ── A7 — occlusion / abstain ─────────────────────────────────────────

    def _check_occlusion(self, detections: List) -> Optional[Tuple[str, bool, str, str, dict]]:
        if len(detections) == 0:
            self._occlusion_streak += 1
        else:
            self._occlusion_streak = 0

        if self._occlusion_streak >= self.OCCLUSION_STREAK_FRAMES and not self._occlusion_active:
            self._occlusion_active = True
            return ("A7", True, "abstain",
                    f"No detections for {self._occlusion_streak} consecutive frames "
                    f"— camera blocked or rack out of view?", {})
        if self._occlusion_streak == 0 and self._occlusion_active:
            self._occlusion_active = False
            return ("A7", False, "cleared", "Detections resumed — no longer occluded", {})
        return None

    # ── driver ────────────────────────────────────────────────────────────

    def step(self, detections: List, current_step: Optional[dict],
            current_step_status: Optional[str]) -> List[Tuple[str, bool, str, str, dict]]:
        """Run all three checks for one frame. Returns a list of
        (code, active, severity, message, extra) tuples — empty most frames."""
        events = []
        for check in (
            self._check_forbidden_zone(detections, current_step),
            self._check_dwell(current_step, current_step_status),
            self._check_occlusion(detections),
        ):
            if check is not None:
                events.append(check)
        return events
