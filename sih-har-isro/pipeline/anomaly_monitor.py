"""
Typed Anomaly Monitor — A4 (forbidden zone), A5 (dwell/timeout), A6 (object
drift), A7 (occlusion)
==================================================================================
Concrete implementations of items from the research brief's anomaly
taxonomy (BAS_Onboard_Edge_Vision_LLM_Context.md section 7). A1 (wrong step)
and A2 (skipped step) already exist via pipeline/state_machine.py's own
skip/out-of-sequence detection; A8 (model disagreement) already exists via
har_pipeline.py's CNN/LSTM ensemble fusion. A3/A9 are future work.

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

  A6 object-drift HOLD — a tracked box (red/yellow/main) is moving fast
    with no hand near it. Boxes don't come to rest on their own in
    microgravity, so fast + untethered means it's drifting away rather than
    being deliberately handled. Edge-triggered per box, so one drifting box
    doesn't mask another; the hold code is per-label ("A6:red_box") so two
    boxes drifting at once clear independently.

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

    # A6 tuning: normalized (0..1 of frame width/height) units.
    DRIFT_SPEED_THRESHOLD = 0.35   # frame-widths/sec to count as "moving fast"
    DRIFT_HAND_PROXIMITY = 0.12    # a hand this close counts as "being handled"
    DRIFT_TRACKED_LABELS = ("red_box", "yellow_box", "main_box")
    DRIFT_MAX_DT = 1.0             # gap this large means the box just reappeared, not teleported
    # A single frame-to-frame HSV centroid delta is noisy (lighting/detector
    # jitter alone clears DRIFT_SPEED_THRESHOLD constantly — measured 707
    # spurious A6 holds in one 94s real clip before this existed). Same fix
    # as A7's OCCLUSION_STREAK_FRAMES: require sustained fast+untethered
    # motion before firing, clear on the first good frame.
    DRIFT_STREAK_FRAMES = 5        # ~0.15-0.2s at 24-30fps

    def __init__(self, frame_width: int, frame_height: int, fps: float = 30.0):
        self.fw, self.fh, self.fps = frame_width, frame_height, fps
        self._occlusion_streak = 0
        self._occlusion_active = False
        self._zone_active = False
        self._step_timer_start: Dict[int, float] = {}
        self._step_soft_prompted: Set[int] = set()
        self._step_escalated: Set[int] = set()
        self._drift_last: Dict[str, Tuple[float, float, float]] = {}  # label -> (x_norm, y_norm, t)
        self._drift_active: Set[str] = set()
        self._drift_streak: Dict[str, int] = {}

    def reset(self) -> None:
        self._occlusion_streak = 0
        self._occlusion_active = False
        self._zone_active = False
        self._step_timer_start.clear()
        self._step_soft_prompted.clear()
        self._step_escalated.clear()
        self._drift_streak.clear()
        self._drift_last.clear()
        self._drift_active.clear()

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
            # Edge-triggered on the same frame the hand entered — detection
            # is bounded only by one frame period, not a real "how long did
            # it take to notice" delay like A5/A7's below.
            return ("A4", True, "hold", f"Hand entered forbidden zone '{hit.name}'",
                    {"zone": hit.name, "detect_latency_ms": round(1000.0 / self.fps, 1)})
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
            # Here "detection" genuinely is "how long we watched the clock" —
            # the hold IS the elapsed dwell time, not a fixed processing cost.
            return ("A5", True, "hold",
                    f"Step {step_id} exceeded its {timeout:.0f}s window with no progress",
                    {"step_id": step_id, "elapsed_s": round(elapsed, 1), "timeout_s": timeout,
                     "detect_latency_ms": round(elapsed * 1000.0, 1)})
        if elapsed >= 0.7 * timeout and step_id not in self._step_soft_prompted:
            self._step_soft_prompted.add(step_id)
            return ("A5", False, "soft",
                    f"Step {step_id} is at 70% of its {timeout:.0f}s window — still there?",
                    {"step_id": step_id, "elapsed_s": round(elapsed, 1), "timeout_s": timeout,
                     "detect_latency_ms": round(elapsed * 1000.0, 1)})
        return None

    def clear_dwell_hold(self, step_id: int) -> Optional[Tuple[str, bool, str, str, dict]]:
        """Call once the FSM actually advances past a step that had escalated
        to a hard A5 hold, so the hold releases."""
        if step_id in self._step_escalated:
            self.clear_step_timer(step_id)
            return ("A5", False, "cleared", f"Step {step_id} progressed — dwell hold cleared", {})
        return None

    # ── A6 — untethered object drift ─────────────────────────────────────

    def _check_drift(self, detections: List) -> List[Tuple[str, bool, str, str, dict]]:
        """A tracked box moving fast with nobody's hand near it. In
        microgravity a released object doesn't settle by itself, so this is
        the closest thing to "it's floating away" this pipeline can see from
        one fixed camera. Per-label state (not one shared flag like A4/A5/A7)
        because more than one box can be drifting at once and they need to
        clear independently — the returned code is "A6:<label>", not "A6"."""
        now = time.perf_counter()
        events: List[Tuple[str, bool, str, str, dict]] = []
        seen: Set[str] = set()

        hand_points = [(d.centroid[0] / self.fw, d.centroid[1] / self.fh)
                       for d in detections if getattr(d, "label", None) == "hand"]

        for d in detections:
            label = getattr(d, "label", None)
            if label not in self.DRIFT_TRACKED_LABELS:
                continue
            seen.add(label)
            x, y = d.centroid[0] / self.fw, d.centroid[1] / self.fh
            prev = self._drift_last.get(label)
            self._drift_last[label] = (x, y, now)
            if prev is None:
                continue
            px, py, pt = prev
            dt = now - pt
            if dt <= 0 or dt > self.DRIFT_MAX_DT:
                continue  # box just reappeared after a gap — not a real speed sample

            speed = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 / dt
            hand_near = any(((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 <= self.DRIFT_HAND_PROXIMITY
                            for hx, hy in hand_points)
            drifting_now = speed >= self.DRIFT_SPEED_THRESHOLD and not hand_near
            self._drift_streak[label] = self._drift_streak.get(label, 0) + 1 if drifting_now else 0

            code = f"A6:{label}"
            if (self._drift_streak[label] >= self.DRIFT_STREAK_FRAMES
                    and label not in self._drift_active):
                self._drift_active.add(label)
                events.append((code, True, "hold",
                              f"{label.replace('_', ' ')} is drifting untethered "
                              f"({speed:.2f} frame-widths/s, no hand nearby)",
                              {"label": label, "speed": round(speed, 3)}))
            elif not drifting_now and label in self._drift_active:
                self._drift_active.discard(label)
                events.append((code, False, "cleared",
                              f"{label.replace('_', ' ')} drift stopped", {"label": label}))

        # A box no longer detected at all can't keep reporting a speed —
        # release its hold so it doesn't get stuck (A7 owns "can't see it").
        for label in list(self._drift_active):
            if label not in seen:
                self._drift_active.discard(label)
                self._drift_last.pop(label, None)
                self._drift_streak.pop(label, None)
                events.append((f"A6:{label}", False, "cleared",
                              f"{label.replace('_', ' ')} out of view — drift hold released",
                              {"label": label}))
        return events

    # ── A7 — occlusion / abstain ─────────────────────────────────────────

    def _check_occlusion(self, detections: List) -> Optional[Tuple[str, bool, str, str, dict]]:
        if len(detections) == 0:
            self._occlusion_streak += 1
        else:
            self._occlusion_streak = 0

        if self._occlusion_streak >= self.OCCLUSION_STREAK_FRAMES and not self._occlusion_active:
            self._occlusion_active = True
            # The deliberate wait for OCCLUSION_STREAK_FRAMES consecutive
            # empty frames (so one noisy frame doesn't false-trigger) IS the
            # detection cost here — report the real window, not a rounding.
            detect_latency_ms = round(self.OCCLUSION_STREAK_FRAMES / self.fps * 1000.0, 1)
            return ("A7", True, "abstain",
                    f"No detections for {self._occlusion_streak} consecutive frames "
                    f"— camera blocked or rack out of view?",
                    {"detect_latency_ms": detect_latency_ms})
        if self._occlusion_streak == 0 and self._occlusion_active:
            self._occlusion_active = False
            return ("A7", False, "cleared", "Detections resumed — no longer occluded", {})
        return None

    # ── driver ────────────────────────────────────────────────────────────

    def step(self, detections: List, current_step: Optional[dict],
            current_step_status: Optional[str]) -> List[Tuple[str, bool, str, str, dict]]:
        """Run all four checks for one frame. Returns a list of
        (code, active, severity, message, extra) tuples — empty most frames."""
        events = []
        for check in (
            self._check_forbidden_zone(detections, current_step),
            self._check_dwell(current_step, current_step_status),
            self._check_occlusion(detections),
        ):
            if check is not None:
                events.append(check)
        events.extend(self._check_drift(detections))  # can fire >1 event/frame (multi-box)
        return events


def _selftest() -> None:
    """Smallest runnable check for A6 (object drift) — the other three
    checks (A4/A5/A7) already had integration coverage through a real
    HARPipeline per the project history; this covers the new one.

    Also regression-covers the debounce: a real run against
    recordings/experiment_20260830_223615.mp4 fired A6 707 times (vs. 1
    real A5 hold) before DRIFT_STREAK_FRAMES existed — pure HSV-centroid
    jitter, one frame at a time, cleared threshold. Test 1 below asserts
    no event fires before the streak is reached, not just that one
    eventually does."""
    import time as _time
    from types import SimpleNamespace as _NS

    def det(label, x, y):
        return _NS(label=label, centroid=(x, y))

    fw = fh = 100  # 1 normalized unit == 100px, keeps the math readable
    N = AnomalyMonitor.DRIFT_STREAK_FRAMES
    dt = 0.02  # 10px/dt = 5 units/s, comfortably over DRIFT_SPEED_THRESHOLD

    # 1) Fast + untethered, sustained for the full streak -> fires (not
    #    before), then clears the instant it stops moving.
    m = AnomalyMonitor(fw, fh, fps=30.0)
    x = 0
    m._check_drift([det("red_box", x, 50)])  # prime — no prior sample yet
    fired: list = []
    for i in range(N):
        x += 10
        _time.sleep(dt)
        fired = m._check_drift([det("red_box", x, 50)])
        if i < N - 1:
            assert fired == [], f"fired early at streak={i + 1}/{N}: {fired}"
    assert len(fired) == 1 and fired[0][:3] == ("A6:red_box", True, "hold"), fired

    _time.sleep(dt)
    cleared = m._check_drift([det("red_box", x, 50)])  # stayed put
    assert len(cleared) == 1 and cleared[0][:3] == ("A6:red_box", False, "cleared"), cleared

    # 2) Same fast motion, but a hand is right there the whole time -> never
    #    fires (being handled, not drifting away) — streak never builds.
    m2 = AnomalyMonitor(fw, fh, fps=30.0)
    x = 0
    m2._check_drift([det("yellow_box", x, 50), det("hand", x + 2, 50)])
    all_events: list = []
    for _ in range(N + 2):
        x += 10
        _time.sleep(dt)
        all_events += m2._check_drift([det("yellow_box", x, 50), det("hand", x + 2, 50)])
    assert all_events == [], all_events

    # 3) Two boxes drifting at once clear INDEPENDENTLY (distinct codes).
    m3 = AnomalyMonitor(fw, fh, fps=30.0)
    rx = ry = 0
    m3._check_drift([det("red_box", rx, 0), det("yellow_box", ry, 0)])
    last: list = []
    for _ in range(N):
        rx += 10
        ry += 10
        _time.sleep(dt)
        last = m3._check_drift([det("red_box", rx, 0), det("yellow_box", ry, 0)])
    assert {e[0] for e in last} == {"A6:red_box", "A6:yellow_box"}, last

    _time.sleep(dt)  # red stops, yellow keeps drifting
    only_red_stops = m3._check_drift([det("red_box", rx, 0), det("yellow_box", ry + 10, 0)])
    assert [e[0] for e in only_red_stops] == ["A6:red_box"], only_red_stops
    assert m3._drift_active == {"yellow_box"}, m3._drift_active

    print("anomaly_monitor A6 selftest: OK")


if __name__ == "__main__":
    _selftest()
