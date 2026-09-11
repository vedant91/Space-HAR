"""
Session Metrics — time-to-detect, false-hold rate, abstain quality (Brief §13)
================================================================================
Wave 2's last build-plan item (G5): concrete numbers for judges/paper tables,
built entirely from the typed-anomaly vocabulary (A4/A5/A7 — hold/abstain/
cleared) this project already established in pipeline/anomaly_monitor.py.
Nothing here changes pipeline behavior; it only observes the same events
already flowing through har_pipeline.py's `_sync_anomaly_event`.

  time_to_detect_ms — how long each anomaly type took to actually notice
    something was wrong, taken from the `detect_latency_ms` AnomalyMonitor
    now attaches to every non-cleared event (A4: ~one frame period, edge-
    triggered; A5: the full elapsed time since the step started, since a
    dwell hold's "detection" *is* watching the clock; A7: the fixed
    OCCLUSION_STREAK_FRAMES window it deliberately waits before deciding
    the camera is actually blocked, not just a noisy frame).

  false_hold_rate — a HEURISTIC PROXY, stated honestly: a hold that clears
    within FALSE_HOLD_BLIP_THRESHOLD_S is counted as a likely detector blip
    rather than a real anomaly the crew had to work through. There is no
    onboard ground-truth "was this hold actually correct" signal, so this is
    the best available proxy, not a verified accuracy number.

  abstain_count — how many times the system refused to guess (A7 occlusion)
    rather than let a stale prediction sneak a wrong step through, per the
    project's own confidence-gated design rule.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

FALSE_HOLD_BLIP_THRESHOLD_S = 1.5


@dataclass
class HoldSpan:
    code: str
    severity: str
    started_at: float
    detect_latency_ms: Optional[float] = None
    ended_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round(self.ended_at - self.started_at, 3)


class SessionMetrics:
    """One instance per pipeline run. Feed it every anomaly event via
    `record_event()` (same (code, active, severity, extra) shape
    AnomalyMonitor.step() / _sync_anomaly_event already use); read a live
    snapshot via `live_summary()` or the final report via `finalize()`/`write()`."""

    def __init__(self):
        self.session_start = time.time()
        self._open: Dict[str, HoldSpan] = {}
        self.closed: List[HoldSpan] = []
        self.detect_latencies_ms: List[float] = []
        self.abstain_count: int = 0

    def reset(self) -> None:
        self.session_start = time.time()
        self._open.clear()
        self.closed.clear()
        self.detect_latencies_ms.clear()
        self.abstain_count = 0

    def record_event(self, code: str, active: bool, severity: str, extra: dict) -> None:
        now = time.time()
        if severity == "cleared":
            span = self._open.pop(code, None)
            if span is not None:
                span.ended_at = now
                self.closed.append(span)
            return

        lat = extra.get("detect_latency_ms")
        if lat is not None:
            self.detect_latencies_ms.append(float(lat))

        if severity == "soft":
            # A soft prompt never holds the FSM (see AnomalyMonitor/
            # force_hold's docstrings) — it contributes a detect-latency
            # sample but isn't a hold span.
            return

        if severity == "abstain":
            self.abstain_count += 1
        if code not in self._open:
            self._open[code] = HoldSpan(code=code, severity=severity, started_at=now,
                                        detect_latency_ms=lat)

    def live_summary(self) -> Dict:
        """Cheap running snapshot for the GUI — doesn't close open spans."""
        return {
            "open_holds": len(self._open),
            "closed_holds": len(self.closed),
            "abstain_count": self.abstain_count,
            "avg_detect_ms": (round(sum(self.detect_latencies_ms) / len(self.detect_latencies_ms), 1)
                             if self.detect_latencies_ms else None),
        }

    def finalize(self, session_duration_s: Optional[float] = None) -> Dict:
        # A hold still open at session end (e.g. the run was stopped mid-
        # hold) still counts — closed at "now" rather than silently dropped.
        now = time.time()
        for span in list(self._open.values()):
            span.ended_at = now
            self.closed.append(span)
        self._open.clear()

        total_holds = len(self.closed)
        false_holds = sum(1 for s in self.closed
                          if s.duration_s is not None and s.duration_s < FALSE_HOLD_BLIP_THRESHOLD_S)
        false_hold_rate = round(false_holds / total_holds, 3) if total_holds else 0.0
        avg_detect_ms = (round(sum(self.detect_latencies_ms) / len(self.detect_latencies_ms), 1)
                         if self.detect_latencies_ms else None)

        return {
            "session_duration_s": round(session_duration_s if session_duration_s is not None
                                        else (now - self.session_start), 1),
            "total_holds": total_holds,
            "false_hold_rate": false_hold_rate,
            "false_hold_note": (
                f"Heuristic proxy: a hold that clears within {FALSE_HOLD_BLIP_THRESHOLD_S}s is "
                "counted as a likely detector blip, not a real anomaly. No onboard ground-truth "
                "'was this hold actually correct' signal exists — this is not a verified "
                "accuracy number."),
            "abstain_count": self.abstain_count,
            "time_to_detect_ms": {
                "avg": avg_detect_ms,
                "samples": [round(x, 1) for x in self.detect_latencies_ms],
            },
            "holds": [
                {"code": s.code, "severity": s.severity, "duration_s": s.duration_s,
                 "detect_latency_ms": s.detect_latency_ms}
                for s in self.closed
            ],
        }

    def write(self, path: str, session_duration_s: Optional[float] = None) -> Dict:
        report = self.finalize(session_duration_s)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, indent=2))
        return report
