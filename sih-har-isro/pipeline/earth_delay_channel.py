"""
Earth Delay Channel — Latency-Race Demo
==========================================
Not a safety path. This module exists to make the core thesis ("Earth is
too slow to be the safety loop") visible and repeatable, not to add any
real capability: it takes a COPY of an alert the onboard system already
fired locally and replays an equivalent "Earth received it" event after a
configurable delay (2/4/8s — the "practical ops/video/teleop" range this
project's research brief settles on, not raw speed-of-light).

Design guarantee: this channel is fed *after* the real local alert has
already fired (see pipeline/har_pipeline.py's _bind_callbacks) and its
output only ever reaches logging/GUI display — it is never read back into
ExperimentStateMachine or any other decision path. Delaying "Earth's
copy" cannot change what the onboard system does, by construction.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

DEFAULT_EARTH_DELAY_S = 4.0        # research brief's "practical ops" midpoint
ALLOWED_DELAYS_S = (2.0, 4.0, 8.0)  # the three the pitch narrative uses


@dataclass
class RaceEvent:
    kind: str                    # "step_skipped" | "out_of_sequence" | "uncertain" | "forbidden_zone" | ...
    payload: dict
    local_fire_time: float       # time.perf_counter() when the local alert fired
    earth_deliver_time: float    # local_fire_time + delay_s
    delivered: bool = False
    delivered_at: Optional[float] = None


class EarthDelayChannel:
    """
    fire(kind, payload) records "the local system just alerted on this" and
    schedules an equivalent "Earth just found out" event `delay_s` later.
    Purely additive/observational — see module docstring.
    """

    def __init__(self, delay_s: float = DEFAULT_EARTH_DELAY_S,
                 on_deliver: Optional[Callable[[RaceEvent], None]] = None):
        self.delay_s = float(delay_s)
        self.on_deliver = on_deliver
        self.events: List[RaceEvent] = []
        self._lock = threading.Lock()
        self._timers: List[threading.Timer] = []

    def set_delay(self, delay_s: float) -> None:
        self.delay_s = float(delay_s)

    def fire(self, kind: str, payload: dict) -> RaceEvent:
        now = time.perf_counter()
        ev = RaceEvent(kind=kind, payload=dict(payload),
                       local_fire_time=now, earth_deliver_time=now + self.delay_s)
        with self._lock:
            self.events.append(ev)
        timer = threading.Timer(self.delay_s, self._deliver, args=(ev,))
        timer.daemon = True
        timer.start()
        with self._lock:
            self._timers.append(timer)
        return ev

    def _deliver(self, ev: RaceEvent) -> None:
        ev.delivered = True
        ev.delivered_at = time.perf_counter()
        if self.on_deliver:
            try:
                self.on_deliver(ev)
            except Exception:
                pass  # display-only channel — must never affect the real pipeline

    def flush(self, timeout_s: float = 0.0) -> None:
        """Block until every scheduled delivery has fired (or timeout). Used
        by the offline race-demo script so it can print a complete report."""
        deadline = time.perf_counter() + max(self.delay_s + 0.5, timeout_s)
        while time.perf_counter() < deadline:
            with self._lock:
                pending = [e for e in self.events if not e.delivered]
            if not pending:
                return
            time.sleep(0.05)

    def summary(self) -> List[Dict]:
        with self._lock:
            events = list(self.events)
        return [{
            "kind": e.kind,
            "payload": e.payload,
            "local_fire_time": e.local_fire_time,
            "earth_deliver_time": e.earth_deliver_time,
            "delay_s": round(e.earth_deliver_time - e.local_fire_time, 3),
            "delivered": e.delivered,
        } for e in events]
