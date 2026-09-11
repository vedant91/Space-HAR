"""
Process Watchdog — flight-honesty heartbeat + degraded-subsystem reporting
============================================================================
Brief §6 rule 6: never let a subsystem go silently dark — a degraded or dead
part of the system must say so loudly (log + voice + GUI), not be discovered
by the crew during an emergency. `_load_models()` already soft-degrades and
logs once at startup when a model file is missing; this extends that to a
live per-session watchdog that keeps checking:

  - **periodic heartbeat** — a "still alive and monitoring" log entry every
    `HEARTBEAT_INTERVAL_S`, so a mission review can tell "the system genuinely
    ran the whole session" from "it silently died and nobody noticed."
  - **stalled camera** — `frame_idx` hasn't advanced in `CAMERA_STALL_S`: the
    capture loop itself is stuck (source unplugged, driver hang). Distinct
    from A7 occlusion (camera is live, frames keep arriving, they're just
    empty of detections) — this is "no new frames at all."
  - **dead background threads** — the voice-alert thread, the network
    streamer.
  - **model availability snapshot** — attached to every heartbeat, so a
    session's log can be checked after the fact for "was PoseNet/LSTM/CNN
    actually loaded the whole time."

Same authority boundary as anomaly_monitor.py: this NEVER touches
ExperimentStateMachine. It's an honesty/observability layer, not a new hold
reason — a genuinely stalled camera already stops new predictions on its own
(no new frames -> no new skeleton_buffer entries -> no new feed_prediction
calls); the watchdog's job is making sure that fact is loud, not silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HealthEvent:
    kind: str            # "heartbeat" | "camera_stalled" | "camera_resumed" |
                          # "voice_thread_dead" | "voice_thread_resumed" |
                          # "streamer_dead" | "streamer_resumed"
    healthy: bool
    message: str
    extra: dict = field(default_factory=dict)


class ProcessWatchdog:
    """One instance per pipeline run. Call `tick()` every frame; it rate-
    limits its own heartbeat/stall checks internally, so calling it often is
    cheap (a handful of comparisons, no I/O)."""

    HEARTBEAT_INTERVAL_S = 30.0
    CAMERA_STALL_S = 5.0

    def __init__(self):
        self._last_heartbeat = time.time()
        self._last_frame_idx_seen = -1
        self._last_frame_change_at = time.time()
        self._camera_stalled = False
        self._voice_thread_was_alive = True
        self._streamer_was_alive: Optional[bool] = None  # None = streaming not enabled

    def reset(self) -> None:
        self.__init__()

    def tick(self, frame_idx: int, voice_thread_alive: bool,
            streamer_alive: Optional[bool], model_status: Dict[str, str]) -> List[HealthEvent]:
        events: List[HealthEvent] = []
        now = time.time()

        # ── camera / capture-loop stall ──────────────────────────────────
        if frame_idx != self._last_frame_idx_seen:
            self._last_frame_idx_seen = frame_idx
            self._last_frame_change_at = now
            if self._camera_stalled:
                self._camera_stalled = False
                events.append(HealthEvent("camera_resumed", True,
                    "Camera/video source resumed producing frames"))
        elif not self._camera_stalled and (now - self._last_frame_change_at) >= self.CAMERA_STALL_S:
            self._camera_stalled = True
            events.append(HealthEvent("camera_stalled", False,
                f"No new frames for {self.CAMERA_STALL_S:.0f}s — capture source "
                f"may be stuck or disconnected", {"frame_idx": frame_idx}))

        # ── voice-alert background thread ────────────────────────────────
        if self._voice_thread_was_alive and not voice_thread_alive:
            self._voice_thread_was_alive = False
            events.append(HealthEvent("voice_thread_dead", False,
                "Voice alert thread is no longer running — alerts will only "
                "reach the structured log and GUI, not audio"))
        elif not self._voice_thread_was_alive and voice_thread_alive:
            self._voice_thread_was_alive = True
            events.append(HealthEvent("voice_thread_resumed", True,
                "Voice alert thread is running again"))

        # ── network streamer (None = streaming not enabled this run) ────
        if streamer_alive is not None:
            if self._streamer_was_alive is not False and not streamer_alive:
                self._streamer_was_alive = False
                events.append(HealthEvent("streamer_dead", False,
                    "Network stream died mid-session — local recording "
                    "(if enabled) is unaffected"))
            elif self._streamer_was_alive is False and streamer_alive:
                self._streamer_was_alive = True
                events.append(HealthEvent("streamer_resumed", True,
                    "Network stream is back up"))
            elif self._streamer_was_alive is None:
                self._streamer_was_alive = streamer_alive

        # ── periodic heartbeat (always healthy=True; a dead heartbeat IS the
        # absence of a log line, not an event we can emit from inside a dead
        # process — the honesty this buys is "prove you were alive," not
        # "detect your own death") ──
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL_S:
            self._last_heartbeat = now
            events.append(HealthEvent("heartbeat", True, "System alive and monitoring",
                                      {"frame_idx": frame_idx, "models": dict(model_status)}))

        return events
