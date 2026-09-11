"""
Smart Clip Uplink — ring-buffer + anomaly-clip mode (Brief §12 bandwidth thesis)
=================================================================================
Full continuous IP streaming (pipeline/stream_sender.py) is the SIH-required
mode and is untouched by this module. This adds a *second*, additive mode:
instead of (or alongside) streaming every frame, only save/uplink short clips
around actual anomaly events — pre-roll (what led up to it) + post-roll (what
happened after). This is the concrete version of the brief's "don't ship 25
Mbps continuously, ship the 4 seconds that matter" argument.

Design:
  - `RingBuffer` holds the last `pre_roll_s` seconds of raw BGR frames,
    always, at near-zero cost (numpy array copies into a bounded deque).
  - `ClipUplinkManager.feed(frame)` is called every processed frame (mirrors
    `AnomalyMonitor.step()`'s "called every frame" convention). It (a) keeps
    the ring buffer current, and (b) if a clip capture is pending (post-roll
    in progress), appends this frame to it and flushes to disk once enough
    post-roll frames have accumulated.
  - `ClipUplinkManager.trigger(code, message)` starts a new clip: pre-roll
    frames come straight from the ring buffer (already have them), post-roll
    frames arrive via subsequent `feed()` calls. Idempotent while a clip for
    the same code is already pending, so a still-active anomaly (e.g. an A5
    hold that hasn't cleared yet) doesn't spawn a new clip every frame.
  - `bandwidth_summary()` gives the actual number for the thesis: bytes a
    continuous stream would have sent for this session vs. bytes the clips
    alone actually used.

This module never touches ExperimentStateMachine or feed_prediction — same
"models/heuristics propose, FSM decides" boundary as anomaly_monitor.py; it
only ever writes files and records metadata.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class ClipRecord:
    code: str
    message: str
    path: str
    frames: int
    duration_s: float
    size_bytes: int
    triggered_at: float


class ClipUplinkManager:
    """One instance per pipeline run. Call `feed()` every processed frame,
    `trigger()` on an anomaly event, `close()` at shutdown to flush any
    clip still mid-post-roll (better a short clip than a lost one)."""

    def __init__(self, frame_width: int, frame_height: int, fps: float,
                 pre_roll_s: float, post_roll_s: float, output_dir: str):
        self.fw, self.fh, self.fps = frame_width, frame_height, fps
        self.pre_roll_s, self.post_roll_s = pre_roll_s, post_roll_s
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ring_len = max(1, int(round(fps * pre_roll_s)))
        self._ring: Deque[np.ndarray] = deque(maxlen=ring_len)

        # Pending capture state (at most one clip records at a time — a
        # second trigger for a *different* code while one is pending just
        # extends nothing; it's dropped, same "don't flood" rule as the
        # anomaly monitor's own edge-triggering).
        self._pending_code: Optional[str] = None
        self._pending_message: str = ""
        self._pending_frames: List[np.ndarray] = []
        self._pending_post_roll_target: int = 0
        self._pending_triggered_at: float = 0.0

        self.clips: List[ClipRecord] = []
        self.frames_fed: int = 0           # for bandwidth_summary()'s "would-be-stream" side

    def reset(self) -> None:
        self._ring.clear()
        self._pending_code = None
        self._pending_frames = []
        self.clips.clear()
        self.frames_fed = 0

    # ── driver ────────────────────────────────────────────────────────────

    def feed(self, frame: np.ndarray) -> Optional[ClipRecord]:
        """Call once per processed frame, whether or not a clip is pending.
        Returns the finished ClipRecord the moment a pending clip flushes,
        else None."""
        self.frames_fed += 1
        self._ring.append(frame.copy())

        if self._pending_code is None:
            return None

        self._pending_frames.append(frame.copy())
        if len(self._pending_frames) >= self._pending_post_roll_target:
            return self._flush_pending()
        return None

    def trigger(self, code: str, message: str) -> None:
        """Start a new clip capture. No-op while one is already pending —
        the caller (har_pipeline._maybe_clip_trigger) is expected to call
        this on the same edge-triggered events anomaly_monitor already
        de-duplicates (entry, not every frame of an ongoing hold)."""
        if self._pending_code is not None:
            return
        self._pending_code = code
        self._pending_message = message
        self._pending_triggered_at = time.time()
        # Pre-roll comes straight from the ring buffer, taken now (a copy —
        # the ring keeps filling independently while post-roll accumulates).
        self._pending_frames = list(self._ring)
        self._pending_post_roll_target = len(self._pending_frames) + max(
            1, int(round(self.fps * self.post_roll_s)))

    def close(self) -> Optional[ClipRecord]:
        """Flush a still-pending clip at shutdown (short post-roll is better
        than losing the clip entirely)."""
        if self._pending_code is not None and self._pending_frames:
            return self._flush_pending()
        return None

    def _flush_pending(self) -> ClipRecord:
        code, message, frames = self._pending_code, self._pending_message, self._pending_frames
        triggered_at = self._pending_triggered_at
        self._pending_code = None
        self._pending_frames = []
        self._pending_post_roll_target = 0

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(triggered_at))
        path = self.output_dir / f"clip_{code}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.fw, self.fh))
        for f in frames:
            writer.write(f)
        writer.release()

        size_bytes = path.stat().st_size if path.exists() else 0
        rec = ClipRecord(
            code=code, message=message, path=str(path), frames=len(frames),
            duration_s=round(len(frames) / self.fps, 2), size_bytes=size_bytes,
            triggered_at=triggered_at,
        )
        self.clips.append(rec)
        return rec

    # ── bandwidth thesis ─────────────────────────────────────────────────

    def bandwidth_summary(self) -> Dict:
        """The actual number behind Brief §12: what a continuous stream
        would have sent for every frame this session actually saw, vs. what
        the clip files alone came to. Raw-frame estimate (not the mp4's own
        compression) for the "would-be-stream" side, since a continuous IP
        stream at this project's settings is effectively raw/lightly
        compressed video (see pipeline/stream_sender.py), not clip-grade
        H.264 with GOP reuse across an entire session."""
        raw_frame_bytes = self.fw * self.fh * 3
        stream_bytes_estimate = self.frames_fed * raw_frame_bytes
        clip_bytes_actual = sum(c.size_bytes for c in self.clips)
        savings_pct = (0.0 if stream_bytes_estimate == 0 else
                       round(100.0 * (1 - clip_bytes_actual / stream_bytes_estimate), 1))
        return {
            "frames_fed": self.frames_fed,
            "num_clips": len(self.clips),
            "stream_bytes_estimate": stream_bytes_estimate,
            "clip_bytes_actual": clip_bytes_actual,
            "savings_pct": savings_pct,
        }

    def summary(self) -> Dict:
        return {
            "clips": [
                {"code": c.code, "message": c.message, "path": c.path, "frames": c.frames,
                 "duration_s": c.duration_s, "size_bytes": c.size_bytes}
                for c in self.clips
            ],
            "bandwidth": self.bandwidth_summary(),
        }
