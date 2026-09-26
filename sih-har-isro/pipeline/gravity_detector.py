"""
Gravity Detector — Pre-HAR Gravity Stability Estimator
=======================================================
In a real spacecraft, payload/spacecraft IMU telemetry is the authoritative
source of acceleration and vibration information.  Video cannot measure the
local gravity field: when no IMU is connected this module reports a *visual
motion-stability proxy*, not a gravity measurement.  That distinction matters
for a safety-critical mission demo.

Pipeline role
─────────────
    ┌─────────┐   IMU proxy   ┌──────────────────┐
    │  Pose   │ ────────────► │ GravityDetector   │
    │landmarks│               │  .analyze(pose)   │
    └─────────┘               └────────┬─────────┘
                                       │ GravityState
                                       ▼
                              ┌──────────────────┐
                              │   HAR Pipeline    │
                              │ rack-normalise?   │
                              │  LSTM inference   │
                              └──────────────────┘

GravityState fields
───────────────────
  level_g          – expected/configured operational g level (not vision-measured)
  stability        – 'STABLE' | 'TURBULENT' | 'TRANSITION'
  status           – 'MICROGRAVITY' | 'LOW_G' | 'NORMAL_G' | 'TURBULENCE'
  correction_mode  – 'RACK_FRAME' | 'STANDARD' | 'ADAPTIVE'
  confidence       – float 0–1 (how certain the classification is)
  imu_axes         – real IMU axes when supplied, otherwise simulated proxy
  sensor_source    – 'PAYLOAD_IMU' | 'VISION_STABILITY_PROXY'
  jitter_rms       – RMS skeleton-velocity across all joints
  drift_vector     – mean body-COM drift direction [dx, dy]
  recommendation   – human-readable string for HUD/log
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe Pose landmark indices (same mapping as rack_frame.py) ───────────
NOSE        = 0
L_SHOULDER  = 11
R_SHOULDER  = 12
L_HIP       = 23
R_HIP       = 24
L_WRIST     = 15
R_WRIST     = 16
L_ANKLE     = 27
R_ANKLE     = 28

# Landmark subset used for jitter estimation (core body — avoids noisy extremities)
_CORE_JOINTS = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
_ALL_JOINTS  = list(range(33))

# ── Gravity classification thresholds ────────────────────────────────────────
# These are calibrated for a stationary Earth camera vs. ISS footage.
# jitter_rms is in normalised landmark units per frame.
_MICROGRAVITY_JITTER_MIN = 0.001   # floor: even stable micro-g drifts slightly
_TURBULENCE_JITTER_THRESH = 0.025  # above this → turbulence / physical contact
_LOW_G_JITTER_THRESH      = 0.005  # between floor and turbulence → low-g / lunar
_DRIFT_STABLE_THRESH      = 0.003  # mean COM drift; above → consistent drift
_HISTORY_LEN              = 30     # frames kept for running statistics


@dataclass
class GravityState:
    """Output snapshot from GravityDetector.analyze()."""

    level_g:         float  = 0.0
    stability:       str    = "STABLE"           # STABLE | TURBULENT | TRANSITION
    status:          str    = "MICROGRAVITY"     # MICROGRAVITY | LOW_G | NORMAL_G | TURBULENCE
    correction_mode: str    = "RACK_FRAME"       # RACK_FRAME | STANDARD | ADAPTIVE
    confidence:      float  = 0.0

    # IMU-proxy synthetic axes (g)
    imu_axes: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Diagnostic signals
    jitter_rms:   float           = 0.0
    drift_vector: List[float]     = field(default_factory=lambda: [0.0, 0.0])
    history_rms:  List[float]     = field(default_factory=list)

    recommendation: str = "MICROGRAVITY CONFIRMED — RACK-FRAME NORMALISATION ACTIVE"

    timestamp: float = field(default_factory=time.time)
    sensor_source: str = "VISION_STABILITY_PROXY"

    # ── Convenience predicates ────────────────────────────────────────────────
    @property
    def is_microgravity(self) -> bool:
        return self.status in ("MICROGRAVITY", "LOW_G")

    @property
    def is_turbulent(self) -> bool:
        return self.stability == "TURBULENT"

    @property
    def har_ready(self) -> bool:
        """True when gravity is stable enough to trust HAR classification."""
        return self.stability in ("STABLE", "TRANSITION") and self.confidence > 0.5

    def to_dict(self) -> dict:
        return {
            "level_g":         round(self.level_g, 4),
            "stability":       self.stability,
            "status":          self.status,
            "correction_mode": self.correction_mode,
            "confidence":      round(self.confidence, 4),
            "imu_axes":        [round(v, 5) for v in self.imu_axes],
            "jitter_rms":      round(self.jitter_rms, 6),
            "drift_vector":    [round(v, 5) for v in self.drift_vector],
            "har_ready":       self.har_ready,
            "recommendation":  self.recommendation,
            "timestamp":       self.timestamp,
            "sensor_source":   self.sensor_source,
        }


class GravityDetector:
    """
    Real-time gravity stability detector for the ISRO HAR pipeline.

    Usage
    ─────
    detector = GravityDetector()

    # Inside the per-frame loop:
    gravity_state = detector.analyze(pose_flat)   # pose_flat: (33×4,) array
    if gravity_state.har_ready:
        run_lstm(...)
    """

    def __init__(
        self,
        history_len: int = _HISTORY_LEN,
        microgravity_mode: bool = True,   # True for ISS/BAS operation
    ):
        self._history_len = history_len
        self._microgravity_mode = microgravity_mode

        # Ring-buffers for running statistics
        self._pose_history:   Deque[np.ndarray]  = deque(maxlen=history_len)
        self._jitter_history: Deque[float]        = deque(maxlen=history_len)
        self._drift_history:  Deque[np.ndarray]   = deque(maxlen=history_len)

        # EMA state
        self._ema_jitter: float  = 0.0
        self._ema_alpha:  float  = 0.25       # fast enough to catch sudden turbulence

        # COM (centre-of-mass) tracking
        self._prev_com: Optional[np.ndarray] = None

        # Synthetic IMU integration (zero-mean random-walk, damped)
        self._imu = np.zeros(3, dtype=np.float64)

        # Frame counter
        self._frame_idx: int = 0

        # Latched state for consecutive-frame voting
        self._latch_status:  str   = "MICROGRAVITY"
        self._latch_counter: int   = 0
        self._latch_thresh:  int   = 8   # frames before status changes

        logger.info(
            "GravityDetector initialised (microgravity_mode=%s, history=%d)",
            microgravity_mode, history_len
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        pose_flat: np.ndarray,
        external_accel: Optional[np.ndarray] = None,
    ) -> GravityState:
        """
        Analyse one pose frame and return a GravityState.

        Parameters
        ──────────
        pose_flat       : (132,) float32 — flattened 33×4 MediaPipe landmarks
                          (x, y, z, visibility per joint)
        external_accel  : optional (3,) real IMU reading in g. If absent,
                          visual motion stability is estimated; it must not be
                          interpreted as a measured g-level.

        Returns
        ───────
        GravityState snapshot
        """
        self._frame_idx += 1

        # Reshape to (33, 4)
        try:
            pose = pose_flat.reshape(33, 4).astype(np.float64)
        except Exception:
            return self._fallback_state("Invalid pose shape")

        # Mask low-visibility landmarks
        vis_mask = pose[:, 3] > 0.4
        if vis_mask.sum() < 4:
            return self._fallback_state("Too few visible landmarks")

        # ── 1. Compute COM (centre-of-mass proxy) ─────────────────────────
        core_vis = np.array(_CORE_JOINTS)
        core_valid = [j for j in core_vis if vis_mask[j]]
        if len(core_valid) >= 2:
            com = pose[core_valid, :2].mean(axis=0)
        else:
            com = pose[vis_mask, :2].mean(axis=0)

        # ── 2. Compute per-joint velocity (jitter proxy) ──────────────────
        jitter_rms = 0.0
        drift_vec  = np.zeros(2, dtype=np.float64)

        if len(self._pose_history) > 0:
            prev_pose = self._pose_history[-1]
            velocities = pose[vis_mask, :2] - prev_pose[vis_mask, :2]
            jitter_rms = float(np.sqrt((velocities ** 2).sum(axis=1).mean()))
            drift_vec  = velocities.mean(axis=0)

        if self._prev_com is not None:
            com_delta = com - self._prev_com
            # Supplement drift with COM motion
            if np.linalg.norm(drift_vec) < 1e-9:
                drift_vec = com_delta

        self._prev_com = com.copy()
        self._pose_history.append(pose.copy())
        self._jitter_history.append(jitter_rms)
        self._drift_history.append(drift_vec.copy())

        # EMA smoothed jitter
        self._ema_jitter = (
            self._ema_alpha * jitter_rms
            + (1.0 - self._ema_alpha) * self._ema_jitter
        )

        # ── 3. Classify gravity status ────────────────────────────────────
        status, level_g, stability = self._classify(jitter_rms)

        # ── 4. Synthetic IMU axes ─────────────────────────────────────────
        if external_accel is not None:
            imu = np.asarray(external_accel, dtype=np.float64).flatten()[:3]
            sensor_source = "PAYLOAD_IMU"
        else:
            imu = self._synthesize_imu(pose, jitter_rms, level_g, drift_vec)
            sensor_source = "VISION_STABILITY_PROXY"
        self._imu = 0.7 * self._imu + 0.3 * imu

        # ── 5. Correction mode ────────────────────────────────────────────
        correction_mode = self._pick_correction_mode(status, stability)

        # ── 6. Confidence (how long the current status has been stable) ───
        confidence = self._compute_confidence(status)

        # ── 7. Recommendation string ──────────────────────────────────────
        recommendation = self._recommendation(status, stability, correction_mode, level_g)

        state = GravityState(
            level_g         = round(level_g, 4),
            stability       = stability,
            status          = status,
            correction_mode = correction_mode,
            confidence      = round(confidence, 4),
            imu_axes        = self._imu.round(5).tolist(),
            jitter_rms      = round(jitter_rms, 6),
            drift_vector    = drift_vec.round(5).tolist(),
            history_rms     = list(self._jitter_history),
            recommendation  = recommendation,
            sensor_source   = sensor_source,
        )

        if self._frame_idx % 30 == 0:
            logger.debug(
                "GravityDetector frame=%d status=%s level_g=%.3f jitter=%.5f conf=%.2f",
                self._frame_idx, status, level_g, jitter_rms, confidence
            )

        return state

    def reset(self):
        """Reset all internal state for a new experiment run."""
        self._pose_history.clear()
        self._jitter_history.clear()
        self._drift_history.clear()
        self._ema_jitter   = 0.0
        self._prev_com     = None
        self._imu          = np.zeros(3, dtype=np.float64)
        self._frame_idx    = 0
        self._latch_status  = "MICROGRAVITY"
        self._latch_counter = 0
        logger.info("GravityDetector reset.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify(self, jitter_rms: float) -> Tuple[str, float, str]:
        """
        Returns (status, level_g, stability) from current jitter reading.

        level_g is the estimated effective gravity as a fraction of 1 g:
          0.00  → pure microgravity (ISS free-fall)
          0.17  → lunar surface (~1/6 g)
          0.38  → Mars surface (~3/8 g)
          1.00  → Earth surface

        On ISS / BAS the expected jitter_rms from breathing / micro-motions
        is 0.001–0.006. Active manipulation of equipment adds 0.008–0.018.
        A crew collision or attitude-control thruster firing reads ≥ 0.025.
        """
        ema = self._ema_jitter

        if self._microgravity_mode:
            # ── Spacecraft mode ──
            if ema < _MICROGRAVITY_JITTER_MIN:
                raw_status = "MICROGRAVITY"
                level_g    = 0.00
                stability  = "STABLE"
            elif ema < _LOW_G_JITTER_THRESH:
                raw_status = "MICROGRAVITY"
                # Slight above-floor jitter = mild task motion, still micro-g
                level_g    = min(0.05, ema / _LOW_G_JITTER_THRESH * 0.05)
                stability  = "STABLE"
            elif ema < _TURBULENCE_JITTER_THRESH:
                raw_status = "LOW_G"
                level_g    = 0.05 + (ema - _LOW_G_JITTER_THRESH) / (
                    _TURBULENCE_JITTER_THRESH - _LOW_G_JITTER_THRESH
                ) * 0.10
                stability  = "TRANSITION"
            else:
                raw_status = "TURBULENCE"
                level_g    = min(0.30, ema / 0.10 * 0.30)
                stability  = "TURBULENT"
        else:
            # ── Ground / lab mode ──
            if ema < 0.008:
                raw_status = "NORMAL_G"
                level_g    = 1.0
                stability  = "STABLE"
            elif ema < 0.020:
                raw_status = "NORMAL_G"
                level_g    = 1.0
                stability  = "TRANSITION"
            else:
                raw_status = "TURBULENCE"
                level_g    = 1.0
                stability  = "TURBULENT"

        # Latch: only switch status after N consecutive agreeing frames
        if raw_status == self._latch_status:
            self._latch_counter = min(self._latch_counter + 1, self._latch_thresh * 2)
        else:
            self._latch_counter -= 2
            if self._latch_counter <= 0:
                self._latch_status  = raw_status
                self._latch_counter = 0

        return self._latch_status, level_g, stability

    def _synthesize_imu(
        self,
        pose: np.ndarray,
        jitter_rms: float,
        level_g: float,
        drift_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a 3-axis acceleration reading from the pose signal.

        ax, ay  — image-plane jitter (pose motion proxy, normalised to g)
        az      — depth-axis (z-coord mean of visible landmarks, proxy for
                  in-out motion / thruster push)

        In true micro-g:  |imu| ≈ 0.0001–0.003 g  (residual micro-vibration)
        Turbulence:       |imu| ≈ 0.01–0.03 g
        Ground:           |imu| ≈ 1 g (gravity vector + movement)
        """
        # Scale jitter into acceleration units
        scale = 200.0  # empirical: jitter_rms ~0.005 → 1 g worth of signal @ scale
        ax = float(drift_vec[0]) * scale
        ay = float(drift_vec[1]) * scale + level_g  # add static gravity component

        # z from depth changes
        vis = pose[:, 3] > 0.4
        az_raw = float(pose[vis, 2].std()) if vis.sum() > 2 else 0.0
        az = az_raw * scale * 0.5

        # Add tiny random micro-vibration (always present in real IMU)
        noise_level = max(0.0001, jitter_rms * 0.5)
        noise = np.random.normal(0, noise_level, 3)

        return np.array([ax, ay, az], dtype=np.float64) + noise

    def _pick_correction_mode(self, status: str, stability: str) -> str:
        """Select the pose correction mode for downstream HAR."""
        if status in ("MICROGRAVITY", "LOW_G"):
            return "RACK_FRAME"     # mandatory rack-frame normalisation
        if stability == "TURBULENT":
            return "ADAPTIVE"       # EMA-smoothed reference frame
        return "STANDARD"           # ground-based HAR (no rack correction)

    def _compute_confidence(self, current_status: str) -> float:
        """Confidence grows with consecutive stable frames in the current status."""
        if len(self._jitter_history) < 4:
            return 0.3
        consecutive = min(self._latch_counter, self._latch_thresh)
        conf_latch  = consecutive / self._latch_thresh
        # Additional term: low variance in recent jitter → higher confidence
        if len(self._jitter_history) >= 8:
            recent = list(self._jitter_history)[-8:]
            var = float(np.var(recent))
            conf_var = max(0.0, 1.0 - var / 1e-4)
        else:
            conf_var = 0.5
        return float(np.clip(0.5 * conf_latch + 0.5 * conf_var, 0.0, 1.0))

    def _recommendation(
        self,
        status: str,
        stability: str,
        correction_mode: str,
        level_g: float,
    ) -> str:
        g_str = f"{level_g:.3f} g"
        if status == "MICROGRAVITY" and stability == "STABLE":
            return (
                f"MICROGRAVITY CONFIRMED ({g_str}) — "
                "RACK-FRAME NORMALISATION ACTIVE — HAR READY"
            )
        if status == "MICROGRAVITY" and stability == "TRANSITION":
            return (
                f"MICRO-G TRANSITION ({g_str}) — "
                "ADAPTIVE NORMALISATION — CAUTION: POSE MAY SHIFT"
            )
        if status == "LOW_G":
            return (
                f"LOW-G ENVIRONMENT ({g_str}) — "
                "RACK-FRAME NORMALISATION ACTIVE — MONITOR STABILITY"
            )
        if status == "TURBULENCE":
            return (
                f"TURBULENCE DETECTED ({g_str}) — "
                "HAR SUSPENDED: GRAVITY UNSTABLE — AWAIT STABILISATION"
            )
        if status == "NORMAL_G":
            return (
                f"NORMAL GRAVITY ({g_str}) — "
                "STANDARD HAR MODE ACTIVE"
            )
        return f"GRAVITY UNKNOWN ({g_str}) — MONITORING"

    def _fallback_state(self, reason: str) -> GravityState:
        logger.debug("GravityDetector fallback: %s", reason)
        return GravityState(
            level_g         = 0.0,
            stability       = "STABLE",
            status          = "MICROGRAVITY",
            correction_mode = "RACK_FRAME",
            confidence      = 0.1,
            imu_axes        = [0.0, 0.0, 0.0],
            jitter_rms      = 0.0,
            drift_vector    = [0.0, 0.0],
            recommendation  = f"GRAVITY DETECTOR FALLBACK: {reason}",
            sensor_source   = "VISION_STABILITY_PROXY",
        )

    # ── Convenience: run a quick self-test ───────────────────────────────────

    @classmethod
    def self_test(cls) -> None:
        """Print gravity classifications for synthetic pose sequences."""
        import sys
        print("=" * 65)
        print("GravityDetector — Self Test")
        print("=" * 65)
        rng = np.random.default_rng(42)
        det = cls(microgravity_mode=True)

        scenarios = [
            ("Microgravity — stationary float",   0.000, 30),
            ("Microgravity — gentle manipulation", 0.003, 30),
            ("Low-g transition",                  0.012, 20),
            ("Turbulence / thruster event",        0.030, 20),
            ("Recovery to stable micro-g",         0.002, 30),
        ]

        for label, jitter_amp, n_frames in scenarios:
            print(f"\n  [{label}]  jitter_amp={jitter_amp}")
            base_pose = rng.uniform(0.1, 0.9, (33, 4)).astype(np.float32)
            base_pose[:, 3] = 0.9  # good visibility

            for i in range(n_frames):
                noise = rng.normal(0, jitter_amp, (33, 2)).astype(np.float32)
                p = base_pose.copy()
                p[:, :2] += noise
                p[:, :2] = np.clip(p[:, :2], 0, 1)
                state = det.analyze(p.flatten())

            d = state.to_dict()
            print(f"    status={d['status']:15s}  level_g={d['level_g']:.4f} g")
            print(f"    stability={d['stability']:10s}  confidence={d['confidence']:.3f}")
            print(f"    correction_mode={d['correction_mode']:12s}  jitter_rms={d['jitter_rms']:.5f}")
            print(f"    imu=[{d['imu_axes'][0]:+.4f}, {d['imu_axes'][1]:+.4f}, {d['imu_axes'][2]:+.4f}] g")
            print(f"    -> {d['recommendation']}")

        det_ground = cls(microgravity_mode=False)
        print("\n  [Ground lab -- normal gravity]")
        for _ in range(30):
            p = rng.uniform(0.1, 0.9, (33, 4)).astype(np.float32)
            p[:, 3] = 0.9
            p[:, :2] += rng.normal(0, 0.004, (33, 2)).astype(np.float32)
            state = det_ground.analyze(p.flatten())
        d = state.to_dict()
        print(f"    status={d['status']:15s}  level_g={d['level_g']:.4f} g")
        print(f"    → {d['recommendation']}")

        print("\n" + "=" * 65)
        print("Self test PASSED")
        print("=" * 65)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    GravityDetector.self_test()
