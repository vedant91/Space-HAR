"""
Rack-Anchored Reference Frame — Orientation-Agnostic Pose (Stage 1)
====================================================================
In microgravity there is no fixed 'up'. Standard 2D/ground-based 3D posture
models assume a gravity-aligned human, so they fail when an astronaut floats
sideways or inverted. The fix without a GPU: stop measuring the body relative
to the image/floor frame and measure it relative to the payload rack instead.

The HSV detector already anchors the rack every frame (red/yellow/main box).
This module turns that anchor into a rigid reference frame:

  1. RACK LINE — roll angle from the main_box minAreaRect (torso as fallback).
  2. POLARITY  — a rectangle defines a line, not a direction: θ and θ+180 are
     the same rack. Body-side polarity would erase real inversion (upright vs
     inverted would map to identical features). So polarity is LATCHED: seeded
     once from the torso at session start (protocol step 1 = astronaut roughly
     upright relative to the rack), then tracked purely by continuity.
  3. ORIGIN — torso center (mid-shoulder / mid-hip midpoint).
  4. SCALE  — torso length, EMA-smoothed.

Landmarks are rotated by −θ_rack, recentered, rescaled. Output stays 33×4 =
132 features, so the LSTM contract is unchanged. Every rigid pose maps to the
same canonical representation regardless of camera/body roll relative to the
rack, while genuine rack-relative inversion is preserved.

IMPORTANT — train/inference consistency: use --rack-normalize on
mediapipe_labeler.py (it runs the same HSV rect extraction) and retrain the
LSTM before enabling RACK_FRAME_NORMALIZE here.

True 3D HMR (SMPL mesh) is Stage 2 — see pipeline/hmr_backend.py.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import SKELETON_FEATURES

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES  # single source of truth: config.SKELETON_FEATURES (132)

# MediaPipe Pose indices (mirrors data_generation/synthetic_pose.py)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def wrap180(a: float) -> float:
    """Wrap angle in degrees to (−180, 180]."""
    return float((a + 180.0) % 360.0 - 180.0)


def wrap90(a: float) -> float:
    """Wrap an angle representing an undirected LINE to (−90, 90]."""
    return float((a + 90.0) % 180.0 - 90.0)


def angle_deg(v: Sequence[float]) -> float:
    """Orientation of vector v in degrees, wrapped to (−180, 180]."""
    return wrap180(math.degrees(math.atan2(float(v[1]), float(v[0]))))


def pick_rack_rect(detections) -> Optional[object]:
    """
    Choose the best rack-anchor rect from HSV detections (duck-typed: any
    objects with .label and .rect). Preference: main_box > red > yellow.
    Returns the minAreaRect tuple or None.
    """
    by_pref = {"main_box": 0, "red_box": 1, "yellow_box": 2}
    best = None
    best_rank = 99
    for d in detections or []:
        rect = getattr(d, "rect", None)
        if rect is None:
            continue
        rank = by_pref.get(getattr(d, "label", ""), 99)
        if rank < best_rank:
            best, best_rank = rect, rank
    return best


class RackFrameNormalizer:
    """
    Re-expresses MediaPipe pose features in the payload-rack frame.

    Per frame:
        feats = normalizer.normalize(skel132, rack_rect)

    rack_rect is the main_box minAreaRect ((cx,cy),(w,h),angle_deg) or None.

    Stateful (EMA-smoothed rack angle and torso scale, latched polarity);
    call reset() between sessions/trials.
    """

    def __init__(self,
                 angle_ema: float = 0.85,
                 scale_ema: float = 0.90,
                 min_torso_norm: float = 0.02):
        self.angle_ema = float(np.clip(angle_ema, 0.0, 0.999))
        self.scale_ema = float(np.clip(scale_ema, 0.0, 0.999))
        self.min_torso_norm = float(min_torso_norm)
        self._rep = None    # continuous rack rotation angle (deg, −180..180)
        self._scale = None  # EMA torso length (image-normalized units)

    # ── public ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._rep = None
        self._scale = None

    @property
    def initialized(self) -> bool:
        return self._rep is not None and self._scale is not None

    @property
    def rack_angle_deg(self) -> Optional[float]:
        return self._rep

    def rack_line_from_rect(self, rect) -> float:
        """
        Rack line angle in (−90, 90] from a cv2.minAreaRect result.
        OpenCV ≥4.5 measures the angle from the horizontal to the FIRST edge;
        with aspect < 1 that edge is the long side's complement, so shift by
        90° to reference the long axis, then wrap as a line.
        """
        (_cx, _cy), (w, h), a = rect
        a = float(a)
        if w < h:
            a -= 90.0
        return wrap90(a)

    def normalize(self, skel: np.ndarray,
                  rack_rect: Optional[object] = None) -> np.ndarray:
        """
        Normalize one frame of pose features (flat 132 or (33,4)) into the
        rack frame. Returns flat 132, NaN-safe (zero vector when the pose is
        missing/unusable; normalizer state is preserved).
        """
        p = np.asarray(skel, dtype=np.float32).reshape(NUM_LANDMARKS, 4)
        out = np.zeros_like(p)

        vis = p[:, 3]
        sh = p[[L_SHOULDER, R_SHOULDER], :2]
        hp = p[[L_HIP, R_HIP], :2]
        sh_mid = sh.mean(axis=0)
        hp_mid = hp.mean(axis=0)
        torso_down = hp_mid - sh_mid          # head → feet vector
        torso_len = float(np.linalg.norm(torso_down))
        torso_valid = bool(np.any(vis > 0.5)) and torso_len > self.min_torso_norm

        theta_body = angle_deg(torso_down) if torso_valid else None

        # ── Rotation source ────────────────────────────────────────────
        if rack_rect is not None:
            line = self.rack_line_from_rect(rack_rect)
            if self._rep is None:
                # Seed polarity once: assume protocol start ≈ upright relative
                # to rack, so pick the lift of the line closest to the body.
                if theta_body is not None:
                    self._rep = self._closest_lift(line, theta_body)
                else:
                    self._rep = line
            else:
                # Continuity: follow the line without ever flipping polarity,
                # smoothed the same way the torso-fallback branch below is —
                # previously this assigned the raw lift with zero damping, so
                # angle_ema had no effect at all whenever a rack was visible.
                raw = self._closest_lift(line, self._rep)
                self._rep = self._ema_angle(self._rep, raw)
        elif theta_body is not None:
            # No rack visible: the torso vector carries full polarity by
            # itself, so rotate it straight to the canonical direction (+y).
            rep = wrap180(theta_body - 90.0)
            self._rep = rep if self._rep is None else self._ema_angle(self._rep, rep)
        else:
            # Nothing usable: keep previous state, output zeros this frame.
            return out.reshape(-1).astype(np.float32)

        # ── Scale: EMA of torso length ─────────────────────────────────
        if torso_valid:
            if self._scale is None:
                self._scale = torso_len
            else:
                self._scale = (self.scale_ema * self._scale
                               + (1.0 - self.scale_ema) * torso_len)

        if self._scale is None:
            # No valid torso has ever been seen yet (degenerate on every frame
            # so far) — there is no reasonable scale to fall back to. Dividing
            # by the current (near-zero, invalid) torso_len instead of
            # refusing output produced finite-but-nonsense feature magnitudes
            # in the hundreds (vs. the expected O(1)) that would silently
            # poison the LSTM. Treat this frame as unusable instead.
            return out.reshape(-1).astype(np.float32)
        scale = self._scale
        phi = math.radians(self._rep)
        cos_t, sin_t = math.cos(phi), math.sin(phi)

        origin = 0.5 * (sh_mid + hp_mid)
        xy = p[:, :2] - origin
        # rotate by −θ (image frame → rack frame)
        rx = cos_t * xy[:, 0] + sin_t * xy[:, 1]
        ry = -sin_t * xy[:, 0] + cos_t * xy[:, 1]
        out[:, 0] = rx / scale
        out[:, 1] = ry / scale

        # z is MediaPipe's weak relative depth: recenter only (its units
        # differ from x/y, so it is not rescaled). Visibility passes through.
        out[:, 2] = p[:, 2] - float(p[:, 2].mean())
        out[:, 3] = vis
        return out.reshape(-1).astype(np.float32)

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _closest_lift(line: float, reference: float) -> float:
        """Of {line, line±180}, return the candidate closest to `reference`."""
        cands = (line, line - 180.0, line + 180.0)
        return min(cands, key=lambda c: abs(wrap180(c - reference)))

    def _ema_angle(self, prev: float, new: float) -> float:
        """EMA over the shortest arc (±180 wrap)."""
        d = wrap180(new - prev)
        return wrap180(prev + (1.0 - self.angle_ema) * d)


# ═══════════════════════════════════════════════════════════════════════════
# Selftest
# ═══════════════════════════════════════════════════════════════════════════

def _make_pose(body_angle_deg: float = 90.0, scale: float = 1.0) -> np.ndarray:
    """
    Synthetic astronaut with the head→feet axis pointing at `body_angle_deg`
    (90° = head up / feet down in image coords, y grows downward).
    """
    p = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    p[:, 3] = 0.95
    p[L_SHOULDER, :2] = (0.40, 0.30)
    p[R_SHOULDER, :2] = (0.60, 0.30)
    p[L_HIP, :2] = (0.42, 0.60)
    p[R_HIP, :2] = (0.58, 0.60)
    p[0, :2] = (0.50, 0.10)            # nose above shoulders
    p[L_ELBOW, :2] = (0.35, 0.45)
    p[R_ELBOW, :2] = (0.65, 0.45)
    p[L_WRIST, :2] = (0.33, 0.58)
    p[R_WRIST, :2] = (0.67, 0.58)
    p[L_KNEE, :2] = (0.43, 0.80)
    p[R_KNEE, :2] = (0.57, 0.80)
    p[L_ANKLE, :2] = (0.44, 0.95)
    p[R_ANKLE, :2] = (0.56, 0.95)

    c = p[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :2].mean(axis=0)
    th = math.radians(body_angle_deg - 90.0)   # canonical pose is at 90°
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]])
    p[:, :2] = (p[:, :2] - c) @ R.T * scale + c
    return p


def _rect_from_line(line_deg: float, scale: float = 1.0):
    """Wide minAreaRect-style rect whose long axis sits at `line_deg`."""
    a = wrap90(line_deg)
    return ((0.5, 0.5), (0.8 * scale, 0.4 * scale), a)


def _selftest() -> None:
    # 1) Scene-rotation equivariance (rack present): rotate the WHOLE scene
    #    (body 20° off the rack axis) step-by-step through a full turn →
    #    every output identical to the first.
    norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    ref = None
    for k in range(0, 360, 5):
        delta = float(k)
        pose = _make_pose(body_angle_deg=wrap180(90.0 + delta))
        rect = _rect_from_line(wrap90(delta))
        feats = norm.normalize(pose.reshape(-1), rack_rect=rect)
        if ref is None:
            ref = feats
        else:
            err = float(np.abs(feats - ref).max())
            assert err < 1e-4, f"equivariance failed at delta={delta}: err={err}"

    # 2) Inversion is preserved: body upright vs inverted relative to the
    #    rack must produce DIFFERENT canonical features.
    n_up = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    n_inv = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    up = n_up.normalize(_make_pose(90.0).reshape(-1), rack_rect=_rect_from_line(0.0))
    inv = n_inv.normalize(_make_pose(-90.0).reshape(-1), rack_rect=_rect_from_line(0.0))
    assert float(np.abs(up - inv).max()) > 0.1, "inversion collapsed to upright"

    # 3) Scale invariance with the rack anchor.
    n = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    ref_s = None
    for s in (0.6, 1.0, 1.7):
        feats = n.normalize(_make_pose(scale=s).reshape(-1),
                            rack_rect=_rect_from_line(0.0, scale=s))
        if ref_s is None:
            ref_s = feats
        else:
            err = float(np.abs(feats - ref_s).max())
            assert err < 1e-4, f"scale={s}: max err {err}"

    # 4) Torso fallback (no rack): whole-scene rotation → identical output,
    #    fed continuously so the EMA sees a smooth sequence.
    n = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    ref_f = None
    for k in range(0, 360, 5):
        feats = n.normalize(_make_pose(wrap180(90.0 + k)).reshape(-1), rack_rect=None)
        if ref_f is None:
            ref_f = feats
        else:
            err = float(np.abs(feats - ref_f).max())
            assert err < 1e-4, f"torso fallback failed at {k}: err={err}"

    # 5) Rack loss & re-acquisition keeps state finite and stays continuous.
    n = RackFrameNormalizer(angle_ema=0.7, scale_ema=0.7)
    outs = []
    for k in range(0, 40):
        rect = _rect_from_line(wrap90(k * 3.0)) if k % 4 else None  # every 4th frame blind
        outs.append(n.normalize(_make_pose(wrap180(90.0 + k * 3.0)).reshape(-1),
                                rack_rect=rect))
    assert all(np.isfinite(o).all() for o in outs), "NaN/inf in dropout sequence"

    # 6) Degenerate / zero inputs stay NaN-safe and state survives.
    z = RackFrameNormalizer().normalize(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert np.isfinite(z).all() and not np.any(z)

    # 7) Angle EMA actually damps rack-line jitter (regression: this branch
    # previously assigned the raw lift with zero smoothing, so angle_ema had
    # no effect whenever a rack was visible — only the no-rack/torso-fallback
    # branch ever used it). An undamped ±8° jitter would swing a full 16°
    # frame to frame; damping must measurably reduce that.
    n = RackFrameNormalizer(angle_ema=0.9, scale_ema=0.0)
    seen = []
    for j in (8.0, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0, -8.0):
        n.normalize(_make_pose(90.0).reshape(-1), rack_rect=_rect_from_line(j))
        seen.append(n.rack_angle_deg)
    swing = max(seen) - min(seen)
    assert swing < 15.0, f"angle EMA not damping rack-visible jitter: swing={swing:.2f} (raw=16.0)"

    # 8) Degenerate torso on the very first frame(s) — before any valid scale
    # has ever been established — must yield exact zeros, not a garbage
    # large-magnitude output from dividing by a near-zero torso length.
    # (Regression: this used to fall back to `max(torso_len, eps)` and
    # produce finite-but-nonsense feature magnitudes in the hundreds.)
    n = RackFrameNormalizer()
    degenerate = _make_pose(90.0)
    degenerate[L_SHOULDER, :2] = degenerate[R_SHOULDER, :2] = (0.5, 0.5)
    degenerate[L_HIP, :2] = degenerate[R_HIP, :2] = (0.5, 0.5 + 1e-6)  # torso_len ~ 1e-6
    out_bad = n.normalize(degenerate.reshape(-1), rack_rect=_rect_from_line(0.0))
    assert not np.any(out_bad), f"degenerate first frame should be all-zero, got max={np.abs(out_bad).max():.2f}"

    logger.info("rack_frame selftest passed: equivariance, inversion, scale, fallbacks, "
               "NaN-safety, angle-EMA damping, degenerate-scale-before-init")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
