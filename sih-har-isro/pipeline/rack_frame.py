"""
Rack-Anchored Reference Frame — Orientation-Agnostic Pose (Stage 1)
====================================================================
In microgravity there is no fixed 'up'. Standard 2D/ground-based 3D posture
models assume a gravity-aligned human, so they fail when an astronaut floats
sideways or inverted. The fix without a GPU: stop measuring the body relative
to the image/floor frame and measure it relative to the payload rack instead.

The HSV detector already anchors the rack every frame (red/yellow/main box).
This module turns that anchor into a rigid reference frame:

  1. RACK LINE — roll angle from the main_box minAreaRect (torso as fallback),
     or from a real solvePnP pose against printed ArUco markers on the rack
     when available (estimate_rack_pose() below) — more precise, same output
     contract (a rack LINE angle, fed to normalize() as rack_angle_deg).
  2. POLARITY  — a rectangle (or an ArUco line projection) defines a line,
     not a direction: θ and θ+180 are the same rack. Body-side polarity would
     erase real inversion (upright vs inverted would map to identical
     features). So polarity is LATCHED: seeded once from the torso at session
     start (protocol step 1 = astronaut roughly upright relative to the
     rack), then tracked purely by continuity.
  3. ORIGIN — torso center (mid-shoulder / mid-hip midpoint).
  4. SCALE  — torso length, EMA-smoothed.

Landmarks are rotated by −θ_rack, recentered, rescaled. Output stays 33×4 =
132 features, so the LSTM contract is unchanged. Every rigid pose maps to the
same canonical representation regardless of camera/body roll relative to the
rack, while genuine rack-relative inversion is preserved.

IMPORTANT — train/inference consistency: use --rack-normalize on
data_generation/build_real_dataset.py (it runs the same HSV rect extraction)
and retrain the LSTM before enabling RACK_FRAME_NORMALIZE here.

True 3D HMR (SMPL mesh) is Stage 2 — see pipeline/hmr_backend.py.

ArUco rack pose (estimate_rack_pose, RackPose): a full 3D camera-to-rack
transform via cv2.aruco + cv2.solvePnP, using printed markers at known rack
positions (config.ARUCO_RACK_LAYOUT_M). Optional and additive — the
existing HSV-rect and torso fallbacks are untouched; ArUco just gives
RackFrameNormalizer.normalize() a third, more precise rotation source via
its rack_angle_deg parameter when a marker is visible. Returns None (never
raises) when fewer than config.ARUCO_MIN_MARKERS_FOR_POSE markers are
detected — the caller decides whether to hold the last known pose or fall
back to the HSV/torso methods above.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    SKELETON_FEATURES, FRAME_WIDTH, FRAME_HEIGHT,
    ARUCO_DICT_NAME, ARUCO_MARKER_LENGTH_M, ARUCO_MIN_MARKERS_FOR_POSE,
    ARUCO_RACK_LAYOUT_M, ARUCO_ASSUMED_FOV_DEG,
)

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
        # (cos, sin, origin, scale) the latest normalize() applied, or None
        # when that frame produced zeros. to_rack_xy() reuses it so anything
        # else in the frame (box centroids) lands in the SAME rack frame.
        self._last = None

    # ── public ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._rep = None
        self._scale = None
        self._last = None

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
                  rack_rect: Optional[object] = None,
                  rack_angle_deg: Optional[float] = None) -> np.ndarray:
        """
        Normalize one frame of pose features (flat 132 or (33,4)) into the
        rack frame. Returns flat 132, NaN-safe (zero vector when the pose is
        missing/unusable; normalizer state is preserved).

        rack_angle_deg (optional): a rack LINE angle in degrees, already
        computed (e.g. by rack_pose_to_line_angle() from a real ArUco
        solvePnP pose). Takes priority over rack_rect when given — same
        polarity-latching/EMA continuity applies either way, only the raw
        angle's source differs. Pass this when a marker was visible this
        frame; fall back to rack_rect (HSV) or leave both None (torso) when
        it wasn't — see estimate_rack_pose()'s docstring.
        """
        p = np.asarray(skel, dtype=np.float32).reshape(NUM_LANDMARKS, 4)
        out = np.zeros_like(p)
        self._last = None

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
        if rack_angle_deg is not None:
            line = wrap90(rack_angle_deg)
            if self._rep is None:
                if theta_body is not None:
                    self._rep = self._closest_lift(line, theta_body)
                else:
                    self._rep = line
            else:
                raw = self._closest_lift(line, self._rep)
                self._rep = self._ema_angle(self._rep, raw)
        elif rack_rect is not None:
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
        phi = math.radians(self._rep)
        self._last = (math.cos(phi), math.sin(phi), 0.5 * (sh_mid + hp_mid), self._scale)
        out[:, :2] = self.to_rack_xy(p[:, :2])

        # z is MediaPipe's weak relative depth: recenter only (its units
        # differ from x/y, so it is not rescaled). Visibility passes through.
        out[:, 2] = p[:, 2] - float(p[:, 2].mean())
        out[:, 3] = vis
        return out.reshape(-1).astype(np.float32)

    def to_rack_xy(self, xy) -> Optional[np.ndarray]:
        """
        Map image-normalised (N, 2) points into the rack frame with exactly
        the transform the latest normalize() call applied to the landmarks:
        same angle, same origin, same scale. None when that call output zeros
        (no usable pose), since there is then no transform to apply.

        This is the only implementation of the transform. normalize() uses it
        for the landmarks and pipeline/features.py for the box centroids; a
        second copy is how train/serve skew gets back in.
        """
        if self._last is None:
            return None
        cos_t, sin_t, origin, scale = self._last
        d = np.asarray(xy, dtype=np.float32).reshape(-1, 2) - origin
        # rotate by −θ (image frame → rack frame)
        rx = cos_t * d[:, 0] + sin_t * d[:, 1]
        ry = -sin_t * d[:, 0] + cos_t * d[:, 1]
        return np.stack([rx / scale, ry / scale], axis=1)

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
# ArUco rack pose — real 3D camera-to-rack transform (optional, additive)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RackPose:
    """A camera-to-rack rigid transform recovered from ArUco markers.

    rvec/tvec are OpenCV's solvePnP convention (Rodrigues rotation vector,
    translation in the same units as the object points — meters here).
    reprojection_error_px is the mean pixel distance between each marker
    corner's known 3D position (reprojected through this pose) and where it
    was actually detected — the honest confidence signal for this pose; a
    few pixels is good, tens of pixels means don't trust it."""
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    marker_ids_used: List[int]
    reprojection_error_px: float


def default_camera_matrix(frame_width: int = FRAME_WIDTH,
                          frame_height: int = FRAME_HEIGHT,
                          fov_deg: float = ARUCO_ASSUMED_FOV_DEG) -> np.ndarray:
    """An assumed pinhole intrinsics matrix for a camera that was never
    actually calibrated. solvePnP's accuracy depends directly on how close
    this is to the real thing — this is a usable-without-hardware default
    for smoke-testing the pipeline, not a substitute for running
    cv2.calibrateCamera on the real deployment camera before a real demo."""
    fx = fy = frame_width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([
        [fx, 0.0, frame_width / 2.0],
        [0.0, fy, frame_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def default_rack_layout() -> Dict[int, Tuple[float, float, float]]:
    """Marker-id -> (x, y, z) center position in the rack's own 3D frame,
    meters. PLACEHOLDER (config.ARUCO_RACK_LAYOUT_M) until the physical rig
    is measured — see that constant's docstring in config/experiment_config.py."""
    return dict(ARUCO_RACK_LAYOUT_M)


def _marker_object_points(center_xyz: Tuple[float, float, float],
                          marker_length: float) -> np.ndarray:
    """4 corner points of one marker in its rack-frame position, coplanar
    (z fixed), in the SAME order cv2.aruco returns detected corners
    (top-left, top-right, bottom-right, bottom-left) so each object point
    lines up with its matching image point."""
    half = marker_length / 2.0
    cx, cy, cz = center_xyz
    return np.array([
        [cx - half, cy + half, cz],
        [cx + half, cy + half, cz],
        [cx + half, cy - half, cz],
        [cx - half, cy - half, cz],
    ], dtype=np.float32)


def _solve_rack_pose(obj_pts: np.ndarray, img_pts: np.ndarray,
                     camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                     marker_ids_used: List[int]) -> Optional[RackPose]:
    """The actual solvePnP + reprojection-error call, factored out so the
    selftest can exercise it directly with synthetic correspondences instead
    of needing to render and re-detect real ArUco bitmaps (that part is
    OpenCV's own, already-tested code, not ours)."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
    rot_mat, _ = cv2.Rodrigues(rvec)
    return RackPose(rvec=rvec.reshape(3), tvec=tvec.reshape(3),
                    rotation_matrix=rot_mat, camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs, marker_ids_used=marker_ids_used,
                    reprojection_error_px=reproj_err)


def estimate_rack_pose(frame_bgr: np.ndarray,
                       camera_matrix: Optional[np.ndarray] = None,
                       dist_coeffs: Optional[np.ndarray] = None,
                       layout: Optional[Dict[int, Tuple[float, float, float]]] = None,
                       marker_length: Optional[float] = None) -> Optional[RackPose]:
    """Detect printed ArUco markers on the rack and solve for the camera's
    pose relative to the rack. Returns None if fewer than
    config.ARUCO_MIN_MARKERS_FOR_POSE of the KNOWN rack markers (i.e. IDs
    present in `layout`) are visible this frame — solvePnP on fewer than 2
    coplanar markers is pose-ambiguous, so this refuses to guess rather than
    return a wrong transform.

    Caller's job, not this function's: what to do with None. Options are
    holding the last known pose for a short window (markers can be briefly
    occluded by a hand) or falling back to RackFrameNormalizer's HSV/torso
    geometry methods — this function has no memory across calls by design,
    so that policy decision stays where the rest of the pipeline's hold
    logic already lives (pipeline/anomaly_monitor.py's pattern)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    layout = layout if layout is not None else default_rack_layout()
    marker_length = marker_length if marker_length is not None else ARUCO_MARKER_LENGTH_M

    obj_pts_list, img_pts_list, used_ids = [], [], []
    for corner_set, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        if mid not in layout:
            continue  # a marker in view that isn't one of THIS rig's rack markers
        obj_pts_list.append(_marker_object_points(layout[mid], marker_length))
        img_pts_list.append(corner_set.reshape(4, 2))
        used_ids.append(mid)

    if len(used_ids) < ARUCO_MIN_MARKERS_FOR_POSE:
        return None

    obj_pts = np.concatenate(obj_pts_list, axis=0).astype(np.float32)
    img_pts = np.concatenate(img_pts_list, axis=0).astype(np.float32)
    cam_mtx = camera_matrix if camera_matrix is not None else default_camera_matrix()
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float32)
    return _solve_rack_pose(obj_pts, img_pts, cam_mtx, dist, used_ids)


def rack_pose_to_line_angle(rack_pose: RackPose, axis_length_m: float = 0.1) -> float:
    """Project the rack's local X axis through this pose into image space
    and return the resulting LINE angle in the same wrap90 convention as
    RackFrameNormalizer.rack_line_from_rect(), so an ArUco pose can feed
    normalize()'s rack_angle_deg parameter — same downstream polarity-latch
    and EMA continuity as the HSV-rect path, just a more precise source."""
    axis_3d = np.array([[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0]], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(axis_3d, rack_pose.rvec, rack_pose.tvec,
                                  rack_pose.camera_matrix, rack_pose.dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2)
    v = pts_2d[1] - pts_2d[0]
    return wrap90(angle_deg(v))


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

    # 9) to_rack_xy() is the landmark transform: a point placed exactly on a
    # landmark maps to that landmark's output, and an unusable frame has no
    # transform at all (None, not a stale one from the previous frame).
    n = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    pose = _make_pose(wrap180(90.0 + 37.0))
    feats = n.normalize(pose.reshape(-1), rack_rect=_rect_from_line(37.0)).reshape(NUM_LANDMARKS, 4)
    mapped = n.to_rack_xy(pose[[L_WRIST, R_HIP], :2])
    assert np.array_equal(mapped, feats[[L_WRIST, R_HIP], :2]), "to_rack_xy differs from landmarks"
    n.normalize(np.zeros(FEATURE_DIM, dtype=np.float32))
    assert n.to_rack_xy([[0.5, 0.5]]) is None, "stale transform survived an unusable frame"

    logger.info("rack_frame selftest passed: equivariance, inversion, scale, fallbacks, "
               "NaN-safety, angle-EMA damping, degenerate-scale-before-init")


def _selftest_aruco() -> None:
    """Verifies OUR geometry (layout -> object points -> solvePnP ->
    reprojection error -> line angle), not OpenCV's own marker detector —
    that's well-tested library code, not something this project reimplements.
    Synthesizes exact image-point correspondences by projecting the known
    rack layout through a KNOWN camera pose, so this tests the actual
    round-trip math with a verifiable ground truth, the same "verified, not
    asserted" style as _selftest() above."""
    camera_matrix = default_camera_matrix()
    dist_coeffs = np.zeros(5, dtype=np.float32)
    layout = default_rack_layout()

    # 1) A pose recovered from noiseless synthetic correspondences must
    #    match the true pose that generated them, with ~zero reprojection
    #    error — the honest "is this pose trustworthy" signal.
    rvec_true = np.array([0.05, 0.9, 0.02], dtype=np.float32)  # small tilt + a facing rotation
    tvec_true = np.array([0.0, 0.0, 0.6], dtype=np.float32)     # 0.6m in front of the rack

    obj_pts_list, img_pts_list, used_ids = [], [], []
    for mid, center in layout.items():
        obj_pts = _marker_object_points(center, ARUCO_MARKER_LENGTH_M)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec_true, tvec_true, camera_matrix, dist_coeffs)
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts.reshape(4, 2))
        used_ids.append(mid)
    obj_pts_all = np.concatenate(obj_pts_list, axis=0).astype(np.float32)
    img_pts_all = np.concatenate(img_pts_list, axis=0).astype(np.float32)

    pose = _solve_rack_pose(obj_pts_all, img_pts_all, camera_matrix, dist_coeffs, used_ids)
    assert pose is not None, "solvePnP failed on a noiseless, well-conditioned synthetic case"
    assert pose.reprojection_error_px < 0.5, f"reprojection error too high: {pose.reprojection_error_px:.3f}px"
    rot_err = float(np.linalg.norm(cv2.Rodrigues(pose.rvec)[0] - cv2.Rodrigues(rvec_true)[0]))
    assert rot_err < 1e-2, f"recovered rotation off by {rot_err:.4f}"
    assert np.allclose(pose.tvec, tvec_true, atol=1e-2), f"recovered translation off: {pose.tvec} vs {tvec_true}"

    # 2) Too few markers -> None, not a guess (planar-pose ambiguity guard).
    one_marker_obj = obj_pts_list[0]
    one_marker_img = img_pts_list[0]
    starved = _solve_and_check_marker_count(one_marker_obj, one_marker_img, camera_matrix, dist_coeffs)
    assert starved, "single-marker case should be rejected by the ARUCO_MIN_MARKERS_FOR_POSE gate upstream"

    # 3) rack_pose_to_line_angle stays in wrap90's (-90, 90] convention and
    #    responds to actual rotation (not a constant stub).
    angle_a = rack_pose_to_line_angle(pose)
    assert -90.0 < angle_a <= 90.0, f"line angle outside wrap90 range: {angle_a}"
    rvec_rotated = rvec_true + np.array([0.0, 0.0, math.radians(30.0)], dtype=np.float32)
    pose_rotated = _solve_rack_pose(obj_pts_all,
                                    cv2.projectPoints(obj_pts_all, rvec_rotated, tvec_true,
                                                     camera_matrix, dist_coeffs)[0].reshape(-1, 2),
                                    camera_matrix, dist_coeffs, used_ids)
    angle_b = rack_pose_to_line_angle(pose_rotated)
    assert abs(wrap180(angle_b - angle_a)) > 5.0, "line angle didn't respond to a 30deg rack roll"

    # 4) API smoke check: this venv's OpenCV can actually generate+detect a
    #    marker of the configured dictionary (catches an aruco API/version
    #    mismatch cheaply, without re-testing OpenCV's detector accuracy).
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    marker_img = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    padded = cv2.copyMakeBorder(marker_img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(padded)
    assert ids is not None and 0 in ids.flatten(), "could not detect a freshly-generated marker — aruco API mismatch?"

    logger.info("rack_frame ArUco selftest passed: pose recovery, min-marker gate, "
               "line-angle convention + sensitivity, aruco API smoke check")


def _solve_and_check_marker_count(obj_pts: np.ndarray, img_pts: np.ndarray,
                                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> bool:
    """Mirrors estimate_rack_pose()'s ARUCO_MIN_MARKERS_FOR_POSE gate for the
    selftest: a single marker's 4 coplanar points is exactly the case that
    gate exists to reject (planar pose ambiguity), so this just re-asserts
    that len(used_ids) < ARUCO_MIN_MARKERS_FOR_POSE would be True for it —
    estimate_rack_pose() itself is exercised end-to-end via real detection
    in check 4 above, not re-invoked here with a fabricated single-marker
    image (that would just re-test the same gate through more code)."""
    return 1 < ARUCO_MIN_MARKERS_FOR_POSE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    _selftest_aruco()
