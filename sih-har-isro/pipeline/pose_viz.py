"""
Display-only perception visualisation: stable 2D skeleton, rigid 3D body,
steady object boxes, and the monitor side panel.

Nothing here feeds the classifier. Features are built from the raw per-frame
pose and detections (pipeline/features.py) exactly as in training; this
module only decides what a human sees, so it is free to filter over time.

Why each piece exists (measured on test-har-3, 492 frames):
  * MediaPipe's raw 2D landmarks jitter ~0.05 torso-lengths/frame on a
    motionless astronaut -> One-Euro filter (smooth when still, fast when
    moving).
  * The previous 3D panel used HMR3DNet, whose output on real footage has
    bone-length variation 0.22, L/R asymmetry 0.24 and a poor fit to the body
    in the image (0.66 Procrustes residual). BlazePose's own world landmarks,
    free from the same inference, score 0.16 / 0.14 / 0.09. Rebuilding them
    along the kinematic tree with fixed bone lengths makes the body rigid by
    construction (no stretching limbs).
  * Boxes flickered in and out frame to frame -> a per-class display tracker
    with appear/hold hysteresis.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# BlazePose indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_IDX, R_IDX = 19, 20
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_FT, R_FT = 31, 32
HIP_MID, SH_MID = 33, 34          # virtual joints appended for the tree

LEFT = {L_SH, L_EL, L_WR, L_IDX, L_HIP, L_KN, L_AN, L_FT, L_EAR}
RIGHT = {R_SH, R_EL, R_WR, R_IDX, R_HIP, R_KN, R_AN, R_FT, R_EAR}

# Parent -> child in topological order, rooted at the hip midpoint.
TREE: Tuple[Tuple[int, int], ...] = (
    (HIP_MID, L_HIP), (HIP_MID, R_HIP), (HIP_MID, SH_MID),
    (SH_MID, L_SH), (SH_MID, R_SH), (SH_MID, NOSE),
    (NOSE, L_EAR), (NOSE, R_EAR),
    (L_SH, L_EL), (L_EL, L_WR), (L_WR, L_IDX),
    (R_SH, R_EL), (R_EL, R_WR), (R_WR, R_IDX),
    (L_HIP, L_KN), (L_KN, L_AN), (L_AN, L_FT),
    (R_HIP, R_KN), (R_KN, R_AN), (R_AN, R_FT),
)
# Left/right bones share one length (a body is symmetric; pooling both
# sides halves the noise of each side's estimate).
MIRROR = {L_HIP: R_HIP, L_SH: R_SH, L_EAR: R_EAR, L_EL: R_EL, L_WR: R_WR, L_IDX: R_IDX,
          L_KN: R_KN, L_AN: R_AN, L_FT: R_FT}
MIRROR.update({v: k for k, v in list(MIRROR.items())})

# 2D overlay topology (no face mesh: nose-ears only, it reads as a head).
BONES_2D: Tuple[Tuple[int, int], ...] = (
    (L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_SH, L_EL), (L_EL, L_WR), (L_WR, L_IDX), (R_SH, R_EL), (R_EL, R_WR), (R_WR, R_IDX),
    (L_HIP, L_KN), (L_KN, L_AN), (L_AN, L_FT), (R_HIP, R_KN), (R_KN, R_AN), (R_AN, R_FT),
    (NOSE, L_EAR), (NOSE, R_EAR),
)
JOINTS_2D = (NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN)

# Colours (BGR)
C_LEFT = (210, 190, 40)     # teal-cyan
C_RIGHT = (40, 150, 255)    # orange
C_MID = (235, 235, 235)
C_BOX = {"red_box": (60, 60, 235), "yellow_box": (40, 205, 250), "main_box": (215, 215, 215)}
BOX_NAME = {"red_box": "RED BOX", "yellow_box": "YELLOW BOX", "main_box": "MAIN BOX"}
BG = (26, 20, 14)
PANEL = (38, 30, 22)
INK = (235, 235, 235)
MUTED = (150, 150, 150)
OK_C, CUR_C, BAD_C = (110, 205, 90), (40, 190, 255), (70, 70, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── filters ──────────────────────────────────────────────────────────────

class OneEuro:
    """Vectorised One-Euro filter (Casiez et al., CHI 2012): a low-pass
    whose cutoff rises with speed, so a still body is steady and a moving
    arm is not dragged behind."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.x.shape != x.shape or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class HipDepthFilter(OneEuro):
    """Smooth camera-width depths on the capture clock, resetting after a tracking gap."""
    def __call__(self, depths, t):
        if self.t is not None and t - self.t > .5:
            self.reset()
        z = super().__call__(depths, t)
        return z - z[[23, 24]].mean()


class Skeleton2DStabilizer:
    """(33, 4) image-normalised landmarks -> smoothed landmarks.

    Two failure modes are handled, both measured on real footage:
      * a missed frame -> coast on the last body briefly instead of blinking;
      * a glitched frame (MediaPipe collapses the torso or flips it for a
        frame) -> rejected as an outlier and coasted over. A torso whose
        length or angle jumps implausibly in one frame, or whose four corner
        joints are barely seen, is not a body that moved - it is a bad
        estimate. `rejected` tells the caller not to update the 3D view."""

    def __init__(self, hold_s: float = 0.6, aspect: float = 16 / 9):
        self.f = OneEuro(min_cutoff=1.5, beta=8.0)   # beta in normalised units/s
        self.vis = None
        self.last = None
        self.last_t = None
        self.hold_s = hold_s
        self.aspect = aspect
        self.torso = None          # (length, angle) of the accepted body
        self.rejected = False

    def reset(self):
        self.f.reset()
        self.vis = self.last = self.last_t = self.torso = None
        self.rejected = False

    def _torso(self, p):
        sh = (p[L_SH, :2] + p[R_SH, :2]) / 2 * (self.aspect, 1.0)
        hp = (p[L_HIP, :2] + p[R_HIP, :2]) / 2 * (self.aspect, 1.0)
        d = hp - sh
        return float(np.hypot(*d)), math.degrees(math.atan2(d[1], d[0]))

    def _plausible(self, p) -> bool:
        if float(p[[L_SH, R_SH, L_HIP, R_HIP], 3].mean()) < 0.5:
            return False
        if self.torso is None:
            return True
        length, ang = self._torso(p)
        l0, a0 = self.torso
        dang = abs((ang - a0 + 180.0) % 360.0 - 180.0)
        return 0.6 < length / max(l0, 1e-6) < 1.6 and dang < 40.0

    def _coast(self, t):
        if self.last is not None and self.last_t is not None and t - self.last_t < self.hold_s:
            out = self.last.copy()
            out[:, 3] *= 0.9                      # fade while coasting
            self.last = out
            return out
        return None

    def __call__(self, skel: Optional[np.ndarray], t: float) -> Optional[np.ndarray]:
        self.rejected = False
        if skel is None or not np.any(skel):
            out = self._coast(t)
            if out is None:
                self.reset()
            return out
        p = np.asarray(skel, dtype=np.float64).reshape(33, 4)
        if not self._plausible(p):
            out = self._coast(t)
            if out is not None:
                self.rejected = True
                return out
            self.reset()                           # persistent: accept the new body afresh
            if float(p[[L_SH, R_SH, L_HIP, R_HIP], 3].mean()) < 0.5:
                return None
        xy = self.f(p[:, :2], t)
        self.vis = p[:, 3] if self.vis is None else 0.6 * self.vis + 0.4 * p[:, 3]
        out = np.c_[xy, p[:, 2], self.vis]
        length, ang = self._torso(out)
        self.torso = (length, ang) if self.torso is None else             (0.8 * self.torso[0] + 0.2 * length, ang)
        self.last, self.last_t = out.copy(), t
        return out


def _with_virtual(j: np.ndarray) -> np.ndarray:
    """(33, >=3) -> (35, 3) with hip/shoulder midpoints appended."""
    xyz = np.asarray(j, dtype=np.float64)[:, :3]
    hip = 0.5 * (xyz[L_HIP] + xyz[R_HIP])
    sh = 0.5 * (xyz[L_SH] + xyz[R_SH])
    return np.vstack([xyz, hip, sh])


def display_body(pose, size):
    """Hip-relative display coordinates at a nominal 0.5 m 3D torso length."""
    width, height = size
    body = np.asarray(pose, dtype=np.float64).reshape(33, 4).copy()
    xyz = body[:, :3]
    xyz *= (width, height, width)
    xyz -= xyz[[L_HIP, R_HIP]].mean(axis=0)
    torso = np.linalg.norm(xyz[[L_SH, R_SH]].mean(axis=0))
    xyz *= 0.5 / max(torso, 1.0)
    return body


class RigidSkeleton3D:
    """Metric world landmarks -> the same pose rebuilt with constant bone
    lengths (running median, left/right pooled), One-Euro smoothed.

    A body's bones do not change length; per-frame 3D estimates do. Keeping
    each bone's DIRECTION from the estimate but its LENGTH from the running
    median removes the stretching that made the old panel look wrong."""

    def __init__(self, window: int = 180, min_vis: float = 0.55):
        self.samples: Dict[Tuple[int, int], deque] = {b: deque(maxlen=window) for b in TREE}
        self.min_vis = min_vis
        self.f = OneEuro(min_cutoff=1.0, beta=1.5)
        self.prev_dir: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self):
        for d in self.samples.values():
            d.clear()
        self.f.reset()
        self.prev_dir.clear()

    def _length(self, bone) -> Optional[float]:
        p, c = bone
        vals = list(self.samples[bone])
        if c in MIRROR:
            twin = self.samples.get((MIRROR.get(p, p), MIRROR[c]))
            if twin is not None and twin is not self.samples[bone]:
                vals += list(twin)
        return float(np.median(vals)) if vals else None

    def __call__(self, world: np.ndarray, t: float) -> np.ndarray:
        w = np.asarray(world, dtype=np.float64).reshape(33, 4)
        vis = np.r_[w[:, 3], min(w[L_HIP, 3], w[R_HIP, 3]), min(w[L_SH, 3], w[R_SH, 3])]
        est = _with_virtual(self.f(w[:, :3], t))
        for bone in TREE:
            p, c = bone
            if vis[p] >= self.min_vis and vis[c] >= self.min_vis:
                self.samples[bone].append(float(np.linalg.norm(est[c] - est[p])))
        out = est.copy()
        for bone in TREE:
            p, c = bone
            d = est[c] - est[p]
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                d = self.prev_dir.get(bone, np.array([0.0, 1.0, 0.0]))
            else:
                d = d / n
                self.prev_dir[bone] = d
            length = self._length(bone)
            out[c] = out[p] + d * (length if length is not None else n)
        res = np.zeros((33, 4))
        res[:, :3] = out[:33]
        res[:, 3] = w[:, 3]
        return res


class BoxDisplayTracker:
    """Per-class display smoothing: a box must be seen on 2 consecutive
    frames to appear, then is held (fading) for a few frames after it is
    lost. Detector output for the FEATURES is untouched."""

    def __init__(self, appear: int = 2, hold: int = 5, ema: float = 0.55):
        self.appear, self.hold, self.ema = appear, hold, ema
        self.state: Dict[str, dict] = {}

    def reset(self):
        self.state.clear()

    def __call__(self, detections) -> List[dict]:
        seen = {}
        for d in detections or []:
            if d.label not in seen or d.confidence > seen[d.label].confidence:
                seen[d.label] = d
        for lab in list(self.state):
            if lab not in seen:
                st = self.state[lab]
                st["miss"] += 1
                if not st["shown"]:
                    st["hits"] = 0
                if st["miss"] > self.hold:
                    del self.state[lab]
        for lab, d in seen.items():
            b = np.array(d.bbox, dtype=np.float64)
            st = self.state.get(lab)
            if st is None:
                self.state[lab] = {"box": b, "hits": 1, "miss": 0, "conf": float(d.confidence), "shown": False}
            else:
                st["box"] = self.ema * b + (1 - self.ema) * st["box"]
                st["hits"] += 1
                st["miss"] = 0
                st["conf"] = 0.5 * st["conf"] + 0.5 * float(d.confidence)
        out = []
        for lab, st in self.state.items():
            if st["hits"] >= self.appear:
                st["shown"] = True
            if st["shown"]:
                out.append({"label": lab, "box": st["box"], "conf": st["conf"],
                            "alpha": 1.0 if st["miss"] == 0 else max(0.25, 1 - st["miss"] / (self.hold + 1))})
        return out


# ── 2D overlay ───────────────────────────────────────────────────────────

def _side_color(a: int, b: int):
    if a in LEFT and b in LEFT:
        return C_LEFT
    if a in RIGHT and b in RIGHT:
        return C_RIGHT
    return C_MID


def _body_regions(cnts, skel, w, h, need: int = 3):
    """Contours (in a w x h frame) that contain at least `need` visible core
    joints of `skel`; without a skeleton, the single largest region."""
    if not cnts:
        return []
    if skel is None:
        return [max(cnts, key=cv2.contourArea)]
    p = np.asarray(skel).reshape(33, 4)
    pts = [(float(p[i, 0] * w), float(p[i, 1] * h)) for i in JOINTS_2D if p[i, 3] >= 0.5]
    return [c for c in cnts if sum(cv2.pointPolygonTest(c, q, False) >= 0 for q in pts) >= need]


def draw_body_overlay(vis: np.ndarray, skel: Optional[np.ndarray],
                      mask: Optional[np.ndarray] = None, min_vis: float = 0.5) -> None:
    """Segmentation tint + outline (covers the whole suited body, not just
    joint points) and the smoothed skeleton, in place."""
    h, w = vis.shape[:2]
    if mask is not None:
        # Work at the mask's own (pose-input) resolution - blurring and
        # contouring the full frame cost ~20 ms/frame; only the final masked
        # copy and the scaled contour touch full-resolution pixels.
        mh, mw = mask.shape[:2]
        body = (cv2.GaussianBlur(mask, (0, 0), 1.5) > 0.5).astype(np.uint8)
        # MediaPipe's mask occasionally lands on a cable or the lid instead of
        # the crew member: keep only regions that hold the skeleton's joints.
        if body.any():
            raw, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            keep = _body_regions(raw, skel, mw, mh)
            body = np.zeros_like(body)
            if keep:
                cv2.drawContours(body, keep, -1, 1, -1)
        if body.any():
            sx, sy = w / float(mw), h / float(mh)
            bx, by, bw, bh = cv2.boundingRect(body)          # tint only the body's box
            x0, y0 = int(bx * sx), int(by * sy)
            x1, y1 = min(w, int((bx + bw) * sx) + 1), min(h, int((by + bh) * sy) + 1)
            roi = vis[y0:y1, x0:x1]
            big = cv2.resize(body[by:by + bh, bx:bx + bw], (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            tinted = cv2.addWeighted(roi, 0.82, np.full_like(roi, (200, 160, 30)), 0.18, 0)
            cv2.copyTo(tinted, big, roi)
            cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [(c.astype(np.float32) * (sx, sy)).astype(np.int32)
                    for c in cnts if cv2.contourArea(c) > 0.004 * mw * mh]
            cnts = _body_regions(cnts, skel, w, h)
            cv2.drawContours(vis, cnts, -1, (235, 205, 60), 2, cv2.LINE_AA)
    if skel is None:
        return
    p = np.asarray(skel).reshape(33, 4)
    pts = np.c_[p[:, 0] * w, p[:, 1] * h].astype(np.int32)
    for a, b in BONES_2D:
        if p[a, 3] < min_vis or p[b, 3] < min_vis:
            continue
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), (20, 20, 20), 6, cv2.LINE_AA)
        cv2.line(vis, tuple(pts[a]), tuple(pts[b]), _side_color(a, b), 3, cv2.LINE_AA)
    for i in JOINTS_2D:
        if p[i, 3] < min_vis:
            continue
        cv2.circle(vis, tuple(pts[i]), 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pts[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_boxes(vis: np.ndarray, boxes: List[dict], task_labels: Sequence[str] = ()) -> None:
    """Corner-bracket boxes with a label chip; the box the current step is
    about is drawn heavier."""
    for bx in boxes:
        x1, y1, x2, y2 = [int(v) for v in bx["box"]]
        col = C_BOX.get(bx["label"], (255, 255, 255))
        task = bx["label"] in task_labels
        a = bx["alpha"]
        col = tuple(int(c * a + 30 * (1 - a)) for c in col)
        thick = 3 if task else 2
        L = max(10, int(0.22 * min(x2 - x1, y2 - y1)))
        over = vis.copy()
        cv2.rectangle(over, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        cv2.addWeighted(over, 0.5, vis, 0.5, 0, vis)
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(vis, (cx, cy), (cx + dx * L, cy), col, thick, cv2.LINE_AA)
            cv2.line(vis, (cx, cy), (cx, cy + dy * L), col, thick, cv2.LINE_AA)
        text = f"{BOX_NAME.get(bx['label'], bx['label'])}  {bx['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        ty = y1 - 8 if y1 > 24 else y2 + th + 10
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), col, -1)
        cv2.putText(vis, text, (x1 + 5, ty - 1), FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


# ── 3D scene ─────────────────────────────────────────────────────────────

class Camera3D:
    """Fixed oblique view, perspective. World: MediaPipe metric frame
    converted to y-up (X right, Y up, Z toward viewer)."""

    def __init__(self, w: int, h: int, yaw_deg: float = -32.0, pitch_deg: float = 14.0,
                 dist: float = 3.6, height_m: float = 2.3):
        self.w, self.h, self.dist = w, h, dist
        y, p = math.radians(yaw_deg), math.radians(pitch_deg)
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        self.R = rx @ ry
        self.f = 0.86 * h * dist / height_m
        self.center = np.zeros(3)
        self._init = False

    def frame(self, pts: np.ndarray, fill: float = 0.78) -> None:
        """Ease look-at point and zoom so body + boxes fill the view: the
        scene is re-framed smoothly, never jumps, never leaves the panel."""
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        target = 0.5 * (lo + hi)
        radius = max(0.5 * float(np.linalg.norm(hi - lo)), 0.5)
        f_target = fill * 0.5 * min(self.w, self.h) * (self.dist - radius) / radius
        if not self._init:
            self.center, self.f, self._init = target.copy(), f_target, True
        else:
            self.center = 0.85 * self.center + 0.15 * target
            self.f = 0.9 * self.f + 0.1 * f_target

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = (np.asarray(P, dtype=np.float64) - self.center) @ self.R.T
        d = np.maximum(self.dist - q[:, 2], 0.3)
        u = self.w / 2 + self.f * q[:, 0] / d
        v = self.h / 2 - self.f * q[:, 1] / d
        return np.c_[u, v], d


def to_yup(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.c_[xyz[:, 0], -xyz[:, 1], -xyz[:, 2]]


def place_objects(skel2d: Optional[np.ndarray], world_yup: np.ndarray, boxes: List[dict],
                  frame_wh: Tuple[int, int]) -> List[dict]:
    """Put each detected box into the body's 3D frame.

    x/y come from an affine fit image-pixels -> body-metres over the joints
    visible in both the 2D and 3D skeletons (the same body in both), so a box
    at the astronaut's right hand lands at the 3D right hand. Depth: a box
    within reach of a wrist takes that wrist's depth (held / being handled);
    otherwise it sits at the body plane. An approximation, labelled as such
    on screen: one camera measures no box depth."""
    if skel2d is None or not boxes:
        return []
    fw, fh = frame_wh
    p2 = np.asarray(skel2d).reshape(33, 4)
    ok = np.where(p2[:, 3] > 0.5)[0]
    if len(ok) < 6:
        return []
    uv = np.c_[p2[ok, 0] * fw, p2[ok, 1] * fh, np.ones(len(ok))]
    A, *_ = np.linalg.lstsq(uv, world_yup[ok, :2], rcond=None)      # (3, 2)
    scale = math.sqrt(abs(np.linalg.det(A[:2, :2]))) or 1e-3
    sh = (p2[L_SH, :2] + p2[R_SH, :2]) / 2 * (fw, fh)
    hp = (p2[L_HIP, :2] + p2[R_HIP, :2]) / 2 * (fw, fh)
    torso_px = float(np.linalg.norm(sh - hp))
    wr_px = np.c_[p2[[L_WR, R_WR], 0] * fw, p2[[L_WR, R_WR], 1] * fh]
    out = []
    for bx in boxes:
        x1, y1, x2, y2 = bx["box"]
        cu, cv_ = (x1 + x2) / 2, (y1 + y2) / 2
        X, Y = np.array([cu, cv_, 1.0]) @ A
        sx, sy = (x2 - x1) * scale, (y2 - y1) * scale
        sz = 0.8 * min(sx, sy)
        dist = np.linalg.norm(wr_px - [cu, cv_], axis=1)
        near = int(np.argmin(dist))
        held = bool(dist[near] < 0.6 * max(torso_px, 1.0) and bx["label"] != "main_box")
        Z = world_yup[[L_WR, R_WR][near], 2] if held else 0.0
        out.append({**bx, "center": np.array([X, Y, Z]), "size": np.array([sx, sy, sz]),
                    "hand": [L_WR, R_WR][near] if held else None})
    return out


def _cube(center, size):
    c, s = np.asarray(center), np.asarray(size) / 2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return corners, faces


def render_scene_3d(canvas: np.ndarray, joints_yup: Optional[np.ndarray], objects: List[dict],
                    cam: Camera3D, task_labels: Sequence[str] = (), caption: str = "") -> None:
    """Deck grid, shadow, depth-sorted body and boxes, in place on `canvas`."""
    h, w = canvas.shape[:2]
    canvas[:] = (PANEL[0] - 8, PANEL[1] - 6, PANEL[2] - 4)
    if joints_yup is None:
        cv2.putText(canvas, "waiting for a body...", (w // 2 - 95, h // 2), FONT, 0.5, MUTED, 1, cv2.LINE_AA)
        if caption:
            cv2.putText(canvas, caption, (10, h - 10), FONT, 0.4, MUTED, 1, cv2.LINE_AA)
        return
    J = np.asarray(joints_yup)
    vis = J[:, 3] if J.shape[1] > 3 else np.ones(len(J))
    P = _with_virtual(J)
    feet_y = float(np.min(P[[L_AN, R_AN, L_FT, R_FT], 1]))
    floor = feet_y - 0.03
    frame_pts = [P, np.array([[P[HIP_MID, 0], floor, P[HIP_MID, 2]], P[NOSE] + [0, 0.15, 0]])]
    for obj in objects:
        frame_pts.append(_cube(obj["center"], obj["size"])[0])
    cam.frame(np.vstack(frame_pts))

    # deck grid around the body
    g = np.linspace(-1.2, 1.2, 9)
    for k in g:
        for a, b in (((k, floor, -1.2), (k, floor, 1.2)), ((-1.2, floor, k), (1.2, floor, k))):
            seg = np.array([a, b]) + [cam.center[0], 0, cam.center[2]]
            (pa, pb), _ = cam.project(seg)
            cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), (70, 58, 44), 1, cv2.LINE_AA)
    # soft shadow
    ang = np.linspace(0, 2 * np.pi, 40)
    ring = np.c_[P[HIP_MID, 0] + 0.35 * np.cos(ang), np.full(40, floor), P[HIP_MID, 2] + 0.22 * np.sin(ang)]
    sp, _ = cam.project(ring)
    over = canvas.copy()
    cv2.fillPoly(over, [sp.astype(np.int32)], (12, 10, 8), cv2.LINE_AA)
    cv2.addWeighted(over, 0.55, canvas, 0.45, 0, canvas)

    uv, depth = cam.project(P)
    items = []
    for p_, c_ in TREE:
        if c_ in (L_EAR, R_EAR):
            continue
        items.append((0.5 * (depth[p_] + depth[c_]), "bone", (p_, c_)))
    for obj in objects:
        corners, faces = _cube(obj["center"], obj["size"])
        cuv, cd = cam.project(corners)
        items.append((float(cd.mean()), "cube", (obj, cuv, cd, faces)))
    items.sort(key=lambda it: -it[0])

    quad = uv[[L_SH, R_SH, R_HIP, L_HIP]].astype(np.int32)
    over = canvas.copy()
    cv2.fillPoly(over, [quad], (150, 140, 130), cv2.LINE_AA)
    cv2.addWeighted(over, 0.8, canvas, 0.2, 0, canvas)

    d_min, d_max = float(depth.min()), float(depth.max()) + 1e-6
    for d, kind, pay in items:
        shade = 1.0 - 0.35 * (d - d_min) / (d_max - d_min)
        if kind == "bone":
            a, b = pay
            if a in (HIP_MID, SH_MID):
                col = C_MID
            else:
                col = _side_color(a, b)
            col = tuple(int(c * shade) for c in col)
            va = vis[a] if a < 33 else 1.0
            vb = vis[b] if b < 33 else 1.0
            if min(va, vb) < 0.35:
                col = tuple(int(c * 0.45) for c in col)
            th = int(np.clip(cam.f * 0.08 / d, 3, 16))     # ~8 cm limbs: reads as a body, not wires
            pa, pb = tuple(uv[a].astype(int)), tuple(uv[b].astype(int))
            cv2.line(canvas, pa, pb, col, th, cv2.LINE_AA)
            cv2.circle(canvas, pb, max(th // 2, 2), col, -1, cv2.LINE_AA)
        else:
            obj, cuv, cd, faces = pay
            col = C_BOX.get(obj["label"], (220, 220, 220))
            task = obj["label"] in task_labels
            solid = obj["label"] != "main_box"
            over = canvas.copy()
            for f in sorted(faces, key=lambda f: -cd[list(f)].mean()):
                cv2.fillPoly(over, [cuv[list(f)].astype(np.int32)], tuple(int(c * 0.55) for c in col), cv2.LINE_AA)
            cv2.addWeighted(over, 0.6 if solid else 0.3, canvas, 0.4 if solid else 0.7, 0, canvas)
            for f in faces:
                cv2.polylines(canvas, [cuv[list(f)].astype(np.int32)], True, col, 2 if task else 1, cv2.LINE_AA)
            if obj.get("hand") is not None:
                hp = tuple(uv[obj["hand"]].astype(int))
                cp = tuple(cuv.mean(axis=0).astype(int))
                cv2.line(canvas, hp, cp, col, 2 if task else 1, cv2.LINE_AA)
    # head: helmet sphere with a visor, drawn last (always nearest-ish)
    head_c = 0.5 * (P[L_EAR] + P[R_EAR]) if min(vis[L_EAR], vis[R_EAR]) > 0.2 else P[NOSE]
    (hc,), (hd,) = cam.project(head_c[None])
    r = int(np.clip(cam.f * 0.13 / hd, 6, 34))
    cv2.circle(canvas, tuple(hc.astype(int)), r, (215, 215, 215), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(hc.astype(int)), r, (120, 120, 120), 1, cv2.LINE_AA)
    (vc,), _ = cam.project((0.35 * P[NOSE] + 0.65 * head_c)[None])
    cv2.ellipse(canvas, tuple(vc.astype(int)), (max(r // 2, 3), max(r // 3, 2)), 0, 0, 360,
                (90, 70, 40), -1, cv2.LINE_AA)
    if caption:
        cv2.putText(canvas, caption, (10, h - 10), FONT, 0.4, MUTED, 1, cv2.LINE_AA)


# ── side panel ───────────────────────────────────────────────────────────

def _text(img, s, org, scale=0.5, col=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, col, thick, cv2.LINE_AA)


def render_side_panel(panel: np.ndarray, status: dict, pred_step: int, pred_conf: float,
                      fps: Optional[float], latency_ms: Optional[float], scene_fn,
                      banner: str = "", alert: str = "") -> None:
    """Right-hand monitor column: header, 3D view (drawn by scene_fn on a
    sub-canvas), current/next step with confidence, the 8-step checklist."""
    h, w = panel.shape[:2]
    panel[:] = BG
    _text(panel, "BAS-HAR MONITOR", (16, 28), 0.62, INK, 2)
    if fps is not None:
        s = f"{fps:4.1f} FPS  {latency_ms or 0:4.0f} ms"
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.45, 1)
        _text(panel, s, (w - tw - 16, 27), 0.45, (130, 220, 130))
    y0 = 42
    view_h = int(h * 0.42)
    view = panel[y0:y0 + view_h, 10:w - 10]
    scene_fn(view)
    cv2.rectangle(panel, (10, y0), (w - 11, y0 + view_h), (70, 60, 50), 1)
    _text(panel, "3D POSE + OBJECTS", (20, y0 + 20), 0.45, (40, 200, 255), 1)

    y = y0 + view_h + 12
    cur, nxt = status.get("current_step"), status.get("next_step")
    card_h = 92
    cv2.rectangle(panel, (10, y), (w - 10, y + card_h), PANEL, -1)
    if cur:
        _text(panel, f"STEP {cur['id']} OF 8", (22, y + 22), 0.45, CUR_C, 1)
        _text(panel, cur["name"], (22, y + 52), 0.72, INK, 2)
    else:
        _text(panel, "PROTOCOL COMPLETE", (22, y + 46), 0.7, OK_C, 2)
    if nxt:
        _text(panel, f"NEXT  {nxt['id']} . {nxt['name']}", (22, y + 80), 0.45, MUTED, 1)
    bx0, bx1, by = w - 150, w - 22, y + 18
    cv2.rectangle(panel, (bx0, by - 8), (bx1, by + 2), (60, 50, 40), -1)
    cv2.rectangle(panel, (bx0, by - 8), (bx0 + int((bx1 - bx0) * max(0.0, min(1.0, pred_conf))), by + 2),
                  OK_C if pred_conf >= 0.65 else CUR_C, -1)
    _text(panel, f"model: step {pred_step}  {pred_conf:.2f}", (bx0, by + 20), 0.38, MUTED, 1)

    y += card_h + 12
    steps = status.get("steps", [])
    rec = status.get("recovery", {}) or {}
    bottom = h - (64 if alert else 30)
    row = max(18, min(26, (bottom - y) // max(len(steps), 1)))
    cur_id = cur["id"] if cur else None
    for st in steps:
        sid, name, s = st["id"], st["name"], st["status"]
        if s == "COMPLETED":
            col, mark = OK_C, "done"
        elif s == "IN_PROGRESS" or sid == cur_id:
            col, mark = CUR_C, "now"
            cv2.rectangle(panel, (12, y + 1), (w - 12, y + row - 1), (48, 40, 30), -1)
        elif s == "SKIPPED" or (rec.get("active") and sid == rec.get("expected_step_id")):
            col, mark = BAD_C, "missed"
        else:
            col, mark = (110, 110, 110), ""
        cy = y + row // 2
        cv2.circle(panel, (26, cy), 7, col, -1 if s in ("COMPLETED", "IN_PROGRESS") else 2, cv2.LINE_AA)
        _text(panel, f"{sid}  {name}", (44, cy + 5), 0.47, INK if (s != "PENDING" or sid == cur_id) else MUTED, 1)
        if mark:
            (tw, _), _ = cv2.getTextSize(mark, FONT, 0.4, 1)
            _text(panel, mark, (w - tw - 20, cy + 5), 0.4, col, 1)
        y += row
    if alert:
        warn = any(k in alert.lower() for k in ("skip", "missed", "sequence", "return", "wrong", "hold"))
        cv2.rectangle(panel, (10, h - 58), (w - 10, h - 30), (40, 30, 150) if warn else (95, 70, 20), -1)
        scale = 0.43
        while cv2.getTextSize(alert, FONT, scale, 1)[0][0] > w - 36 and scale > 0.3:
            scale -= 0.02
        _text(panel, alert, (18, h - 39), scale, (230, 230, 255) if warn else (240, 235, 220), 1)
    if banner:
        scale = 0.36
        while cv2.getTextSize(banner, FONT, scale, 1)[0][0] > w - 24 and scale > 0.26:
            scale -= 0.01
        _text(panel, banner, (12, h - 10), scale, (120, 120, 120), 1)


if __name__ == "__main__":
    # Self-checks: filter settles, rigid bones keep length, renderers run.
    f = OneEuro()
    x = None
    for k in range(60):
        x = f(np.array([1.0 + 0.01 * ((-1) ** k)]), k / 24)
    assert abs(float(x[0]) - 1.0) < 0.01, x

    rng = np.random.default_rng(0)
    base = np.zeros((33, 4)); base[:, 3] = 1.0
    base[[L_SH, R_SH], :3] = [[-0.2, -0.5, 0], [0.2, -0.5, 0]]
    base[[L_HIP, R_HIP], :3] = [[-0.1, 0, 0], [0.1, 0, 0]]
    base[[L_EL, R_EL], :3] = [[-0.3, -0.25, 0], [0.3, -0.25, 0]]
    base[[L_WR, R_WR], :3] = [[-0.35, 0.0, 0], [0.35, 0.0, 0]]
    base[[L_KN, R_KN], :3] = [[-0.1, 0.45, 0], [0.1, 0.45, 0]]
    base[[L_AN, R_AN], :3] = [[-0.1, 0.85, 0], [0.1, 0.85, 0]]
    base[NOSE, :3] = [0, -0.72, -0.05]
    rig = RigidSkeleton3D()
    lengths = []
    for k in range(80):
        noisy = base.copy(); noisy[:, :3] += rng.normal(0, 0.02, (33, 3))
        out = rig(noisy, k / 24)
        lengths.append(np.linalg.norm(out[L_EL, :3] - out[L_SH, :3]))
    late = np.array(lengths[40:])
    assert late.std() / late.mean() < 0.02, late.std() / late.mean()

    img = np.zeros((360, 400, 3), np.uint8)
    render_scene_3d(img, np.c_[to_yup(out[:, :3]), out[:, 3]], [], Camera3D(400, 360))
    assert img.any()
    panel = np.zeros((720, 420, 3), np.uint8)
    render_side_panel(panel, {"current_step": {"id": 1, "name": "Approach Main Box"}, "next_step": None,
                              "steps": [{"id": 1, "name": "Approach Main Box", "status": "IN_PROGRESS"}]},
                      1, 0.9, 15.0, 66.0, lambda v: v.__setitem__(slice(None), 40))
    assert panel.any()
    print("pose_viz self-checks passed")
