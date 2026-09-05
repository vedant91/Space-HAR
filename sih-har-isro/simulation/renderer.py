"""
ISS / Bharatiya Antariksh Station payload-rack renderer.

Draws a 2D spacecraft science-module scene with:
  - beige equipment racks, handrails, fluorescent lighting
  - white main experiment container + red / yellow inner boxes
  - a suited humanoid whose joints follow a MediaPipe-style pose
  - floating particles and slow camera drift (microgravity)

Used to (a) train the CNN on labeled step frames and (b) drive the
headless HAR pipeline for latency / HSV / sequence tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_generation.synthetic_pose import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    NUM_LANDMARKS,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    generate_sequence,
)

# Saturated BGR colors that sit inside config HSV ranges
RED_BGR = (12, 12, 230)       # hue ~0, high S/V  → HSV_RED
YELLOW_BGR = (0, 255, 255)    # hue 30, sat 255   → HSV_YELLOW
WHITE_BGR = (236, 236, 236)   # low S, high V     → HSV_WHITE
SUIT_BGR = (232, 232, 228)
SKIN_BGR = (110, 155, 200)
VISOR_BGR = (40, 30, 20)
RACK_BGR = (92, 98, 108)
PANEL_BGR = (118, 122, 128)

POSE_EDGES = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]


@dataclass
class FloatingPayload:
    """A deliberately small, deterministic microgravity dynamics model.

    It is a visual/test double, not a claim of flight-qualified orbital-fluid
    physics.  It models the behaviours that matter to HAR: an item follows a
    hand while grasped, keeps drifting after release, and is arrested only when
    it reaches a restrained rack zone.
    """
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    held_by: Optional[str] = None
    latched: bool = False


class MicrogravityWorld:
    """Persistent payload and particle state shared by a rendered protocol."""

    def __init__(self, width: int, height: int, rng: np.random.Generator):
        self.width, self.height, self.rng = width, height, rng
        self.red = FloatingPayload(np.array([0.45, 0.64], dtype=np.float32))
        self.yellow = FloatingPayload(np.array([0.55, 0.64], dtype=np.float32))
        self.main_open = False
        self.frame = 0
        self.particles = np.column_stack((
            rng.uniform(0.13, 0.87, 48), rng.uniform(0.08, 0.88, 48),
            rng.uniform(-0.0012, 0.0012, 48), rng.uniform(-0.0008, 0.0008, 48),
        )).astype(np.float32)

    @staticmethod
    def _wrist(pose: np.ndarray, side: str) -> np.ndarray:
        idx = L_WRIST if side == "L" else R_WRIST
        return pose[idx, :2].astype(np.float32).copy()

    def _release_toward(self, item: FloatingPayload, target: Tuple[float, float]):
        item.held_by = None
        item.latched = False
        direction = np.array(target, dtype=np.float32) - item.position
        item.velocity = 0.035 * direction + self.rng.normal(0, 0.0015, 2).astype(np.float32)

    def _integrate(self, item: FloatingPayload, pose: np.ndarray, dt: float):
        if item.held_by:
            desired = self._wrist(pose, item.held_by)
            item.velocity = (desired - item.position) / max(dt, 1e-4)
            item.position = desired
            return
        if item.latched:
            item.velocity[:] = 0
            return
        # Near-zero gravity: no downward term; tiny air/vent disturbance and
        # very low damping make a released item continue to drift.
        disturbance = np.array([
            0.00018 * np.sin(self.frame * 0.07),
            0.00012 * np.cos(self.frame * 0.05),
        ], dtype=np.float32)
        item.velocity += disturbance
        item.position += item.velocity * dt * 30.0
        item.velocity *= 0.992
        for axis, low, high in ((0, 0.14, 0.86), (1, 0.10, 0.84)):
            if item.position[axis] < low or item.position[axis] > high:
                item.position[axis] = np.clip(item.position[axis], low, high)
                item.velocity[axis] *= -0.55

    def advance(self, step_id: int, t: float, pose: np.ndarray, dt: float = 1 / 30):
        self.frame += 1
        self.particles[:, :2] += self.particles[:, 2:] * (dt * 30.0)
        self.particles[:, :2] = np.mod(self.particles[:, :2] - 0.02, 0.96) + 0.02

        if step_id >= 2:
            self.main_open = True
        if step_id in (3, 4):
            self.red.held_by, self.red.latched = "L", False
        elif step_id == 5:
            if t < 0.52:
                self.red.held_by, self.red.latched = "L", False
            elif self.red.held_by:
                self._release_toward(self.red, (0.22, 0.66))
            if t > 0.86:
                self.red.position = np.array([0.22, 0.66], dtype=np.float32)
                self.red.latched = True

        if step_id in (6, 7):
            self.yellow.held_by, self.yellow.latched = "R", False
        elif step_id == 8:
            if t < 0.52:
                self.yellow.held_by, self.yellow.latched = "R", False
            elif self.yellow.held_by:
                self._release_toward(self.yellow, (0.78, 0.66))
            if t > 0.86:
                self.yellow.position = np.array([0.78, 0.66], dtype=np.float32)
                self.yellow.latched = True

        self._integrate(self.red, pose, dt)
        self._integrate(self.yellow, pose, dt)

    def bbox(self, item: FloatingPayload, size: int = 64) -> Tuple[int, int, int, int]:
        cx, cy = int(item.position[0] * self.width), int(item.position[1] * self.height)
        half = size // 2
        return (cx - half, cy - half, cx + half, cy + half)


def _box_layout(step_id: int, t: float, pose: np.ndarray, w: int, h: int) -> Dict[str, Tuple[int, int, int, int]]:
    """Return pixel bboxes for main/red/yellow given protocol time t in [0,1]."""
    main = (int(w * 0.38), int(h * 0.52), int(w * 0.62), int(h * 0.78))
    # Default: both inner boxes nested in the main container
    red = (int(w * 0.41), int(h * 0.58), int(w * 0.49), int(h * 0.70))
    yel = (int(w * 0.51), int(h * 0.58), int(w * 0.59), int(h * 0.70))
    left_zone = (int(w * 0.16), int(h * 0.58), int(w * 0.28), int(h * 0.74))
    right_zone = (int(w * 0.72), int(h * 0.58), int(w * 0.84), int(h * 0.74))

    def wrist_box(side: str, size: int = 96):
        idx = L_WRIST if side == "L" else R_WRIST
        cx = int(pose[idx, 0] * w)
        cy = int(pose[idx, 1] * h)
        s = size // 2
        return (cx - s, cy - s, cx + s, cy + s)

    if step_id == 1:
        # Lid closed — hide inner boxes by overlapping main (still painted under lid)
        pass
    elif step_id == 2:
        pass
    elif step_id == 3:
        red = wrist_box("L", 64)
    elif step_id == 4:
        red = wrist_box("L", 64)
    elif step_id == 5:
        if t < 0.6:
            red = wrist_box("L", 64)
        else:
            red = left_zone
    elif step_id == 6:
        yel = wrist_box("R", 64)
        red = left_zone
    elif step_id == 7:
        yel = wrist_box("R", 64)
        red = left_zone
    elif step_id == 8:
        red = left_zone
        if t < 0.6:
            yel = wrist_box("R", 64)
        else:
            yel = right_zone

    return {"main_box": main, "red_box": red, "yellow_box": yel,
            "left_zone": left_zone, "right_zone": right_zone}


def _fill_rect(img, box, color, thickness=-1):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_module_background(frame: np.ndarray, rng: np.random.Generator,
                            particles: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    # Vertical gradient: darker ceiling, cooler floor
    for y in range(h):
        c = int(38 + 28 * (y / h))
        frame[y, :] = (c + 8, c + 6, c)

    # Equipment racks left / right
    cv2.rectangle(frame, (0, 40), (int(w * 0.12), h - 20), RACK_BGR, -1)
    cv2.rectangle(frame, (int(w * 0.88), 40), (w, h - 20), RACK_BGR, -1)
    for x in (int(w * 0.03), int(w * 0.08), int(w * 0.91), int(w * 0.96)):
        cv2.line(frame, (x, 50), (x, h - 30), (70, 74, 80), 3)

    # Handrails
    cv2.rectangle(frame, (int(w * 0.14), int(h * 0.18)), (int(w * 0.86), int(h * 0.22)),
                  (160, 160, 155), -1)
    cv2.rectangle(frame, (int(w * 0.14), int(h * 0.86)), (int(w * 0.86), int(h * 0.90)),
                  (160, 160, 155), -1)

    # Fluorescent strips
    cv2.rectangle(frame, (int(w * 0.2), 8), (int(w * 0.8), 22), (210, 215, 220), -1)

    # Floating dust
    if particles is None:
        particles = np.column_stack((rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)))
    for x, y in particles[:, :2]:
        x, y = int(x * w), int(y * h)
        cv2.circle(frame, (int(x), int(y)), 1, (180, 185, 190), -1)
    return frame


def _lm_xy(pose: np.ndarray, idx: int, w: int, h: int) -> Tuple[int, int]:
    return int(np.clip(pose[idx, 0], 0, 1) * w), int(np.clip(pose[idx, 1], 0, 1) * h)


def _draw_astronaut(frame: np.ndarray, pose: np.ndarray):
    h, w = frame.shape[:2]
    # Torso
    pts = np.array([
        _lm_xy(pose, L_SHOULDER, w, h),
        _lm_xy(pose, R_SHOULDER, w, h),
        _lm_xy(pose, R_HIP, w, h),
        _lm_xy(pose, L_HIP, w, h),
    ], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, SUIT_BGR)
    cv2.polylines(frame, [pts], True, (200, 200, 195), 2)

    # Limbs as thick suit tubes
    limbs = [
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    ]
    for a, b, c in limbs:
        pa, pb, pc = _lm_xy(pose, a, w, h), _lm_xy(pose, b, w, h), _lm_xy(pose, c, w, h)
        cv2.line(frame, pa, pb, SUIT_BGR, 18)
        cv2.line(frame, pb, pc, SUIT_BGR, 16)
        cv2.circle(frame, pb, 10, SUIT_BGR, -1)

    # Gloves
    cv2.circle(frame, _lm_xy(pose, L_WRIST, w, h), 16, (245, 245, 240), -1)
    cv2.circle(frame, _lm_xy(pose, R_WRIST, w, h), 16, (245, 245, 240), -1)

    # Helmet
    hx, hy = _lm_xy(pose, NOSE, w, h)
    cv2.circle(frame, (hx, hy - 8), 28, SUIT_BGR, -1)
    cv2.circle(frame, (hx, hy - 4), 18, VISOR_BGR, -1)
    cv2.circle(frame, (hx - 6, hy - 8), 4, SKIN_BGR, -1)  # faint face cue
    cv2.circle(frame, (hx + 6, hy - 8), 4, SKIN_BGR, -1)


def render_frame(
    pose_flat: np.ndarray,
    step_id: int,
    t: float,
    width: int = 1280,
    height: int = 720,
    rng: Optional[np.random.Generator] = None,
    closed_lid: Optional[bool] = None,
    world: Optional[MicrogravityWorld] = None,
) -> Tuple[np.ndarray, Dict]:
    """Render one BGR frame. Returns (frame, meta with bboxes and visible flags)."""
    if rng is None:
        rng = np.random.default_rng()
    pose = pose_flat.reshape(NUM_LANDMARKS, 4)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if world is not None:
        world.advance(step_id, t, pose)
    _draw_module_background(frame, rng, world.particles if world is not None else None)

    boxes = _box_layout(step_id, t, pose, width, height)
    if world is not None:
        boxes["red_box"] = world.bbox(world.red)
        boxes["yellow_box"] = world.bbox(world.yellow)
    if closed_lid is None:
        closed_lid = not world.main_open if world is not None else step_id <= 1

    # Placement zone outlines
    _fill_rect(frame, boxes["left_zone"], (50, 50, 90), 2)
    _fill_rect(frame, boxes["right_zone"], (50, 90, 90), 2)

    _fill_rect(frame, boxes["main_box"], WHITE_BGR, -1)
    cv2.rectangle(frame, boxes["main_box"][:2], boxes["main_box"][2:], (180, 180, 180), 3)

    red_visible = step_id >= 2 and not closed_lid
    yel_visible = step_id >= 2 and not closed_lid
    if closed_lid:
        x1, y1, x2, y2 = boxes["main_box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE_BGR, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 3)

    _draw_astronaut(frame, pose)

    # Paint payload boxes on top so HSV sees a clean saturated region
    # (held boxes would otherwise be covered by white gloves).
    if not closed_lid and step_id >= 2:
        _fill_rect(frame, boxes["red_box"], RED_BGR, -1)
        _fill_rect(frame, boxes["yellow_box"], YELLOW_BGR, -1)
        cv2.rectangle(frame, boxes["red_box"][:2], boxes["red_box"][2:], (0, 0, 80), 2)
        cv2.rectangle(frame, boxes["yellow_box"][:2], boxes["yellow_box"][2:], (0, 90, 90), 2)

    # Cool lighting vignette
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 40), (200, 210, 220), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    meta = {
        "step_id": int(step_id),
        "t": float(t),
        "boxes": {k: tuple(int(v) for v in box) for k, box in boxes.items()},
        "red_visible": bool(red_visible),
        "yellow_visible": bool(yel_visible),
        "main_visible": True,
        "pose": pose_flat.astype(np.float32),
        "microgravity": {
            "red_velocity": world.red.velocity.round(5).tolist() if world else [0.0, 0.0],
            "yellow_velocity": world.yellow.velocity.round(5).tolist() if world else [0.0, 0.0],
            "red_latched": bool(world.red.latched) if world else False,
            "yellow_latched": bool(world.yellow.latched) if world else False,
        },
    }
    return frame, meta


def render_step_clip(
    step_id: int,
    n_frames: int = 45,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    orientation: int = 0,
) -> Tuple[List[np.ndarray], List[dict]]:
    rng = np.random.default_rng(seed + step_id * 17)
    world = MicrogravityWorld(width, height, rng)
    seq = generate_sequence(step_id, n_frames=n_frames, rng=rng, orientation=orientation, drift=True)
    frames, metas = [], []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frame, meta = render_frame(seq[i], step_id, t, width, height, rng=rng, world=world)
        frames.append(frame)
        metas.append(meta)
    return frames, metas


def render_protocol(
    frames_per_step: int = 45,
    width: int = 1280,
    height: int = 720,
    seed: int = 1,
    skip_step: Optional[int] = None,
    sequence: Optional[List[int]] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """Render a persistent microgravity protocol or an injected-error sequence."""
    frames, metas = [], []
    rng = np.random.default_rng(seed)
    world = MicrogravityWorld(width, height, rng)
    sequence = sequence or list(range(1, 9))
    for occurrence, sid in enumerate(sequence):
        if skip_step is not None and sid == skip_step:
            continue
        # Different deterministic body orientation for each recording, while
        # object locations remain rack-relative rather than floor-relative.
        orientation = (occurrence // 3) % 4
        seq = generate_sequence(sid, n_frames=frames_per_step, rng=rng,
                                orientation=orientation, drift=True)
        for i in range(frames_per_step):
            t = i / max(frames_per_step - 1, 1)
            frame, meta = render_frame(seq[i], sid, t, width, height, rng=rng, world=world)
            frames.append(frame)
            metas.append(meta)
    return frames, metas
