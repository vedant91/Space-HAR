"""
The 8-step BAS payload protocol, as keyframed 3-D motion.

This is the piece that makes the whole dataset meaningful. The repo's
existing `data_generation/synthetic_pose.py` defines each step as three 2-D
wrist waypoints in normalised image coordinates and interpolates between
them; a sequence model trained on that is learning to invert a 24-number
closed-form function, which is why its held-out accuracy is ~98% and means
nothing.

Here a step is instead defined by where the hands have to be *in the module*
to physically do the task - reach the container, lift the lid, grasp the red
box at its actual modelled position, carry it to the actual restraint tray.
Arm configuration then falls out of IK against the real skeleton, the body
floats on its own drift curve, and the 2-D landmark trajectory is whatever
the camera happens to see. That trajectory is an *observation* of a
physical process rather than the generator the labels came from, so
measuring a model against it is a real test.

Sequences other than 1..8 are supported so the same generator produces the
error cases the state machine has to catch: a skipped step, and a skip that
is later corrected.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Euler, Quaternion, Vector

from . import common
from . import module as MOD

FPS = 30

# Hand parking position when an arm is not doing anything: loosely on the
# flanking handrail, which is where a crew member's spare hand actually goes.
PARK_L = Vector((-0.56, -0.28, 1.32))
PARK_R = Vector((0.56, -0.28, 1.32))

# Grasp offset: the IK goal is the wrist, but the *palm* is what contacts the
# object, roughly 95 mm further along the hand.
WRIST_BACKOFF = 0.095

# Newton's third law: an arm swung one way rotates the torso the other way,
# and the foot restraint bleeds that rotation back out over a second or so.
# ponytail: point-mass arms about a fixed body centre, not a multibody sim;
# good for a few degrees of physically-signed counter-rotation, no more.
ARM_MASS_KG = 4.0
BODY_INERTIA_KGM2 = 12.0
RESTRAINT_DECAY = 0.97

# Newton's first law: a box let go of mid-air keeps its velocity, less a
# little cabin-air drag per frame.
CABIN_DRAG = 0.995

# Peak hand speed, ~0.9 m/s at 30 fps: a deliberate reach from a restraint.
# Smootherstep peaks at 1.875x its mean speed, which the reach sizing uses.
HAND_PEAK_M_PER_FRAME = 0.03


def _lid_open_angle() -> float:
    return math.radians(-104.0)


def _ease(t: float) -> float:
    """Smootherstep. Real limb motion has zero velocity *and* zero
    acceleration at the endpoints; plain smoothstep still starts with a
    velocity discontinuity that shows up as a visible tick in the rendered
    trajectory (and as an impulse in the derived features)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: Vector, b: Vector, t: float) -> Vector:
    return a + (b - a) * t


class ProtocolAnimator:
    """Keyframes the crew rig, the payload and the container lid."""

    def __init__(self, crew: Dict, payload: Dict, rng: random.Random,
                 frames_per_step: int = 72):
        self.crew = crew
        self.payload = payload
        self.rng = rng
        self.fps = frames_per_step

        self.goal_l = crew["ik"]["hand.L"]
        self.goal_r = crew["ik"]["hand.R"]
        self.root = crew["root"]

        self.container = payload["container"]
        self.lid = payload["lid"]
        self.red = payload["red"]
        self.yellow = payload["yellow"]

        # Per-take jitter: a different crew member on a different day. Small,
        # but enough that two takes are not frame-identical.
        self.jitter = Vector((rng.gauss(0.0, 0.012),
                              rng.gauss(0.0, 0.012),
                              rng.gauss(0.0, 0.010)))
        self.speed = rng.uniform(0.88, 1.14)

        # Per-take performance style: the same eight steps, done the way a
        # different person would do them. Without this every take replays one
        # scripted trajectory and a classifier can only memorise it.
        u = rng.uniform
        self.style = {
            "reach": u(0.15, 0.30),     # share of a step spent getting the hands into position
            "lift": u(0.18, 0.34),      # how high a box is lifted clear of the container (m)
            "pull": u(0.06, 0.14),      # how far it is drawn toward the body (m)
            "inspect": Vector((u(-0.08, 0.08), u(-0.06, 0.04), u(-0.08, 0.06))),
            "arrive": u(0.25, 0.50),    # share of an examine step spent reaching visor height
            "descend": u(0.50, 0.72),   # share of a place step before lowering into the tray
            "release": u(0.72, 0.88),   # when the carrying hand lets go
            "turn": u(0.6, 1.4),        # how much the box is turned while examined
            "two_handed": rng.random() < 0.7,  # steady the box with the spare hand?
        }

        self.red_home = self.red.location.copy()
        self.yellow_home = self.yellow.location.copy()

        self.labels: List[int] = []

    # ── keyframe helpers ────────────────────────────────────────────────

    @staticmethod
    def _key_loc(obj: bpy.types.Object, frame: int, loc: Vector) -> None:
        obj.location = loc
        obj.keyframe_insert("location", frame=frame)

    @staticmethod
    def _key_rot(obj: bpy.types.Object, frame: int, euler: Euler) -> None:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = euler
        obj.keyframe_insert("rotation_euler", frame=frame)

    def _jit(self, v: Vector, scale: float = 1.0) -> Vector:
        return v + self.jitter * scale

    # ── per-step hand trajectories ──────────────────────────────────────

    def _hand_targets(self, step: int, t: float) -> Tuple[Vector, Vector,
                                                          Optional[Vector], Optional[Vector]]:
        """Return (left_wrist, right_wrist, red_pos, yellow_pos) at phase t.

        red_pos / yellow_pos are None when that box is not being carried
        this step (it stays wherever the previous step left it).
        """
        c = MOD.CONTAINER_ORIGIN
        red_slot = self.red_home
        yel_slot = self.yellow_home
        lz, rz = MOD.LEFT_ZONE, MOD.RIGHT_ZONE
        e = _ease(t)

        # Approach vector: the wrist sits back from the palm contact point.
        def wrist_for(target: Vector, back: float = WRIST_BACKOFF) -> Vector:
            return Vector((target.x, target.y - back, target.z + 0.020))

        if step == 1:  # Approach main box
            l = _lerp(PARK_L, wrist_for(Vector((c.x - 0.17, c.y, c.z + 0.05))), e)
            r = _lerp(PARK_R, wrist_for(Vector((c.x + 0.17, c.y, c.z + 0.05))), e)
            return self._jit(l), self._jit(r), None, None

        if step == 2:  # Open the lid: hands take the front edge and swing up
            lid_front = Vector((c.x, c.y - MOD.CONTAINER_SIZE[1] * 0.5,
                                c.z + MOD.CONTAINER_SIZE[2] * 0.5))
            hinge = Vector((c.x, c.y + MOD.CONTAINER_SIZE[1] * 0.5,
                            c.z + MOD.CONTAINER_SIZE[2] * 0.5))
            ang = _lid_open_angle() * e
            radius = (lid_front - hinge)
            swung = hinge + Vector((0.0,
                                    radius.y * math.cos(ang) - radius.z * math.sin(ang),
                                    radius.y * math.sin(ang) + radius.z * math.cos(ang)))
            l = wrist_for(Vector((swung.x - 0.11, swung.y, swung.z)), 0.06)
            r = wrist_for(Vector((swung.x + 0.11, swung.y, swung.z)), 0.06)
            return self._jit(l), self._jit(r), None, None

        st = self.style
        red_lift = Vector((red_slot.x - 0.02, red_slot.y - st["pull"], red_slot.z + st["lift"]))
        yel_lift = Vector((yel_slot.x + 0.02, yel_slot.y - st["pull"], yel_slot.z + st["lift"]))
        ins = st["inspect"]
        red_look = Vector((-0.05, -0.22, 1.50)) + ins
        yel_look = Vector((0.05, -0.22, 1.48)) + Vector((-ins.x, ins.y, ins.z))
        arrive, descend, release, two = st["arrive"], st["descend"], st["release"], st["two_handed"]

        if step == 3:  # Pick the red box (left hand) and lift it clear
            # From wherever the box actually is: home in nominal takes, but a
            # box abandoned mid-air in a recovery sequence is picked up there.
            pos = _lerp(self.red_at, red_lift, e)
            l = wrist_for(pos, 0.075)
            r = _lerp(self._jit(wrist_for(Vector((c.x + 0.17, c.y, c.z + 0.05)))),
                      PARK_R, _ease(max(0.0, (t - 0.55) / 0.45)))
            return self._jit(l), r, pos, None

        if step == 4:  # Examine: bring it to visor height and rotate it
            pos = _lerp(red_lift, red_look, _ease(min(t / arrive, 1.0)))
            l = wrist_for(pos, 0.075)
            r = PARK_R
            if two:  # most crew steady it with the spare hand
                r_in = _ease(min(max((t - 0.15) / 0.35, 0.0), 1.0))
                r = _lerp(PARK_R, Vector((pos.x + 0.15, pos.y + 0.01, pos.z - 0.01)), r_in)
            return self._jit(l), self._jit(r), pos, None

        if step == 5:  # Place the red box in the left restraint tray
            above = Vector((lz.x, lz.y - 0.06, lz.z + 0.20))
            seated = Vector((lz.x, lz.y, lz.z))
            if t < descend:
                pos = _lerp(red_look, above, _ease(t / descend))
            else:
                pos = _lerp(above, seated, _ease((t - descend) / (1.0 - descend)))
            l = wrist_for(pos, 0.075)
            l = _lerp(l, PARK_L, 0.25 * _ease(max(0.0, (t - release) / (1.0 - release))))  # partial withdraw; next reach finishes it
            r0 = Vector((red_look.x + 0.15, red_look.y + 0.01, red_look.z - 0.01)) if two else PARK_R
            r = _lerp(r0, PARK_R, _ease(min(t / 0.4, 1.0)))
            return self._jit(l), self._jit(r), pos, None

        if step == 6:  # Pick the yellow box (right hand)
            pos = _lerp(self.yel_at, yel_lift, e)
            r = wrist_for(pos, 0.075)
            return self._jit(PARK_L), self._jit(r), None, pos

        if step == 7:  # Examine yellow
            pos = _lerp(yel_lift, yel_look, _ease(min(t / arrive, 1.0)))
            r = wrist_for(pos, 0.075)
            l = PARK_L
            if two:
                l_in = _ease(min(max((t - 0.15) / 0.35, 0.0), 1.0))
                l = _lerp(PARK_L, Vector((pos.x - 0.15, pos.y + 0.01, pos.z - 0.01)), l_in)
            return self._jit(l), self._jit(r), None, pos

        if step == 8:  # Place yellow in the right restraint tray
            above = Vector((rz.x, rz.y - 0.06, rz.z + 0.20))
            seated = Vector((rz.x, rz.y, rz.z))
            if t < descend:
                pos = _lerp(yel_look, above, _ease(t / descend))
            else:
                pos = _lerp(above, seated, _ease((t - descend) / (1.0 - descend)))
            r = wrist_for(pos, 0.075)
            r = _lerp(r, PARK_R, 0.25 * _ease(max(0.0, (t - release) / (1.0 - release))))  # partial withdraw; next reach finishes it
            l0 = Vector((yel_look.x - 0.15, yel_look.y + 0.01, yel_look.z - 0.01)) if two else PARK_L
            l = _lerp(l0, PARK_L, _ease(min(t / 0.4, 1.0)))
            return self._jit(l), self._jit(r), None, pos

        return PARK_L, PARK_R, None, None

    # ── body float ──────────────────────────────────────────────────────

    def _body_drift(self, frame_abs: int, base_loc: Vector,
                    base_rot: Euler) -> Tuple[Vector, Euler]:
        """Low-frequency six-axis drift of the torso.

        Amplitudes are small (~2 cm, ~3 deg) and the periods are mutually
        irrational so the motion never visibly loops. This is the component
        that makes the rendered pose sequence non-trivial: the same nominal
        arm motion produces a different landmark trajectory every take.
        """
        s = frame_abs / float(FPS)
        loc = base_loc + Vector((
            0.020 * math.sin(s * 0.41 + self.phase[0]),
            0.014 * math.sin(s * 0.29 + self.phase[1]),
            0.022 * math.sin(s * 0.37 + self.phase[2]),
        ))
        rot = Euler((
            base_rot.x + math.radians(2.4) * math.sin(s * 0.33 + self.phase[3]),
            base_rot.y + math.radians(3.1) * math.sin(s * 0.23 + self.phase[4]),
            base_rot.z + math.radians(2.0) * math.sin(s * 0.31 + self.phase[5]),
        ))
        return loc, rot

    # ── main entry ──────────────────────────────────────────────────────

    def animate(self, sequence: Optional[Sequence[int]] = None) -> Dict:
        """Keyframe the whole take. Returns metadata including per-frame labels."""
        sequence = list(sequence or range(1, 9))
        self.phase = [self.rng.uniform(0.0, math.tau) for _ in range(6)]

        base_loc = self.root.location.copy()
        base_rot = Euler(tuple(self.root.rotation_euler))

        red_pos = self.red_home.copy()
        yel_pos = self.yellow_home.copy()
        lid_angle = 0.0
        red_rot = Euler((0.0, 0.0, 0.0))
        yel_rot = Euler((0.0, 0.0, 0.0))
        red_vel, yel_vel = Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))
        red_free = yel_free = False   # released mid-air, i.e. not seated in a tray
        react = Vector((0.0, 0.0, 0.0))
        l_prev, r_prev = PARK_L.copy(), PARK_R.copy()
        reach = self.style["reach"]
        turn = self.style["turn"]

        frame = 1
        self.labels = []
        step_spans: List[Dict] = []

        for occurrence, step in enumerate(sequence):
            n = max(8, int(round(self.fps * self.speed * self.rng.uniform(0.94, 1.06))))
            span_start = frame
            self.red_at, self.yel_at = red_pos.copy(), yel_pos.copy()
            l_from, r_from = l_prev.copy(), r_prev.copy()
            # A long reach takes more of the step, so the hand never exceeds
            # HAND_PEAK_M_PER_FRAME however far it has to travel.
            l0, r0, _, _ = self._hand_targets(step, 0.0)
            dist = max((l0 - l_from).length, (r0 - r_from).length)
            rch = min(0.6, max(reach, 1.875 * dist / (HAND_PEAK_M_PER_FRAME * max(n - 1, 1))))
            for i in range(n):
                u = i / max(n - 1, 1)
                # Reach phase: the hands travel from wherever the previous
                # step left them before this step's motion starts. Without it
                # the IK goal teleports on the first frame of every step,
                # a one-frame spike that marks each label boundary for free.
                t = max(0.0, (u - rch) / (1.0 - rch))
                l, r, rp, yp = self._hand_targets(step, t)
                if u < rch:
                    k = _ease(u / rch)
                    l, r = _lerp(l_from, l, k), _lerp(r_from, r, k)

                self._key_loc(self.goal_l, frame, l)
                self._key_loc(self.goal_r, frame, r)

                # Newton's third law: angular momentum the arms gain, the torso
                # loses (L = m r x v, so d_theta = -(m / I) r x dx per frame).
                react -= (ARM_MASS_KG / BODY_INERTIA_KGM2) * (
                    (l - base_loc).cross(l - l_prev) + (r - base_loc).cross(r - r_prev))
                react *= RESTRAINT_DECAY
                l_prev, r_prev = l, r

                loc, rot = self._body_drift(frame, base_loc, base_rot)
                rot = Euler((rot.x + react.x, rot.y + react.y, rot.z + react.z))
                self._key_loc(self.root, frame, loc)
                self._key_rot(self.root, frame, rot)

                # Lid: opens during step 2 and then stays open. Re-keying the
                # held value every frame (rather than only at the transition)
                # is what keeps an out-of-order sequence like [1, 3, 2, 3, ...]
                # physically coherent - the lid must not spring shut just
                # because the timeline moved on to a step that never mentions it.
                if step == 2:
                    lid_angle = _lid_open_angle() * _ease(t)
                self._key_rot(self.lid, frame, Euler((lid_angle, 0.0, 0.0)))

                if rp is not None:
                    red_vel = rp - red_pos
                    red_pos = rp
                    red_free = step != 5   # only the tray restrains it
                    if step == 4:
                        red_rot = Euler((math.radians(38.0 * turn * math.sin(t * math.tau)),
                                         math.radians(26.0 * turn * math.sin(t * math.tau * 1.3)),
                                         0.0))
                    elif step == 5:
                        red_rot = Euler((red_rot.x * (1.0 - _ease(t)),
                                         red_rot.y * (1.0 - _ease(t)), 0.0))
                elif red_free:
                    # Newton's first law: an abandoned box keeps drifting.
                    red_pos = red_pos + red_vel
                    red_vel = red_vel * CABIN_DRAG
                if yp is not None:
                    yel_vel = yp - yel_pos
                    yel_pos = yp
                    yel_free = step != 8
                    if step == 7:
                        yel_rot = Euler((math.radians(34.0 * turn * math.sin(t * math.tau)),
                                         math.radians(-24.0 * turn * math.sin(t * math.tau * 1.2)),
                                         0.0))
                    elif step == 8:
                        yel_rot = Euler((yel_rot.x * (1.0 - _ease(t)),
                                         yel_rot.y * (1.0 - _ease(t)), 0.0))
                elif yel_free:
                    yel_pos = yel_pos + yel_vel
                    yel_vel = yel_vel * CABIN_DRAG

                self._key_loc(self.red, frame, red_pos)
                self._key_rot(self.red, frame, red_rot)
                self._key_loc(self.yellow, frame, yel_pos)
                self._key_rot(self.yellow, frame, yel_rot)

                self.labels.append(step)
                frame += 1

            step_spans.append({"step_id": step, "occurrence": occurrence,
                               "start_frame": span_start, "end_frame": frame - 1})

        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = frame - 1
        scene.render.fps = FPS

        # Linear interpolation between the per-frame keys: the easing is
        # already baked into the sampled positions, so letting Bezier handles
        # re-ease between adjacent frames would double-apply it and round off
        # the very motion extremes the classifier needs.
        for obj in (self.goal_l, self.goal_r, self.root, self.lid,
                    self.red, self.yellow):
            for fcurve in common.iter_fcurves(obj):
                for kp in fcurve.keyframe_points:
                    kp.interpolation = "LINEAR"

        return {
            "n_frames": frame - 1,
            "labels": list(self.labels),
            "sequence": list(sequence),
            "spans": step_spans,
            "fps": FPS,
            "speed": self.speed,
            "style": {k: (list(v) if isinstance(v, Vector) else v)
                      for k, v in self.style.items()},
        }
