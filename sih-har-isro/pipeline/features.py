"""
The model input contract: ONE per-frame feature builder, ONE windowing rule.

Training (tools/build_sequences.py), scoring (tools/evaluate_real.py) and the
live pipeline (pipeline/har_pipeline.py) build model inputs through this file
and nowhere else. The shipped model sat at chance partly because those paths
had drifted apart - different purity filters, a sample rate nobody pinned,
rack normalisation on at serve time and off at train time. The fix is to leave
no second implementation to drift.

    camera frame t (BGR), capture time t_s
          |
          |  har_pipeline.perceive()  (shared by all three paths)
          +------------------------------+
          |                              |
    HSVBoxDetector.detect(frame t)   OptimizedMPWrapper.process(frame t,
          |  detections(t)             hint = detections(t-1))
          |                              |  pose 132 = 33 x (x, y, z, vis),
          |                              |  image-normalised, POSE_INPUT_WIDTH px
          |                              |
          |        build_frame_features()   <- this file
          |   pick_rack_rect(dets(t)) ------+
          |                                 v
          |              RackFrameNormalizer.normalize(pose)   (optional)
          |                                 |  pose in the rack frame
          +--> box centroids --> .to_rack_xy()  (the SAME transform)
                                            |
                  layout "pose132"        : pose only               132
                  layout "pose132+obj11"  : pose + 3 boxes x
                                            (cx, cy, vis) + 2 wrist-
                                            to-nearest-box distances 143
                  absent box -> vis 0 + ABSENT, written AFTER the
                  transform so absence is constant, never rack-relative
                                            |
                             (vector, t_s) per camera frame
                                            |
                  SampleWindow: resample onto a fixed clock, period
                  sample_every / FPS (100 ms). Pose dims interpolated between
                  the two bracketing frames only if BOTH saw a pose, else
                  nearest; object dims always nearest. Gap > MAX_FRAME_GAP_S
                  -> clear and warm up again. Last SEQUENCE_WINDOW samples.
                                            |
          +---------------------------------+----------------------------+
          |                                 |                            |
    windows() over a take           windows() over a take        HARPipeline buffer
    tools/build_sequences.py        tools/evaluate_real.py       run() AND
    (training, t = i / fps)         (scoring, t = i / fps)       process_frame()

`windows()` is literally the live buffer replayed over an array, so an offline
window is by construction one the live pipeline could have held. On rendered
frames (t exactly i/30) every grid point lands on a frame, and resampling is
byte-identical to keeping every sample_every-th frame.

A checkpoint records the contract it was trained under (`feature_layout`,
`sample_every`, `rack_normalized`); `contract_of()` reads it back, treating a
checkpoint with no fields as the legacy contract (pose132, every frame).
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (  # noqa: E402
    FEATURE_LAYOUT, FPS, MAX_FRAME_GAP_S, MODEL_FEATURE_DIM, OBJECT_FEATURES,
    SAMPLE_EVERY, SEQUENCE_WINDOW, SKELETON_FEATURES, WINDOW_PURITY,
)
from pipeline.rack_frame import L_WRIST, R_WRIST, NUM_LANDMARKS, pick_rack_rect  # noqa: E402

LEGACY_LAYOUT = "pose132"          # every checkpoint written before layouts were recorded
OBJECT_LAYOUT = "pose132+obj11"
LAYOUT_DIMS = {LEGACY_LAYOUT: SKELETON_FEATURES, OBJECT_LAYOUT: MODEL_FEATURE_DIM}
assert FEATURE_LAYOUT in LAYOUT_DIMS, f"config FEATURE_LAYOUT {FEATURE_LAYOUT!r} unknown"

BOX_LABELS = ("red_box", "yellow_box", "main_box")
# Written into every non-visibility object slot of an absent box, and into a
# hand distance with no box (or no pose) to measure to. -1 is impossible for a
# distance and off-frame for an image coordinate; the visibility bit is what
# actually disambiguates it in rack coordinates.
ABSENT = -1.0
_VIS_SLOTS = [3 * k + 2 for k in range(len(BOX_LABELS))]
assert 3 * len(BOX_LABELS) + 2 == OBJECT_FEATURES

# A frame this close to a grid point IS that grid point (float noise between
# i/30 and k * 3/30 is ~1e-15 s; real cameras jitter by milliseconds).
_T_EPS = 1e-6


def build_frame_features(pose: np.ndarray, detections, frame_wh: Tuple[int, int],
                         normalizer=None, layout: str = FEATURE_LAYOUT) -> np.ndarray:
    """One camera frame -> one model input vector for `layout`.

    `pose` is the raw 132-vector from the pose backend (image-normalised),
    `detections` this frame's HSV output, `frame_wh` the (width, height) the
    detector saw. `normalizer` is a RackFrameNormalizer or None; it is
    stateful (EMA, latched polarity), so each stream needs its own and it must
    see every frame in order.
    """
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if normalizer is not None:
        pose = normalizer.normalize(pose, rack_rect=pick_rack_rect(detections))
    if layout == LEGACY_LAYOUT:
        return pose
    if layout != OBJECT_LAYOUT:
        raise ValueError(f"unknown feature layout {layout!r}")

    obj = np.full(OBJECT_FEATURES, ABSENT, dtype=np.float32)
    obj[_VIS_SLOTS] = 0.0
    first = {}
    for d in detections or []:
        first.setdefault(d.label, d)          # HSV keeps top-1 per colour
    seen = [lab for lab in BOX_LABELS if lab in first]
    if seen:
        w, h = frame_wh
        pts = np.array([[first[lab].centroid[0] / w, first[lab].centroid[1] / h]
                        for lab in seen], dtype=np.float32)
        if normalizer is not None:
            # None when this frame's pose was unusable: no transform exists,
            # so the boxes are reported absent rather than in a stale frame.
            pts = normalizer.to_rack_xy(pts)
        if pts is not None:
            for lab, p in zip(seen, pts):
                k = 3 * BOX_LABELS.index(lab)
                obj[k:k + 2] = p
                obj[k + 2] = 1.0
            if np.any(pose):
                wrists = pose.reshape(NUM_LANDMARKS, 4)[[L_WRIST, R_WRIST], :2]
                obj[-2:] = np.linalg.norm(wrists[:, None, :] - pts[None], axis=-1).min(axis=1)
    return np.concatenate([pose, obj])


class SampleWindow:
    """The windowing rule, in streaming form.

    Frames arrive with timestamps (seconds) at whatever rate the camera and
    the pipeline manage (~15-22 fps live, exactly 30 on renders). Samples are
    emitted on a fixed grid t0 + k * period, period = sample_every / fps,
    where t0 is the first frame after construction / clear(). A grid sample
    is emitted when the first frame at or after it arrives:

      * a frame within _T_EPS of the grid point is taken as-is (so on i/30 s
        renders this is exactly keep-every-sample_every-th-frame);
      * otherwise pose dims [0, interp_dims) are linearly interpolated
        between the two bracketing frames, but ONLY if both have a detected
        pose - never toward an all-zero no-detection vector - else the
        nearest frame is copied;
      * object dims [interp_dims, end) and the label always come from the
        nearest frame: visibility flags and the ABSENT sentinel stay exact.

    A gap > max_gap between consecutive frames (or time going backwards)
    clears the window and restarts the grid instead of inventing motion.
    `push()` returns how many samples it emitted and calls `on_sample()`
    after each one, so a caller can classify every sample, not every frame.
    """

    def __init__(self, window: int = SEQUENCE_WINDOW, sample_every: int = SAMPLE_EVERY,
                 fps: float = FPS, max_gap: float = MAX_FRAME_GAP_S,
                 interp_dims: int = SKELETON_FEATURES):
        if sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every}")
        self.sample_every = int(sample_every)
        self.period = self.sample_every / float(fps)
        self.max_gap = float(max_gap)
        self.interp_dims = int(interp_dims)
        self.gap_resets = 0
        self._buf: deque = deque(maxlen=int(window))
        self._labels: deque = deque(maxlen=int(window))
        self._prev = None            # (t, x, label) of the last frame pushed
        self._t0 = 0.0
        self._k = 0

    def push(self, x, t: float, label=None,
             on_sample: Optional[Callable[[], None]] = None) -> int:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        t = float(t)
        if self._prev is not None and not (0.0 <= t - self._prev[0] <= self.max_gap):
            self.clear()
            self.gap_resets += 1
        cur = (t, x, label)
        if self._prev is None:
            self._t0, self._k = t, 0
        emitted = 0
        while self._t0 + self._k * self.period <= t + _T_EPS:
            self._emit(self._t0 + self._k * self.period, self._prev, cur)
            self._k += 1
            emitted += 1
            if on_sample is not None:
                on_sample()
        self._prev = cur
        return emitted

    def _emit(self, g: float, a, b) -> None:
        if a is None or abs(b[0] - g) <= _T_EPS:
            x, label = b[1].copy(), b[2]
        elif abs(g - a[0]) <= _T_EPS:
            x, label = a[1].copy(), a[2]
        else:
            near = a if (g - a[0]) < (b[0] - g) else b
            x, label = near[1].copy(), near[2]
            d = self.interp_dims
            if np.any(a[1][:d]) and np.any(b[1][:d]):
                w = np.float32((g - a[0]) / (b[0] - a[0]))
                x[:d] = a[1][:d] + w * (b[1][:d] - a[1][:d])
        self._buf.append(x)
        self._labels.append(label)

    @property
    def full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    def __len__(self) -> int:
        return len(self._buf)

    def labels(self) -> list:
        return list(self._labels)

    def array(self) -> np.ndarray:
        return np.stack(self._buf).astype(np.float32)

    def clear(self) -> None:
        self._buf.clear()
        self._labels.clear()
        self._prev = None
        self._k = 0


def windows(seq: np.ndarray, labels: Optional[np.ndarray] = None, stride: int = 1,
            purity: float = WINDOW_PURITY, sample_every: int = SAMPLE_EVERY,
            window: int = SEQUENCE_WINDOW, fps: float = FPS,
            timestamps: Optional[Sequence[float]] = None
            ) -> List[Tuple[np.ndarray, Optional[int]]]:
    """[(window (T, F) float32, dominant label or None)] over a per-frame
    sequence, exactly as the live buffer would have held them.

    `timestamps` default to i / fps (a render). `stride` counts SAMPLES, so
    stride 1 is one window per sample the live pipeline classifies. With
    `labels`, a window is kept only when at least `purity` of it shares the
    majority label, and is labelled by that majority: a window 80% step 3 that
    ends on step 4's first frame is a picture of step 3. Mixed windows are kept
    (purity < 1) because the live buffer sees them at every step boundary;
    STEP_CONFIRM_FRAMES debounces them.
    """
    buf = SampleWindow(window, sample_every, fps)
    out: List[Tuple[np.ndarray, Optional[int]]] = []
    full_seen = [0]

    def on_sample():
        if not buf.full:
            return
        full_seen[0] += 1
        if (full_seen[0] - 1) % stride:
            return
        dominant = None
        if labels is not None:
            values, counts = np.unique(buf.labels(), return_counts=True)
            if counts.max() / float(window) < purity:
                return
            dominant = int(values[int(np.argmax(counts))])
        out.append((buf.array(), dominant))

    for i in range(len(seq)):
        t = (i / float(fps)) if timestamps is None else timestamps[i]
        buf.push(seq[i], t, None if labels is None else int(labels[i]), on_sample)
    return out


def stack_windows(seq: np.ndarray, labels: np.ndarray, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """windows() collected into (N, T, F) float32 and (N,) int64 arrays."""
    ws = windows(seq, labels, **kw)
    if not ws:
        window = kw.get("window", SEQUENCE_WINDOW)
        return (np.zeros((0, window, seq.shape[-1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.stack([x for x, _ in ws]), np.array([y for _, y in ws], dtype=np.int64)


def contract_of(meta: dict) -> dict:
    """The input contract a checkpoint or sequence set was built under.
    Missing fields mean it predates layouts: the legacy contract (132-dim
    pose, every frame). rack_normalized None = not recorded."""
    return {"feature_layout": meta.get("feature_layout", LEGACY_LAYOUT),
            "sample_every": int(meta.get("sample_every", 1)),
            "rack_normalized": meta.get("rack_normalized")}


# ── selftest ─────────────────────────────────────────────────────────────────


def _selftest() -> None:
    from types import SimpleNamespace as Det
    from pipeline.rack_frame import RackFrameNormalizer, _make_pose, _rect_from_line

    wh = (1000, 500)

    def dets(angle, with_yellow):
        d = [Det(label="main_box", centroid=(500, 250), rect=_rect_from_line(angle)),
             Det(label="red_box", centroid=(300, 200), rect=None)]
        if with_yellow:
            d.append(Det(label="yellow_box", centroid=(700, 300), rect=None))
        return d

    # 1) Legacy layout is exactly the normaliser's output, nothing appended.
    a, b = RackFrameNormalizer(), RackFrameNormalizer()
    pose = _make_pose(90.0).reshape(-1)
    f = build_frame_features(pose, dets(0.0, True), wh, a, LEGACY_LAYOUT)
    assert f.shape == (SKELETON_FEATURES,)
    assert np.array_equal(f, b.normalize(pose, rack_rect=_rect_from_line(0.0)))

    # 2) Absent yellow box is byte-identical across frames however the rack
    #    moves (T13 verify: sentinel written AFTER normalisation).
    n = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    absent = set()
    for k in range(0, 360, 15):
        f = build_frame_features(_make_pose(90.0 + k).reshape(-1), dets(k, False), wh, n)
        assert f.shape == (MODEL_FEATURE_DIM,)
        absent.add(f[SKELETON_FEATURES + 3:SKELETON_FEATURES + 6].tobytes())
        assert f[SKELETON_FEATURES + 5] == 0.0 and f[SKELETON_FEATURES + 2] == 1.0
    assert len(absent) == 1, "absent-box sentinel moved with the rack"

    # 3) A box sitting exactly on the left wrist gets the wrist's rack
    #    coordinates and a zero left-hand distance: same transform, not a copy.
    n = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
    pose = _make_pose(90.0 + 40.0)
    wx, wy = pose[L_WRIST, 0] * wh[0], pose[L_WRIST, 1] * wh[1]
    f = build_frame_features(pose.reshape(-1),
                             [Det(label="red_box", centroid=(wx, wy), rect=_rect_from_line(40.0))],
                             wh, n)
    box = f[SKELETON_FEATURES:SKELETON_FEATURES + 2]
    wrist = f[:SKELETON_FEATURES].reshape(NUM_LANDMARKS, 4)[L_WRIST, :2]
    assert np.allclose(box, wrist, atol=1e-6) and abs(f[-2]) < 1e-6, (box, wrist, f[-2])

    # 4) No pose -> no transform -> every box absent, every distance ABSENT.
    f = build_frame_features(np.zeros(SKELETON_FEATURES), dets(0.0, True), wh,
                             RackFrameNormalizer())
    assert not np.any(f[:SKELETON_FEATURES])
    assert np.all(f[SKELETON_FEATURES:][_VIS_SLOTS] == 0.0) and np.all(f[-2:] == ABSENT)

    # 5) On render timestamps (i/30 s) the resampler is byte-identical to
    #    plain decimation seq[::k], stride included - including frames with
    #    no pose, which must pass through untouched.
    rng = np.random.default_rng(0)
    T, k, D = 200, 3, MODEL_FEATURE_DIM
    seq = rng.normal(size=(T, D)).astype(np.float32)
    seq[rng.random(T) < 0.2, :SKELETON_FEATURES] = 0.0
    dec = seq[::k]
    ref = [dec[s:s + SEQUENCE_WINDOW] for s in range(len(dec) - SEQUENCE_WINDOW + 1)]
    got = [x for x, _ in windows(seq, None, purity=0.0, sample_every=k)]
    assert len(got) == len(ref) and all(np.array_equal(g, r) for g, r in zip(got, ref))
    got2 = [x for x, _ in windows(seq, None, stride=2, purity=0.0, sample_every=k)]
    assert all(np.array_equal(g, r) for g, r in zip(got2, ref[::2])) and len(got2) == len(ref[::2])
    legacy = [x for x, _ in windows(seq, None, purity=0.0, sample_every=1)]
    assert all(np.array_equal(g, seq[i:i + SEQUENCE_WINDOW]) for i, g in enumerate(legacy))

    # 6) Off-grid frames: pose interpolated only between two detected poses,
    #    objects and labels always nearest, never toward a no-pose frame.
    sw = SampleWindow(window=4, sample_every=3, fps=30, max_gap=0.5)
    fa = np.zeros(D, np.float32); fa[:SKELETON_FEATURES] = 1.0; fa[SKELETON_FEATURES:] = ABSENT
    fb = np.zeros(D, np.float32); fb[:SKELETON_FEATURES] = 3.0; fb[SKELETON_FEATURES:] = 7.0
    assert sw.push(fa, 0.00, label=1) == 1                 # grid 0.0 = this frame
    assert sw.push(fb, 0.14, label=2) == 1                 # grid 0.1: 0.10 of 0.14 -> w=5/7
    s = sw.array()[-1]
    assert np.allclose(s[:SKELETON_FEATURES], 1.0 + (0.1 / 0.14) * 2.0, atol=1e-6)
    assert np.array_equal(s[SKELETON_FEATURES:], fb[SKELETON_FEATURES:]), "objects must be nearest"
    assert sw.labels()[-1] == 2
    nopose = fb.copy(); nopose[:SKELETON_FEATURES] = 0.0
    assert sw.push(nopose, 0.21, label=3) == 1             # grid 0.2, nearest = 0.21 (no pose)
    assert np.array_equal(sw.array()[-1], nopose), "interpolated toward a no-detection frame"
    assert sw.push(fb, 0.33, label=4) == 1                 # grid 0.3, a has no pose -> nearest
    assert np.array_equal(sw.array()[-1], fb)
    # A slow frame rate emits several samples from one frame.
    assert sw.push(fa, 0.53, label=5) == 2
    # 7) Gap > max_gap clears and warms up again.
    assert len(sw) == 4 and sw.full
    assert sw.push(fa, 1.2, label=6) == 1 and len(sw) == 1 and sw.gap_resets == 1

    # 8) Purity: a 50/50 window is dropped at 0.75, kept (majority) at 0.5.
    lab = np.array([1] * 15 + [2] * 15)
    x30 = np.zeros((30, 2), dtype=np.float32)
    assert windows(x30, lab, sample_every=1, purity=0.75) == []
    assert [y for _, y in windows(x30, lab, sample_every=1, purity=0.5)] == [1]

    # 9) Contracts: no fields = legacy.
    assert contract_of({"feature_dim": 132}) == {
        "feature_layout": LEGACY_LAYOUT, "sample_every": 1, "rack_normalized": None}
    print("features selftest passed: legacy identity, constant sentinel, shared transform, "
          "no-pose absence, resampling == decimation on renders, interpolation rules, "
          "gap reset, purity, contracts")


if __name__ == "__main__":
    _selftest()
