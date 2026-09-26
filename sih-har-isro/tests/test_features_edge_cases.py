"""build_frame_features / windows() edge cases the shared selftest doesn't
already cover byte-for-byte. Paths from the design doc's 24-path diagram:
all boxes detected, no boxes detected, yellow-absent-only (the ~88% real
case), pick_rack_rect() -> None mid-take, pose all-zeros, windows shorter
than the window, purity below 0.75, a decimation remainder."""
from types import SimpleNamespace as Det

import numpy as np
import pytest

from config.experiment_config import MODEL_FEATURE_DIM, SEQUENCE_WINDOW, SKELETON_FEATURES
from pipeline.features import ABSENT, OBJECT_FEATURES, build_frame_features, windows
from pipeline.rack_frame import (
    L_WRIST, R_WRIST, NUM_LANDMARKS, RackFrameNormalizer, _make_pose, _rect_from_line,
)

WH = (1000, 500)


def _dets(main=True, red=True, yellow=True, rect_angle=0.0):
    out = []
    if main:
        out.append(Det(label="main_box", centroid=(500, 250), rect=_rect_from_line(rect_angle)))
    if red:
        out.append(Det(label="red_box", centroid=(300, 200), rect=None))
    if yellow:
        out.append(Det(label="yellow_box", centroid=(700, 300), rect=None))
    return out


def _vis_slots(f, label):
    labels = ("red_box", "yellow_box", "main_box")
    k = 3 * labels.index(label)
    return f[SKELETON_FEATURES + k:SKELETON_FEATURES + k + 3]


class TestBoxPresence:
    def test_all_boxes_detected(self):
        norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
        pose = _make_pose(90.0).reshape(-1)
        f = build_frame_features(pose, _dets(True, True, True), WH, norm)
        assert f.shape == (MODEL_FEATURE_DIM,)
        for label in ("main_box", "red_box", "yellow_box"):
            assert _vis_slots(f, label)[2] == 1.0
        # Hand-distance slots are finite, not the ABSENT sentinel, when a
        # box was actually seen.
        assert f[-2] != ABSENT and f[-1] != ABSENT

    def test_no_boxes_detected(self):
        norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
        pose = _make_pose(90.0).reshape(-1)
        f = build_frame_features(pose, [], WH, norm)
        obj = f[SKELETON_FEATURES:]
        assert np.all(obj[[2, 5, 8]] == 0.0)          # every visibility bit off
        assert np.all(obj[[0, 1, 3, 4, 6, 7]] == ABSENT)
        assert np.all(obj[-2:] == ABSENT)

    def test_yellow_absent_only(self):
        """The ~88% real case (logs/RESULTS.md section 3): red/main seen,
        yellow's HSV recall is 0.118 so it is usually the only absent box."""
        norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
        pose = _make_pose(90.0).reshape(-1)
        f = build_frame_features(pose, _dets(True, True, False), WH, norm)
        assert _vis_slots(f, "main_box")[2] == 1.0
        assert _vis_slots(f, "red_box")[2] == 1.0
        yellow = _vis_slots(f, "yellow_box")
        assert yellow[2] == 0.0 and yellow[0] == ABSENT and yellow[1] == ABSENT

    def test_absent_sentinel_byte_identical_regardless_of_rack_motion(self):
        """T13 verify, independently of features.py's own selftest: the
        absent-box slot must not move as the rack rotates."""
        norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
        seen = set()
        for angle in range(0, 360, 20):
            pose = _make_pose(90.0 + angle).reshape(-1)
            f = build_frame_features(pose, _dets(True, True, False, rect_angle=angle), WH, norm)
            seen.add(_vis_slots(f, "yellow_box").tobytes())
        assert len(seen) == 1

    def test_pick_rack_rect_returns_none_mid_take(self):
        """No detection carries a .rect (pick_rack_rect -> None): the
        normalizer falls back to the torso and must not crash or silently
        stop producing a pose."""
        norm = RackFrameNormalizer(angle_ema=0.0, scale_ema=0.0)
        dets_no_rect = [Det(label="main_box", centroid=(500, 250), rect=None),
                        Det(label="red_box", centroid=(300, 200), rect=None)]
        pose = _make_pose(90.0).reshape(-1)
        f = build_frame_features(pose, dets_no_rect, WH, norm)
        assert f.shape == (MODEL_FEATURE_DIM,)
        assert np.any(f[:SKELETON_FEATURES])          # pose still produced (torso fallback)
        assert _vis_slots(f, "main_box")[2] == 1.0     # box still placed via to_rack_xy

    def test_pose_all_zeros(self):
        """No pose this frame -> no transform -> every box reported absent,
        not merely unpositioned (features.py selftest #4, restated with a
        legacy-layout variant too)."""
        norm = RackFrameNormalizer()
        zero_pose = np.zeros(SKELETON_FEATURES, dtype=np.float32)
        f = build_frame_features(zero_pose, _dets(True, True, True), WH, norm)
        assert not np.any(f[:SKELETON_FEATURES])
        assert np.all(f[SKELETON_FEATURES:][[2, 5, 8]] == 0.0)
        assert np.all(f[-2:] == ABSENT)
        # Legacy layout: all-zero pose passes straight through, untouched.
        f_legacy = build_frame_features(zero_pose, _dets(True, True, True), WH, norm, layout="pose132")
        assert f_legacy.shape == (SKELETON_FEATURES,)
        assert not np.any(f_legacy)


class TestWindowing:
    def test_shorter_than_window(self):
        """A sequence with fewer samples than SEQUENCE_WINDOW never fills a
        window -> zero windows, not a short one."""
        seq = np.zeros((SEQUENCE_WINDOW - 5, 4), dtype=np.float32)
        out = windows(seq, None, sample_every=1, purity=0.0)
        assert out == []

    def test_purity_below_threshold_dropped(self):
        """A window that is 70% one label is dropped at purity=0.75 but kept
        at 0.70 (config WINDOW_PURITY = 0.75 is the standardised value)."""
        n = 20
        lab = np.array([1] * 14 + [2] * 6)   # 70% majority
        x = np.zeros((n, 2), dtype=np.float32)
        assert windows(x, lab, sample_every=1, window=n, purity=0.75) == []
        kept = windows(x, lab, sample_every=1, window=n, purity=0.70)
        assert len(kept) == 1 and kept[0][1] == 1

    def test_decimation_remainder(self):
        """A sequence length not a multiple of sample_every must not crash
        and must not silently drop the trailing partial group -- the last
        grid points still resolve to the nearest real frame."""
        rng = np.random.default_rng(1)
        T, k = 47, 5          # 47 / 5 leaves a remainder of 2
        seq = rng.normal(size=(T, 3)).astype(np.float32)
        dec = seq[::k]
        window = len(dec)     # a single window spanning every decimated sample
        out = windows(seq, None, sample_every=k, window=window, purity=0.0)
        assert len(out) == 1
        assert np.array_equal(out[0][0], dec)


class TestPoseInterpolationLatchedState:
    def test_absent_box_after_rack_rotation_stays_zero_visibility(self):
        """Regression guard for T13: even with EMA smoothing engaged (not
        angle_ema=0 as in the selftest), the visibility bit for an absent box
        is exactly 0.0, never a smoothed near-zero."""
        norm = RackFrameNormalizer(angle_ema=0.85, scale_ema=0.90)
        for angle in range(0, 90, 7):
            pose = _make_pose(90.0 + angle).reshape(-1)
            f = build_frame_features(pose, _dets(True, True, False, rect_angle=angle), WH, norm)
            assert _vis_slots(f, "yellow_box")[2] == 0.0
