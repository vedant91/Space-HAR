"""Regression #2 (T7, the critical silent gap) and #3: SAMPLE_EVERY
decimation must happen in BOTH HARPipeline.run() and .process_frame() --
both go through the single sampling gate `_ingest` -> SampleWindow.push, so
both must need SEQUENCE_WINDOW * SAMPLE_EVERY raw frames to warm up and both
windows must span the same 3.0 s.

HSV/pose are stubbed out here (a fixed dummy 143-dim feature per frame) so
the test is about the DECIMATION AND WARM-UP ARITHMETIC, not perception
quality -- that is pinned separately by test_sequence_parity.py with real
MediaPipe.
"""
import numpy as np
import pytest

import pipeline.har_pipeline as har_pipeline
from config.experiment_config import FPS, SAMPLE_EVERY, SEQUENCE_WINDOW, SKELETON_FEATURES
from pipeline.har_pipeline import HARPipeline

RAW_FRAMES_TO_WARM_UP = 88   # measured: SampleWindow(30, 3, 30) fills at raw index 87 (0-based)
NOMINAL_WARMUP = SEQUENCE_WINDOW * SAMPLE_EVERY   # the design doc's stated "90 raw frames"


def _stub_pipeline(monkeypatch, tiny_lstm_ckpt) -> HARPipeline:
    # Pin the contract explicitly (feature_layout, sample_every=3,
    # rack_normalized) rather than relying on whatever happens to be
    # promoted in models/ right now -- that model's own recorded contract is
    # what HARPipeline actually adopts (regression #1), so a legacy
    # (sample_every=1) promoted checkpoint would silently defeat this test.
    ckpt = tiny_lstm_ckpt("deployed_contract.pt", feature_dim=143,
                          feature_layout="pose132+obj11", sample_every=SAMPLE_EVERY,
                          rack_normalized=True)
    p = HARPipeline(lstm_path=str(ckpt), headless=True, enable_voice=False,
                    enable_recording=False, use_threaded=False)
    assert p.sample_every == SAMPLE_EVERY
    # _perceive() always returns the raw SKELETON_FEATURES(132)-dim pose --
    # build_frame_features() is what appends the object channels downstream,
    # regardless of which layout is deployed.
    dummy = np.zeros(SKELETON_FEATURES, dtype=np.float32)

    def fake_perceive(frame, timings=None):
        return [], dummy   # (detections, pose)

    monkeypatch.setattr(p, "_perceive", fake_perceive)
    monkeypatch.setattr(p, "_predict_and_feed_state_machine", lambda: (0, 0.0))
    return p


class _FakeCapture:
    """Enough of cv2.VideoCapture's interface for run() to drive frames
    through without a camera or a video file."""
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def isOpened(self):
        return True

    def set(self, *a, **kw):
        return True

    def get(self, *a, **kw):
        return -1.0   # not a str source in these tests, so unused

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


def test_process_frame_warms_up_at_sequence_window_times_sample_every(monkeypatch, tiny_lstm_ckpt):
    p = _stub_pipeline(monkeypatch, tiny_lstm_ckpt)
    try:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(RAW_FRAMES_TO_WARM_UP - 1):
            p.process_frame(frame, timestamp=i / float(FPS))
        assert not p.skeleton_buffer.full, "warmed up on too few raw frames"
        p.process_frame(frame, timestamp=(RAW_FRAMES_TO_WARM_UP - 1) / float(FPS))
        assert p.skeleton_buffer.full
        assert p.skeleton_buffer.sample_every == SAMPLE_EVERY
        span_s = SEQUENCE_WINDOW * p.skeleton_buffer.sample_every / float(FPS)
        assert span_s == pytest.approx(3.0)
    finally:
        p.close()


def test_run_warms_up_at_same_frame_count_as_process_frame(monkeypatch, tiny_lstm_ckpt):
    """run() must decimate too -- not just process_frame(). Both are driven
    through the SAME _ingest()/SampleWindow, so this pins that run() has not
    grown a second, undecimated path (the exact regression the design doc
    calls "silent -- this is the critical one")."""
    p = _stub_pipeline(monkeypatch, tiny_lstm_ckpt)
    try:
        # run() timestamps a webcam source with time.monotonic(); make that
        # clock advance in lock-step with the nominal FPS, like a real camera.
        clock = {"t": 0.0}

        def fake_monotonic():
            t = clock["t"]
            clock["t"] += 1.0 / FPS
            return t

        monkeypatch.setattr(har_pipeline.time, "monotonic", fake_monotonic)

        n_frames = RAW_FRAMES_TO_WARM_UP + 5
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(n_frames)]
        fake_cap = _FakeCapture(frames)
        monkeypatch.setattr(har_pipeline.cv2, "VideoCapture", lambda *a, **kw: fake_cap)

        full_at = {"frame": None}
        orig_ingest = p._ingest

        def spying_ingest(features, t, label=None):
            n = orig_ingest(features, t, label)
            if p.skeleton_buffer.full and full_at["frame"] is None:
                full_at["frame"] = p.frame_idx
            return n

        monkeypatch.setattr(p, "_ingest", spying_ingest)
        p.source = 0
        p.run()

        assert full_at["frame"] == RAW_FRAMES_TO_WARM_UP, (
            "run() filled the model window on a different raw-frame count than "
            "process_frame() -- SAMPLE_EVERY decimation has drifted apart between "
            "the two paths")
    finally:
        p.close()
