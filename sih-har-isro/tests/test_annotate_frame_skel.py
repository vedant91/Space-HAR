"""_annotate_frame()'s optional 5th `skel` param draws the MediaPipe pose
skeleton on the camera view via pipeline/pose_viz.py's hardcoded BONES_2D
edge list (not imported from mediapipe: that import path broke the whole
pipeline's import once before).

Three things pinned here:
  (a) skel omitted (or None) behaves exactly as it did before this param
      existed -- no crash, output frame unaffected by the new code path.
  (b) an all-zeros skel is a legal input (np.any(skel) is False) that draws
      nothing -- not a crash, not a silent garbage overlay.
  (c) a realistic non-zero, mostly-visible skel actually modifies pixels --
      proving the overlay draws something rather than silently no-op'ing.
"""
import numpy as np

import pipeline.har_pipeline as har_pipeline
from config.experiment_config import FRAME_HEIGHT, FRAME_WIDTH
from pipeline.har_pipeline import SIDE_PANEL_W, HARPipeline


def _pipeline(tiny_lstm_ckpt) -> HARPipeline:
    ckpt = tiny_lstm_ckpt("annotate.pt", feature_dim=143, feature_layout="pose132+obj11",
                          sample_every=3, rack_normalized=True)
    return HARPipeline(lstm_path=str(ckpt), headless=True, enable_voice=False,
                       enable_recording=False, use_threaded=False)


def _frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _annotate_deterministic(p, monkeypatch, skel):
    """The monitor view keeps display-only temporal state (skeleton filter,
    box tracker, 3D camera); reset it so two calls are byte-comparable and
    any pixel difference is attributable to `skel` alone. The panel's
    elapsed-time readout uses time.time(), so freeze that too."""
    monkeypatch.setattr(har_pipeline.time, "time", lambda: 1000.0)
    p.reset_runtime()
    return p._annotate_frame(_frame(), [], 0, 0.5, skel)


def test_skel_omitted_still_works(tiny_lstm_ckpt):
    p = _pipeline(tiny_lstm_ckpt)
    try:
        vis = p._annotate_frame(_frame(), [], 0, 0.5)
        # camera view + the monitor side panel
        assert vis.shape == (FRAME_HEIGHT, FRAME_WIDTH + SIDE_PANEL_W, 3)
    finally:
        p.close()


def test_all_zero_skel_draws_nothing(tiny_lstm_ckpt, monkeypatch):
    p = _pipeline(tiny_lstm_ckpt)
    try:
        zero_skel = np.zeros(132, dtype=np.float32)
        vis_no_skel = _annotate_deterministic(p, monkeypatch, None)
        vis_zero_skel = _annotate_deterministic(p, monkeypatch, zero_skel)
        # np.any(skel) is False for all-zeros -> the skeleton-drawing branch
        # is skipped entirely, so the two frames must be pixel-identical.
        assert np.array_equal(vis_no_skel, vis_zero_skel)
    finally:
        p.close()


def test_real_skel_actually_draws(tiny_lstm_ckpt, monkeypatch):
    p = _pipeline(tiny_lstm_ckpt)
    try:
        # 33 landmarks x (x, y, z, visibility), normalized image coords,
        # visibility comfortably above the 0.5 threshold in _annotate_frame.
        skel = np.zeros((33, 4), dtype=np.float32)
        skel[:, 0] = np.linspace(0.2, 0.8, 33)   # x
        skel[:, 1] = np.linspace(0.1, 0.9, 33)   # y
        skel[:, 2] = 0.0                          # z (unused for drawing)
        skel[:, 3] = 0.9                          # visibility
        skel = skel.reshape(132)

        vis_no_skel = _annotate_deterministic(p, monkeypatch, None)
        vis_with_skel = _annotate_deterministic(p, monkeypatch, skel)

        # the skeleton is drawn on the camera view, not the side panel
        cam = slice(0, FRAME_WIDTH)
        assert not np.array_equal(vis_no_skel[:, cam], vis_with_skel[:, cam]), (
            "a realistic non-zero skel changed no pixels -- the overlay silently "
            "no-op'ed")
    finally:
        p.close()
