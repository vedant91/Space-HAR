"""THE test: a full-take, SEQUENCE-level parity check between the training
feature path (tools/build_sequences.py's own perceive()+take_features()) and
the live pipeline (pipeline/har_pipeline.HARPipeline.process_frame()).

Single-frame parity is not enough (Review Outcomes, "Parity test,
corrected"): RackFrameNormalizer carries angle_ema/scale_ema and a latched
polarity, UprightPoseEstimator latches its working angle, and MediaPipe
tracks across frames (static_image_mode=False). This test runs ~24 real
frames through both paths, with reset semantics pinned identically at the
start of each (a fresh RackFrameNormalizer / a freshly-constructed
UprightPoseEstimator / HARPipeline.reset_runtime()), and asserts the
per-sample resampled sequences are elementwise identical.

Kept to a handful of frames per the environment note (a GPU render may be
running): 24 frames, each run through MediaPipe exactly twice (once per
path).
"""
import numpy as np
import pytest

from tests.conftest import require_blender_twin_frames

N_FRAMES = 24


@pytest.fixture(scope="module")
def frame_paths():
    return require_blender_twin_frames(N_FRAMES)


class _RecordingSampleWindow:
    """Wraps a real SampleWindow but also records every emitted sample,
    independent of the window's own (small) maxlen eviction -- so the full
    per-sample history of a take survives even though the live buffer only
    ever keeps the last SEQUENCE_WINDOW samples."""

    def __init__(self, window, sample_every, fps):
        from pipeline.features import SampleWindow
        self._sw = SampleWindow(window, sample_every, fps)
        self.history = []
        orig_emit = self._sw._emit

        def spying_emit(g, a, b):
            orig_emit(g, a, b)
            self.history.append(self._sw._buf[-1].copy())

        self._sw._emit = spying_emit

    def __getattr__(self, name):
        return getattr(self._sw, name)


def _training_path_samples(frame_paths, sample_every, fps, layout, rack_normalize, upright):
    """Exactly tools/build_sequences.py's per-take pipeline: perceive() frame
    by frame (frame 0 gets no pose hint, same as live), then
    take_features(), then resampled onto the SampleWindow grid -- the same
    two functions build_sequences.build() calls."""
    import cv2

    from config.experiment_config import FRAME_HEIGHT, FRAME_WIDTH
    from pipeline.har_pipeline import OptimizedMPWrapper, perceive, to_frame_size
    from pipeline.hsv_detector import HSVBoxDetector
    from tests.conftest import import_tool
    take_features = import_tool("build_sequences").take_features

    detector = HSVBoxDetector(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    backend = OptimizedMPWrapper(frame_wh=(FRAME_WIDTH, FRAME_HEIGHT), upright=upright)
    try:
        poses, dets_all, prev = [], [], None
        for path in frame_paths:
            img = to_frame_size(cv2.imread(str(path)))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets, feats = perceive(img, rgb, detector, backend, prev)
            prev = dets
            poses.append(feats)
            dets_all.append(dets)
    finally:
        backend.close()

    poses = np.stack(poses)
    feats = take_features(poses, dets_all, (FRAME_WIDTH, FRAME_HEIGHT), rack_normalize, layout)

    sw = _RecordingSampleWindow(len(frame_paths), sample_every, fps)
    for i in range(len(feats)):
        sw.push(feats[i], i / float(fps))
    return sw.history


def _live_path_samples(frame_paths, sample_every, fps, layout, rack_normalize, upright):
    """The live pipeline itself: HARPipeline.process_frame(), with an
    explicit contract matching the training-path arguments above, and reset
    semantics pinned exactly as a real session start would."""
    import cv2

    from pipeline.har_pipeline import HARPipeline, to_frame_size

    ckpt = _make_matching_ckpt(layout, sample_every, rack_normalize)
    p = HARPipeline(lstm_path=str(ckpt), headless=True, enable_voice=False,
                    enable_recording=False, use_threaded=False)
    try:
        assert p.feature_layout == layout
        assert p.sample_every == sample_every
        assert (p.rack_normalizer is not None) == rack_normalize
        p.reset_runtime()   # pin: fresh EMA, fresh latched polarity/angle
        rec = _RecordingSampleWindow(len(frame_paths), sample_every, fps)
        p.skeleton_buffer = rec
        for i, path in enumerate(frame_paths):
            frame = to_frame_size(cv2.imread(str(path)))
            p.process_frame(frame, timestamp=i / float(fps))
        return rec.history
    finally:
        p.close()


def _make_matching_ckpt(layout, sample_every, rack_normalize):
    import torch

    from config.experiment_config import MODEL_FEATURE_DIM, SKELETON_FEATURES
    from pipeline.features import LAYOUT_DIMS
    from train.train_lstm import HARLSTMClassifier

    dim = LAYOUT_DIMS[layout]
    model = HARLSTMClassifier(feature_dim=dim, hidden_size=4, num_layers=1, num_classes=2)
    path = _make_matching_ckpt._dir / "parity_model.pt"
    torch.save({
        "model_state": model.state_dict(), "feature_dim": dim, "hidden_size": 4,
        "num_layers": 1, "num_classes": 2, "idx_to_label": {0: 1, 1: 2},
        "val_acc": 0.5, "window": 30,
        "feature_layout": layout, "sample_every": sample_every,
        "rack_normalized": rack_normalize,
    }, path)
    return path


@pytest.fixture(autouse=True)
def _ckpt_dir(tmp_path):
    _make_matching_ckpt._dir = tmp_path
    yield


@pytest.mark.parametrize("layout,rack_normalize", [
    ("pose132+obj11", True),
    ("pose132", False),
])
def test_full_take_sequence_parity(frame_paths, layout, rack_normalize):
    from config.experiment_config import FPS, SAMPLE_EVERY

    train = _training_path_samples(frame_paths, SAMPLE_EVERY, FPS, layout, rack_normalize, True)
    live = _live_path_samples(frame_paths, SAMPLE_EVERY, FPS, layout, rack_normalize, True)

    assert len(train) == len(live) > 0
    for i, (t, l) in enumerate(zip(train, live)):
        assert np.array_equal(t, l), f"sample {i} diverged between training and live paths"
