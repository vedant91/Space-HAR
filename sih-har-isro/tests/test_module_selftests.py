"""Run the modules' own `python -m` self-tests inside pytest, so a broken
selftest fails CI/local `pytest` too, not only a manual `python
pipeline/features.py`. These selftests already cover most of paths 1-3, 7-9
of the 24-path diagram (legacy-layout identity, constant absent-box sentinel,
shared rack transform for boxes, no-pose absence, decimation==resampling on
render timestamps, interpolation rules, gap reset, purity, contracts,
equivariance/inversion/scale/fallbacks/NaN-safety for the rack frame)."""
from pipeline.features import _selftest as features_selftest
from pipeline.rack_frame import _selftest as rack_frame_selftest


def test_features_selftest():
    features_selftest()


def test_rack_frame_selftest():
    rack_frame_selftest()


def test_upright_pose_smoke():
    import numpy as np
    from pipeline.upright_pose import UprightPoseEstimator, FEATURE_DIM

    est = UprightPoseEstimator(complexity=0, frame_width=640, frame_height=360)
    out = est.process(np.zeros((360, 640, 3), dtype=np.uint8))
    assert out.shape == (FEATURE_DIM,)
    assert np.isfinite(out).all()
    est.close()
