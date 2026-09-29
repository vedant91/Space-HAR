"""Regression pin (companion to test_sample_rate_regression.py's constant
pins): tools/build_sequences.py's HMR_TRAIN_BACKEND and HMR_UNRELIABLE_TAGS
record which HMR backend a model is trained under and which tags are still
too unreliable to train on. Both were updated when the custom HMR3DNet
replaced ROMP (docs/architecture.html section 3) -- pin the CURRENT values
directly so an accidental revert (or a copy-paste of the old 8-tag ROMP-era
list) is caught immediately rather than silently degrading training data.
"""
from tests.conftest import import_tool


def test_hmr_train_backend_is_custom():
    build_sequences = import_tool("build_sequences")
    assert build_sequences.HMR_TRAIN_BACKEND == "custom"


def test_hmr_unreliable_tags_is_the_three_confirmed_gaps():
    build_sequences = import_tool("build_sequences")
    # orient_60/30/180, pitch_60 and light_single_fail_180 are no longer
    # unreliable for the custom model -- only exact 90 deg roll (either
    # direction) and dim light at 90 deg remain unclosed gaps.
    assert set(build_sequences.HMR_UNRELIABLE_TAGS) == {
        "orient_90", "orient_-90", "light_dim_90",
    }
