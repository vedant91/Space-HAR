"""Regression #1 (T8): model selection loads the configured model path, and
a model whose input dim contradicts its own recorded layout fails at startup
naming both dims. Also: ONNX and .pt disagreeing refuses to start.

HARPipeline() is real -- no source is opened (no camera), so this is safe to
construct without hardware; only pipeline.har_pipeline.LSTM_PATH /
LSTM_ONNX_PATH module globals are monkeypatched, never config files.
"""
import pytest

import pipeline.har_pipeline as har_pipeline
from pipeline.har_pipeline import HARPipeline


def _pipeline_kwargs():
    return dict(headless=True, enable_voice=False, enable_recording=False,
                use_threaded=False)


def test_explicit_lstm_path_is_loaded(tiny_lstm_ckpt):
    """An explicit lstm_path= is the model that gets served, named on the
    banner -- not the promoted models/lstm_classifier.pt."""
    ckpt = tiny_lstm_ckpt("mine.pt", feature_dim=143, hidden_size=7, num_layers=1,
                          num_classes=3, feature_layout="pose132+obj11",
                          sample_every=3, rack_normalized=True)
    p = HARPipeline(lstm_path=str(ckpt), **_pipeline_kwargs())
    try:
        assert "mine.pt" in p.model_banner
        assert p.lstm_model is not None
        assert p.lstm_model.hidden_size == 7
        assert p.feature_layout == "pose132+obj11"
        assert p.sample_every == 3
    finally:
        p.close()


def test_mismatched_dim_fails_naming_both_dims(tiny_lstm_ckpt):
    """A checkpoint whose actual feature_dim contradicts its own recorded
    layout (pose132+obj11 wants 143) refuses to start, and the error names
    BOTH numbers -- not just one -- so the failure is diagnosable without
    opening the checkpoint."""
    ckpt = tiny_lstm_ckpt("bad.pt", feature_dim=99, feature_layout="pose132+obj11",
                          sample_every=3, rack_normalized=True)
    with pytest.raises(RuntimeError) as exc:
        HARPipeline(lstm_path=str(ckpt), **_pipeline_kwargs())
    msg = str(exc.value)
    assert "99" in msg and "143" in msg


def test_legacy_contract_no_fields_means_pose132(tiny_lstm_ckpt):
    """A checkpoint recording no contract fields is the legacy contract:
    132-dim pose, every frame -- and a legacy checkpoint with feature_dim=132
    loads cleanly under that assumption."""
    ckpt = tiny_lstm_ckpt("legacy.pt", feature_dim=132, feature_layout=None,
                          sample_every=None, rack_normalized=None)
    p = HARPipeline(lstm_path=str(ckpt), **_pipeline_kwargs())
    try:
        assert p.feature_layout == "pose132"
        assert p.sample_every == 1
    finally:
        p.close()


def test_legacy_contract_dim_mismatch_still_names_both_dims(tiny_lstm_ckpt):
    ckpt = tiny_lstm_ckpt("legacy_bad.pt", feature_dim=50, feature_layout=None,
                          sample_every=None, rack_normalized=None)
    with pytest.raises(RuntimeError) as exc:
        HARPipeline(lstm_path=str(ckpt), **_pipeline_kwargs())
    msg = str(exc.value)
    assert "50" in msg and "132" in msg


def test_onnx_and_pt_disagree_refuses_to_start(tmp_path, tiny_lstm_ckpt, monkeypatch):
    """The promoted-model path (_load_promoted_lstm): a .pt recording one
    layout beside an .onnx built for a different input dim must not be
    silently served -- "the pair was not promoted together"."""
    from train.train_lstm import _export_lstm_onnx

    pt_path = tiny_lstm_ckpt("promoted.pt", feature_dim=143, feature_layout="pose132+obj11",
                             sample_every=3, rack_normalized=True)
    # A DIFFERENT checkpoint's ONNX (132-dim), placed at the promoted .onnx
    # path -- exactly the "pair not promoted together" failure mode.
    mismatched_pt = tiny_lstm_ckpt("other.pt", feature_dim=132, feature_layout="pose132",
                                   sample_every=1, rack_normalized=False)
    onnx_path = tmp_path / "promoted.onnx"
    _export_lstm_onnx(str(mismatched_pt), str(onnx_path))
    assert onnx_path.exists()

    monkeypatch.setattr(har_pipeline, "LSTM_PATH", str(pt_path))
    monkeypatch.setattr(har_pipeline, "LSTM_ONNX_PATH", str(onnx_path))

    with pytest.raises(RuntimeError) as exc:
        HARPipeline(**_pipeline_kwargs())
    msg = str(exc.value)
    assert "132" in msg and "143" in msg
    assert "not promoted together" in msg
