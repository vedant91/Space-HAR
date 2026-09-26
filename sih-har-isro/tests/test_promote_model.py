"""tools/promote_model.py (release-engineer). Every test points --models-dir
at tmp_path -- never the real models/."""
import torch
import pytest

from tests.conftest import import_tool

promote_model = import_tool("promote_model")


def _baseline_ckpt(tiny_lstm_ckpt, name="lstm_final_rack.pt"):
    """A minimal but valid LSTM checkpoint to stand in for the published
    baseline, so promote() has something to preserve as baseline_rack_0201.pt."""
    return tiny_lstm_ckpt(name, feature_dim=132, hidden_size=4, num_layers=1,
                          num_classes=2, feature_layout=None, sample_every=None,
                          rack_normalized=None)


def test_refuses_non_lstm_checkpoint(tmp_path):
    bad = tmp_path / "not_lstm.pt"
    torch.save({"some_other_key": 1}, bad)
    with pytest.raises(SystemExit) as exc:
        promote_model.load_lstm(bad)
    assert "not an LSTM checkpoint" in str(exc.value)


def test_promote_creates_baseline_once(tmp_path, tiny_lstm_ckpt):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _baseline_ckpt(tiny_lstm_ckpt).rename(models_dir / "lstm_final_rack.pt")

    ckpt = tiny_lstm_ckpt("candidate.pt", feature_dim=132, hidden_size=4, num_layers=1,
                          num_classes=2, feature_layout=None, sample_every=None,
                          rack_normalized=None)
    promote_model.promote(ckpt, models_dir)

    baseline = models_dir / "baseline_rack_0201.pt"
    assert baseline.exists()
    first_bytes = baseline.read_bytes()

    # A second promotion (of a DIFFERENT checkpoint) must not overwrite it.
    ckpt2 = tiny_lstm_ckpt("candidate2.pt", feature_dim=132, hidden_size=4, num_layers=1,
                           num_classes=2, feature_layout=None, sample_every=None,
                           rack_normalized=None)
    promote_model.promote(ckpt2, models_dir)
    assert baseline.read_bytes() == first_bytes


def test_promote_refuses_without_baseline_present(tmp_path, tiny_lstm_ckpt):
    """Before the first promotion ever, no baseline_rack_0201.pt AND no
    lstm_final_rack.pt to copy it from -> refuse rather than promote blind."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt = tiny_lstm_ckpt("candidate.pt", feature_dim=132, hidden_size=4, num_layers=1,
                          num_classes=2, feature_layout=None, sample_every=None,
                          rack_normalized=None)
    with pytest.raises(SystemExit) as exc:
        promote_model.promote(ckpt, models_dir)
    assert "baseline" in str(exc.value)


def test_promote_writes_self_contained_matching_onnx(tmp_path, tiny_lstm_ckpt):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _baseline_ckpt(tiny_lstm_ckpt).rename(models_dir / "lstm_final_rack.pt")

    ckpt = tiny_lstm_ckpt("candidate.pt", feature_dim=132, hidden_size=4, num_layers=1,
                          num_classes=2, feature_layout=None, sample_every=None,
                          rack_normalized=None)
    promote_model.promote(ckpt, models_dir)

    from config.experiment_config import LSTM_ONNX_PATH, LSTM_PATH, SEQUENCE_WINDOW
    from pathlib import Path
    import numpy as np
    import onnxruntime as ort

    dst_onnx = models_dir / Path(LSTM_ONNX_PATH).name
    dst_pt = models_dir / Path(LSTM_PATH).name
    assert dst_onnx.exists() and dst_pt.exists()

    sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, SEQUENCE_WINDOW, 132)).astype(np.float32)
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    model, ck = promote_model.load_lstm(dst_pt)
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    assert np.allclose(got, want, atol=1e-4)


def test_promote_refuses_mismatched_onnx_beside_checkpoint(tmp_path, tiny_lstm_ckpt):
    """A .onnx sitting next to the checkpoint that was exported from a
    DIFFERENT model (mismatched weights, same shape) must be refused, not
    silently promoted as if it were this checkpoint's own export."""
    from train.train_lstm import _export_lstm_onnx

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _baseline_ckpt(tiny_lstm_ckpt).rename(models_dir / "lstm_final_rack.pt")

    ckpt = tiny_lstm_ckpt("candidate.pt", feature_dim=132, hidden_size=4, num_layers=1,
                          num_classes=2, feature_layout=None, sample_every=None,
                          rack_normalized=None)
    other = tiny_lstm_ckpt("other.pt", feature_dim=132, hidden_size=4, num_layers=1,
                           num_classes=2, feature_layout=None, sample_every=None,
                           rack_normalized=None)
    # A same-shape, different-weights ONNX placed at candidate.onnx.
    _export_lstm_onnx(str(other), str(ckpt.with_suffix(".onnx")))

    with pytest.raises(SystemExit) as exc:
        promote_model.promote(ckpt, models_dir)
    assert "not this checkpoint" in str(exc.value)
