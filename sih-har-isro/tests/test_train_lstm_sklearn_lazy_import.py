"""`from sklearn.model_selection import GroupShuffleSplit` was moved off the
module level of train/train_lstm.py into the one function that uses it (the
dataset-split branch, guarded by `groups_path.exists()`), specifically so
`from train.train_lstm import HARLSTMClassifier` -- what pipeline/har_pipeline.py
does to load a trained model -- never needs scikit-learn installed just to
serve a checkpoint. Block sklearn in sys.modules and prove the import still
succeeds; a module-level re-import of sklearn would raise ImportError here.
"""
import sys

import pytest


@pytest.fixture
def block_sklearn(monkeypatch):
    """Make any `import sklearn` (or `from sklearn... import ...`) fail,
    without requiring sklearn to actually be uninstalled on this machine."""
    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sklearn", None)


def test_harlstmclassifier_importable_without_sklearn(block_sklearn, monkeypatch):
    # Force train.train_lstm to actually execute its module body again under
    # the sklearn block, rather than reusing whatever another test file
    # already cached in sys.modules.
    for name in list(sys.modules):
        if name == "train.train_lstm" or name.startswith("train.train_lstm."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from train.train_lstm import HARLSTMClassifier  # noqa: F401 -- import itself is the assertion

    model = HARLSTMClassifier(feature_dim=143, hidden_size=4, num_layers=1, num_classes=2)
    assert model.hidden_size == 4
