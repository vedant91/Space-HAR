"""tools/check_dataset.py: gate a dataset before training/render commit.
Build tiny synthetic take folders in tmp_path and assert each failure class
fires -- label/frame mismatch, missing step, wrist teleport, empty holdout --
and a clean take passes. Regression #5: a holdout tag matching zero takes
fails loudly."""
import numpy as np
import pytest

from tests.conftest import import_tool

check_dataset = import_tool("check_dataset")


def test_clean_take_passes(make_take, tmp_path):
    root = tmp_path / "ds"
    make_take(root, index=0, tag="nominal", n_frames=12, sequence=(1, 2))
    errors, tags = check_dataset.check(root, holdout=[])
    assert errors == []
    assert tags == {"nominal": ["take_0000"]}


def test_no_finished_takes(tmp_path):
    errors, tags = check_dataset.check(tmp_path / "empty", holdout=[])
    assert len(errors) == 1 and "no finished takes" in errors[0]


def test_label_frame_count_mismatch(make_take, tmp_path):
    root = tmp_path / "ds"
    take_dir = make_take(root, index=0, n_frames=12, sequence=(1, 2))
    cam_dir = take_dir / "payload_a"
    # Corrupt: labels no longer agree with meta's n_frames / pose arrays.
    np.save(cam_dir / "labels.npy", np.ones(9, dtype=np.int64))
    errors, _ = check_dataset.check(root, holdout=[])
    assert any("disagree" in e for e in errors)


def test_missing_step_in_sequence(make_take, tmp_path):
    root = tmp_path / "ds"
    take_dir = make_take(root, index=0, n_frames=12, sequence=(1, 2))
    cam_dir = take_dir / "payload_a"
    # labels only ever say step 1, but meta's sequence claims 1 AND 2.
    np.save(cam_dir / "labels.npy", np.ones(12, dtype=np.int64))
    errors, _ = check_dataset.check(root, holdout=[])
    assert any("!= sequence" in e for e in errors)


def test_wrist_teleport(make_take, tmp_path):
    root = tmp_path / "ds"
    make_take(root, index=0, n_frames=12, sequence=(1, 2), wrist_teleport=True)
    errors, _ = check_dataset.check(root, holdout=[])
    assert any("wrist jumps" in e for e in errors)


def test_empty_holdout_tag_fails_loudly(make_take, tmp_path):
    """Regression #5: a holdout tag that matches zero takes must fail, not
    silently produce an empty holdout bucket."""
    root = tmp_path / "ds"
    make_take(root, index=0, tag="nominal", n_frames=12, sequence=(1, 2))
    errors, tags = check_dataset.check(root, holdout=["orient_180"])
    assert any("matches no take" in e for e in errors)
    assert "orient_180" not in tags


def test_holdout_tag_present_does_not_fail(make_take, tmp_path):
    root = tmp_path / "ds"
    make_take(root, index=0, tag="orient_180", n_frames=12, sequence=(1, 2))
    errors, _ = check_dataset.check(root, holdout=["orient_180"])
    assert errors == []


def test_real_take_checks_frame_count_only(make_take, tmp_path):
    """Recorded (not rendered) takes have no pose ground truth -- only the
    labels/PNG/n_frames triangle is checked."""
    root = tmp_path / "ds"
    make_take(root, index=0, tag="real_nominal", n_frames=6, sequence=(1,), real=True)
    errors, _ = check_dataset.check(root, holdout=[])
    assert errors == []


def test_main_exits_nonzero_on_failure(make_take, tmp_path, capsys):
    root = tmp_path / "ds"
    make_take(root, index=0, tag="nominal", n_frames=12, sequence=(1, 2))
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["check_dataset.py", str(root), "--holdout", "orient_180"]
        with pytest.raises(SystemExit) as exc:
            check_dataset.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old_argv
    out = capsys.readouterr().out
    assert "DATASET_CHECK FAILED" in out
