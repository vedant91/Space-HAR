"""tools/record_take.py: the meta derivation (run-length sequence, measured
fps). Nothing is factored out of record_take.py for testability -- it is
exercised end to end through its real main(), with cv2 camera I/O faked out,
into a small take written under tmp_path.
"""
import sys

import numpy as np
import pytest

from tests.conftest import import_tool

record_take = import_tool("record_take")


class _FakeCapture:
    def __init__(self, n_frames=8):
        self.n = n_frames
        self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def set(self, *a, **kw):
        return True

    def read(self):
        return True, self._frame.copy()

    def release(self):
        pass


def _fake_clock(step=0.1):
    """time.perf_counter() replacement: a fixed step per call."""
    state = {"t": 0.0}

    def _tick():
        t = state["t"]
        state["t"] += step
        return t
    return _tick


@pytest.mark.parametrize("step_ids", [(1, 2), (30, 10)])
def test_meta_run_length_sequence_and_measured_fps(tmp_path, monkeypatch, step_ids):
    # Keys pressed, one per read() iteration: start recording, hold step 1
    # for several frames, switch to step 2 for several frames, then quit.
    # The exact frame-to-keypress alignment (a key changes `step` for the
    # NEXT append, not the current one) is record_take.py's own internal
    # detail -- deliberately not re-derived here -- so this only pins the
    # two things record_take.py itself derives: run-length sequence and
    # measured fps, not a hand-traced frame count.
    keys = iter([ord(" ")] + [ord("1")] * 4 + [ord("2")] * 4 + [ord("q")])

    monkeypatch.setattr(record_take.cv2, "VideoCapture", lambda *a, **kw: _FakeCapture())
    monkeypatch.setattr(record_take.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(record_take.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(record_take.cv2, "waitKey", lambda *a, **kw: next(keys))
    monkeypatch.setattr(record_take.time, "perf_counter", _fake_clock(step=0.1))

    out_dir = tmp_path / "real"
    argv = ["record_take.py", "--out", str(out_dir), "--tag", "real_nominal"]
    if step_ids != (1, 2):
        import json
        project = tmp_path / "project.json"
        project.write_text(json.dumps({"id": "inspection", "name": "Inspection", "checkpoint": "model.pt",
                                       "steps": [{"id": step_id, "name": "Inspect"} for step_id in step_ids]}))
        argv += ["--project", str(project)]
    monkeypatch.setattr(sys, "argv", argv)
    record_take.main()

    take_dir = out_dir / "take_0000"
    cam_dir = take_dir / "webcam"
    labels = np.load(cam_dir / "labels.npy")
    times = np.load(cam_dir / "times.npy")
    meta = __import__("json").loads((take_dir / "meta.json").read_text(encoding="utf-8"))

    # Run-length sequence: [k for k, _ in groupby(labels)], not sorted(set()).
    from itertools import groupby
    assert [k for k, _ in groupby(labels.tolist())] == list(step_ids)
    assert meta["sequence"] == list(step_ids)
    if step_ids != (1, 2):
        assert meta["project_id"] == "inspection"
    assert meta["n_frames"] == len(labels) == len(times)
    assert meta["real"] is True
    assert meta["tag"] == "real_nominal"

    duration = times[-1]
    assert meta["fps"] == round(len(labels) / duration, 2)   # measured, not FPS config
    assert len(list((cam_dir / "frames").glob("*.png"))) == len(labels)


def test_no_frames_recorded_writes_nothing(tmp_path, monkeypatch, capsys):
    keys = iter([ord("q")])   # quit immediately, never started recording
    monkeypatch.setattr(record_take.cv2, "VideoCapture", lambda *a, **kw: _FakeCapture())
    monkeypatch.setattr(record_take.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(record_take.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(record_take.cv2, "waitKey", lambda *a, **kw: next(keys))
    monkeypatch.setattr(record_take.time, "perf_counter", _fake_clock())

    out_dir = tmp_path / "real"
    monkeypatch.setattr(sys, "argv", ["record_take.py", "--out", str(out_dir)])
    record_take.main()

    # frame_dir.mkdir() runs before the capture loop, so the (empty) folder
    # exists either way; what "no take written" actually means is no meta.
    assert not (out_dir / "take_0000" / "meta.json").exists()
    assert not (out_dir / "take_0000" / "webcam" / "labels.npy").exists()
