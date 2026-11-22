import numpy as np
import torch

from pipeline.pose_refiner import PoseRefiner


def test_zero_residual_preserves_image_coordinates_and_centers_depth():
    model = PoseRefiner(width=16).eval()
    rng = np.random.default_rng(7)
    pose = rng.uniform(0, 1, 132).astype(np.float32)
    refined = model.refine(pose, pose, pose, (640, 360)).reshape(33, 4)
    original = pose.reshape(33, 4)
    np.testing.assert_allclose(refined[:, [0, 1, 3]], original[:, [0, 1, 3]], atol=1e-6)
    assert abs(refined[[23, 24], 2].mean()) < 1e-6


def test_unavailable_rgb_keeps_lifted_pose():
    model = PoseRefiner(width=16)
    pose = np.ones(132, dtype=np.float32)
    for rgb in (None, np.zeros_like(pose), np.full_like(pose, np.nan)):
        assert model.refine(pose, rgb, pose, (640, 360)) is pose


def test_load_uses_saved_width(tmp_path):
    path = tmp_path / "refiner.pt"
    model = PoseRefiner(width=32).eval()
    torch.save(dict(format="bas-pose-refiner-v1", width=32, model=model.state_dict()), path)
    x = torch.randn((1, 33, 12))
    torch.testing.assert_close(PoseRefiner.load(path)(x), model(x))


def test_yolo_class_profile_and_explicit_override(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ultralytics
    from pipeline.yolo_detector import CLASSES, YOLOBoxDetector

    path = tmp_path / "payload.pt"
    path.touch()
    path.with_suffix(".thresholds.json").write_text('{"red_box": 0.55, "main_box": 0.15}')
    model = SimpleNamespace(names=dict(enumerate(CLASSES)))
    monkeypatch.setattr(ultralytics, "YOLO", lambda _: model)
    assert YOLOBoxDetector(weights=path).thresholds == {
        "red_box": .55, "yellow_box": .25, "main_box": .15,
    }
    assert set(YOLOBoxDetector(weights=path, conf=.25).thresholds.values()) == {.25}
