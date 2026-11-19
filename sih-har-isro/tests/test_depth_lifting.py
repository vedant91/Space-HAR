import numpy as np
import pytest

from pipeline.depth_lifter import DepthLifter, encode, refine_image_pose
from pipeline.pose_viz import HipDepthFilter, display_body
from pipeline.hmr_backend import HMRBackend


def pose():
    p = np.random.default_rng(8).uniform(.2, .8, size=(33, 4)).astype(np.float32)
    p[:, 3] = 1
    p[[11, 12], :2] = [[.4, .3], [.6, .3]]
    p[[23, 24], :2] = [[.45, .7], [.55, .7]]
    return p


def test_lift_is_roll_invariant_and_does_not_read_input_depth():
    p = pose()
    xy = p[:, :2] * (640, 360)
    q = p.copy()
    q[:, :2] = np.column_stack((360 - xy[:, 1], xy[:, 0])) / (360, 640)
    q[:, 2] = 100
    a, scale_a, _ = encode(p, (640, 360))
    b, scale_b, _ = encode(q, (360, 640))
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(scale_a * 640, scale_b * 360)
    model = DepthLifter().eval()
    r = model.lift(q, (360, 640)).reshape(33, 4)
    np.testing.assert_array_equal(q[:, [0, 1, 3]], r[:, [0, 1, 3]])
    assert abs(r[[23, 24], 2].mean()) < 1e-6


def test_missing_torso_and_bad_joints():
    model = DepthLifter().eval()
    p = pose()
    p[15, 0] = np.nan
    result = model.lift(p, (640, 360)).reshape(33, 4)
    assert np.isfinite(result).all() and result[15, 3] == 0
    p[23, 3] = 0
    assert model.lift(p, (640, 360)) is None


def test_display_scale_includes_out_of_plane_torso():
    p = pose()
    p[[11, 12], :2] = p[[23, 24], :2]
    p[[11, 12], 2] = 1.5
    body = display_body(p, (640, 360))
    np.testing.assert_allclose(body[[23, 24], :3].mean(0), 0, atol=1e-6)
    assert np.linalg.norm(body[[11, 12], :3].mean(0)) == pytest.approx(.5)


def test_fusion_uses_same_coordinates_as_evaluation():
    model = DepthLifter().eval()
    model.skeleton_weight = 0.4
    p = pose()
    rgb = p.copy()
    rgb[:, 0] += .02
    candidate = model.lift(p, (640, 360), rgb).reshape(33, 4)
    expected = refine_image_pose(p, rgb, (640, 360))
    np.testing.assert_allclose(candidate[:, :2], expected[:, :2])
    alone = model.lift(p, (640, 360)).reshape(33, 4)
    z = rgb[:, 2] - rgb[[23, 24], 2].mean()
    np.testing.assert_allclose(candidate[:, 2], .4 * alone[:, 2] + .6 * z, atol=1e-6)


def test_custom_hmr_follows_reach_at_constant_roll_and_clears_lost_track():
    hb = HMRBackend("none")
    hb.reset()
    hb._person = None
    boxes = []
    def infer(image, box, angle):
        boxes.append(tuple(box))
        return np.ones(132), 1
    hb._custom_at = infer
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    p = pose()
    hb.custom_from_pose(frame, p)
    p[15, 0] = .95
    hb.custom_from_pose(frame, p)
    assert len(boxes) == 2 and boxes[1][2] > boxes[0][2]
    for _ in range(8):
        absent = hb.custom_from_pose(frame, np.zeros_like(p))
    assert not absent.any()


def test_depth_filter_restarts_after_gap_and_keeps_hips_centered():
    filt = HipDepthFilter()
    filt(np.zeros(33), 0)
    z = np.linspace(-.5, .5, 33)
    near = filt(z, .1)
    expected = z - z[[23, 24]].mean()
    assert not np.allclose(near, expected)
    np.testing.assert_allclose(filt(z, 2), expected)


def test_six_class_checkpoint_maps_lid_states_without_relabelling_people(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ultralytics
    from pipeline.yolo_detector import YOLOBoxDetector

    weights = tmp_path / "six.pt"
    weights.touch()
    names = dict(enumerate(("hand", "red_box", "yellow_box", "main_box_open", "main_box_closed", "person")))
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: SimpleNamespace(names=names))
    assert YOLOBoxDetector(weights=weights).labels == {1: "red_box", 2: "yellow_box", 3: "main_box", 4: "main_box"}


def test_missing_render_frame_does_not_shift_box_annotations(tmp_path):
    import json
    import cv2
    from tools.export_yolo_labels import write_labels_for_take

    cam = tmp_path / "payload_a"
    (cam / "frames").mkdir(parents=True)
    (tmp_path / "meta.json").write_text(json.dumps({"cameras": ["payload_a"]}))
    (cam / "payload.json").write_text(json.dumps([{}, {}, {"yellow_box": {"visible": True, "bbox": [1, 2, 6, 9]}}]))
    frame = cam / "frames/00002.png"
    cv2.imwrite(str(frame), np.zeros((10, 10, 3), dtype=np.uint8))
    assert write_labels_for_take(tmp_path) == (1, [frame])
    assert frame.with_suffix(".txt").read_text().split()[0] == "1"
