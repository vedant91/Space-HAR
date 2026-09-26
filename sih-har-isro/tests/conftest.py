"""Shared fixtures. Only fixtures genuinely reused by 2+ test modules live
here -- see docs/designs/step-classifier-generalisation.md, Review Outcomes,
for what these tests are pinning down."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BLENDER_TWIN_FRAMES = sorted(
    (REPO / "dataset" / "blender_twin" / "raw_frames").glob("frame_*.png")
)


@pytest.fixture
def repo_root() -> Path:
    return REPO


def _tiny_png(path: Path, w: int = 48, h: int = 32) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


@pytest.fixture
def make_take(tmp_path):
    """Factory: build one synthetic rendered take directory under `root`
    (default a fresh tmp_path subdir), matching the schema
    blender/render_dataset.py writes and tools/check_dataset.py /
    tools/build_sequences.py read.

    Returns the take directory path.
    """
    def _make(root: Path | None = None, index: int = 0, tag: str = "nominal",
              n_frames: int = 12, sequence=(1, 2), camera: str = "payload_a",
              real: bool = False, wrist_teleport: bool = False,
              frames_written: bool = True, fps: int = 30) -> Path:
        root = root or (tmp_path / "dataset")
        take_dir = root / f"take_{index:04d}"
        cam_dir = take_dir / camera
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Labels: split n_frames evenly across `sequence`, run-length encoded
        # (each step is a contiguous block), so set(labels) == set(sequence).
        labels = np.zeros(n_frames, dtype=np.int64)
        bounds = np.linspace(0, n_frames, len(sequence) + 1, dtype=int)
        for k, step in enumerate(sequence):
            labels[bounds[k]:bounds[k + 1]] = step
        np.save(cam_dir / "labels.npy", labels)

        meta = {
            "index": index, "tag": tag, "cameras": [camera],
            "n_frames": n_frames, "fps": fps, "sequence": list(sequence),
            "frames_written": frames_written, "real": real,
            "body_roll_deg": 0.0,
        }

        if real:
            if frames_written:
                for i in range(n_frames):
                    _tiny_png(cam_dir / "frames" / f"{i:05d}.png")
        else:
            pose2d = np.zeros((n_frames, 132), dtype=np.float32)
            pose2d[:, 3::4] = 0.9  # visibility channel
            world = np.zeros((n_frames, 33, 3), dtype=np.float32)
            if wrist_teleport and n_frames > 1:
                world[n_frames // 2, 15, :] = 5.0  # a 5 m jump in one frame
            np.save(cam_dir / "pose_2d.npy", pose2d)
            np.save(cam_dir / "pose_world.npy", world)
            payload = [{
                "red_box": {"visible": False, "bbox": [0, 0, 1, 1]},
                "yellow_box": {"visible": False, "bbox": [0, 0, 1, 1]},
                "main_box": {"visible": False, "bbox": [0, 0, 1, 1]},
            } for _ in range(n_frames)]
            (cam_dir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
            if frames_written:
                for i in range(n_frames):
                    _tiny_png(cam_dir / "frames" / f"{i:05d}.png")

        (take_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return take_dir

    return _make


@pytest.fixture
def tiny_lstm_ckpt(tmp_path):
    """Factory: save a small HARLSTMClassifier checkpoint with a given
    (possibly self-contradictory, for regression tests) contract."""
    def _make(name: str, feature_dim: int = 143, hidden_size: int = 4,
              num_layers: int = 1, num_classes: int = 2,
              feature_layout: str | None = "pose132+obj11",
              sample_every: int = 3, rack_normalized: bool | None = True,
              window: int = 30) -> Path:
        import torch
        from train.train_lstm import HARLSTMClassifier

        model = HARLSTMClassifier(feature_dim=feature_dim, hidden_size=hidden_size,
                                  num_layers=num_layers, num_classes=num_classes)
        ckpt = {
            "model_state": model.state_dict(), "feature_dim": feature_dim,
            "hidden_size": hidden_size, "num_layers": num_layers,
            "num_classes": num_classes,
            "idx_to_label": {i: i + 1 for i in range(num_classes)},
            "val_acc": 0.5, "window": window,
        }
        if feature_layout is not None:
            ckpt["feature_layout"] = feature_layout
        if sample_every is not None:
            ckpt["sample_every"] = sample_every
        if rack_normalized is not None:
            ckpt["rack_normalized"] = rack_normalized
        path = tmp_path / name
        torch.save(ckpt, path)
        return path

    return _make


def require_blender_twin_frames(n: int):
    if len(BLENDER_TWIN_FRAMES) < n:
        pytest.skip(f"need {n} frames under dataset/blender_twin/raw_frames, "
                    f"found {len(BLENDER_TWIN_FRAMES)}")
    return BLENDER_TWIN_FRAMES[:n]


def import_tool(name: str):
    """Import tools/<name>.py by file path, bypassing `import tools.<name>`.

    This machine has an unrelated `tools` regular package installed in
    site-packages (MCP server tooling); regular packages win over this
    repo's namespace-package `tools/` directory (no __init__.py) regardless
    of sys.path order, so `import tools.build_sequences` resolves to the
    wrong package here. Loading by explicit file path sidesteps that.
    """
    import importlib.util

    mod_name = f"_repo_tools_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
