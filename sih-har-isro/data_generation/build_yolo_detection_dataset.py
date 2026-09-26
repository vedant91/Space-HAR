"""
Synthetic YOLO Detection Dataset Builder
==========================================
Renders (image, bboxes) pairs for the 4 DETECTION_CLASSES (hand/red_box/
yellow_box/main_box) via simulation/renderer.py — same "free, exact,
renderer-driven ground truth" trick train/train_posenet.py uses for pose:
the renderer draws every box and joint from a known position, so pixel-exact
bounding boxes cost nothing extra to produce.

No `world` (MicrogravityWorld) passed to render_frame() here, matching
train_posenet.py's SyntheticPoseDataset — box pixels come purely from
_box_layout()'s deterministic step_id/t/pose function, so the bbox we
compute from that same function always matches what's actually drawn.

Usage:
    python data_generation/build_yolo_detection_dataset.py
    python data_generation/build_yolo_detection_dataset.py --n-samples 500  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    FRAME_WIDTH, FRAME_HEIGHT, DETECTION_CLASSES, YOLO_SYNTH_SAMPLES, YOLO_VAL_SAMPLES,
)
from simulation.renderer import render_frame
from data_generation.synthetic_pose import generate_sequence, L_WRIST, R_WRIST

_LABEL_TO_ID = {v: k for k, v in DETECTION_CLASSES.items()}
HAND_BOX_SIZE = 40  # px, half-extent around each wrist — matches renderer's 16px glove radius + margin


def _hand_bboxes(pose_flat: np.ndarray, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """Both wrists' pixel bboxes — the renderer always draws a glove at each,
    regardless of step, so hand is always "visible" (unlike the boxes)."""
    pose = pose_flat.reshape(-1, 4)
    boxes = []
    for idx in (L_WRIST, R_WRIST):
        cx, cy = int(pose[idx, 0] * w), int(pose[idx, 1] * h)
        s = HAND_BOX_SIZE // 2
        boxes.append((cx - s, cy - s, cx + s, cy + s))
    return boxes


def _clip_and_normalize(box: Tuple[int, int, int, int], w: int, h: int):
    """Clip to frame, return None if degenerate/off-frame, else YOLO-format
    (x_center, y_center, width, height) normalized to [0,1]."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return cx / w, cy / h, bw / w, bh / h


def _yolo_labels_for_frame(meta: Dict, w: int, h: int) -> List[str]:
    lines = []
    boxes = meta["boxes"]
    if meta.get("main_visible"):
        # main_box_open/main_box_closed, not a single main_box class: the
        # renderer now draws a real distinct lid graphic for closed_lid
        # (see simulation/renderer.py), so this is a genuinely learnable
        # visual distinction, not just a label split with identical pixels.
        main_label = "main_box_closed" if meta.get("closed_lid") else "main_box_open"
        norm = _clip_and_normalize(boxes["main_box"], w, h)
        if norm:
            lines.append(f"{_LABEL_TO_ID[main_label]} " + " ".join(f"{v:.6f}" for v in norm))
    person_bbox = meta.get("person_bbox")
    if person_bbox is not None:
        norm = _clip_and_normalize(person_bbox, w, h)
        if norm:
            lines.append(f"{_LABEL_TO_ID['person']} " + " ".join(f"{v:.6f}" for v in norm))
    if meta.get("red_visible"):
        norm = _clip_and_normalize(boxes["red_box"], w, h)
        if norm:
            lines.append(f"{_LABEL_TO_ID['red_box']} " + " ".join(f"{v:.6f}" for v in norm))
    if meta.get("yellow_visible"):
        norm = _clip_and_normalize(boxes["yellow_box"], w, h)
        if norm:
            lines.append(f"{_LABEL_TO_ID['yellow_box']} " + " ".join(f"{v:.6f}" for v in norm))
    for hb in _hand_bboxes(meta["pose"], w, h):
        norm = _clip_and_normalize(hb, w, h)
        if norm:
            lines.append(f"{_LABEL_TO_ID['hand']} " + " ".join(f"{v:.6f}" for v in norm))
    return lines


def _generate_split(out_dir: Path, split: str, n_samples: int, seed: int,
                     width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> int:
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    steps = list(range(1, 9))
    per_step = max(1, n_samples // len(steps))
    written = 0
    for step_id in steps:
        made = 0
        while made < per_step:
            ori = int(rng.integers(0, 4))
            n_frames = int(rng.integers(12, 30))
            seq = generate_sequence(step_id, n_frames=n_frames, rng=rng, orientation=ori, drift=True)
            pick = rng.choice(n_frames, size=min(4, n_frames), replace=False)
            for i in pick:
                if made >= per_step:
                    break
                t = float(i) / max(n_frames - 1, 1)
                frame, meta = render_frame(seq[i], step_id, t, width=width, height=height, rng=rng)
                lines = _yolo_labels_for_frame(meta, width, height)
                stem = f"s{step_id}_{written:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                written += 1
                made += 1
    return written


def build(out_dir: str = "dataset/yolo_detection", n_train: int = YOLO_SYNTH_SAMPLES,
          n_val: int = YOLO_VAL_SAMPLES, seed: int = 21) -> Dict:
    out = Path(out_dir)
    n_written_train = _generate_split(out, "train", n_train, seed=seed)
    n_written_val = _generate_split(out, "val", n_val, seed=seed + 1000)

    names = {i: name for i, name in DETECTION_CLASSES.items()}
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {name}\n" for i, name in names.items())
    )
    return {"n_train": n_written_train, "n_val": n_written_val, "data_yaml": str(data_yaml)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic YOLO detection dataset")
    parser.add_argument("--out-dir", type=str, default="dataset/yolo_detection")
    parser.add_argument("--n-samples", type=int, default=YOLO_SYNTH_SAMPLES)
    parser.add_argument("--n-val", type=int, default=YOLO_VAL_SAMPLES)
    args = parser.parse_args()

    import json
    result = build(out_dir=args.out_dir, n_train=args.n_samples, n_val=args.n_val)
    print(json.dumps(result, indent=2))
