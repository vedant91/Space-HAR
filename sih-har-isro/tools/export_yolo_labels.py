"""
Convert Blender's ground-truth payload.json into YOLO detection labels.

Classes are exactly what the ground truth actually tracks - red_box (0),
yellow_box (1), main_box (2). No lid-open/closed distinction: groundtruth.py's
sample_payload() (blender/bas_har/groundtruth.py:225) has no such field, only
a bbox + visible_fraction + visible + on_screen per box. Inventing a 4th/5th
class the data doesn't back would just train on labels nobody generated.

A box's own `visible` flag (already thresholded in groundtruth.py at the
smallest visible_fraction that still yields a usable colour patch) decides
whether a YOLO label is written for it on that frame - the same real/absent
distinction the detector will have to make live.

Writes one .txt label per frame, in the SAME directory as the frame's PNG
(dataset/blender_core/take_*/payload_a/frames/*.txt next to *.png) - not a
separate labels/ tree - because ultralytics' path-based label lookup
(img2label_paths) only rewrites an "images" segment in the path; since ours
is ".../frames/...", it falls through to same-directory-same-basename
lookup, which this matches exactly, without touching the existing dataset
layout every other tool in this repo already depends on.

Split by TAKE, never by frame - the take_0022 (different camera, "heldout_view")
generalisation take is always in val, same convention as train/train_hmr3d.py
and train/train_lstm.py, and for the same reason: a held-out FRAME from a
take the model has otherwise trained on tells you almost nothing.

Run: python tools/export_yolo_labels.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASSES = ["red_box", "yellow_box", "main_box"]
HELDOUT_VIEW_TAKE = 22


def write_labels_for_take(take_dir: Path) -> tuple[int, list[Path]]:
    meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
    cam = None
    for cam_name in meta.get("cameras", []):
        d = take_dir / cam_name
        if d.is_dir():
            cam = d
            break
    if cam is None:
        raise FileNotFoundError(f"{take_dir}: none of meta['cameras']={meta.get('cameras')} exist")

    payload = json.loads((cam / "payload.json").read_text(encoding="utf-8"))
    frames = sorted((cam / "frames").glob("*.png"))
    n = min(len(frames), len(payload))

    written = []
    for i in range(n):
        w, h = _image_size(frames[i])
        lines = []
        for cls_id, key in enumerate(CLASSES):
            box = payload[i].get(key)
            if box is None or not box.get("visible"):
                continue
            x1, y1, x2, y2 = box["bbox"]
            x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
            y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            lines.append(f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        label_path = frames[i].with_suffix(".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        written.append(frames[i])
    return n, written


_size_cache: dict = {}


def _image_size(path: Path) -> tuple[int, int]:
    if path.parent not in _size_cache:
        import cv2
        img = cv2.imread(str(path))
        _size_cache[path.parent] = (img.shape[1], img.shape[0])
    return _size_cache[path.parent]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "dataset" / "blender_core"))
    ap.add_argument("--out", default=str(REPO / "dataset" / "yolo_boxes"))
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    take_dirs = sorted(Path(args.dataset).glob("take_*"))
    all_images: dict[int, list[Path]] = {}
    for d in take_dirs:
        idx = int(d.name.split("_")[1])
        meta_path = d / "meta.json"
        if not meta_path.exists():
            # render_dataset.py writes meta.json last, after every frame and
            # ground-truth file for the take - missing it means a render is
            # still in progress on this take. Skip it now; re-running this
            # script later (it's idempotent) picks it up once it's done.
            print(f"take_{idx:04d}: meta.json not written yet (render in progress), skipping")
            continue
        # Skip protocol-error takes (skip/recover) for the detector: they
        # exercise the state machine, not new box appearances, so they add
        # training time without teaching the detector anything new.
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("tag") in ("skip", "recover"):
            continue
        n, images = write_labels_for_take(d)
        all_images[idx] = images
        print(f"take_{idx:04d} ({meta.get('tag', '?')}): {n} frames labelled")

    import numpy as np
    takes = sorted(all_images.keys())
    rng = np.random.default_rng(args.seed)
    others = [t for t in takes if t != HELDOUT_VIEW_TAKE]
    rng.shuffle(others)
    n_val = max(1, int(round(args.val_fraction * len(takes)))) - 1
    val_takes = set([HELDOUT_VIEW_TAKE] + others[:max(n_val, 0)]) & set(takes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [p for t in takes if t not in val_takes for p in all_images[t]]
    val_paths = [p for t in val_takes for p in all_images[t]]
    (out_dir / "train.txt").write_text("\n".join(str(p.resolve()) for p in train_paths), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(str(p.resolve()) for p in val_paths), encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        f"train: {(out_dir / 'train.txt').resolve()}\n"
        f"val: {(out_dir / 'val.txt').resolve()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8")

    print(f"\ntrain images: {len(train_paths)}  val images: {len(val_paths)}")
    print(f"val takes: {sorted(val_takes)}")
    print(f"wrote {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
