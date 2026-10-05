"""
Fine-tune the YOLOv8n box detector on real frames mixed with Blender renders.

Why mix: trained on renders alone the detector scores 0.93 mAP@50 on rendered
validation but only 0.36 on a real clip it has never seen (it boxes the helmet
as the case). Real frames, labelled offline by tools/pseudo_label_boxes.py,
close much of that gap: 0.63 on the same unseen clip with this recipe (0.80 with an
earlier, less strict labelling). Real frames are a few hundred
against thousands of renders, so they are repeated (--repeat-real) to keep the
new domain from being drowned out.

Honest evaluation: `--holdout <clip>` keeps a WHOLE clip out of training and
validates on it. Without --holdout the real frames are split into contiguous
24-frame chunks (85 / 15), never frame by frame, because neighbouring frames
are near-duplicates and a frame split would leak.

    # measure generalisation: hold one clip out
    python train/train_yolo_boxes.py --real dataset/yolo_boxes_real/labelled \
        --synthetic dataset/yolo_boxes/train.txt --holdout test-har-2 --epochs 20

    # final deployment weights: use every clip (the score is then NOT held out)
    python train/train_yolo_boxes.py --real dataset/yolo_boxes_real/labelled \
        --synthetic dataset/yolo_boxes/train.txt --holdout none --epochs 8

Writes <out> (default models/yolo_boxes_finetuned.pt); it never overwrites the
deployed models/yolo_boxes.pt by itself. Promote deliberately.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["red_box", "yellow_box", "main_box"]


def clip_images(real_dir: Path) -> Dict[str, List[str]]:
    """{clip name: sorted image paths} for every labelled clip folder."""
    clips = {}
    for d in sorted(p for p in real_dir.iterdir() if p.is_dir()):
        imgs = sorted(str(p.resolve()) for p in d.glob("*.jpg") if p.with_suffix(".txt").exists())
        if imgs:
            clips[d.name] = imgs
    return clips


def chunk_split(images: List[str], val_fraction: float, rng: random.Random, chunk: int = 24):
    """Split by contiguous chunks of frame numbers so near-duplicate
    neighbours never straddle train / val."""
    groups: Dict[tuple, List[str]] = {}
    for f in images:
        clip, idx = Path(f).stem.rsplit("_", 1)
        groups.setdefault((clip, int(idx) // chunk), []).append(f)
    keys = sorted(groups)
    rng.shuffle(keys)
    val_keys = set(keys[:max(1, int(val_fraction * len(keys)))])
    train = [f for k in keys if k not in val_keys for f in groups[k]]
    val = [f for k in val_keys for f in groups[k]]
    return train, val


def read_list(path: Optional[Path]) -> List[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()] if path and path.exists() else []


def build(args, rng: random.Random) -> Path:
    clips = clip_images(args.real)
    if not clips:
        raise SystemExit(f"no labelled clips under {args.real} (run tools/pseudo_label_boxes.py first)")
    if args.holdout.lower() != "none":
        if args.holdout not in clips:
            raise SystemExit(f"--holdout {args.holdout!r} not among {sorted(clips)}")
        real_val = clips[args.holdout]
        real_train = [f for c, imgs in clips.items() if c != args.holdout for f in imgs]
        print(f"held out whole clip {args.holdout!r}: {len(real_val)} validation frames")
    else:
        real_train, real_val = chunk_split([f for imgs in clips.values() for f in imgs], 0.15, rng)
        print("no clip held out: chunked 85/15 split (NOT a held-out score)")

    synth = read_list(args.synthetic)
    synth_train = rng.sample(synth, min(args.synthetic_count, len(synth))) if synth else []
    synth_val = read_list(args.synthetic_val)
    synth_val = rng.sample(synth_val, min(args.synthetic_val_count, len(synth_val))) if synth_val else []

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    (work / "train.txt").write_text("\n".join(real_train * args.repeat_real + synth_train))
    (work / "val.txt").write_text("\n".join(real_val + synth_val))
    yaml = work / "data.yaml"
    yaml.write_text(f"train: {(work / 'train.txt').resolve()}\nval: {(work / 'val.txt').resolve()}\n"
                    f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    print(f"train: {len(real_train)} real x{args.repeat_real} + {len(synth_train)} synthetic | "
          f"val: {len(real_val)} real + {len(synth_val)} synthetic")
    return yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, default=ROOT / "dataset/yolo_boxes_real/labelled")
    ap.add_argument("--synthetic", type=Path, default=ROOT / "dataset/yolo_boxes/train.txt",
                    help="text file: one rendered-image path per line")
    ap.add_argument("--synthetic-val", type=Path, default=ROOT / "dataset/yolo_boxes/val.txt")
    ap.add_argument("--synthetic-count", type=int, default=1200)
    ap.add_argument("--synthetic-val-count", type=int, default=200)
    ap.add_argument("--repeat-real", type=int, default=3)
    ap.add_argument("--holdout", default="none", help="clip name to hold out entirely, or 'none'")
    ap.add_argument("--init", type=Path, default=ROOT / "models/yolo_boxes.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "models/yolo_boxes_finetuned.pt")
    ap.add_argument("--work", type=Path, default=ROOT / "dataset/yolo_boxes_real/_train")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2,
                    help="dataloader workers; 4+ crashes a 16 GB Windows laptop, 0 is ~5x slower")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_yaml = build(args, random.Random(args.seed))

    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # offline-safe: never let ultralytics pip-install
    from ultralytics import YOLO
    model = YOLO(str(args.init))
    run_dir = ROOT / "runs" / "yolo_boxes"
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=16, workers=args.workers,
                patience=args.patience, lr0=0.003, cos_lr=True, close_mosaic=4, degrees=10, scale=0.5,
                project=str(run_dir), name="finetune", exist_ok=True, plots=False, verbose=False)
    best = run_dir / "finetune" / "weights" / "best.pt"
    last = run_dir / "finetune" / "weights" / "last.pt"
    chosen = best if best.exists() else last
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(chosen, args.out)
    metrics = YOLO(str(args.out)).val(data=str(data_yaml), imgsz=640, batch=16, workers=0, verbose=False, plots=False)
    print(f"saved {args.out} | validation mAP@50 {metrics.box.map50:.3f}, mAP@50-95 {metrics.box.map:.3f}")


def _selftest() -> None:
    rng = random.Random(0)
    imgs = [f"/x/clipA_{i:05d}.jpg" for i in range(0, 240, 2)] + [f"/x/clipB_{i:05d}.jpg" for i in range(0, 240, 2)]
    train, val = chunk_split(imgs, 0.15, rng)
    assert set(train) | set(val) == set(imgs) and not set(train) & set(val)

    def chunks(fs):
        return {(Path(f).stem.rsplit("_", 1)[0], int(Path(f).stem.rsplit("_", 1)[1]) // 24) for f in fs}
    assert not chunks(train) & chunks(val), "neighbouring frames leaked across the split"
    print("train_yolo_boxes selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
