"""Fine-tune the payload detector on independently seeded, visibility-labelled takes."""
import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.yolo_detector import CLASSES, DEFAULT_WEIGHTS


def train(root, work, epochs, device, initial_weights=DEFAULT_WEIGHTS):
    from ultralytics import YOLO
    work.mkdir(parents=True, exist_ok=True)
    config = {"names": CLASSES}
    for split in ("train", "val"):
        files = sorted((root / split).glob("take_*/*.png"))
        if split == "train":
            files.extend(sorted((root / "refine_train").glob("take_*/*.png")))
        if not files or any(not f.with_suffix(".txt").is_file() for f in files):
            raise ValueError(f"{split} requires RGB and aligned boxes from export_depth_takes.py")
        listing = work / f"{split}.txt"
        listing.write_text("".join(f"{p.resolve()}\n" for p in files))
        config[split] = str(listing.resolve())
    data = work / "data.yaml"
    data.write_text(yaml.safe_dump(config))
    model = YOLO(initial_weights)
    model.train(data=str(data), epochs=epochs, patience=8, imgsz=448, batch=8,
                workers=0, device=device, amp=False, seed=17, optimizer="AdamW",
                lr0=0.001, freeze=6, degrees=20, mosaic=0.4, close_mosaic=3,
                project=str(work.resolve()), name="payload", plots=False, verbose=False)
    print(f"Validation-selected weights: {model.trainer.best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--init", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    train(args.dataset, args.work, args.epochs, args.device, args.init)
