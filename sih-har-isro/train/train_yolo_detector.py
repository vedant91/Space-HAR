"""
YOLO Detector — Fine-tune Script (opt-in, DETECTOR_BACKEND="yolo")
====================================================================
Fine-tunes a pretrained COCO YOLO checkpoint (config.YOLO_BASE_CHECKPOINT) on
our own synthetic renderer bboxes for the 6 DETECTION_CLASSES (hand,
red_box, yellow_box, main_box_open, main_box_closed, person) — COCO itself
has no red_box/yellow_box/main_box classes, so the pretrained weights are
only ever a feature-extractor starting point, never used as-is at
inference time. COCO DOES have a person class, but we don't rely on that
surviving fine-tuning — this dataset provides our own explicit synthetic
person bboxes (simulation/renderer.py's person_bbox meta field) as one of
this fine-tune's own trained classes, same as every other class here, so
there's no catastrophic-forgetting risk to guard against.

hsv_detector.py's own docstring already concludes HSV beats a trained
detector for this exact task (measured 1.000/1.000 precision/recall). This
script exists to measure YOLO honestly against that, not to replace it by
default — DETECTOR_BACKEND stays "hsv" unless explicitly flipped.

Usage:
    python train/train_yolo_detector.py                  # build dataset (if missing) + train + benchmark
    python train/train_yolo_detector.py --skip-build      # reuse an existing dataset/yolo_detection/
    python train/train_yolo_detector.py --n-samples 300 --epochs 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    FRAME_WIDTH, FRAME_HEIGHT, YOLO_BASE_CHECKPOINT, YOLO_IMG_SIZE,
    YOLO_EPOCHS, YOLO_BATCH_SIZE, YOLO_SYNTH_SAMPLES, YOLO_VAL_SAMPLES,
    YOLO_MODEL_PATH, E2E_MAX_MEAN_LATENCY_MS, E2E_MAX_P95_LATENCY_MS,
)


def _ensure_dataset(out_dir: str, n_samples: int, n_val: int) -> str:
    data_yaml = Path(out_dir) / "data.yaml"
    if data_yaml.exists():
        print(f"Reusing existing dataset at {data_yaml}")
        return str(data_yaml)
    from data_generation.build_yolo_detection_dataset import build
    result = build(out_dir=out_dir, n_train=n_samples, n_val=n_val)
    print(f"Built synthetic detection dataset: {json.dumps(result)}")
    return result["data_yaml"]


def _benchmark_latency(model_path: str, n_frames: int = 100) -> dict:
    """Honest CPU per-frame inference latency — same measurement spirit as
    the rest of this repo's models (README's mean/p95 table). Runs on
    random full-resolution frames (worst case: no letterbox cache warm-up
    beyond the first call)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Warm-up (first call pays one-time graph/cache setup cost — exclude it,
    # same convention as benchmarking any other model in this repo).
    model.predict(frame, imgsz=YOLO_IMG_SIZE, device="cpu", verbose=False)

    times_ms = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=YOLO_IMG_SIZE, device="cpu", verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    mean_ms = float(np.mean(times_ms))
    p95_ms = float(times_ms[int(0.95 * len(times_ms))])
    return {"mean_ms": mean_ms, "p95_ms": p95_ms, "n": n_frames}


def train_yolo_detector(out_dir: str = "dataset/yolo_detection",
                         n_samples: int = YOLO_SYNTH_SAMPLES,
                         n_val: int = YOLO_VAL_SAMPLES,
                         epochs: int = YOLO_EPOCHS,
                         batch: int = YOLO_BATCH_SIZE,
                         imgsz: int = YOLO_IMG_SIZE,
                         base_checkpoint: str = YOLO_BASE_CHECKPOINT,
                         output_path: str = YOLO_MODEL_PATH,
                         skip_build: bool = False) -> dict:
    from ultralytics import YOLO

    if skip_build and not Path(out_dir, "data.yaml").exists():
        raise FileNotFoundError(f"--skip-build given but {out_dir}/data.yaml doesn't exist")
    data_yaml = str(Path(out_dir) / "data.yaml") if skip_build else _ensure_dataset(out_dir, n_samples, n_val)

    model = YOLO(base_checkpoint)  # pretrained COCO weights — fine-tune starting point only
    run_name = "har_yolo_detector"
    train_result = model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device="cpu",
        project="runs/yolo_detect", name=run_name, exist_ok=True, verbose=True,
        patience=8,
    )
    best_pt = Path(train_result.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but no best.pt found at {best_pt}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_pt, output_path)
    print(f"Best checkpoint copied to {output_path}")

    metrics = model.val(data=data_yaml, imgsz=imgsz, device="cpu")
    map50 = float(metrics.box.map50)
    print(f"val mAP50={map50:.3f}")

    latency = _benchmark_latency(output_path)
    print(f"CPU inference latency: mean={latency['mean_ms']:.1f}ms p95={latency['p95_ms']:.1f}ms "
          f"(n={latency['n']}) — for comparison, the WHOLE existing pipeline's gate is "
          f"mean<={E2E_MAX_MEAN_LATENCY_MS}ms/p95<={E2E_MAX_P95_LATENCY_MS}ms, "
          f"currently met at ~25ms/~27ms without any detector cost (HSV is near-zero).")

    return {"map50": map50, "latency": latency, "checkpoint": output_path,
            "base_checkpoint": base_checkpoint}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on synthetic box/hand bboxes")
    parser.add_argument("--out-dir", type=str, default="dataset/yolo_detection")
    parser.add_argument("--n-samples", type=int, default=YOLO_SYNTH_SAMPLES)
    parser.add_argument("--n-val", type=int, default=YOLO_VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    parser.add_argument("--batch", type=int, default=YOLO_BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMG_SIZE)
    parser.add_argument("--base-checkpoint", type=str, default=YOLO_BASE_CHECKPOINT)
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse an existing dataset/yolo_detection/ instead of rebuilding")
    args = parser.parse_args()

    result = train_yolo_detector(
        out_dir=args.out_dir, n_samples=args.n_samples, n_val=args.n_val,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        base_checkpoint=args.base_checkpoint, skip_build=args.skip_build,
    )
    print(json.dumps(result, indent=2, default=str))
