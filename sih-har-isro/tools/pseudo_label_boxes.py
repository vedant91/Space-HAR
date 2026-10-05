"""
Label real footage for the box detector without hand annotation.

The box detector (pipeline/yolo_detector.py) is first trained on Blender
renders. On real or AI-generated video it then fails in a specific way: it
boxes the astronaut's helmet as "the case" and misses the real boxes (see
docs/readme/detector-before-after.png). Fixing that needs labelled real frames,
and nobody wants to hand-label hundreds of them.

This tool labels them offline with a large open-vocabulary detector
(Grounding DINO) used ONLY as a teacher: far too heavy for the live loop, but
good enough to author training labels. A teacher label is kept only if the
pixels agree with it, because the teacher is wrong often enough to matter:

  * red_box / yellow_box: that hue must actually fill the box (HSV test).
  * main_box: must be unsaturated metal, must not contain mostly red/yellow,
    and must be the case BODY - never the open lid and never body+lid. The lid
    stands upright above the body, so of several candidates the smallest one
    holding the boxes wins, else the lowest wide one.
  * the case barely moves inside a clip, so its box is median-filtered over
    +-4 frames to remove one-off lid / oversized picks.
  * a frame where a box vanishes between two frames that both show it is a
    teacher miss, not an empty scene, and is dropped.

Output per clip (YOLO format, classes: 0 red_box, 1 yellow_box, 2 main_box):

    <out>/<clip>/<frame>.jpg   <out>/<clip>/<frame>.txt   <out>/qa_<clip>.jpg

Always look at the qa_<clip>.jpg contact sheet before training on a clip.

    python tools/pseudo_label_boxes.py --video clipA.mp4 clipB.mp4 \
        --out dataset/yolo_boxes_real/labelled --stride 2

Needs `transformers` and, once, internet to fetch the teacher weights
(authoring time only: nothing here runs on board).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASSES = ["red_box", "yellow_box", "main_box"]
TEACHER = "IDEA-Research/grounding-dino-base"
PROMPT = "a red box. a yellow box. a metal storage case."
Box = List[float]


def hue_fractions(img: np.ndarray, box: Box) -> Tuple[float, float, float]:
    """Fraction of the box that is red, yellow, and unsaturated metal."""
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    red = (((h < 10) | (h > 168)) & (s > 90) & (v > 50)).mean()
    yel = ((h > 14) & (h < 38) & (s > 90) & (v > 80)).mean()
    grey = ((s < 60) & (v > 40)).mean()
    return float(red), float(yel), float(grey)


def inside(a: Box, o: Box) -> float:
    """Share of box `a`'s area that lies inside box `o`."""
    ix = max(0, min(a[2], o[2]) - max(a[0], o[0])) * max(0, min(a[3], o[3]) - max(a[1], o[1]))
    return ix / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def area(b: Box) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def verify_frame(img: np.ndarray, dets: List[dict]) -> Dict[int, Tuple[float, Box]]:
    """Teacher detections for one frame -> {class_id: (quality, box)}."""
    h_img, w_img = img.shape[:2]
    best: Dict[int, Tuple[float, Box]] = {}
    cases: List[Tuple[float, Box]] = []
    for d in dets:
        text, score, box = d["text"], d["score"], d["xyxy"]
        frac = area(box) / (w_img * h_img)
        red, yel, grey = hue_fractions(img, box)
        if "case" in text:
            if score >= 0.2 and grey >= 0.45 and red < 0.25 and yel < 0.25 and 0.02 < frac < 0.5:
                cases.append((score, box))
            continue
        if "red" in text and "yellow" not in text:
            ok, cls, quality = score >= 0.4 and red >= 0.2 and 0.005 < frac < 0.25, 0, score * red
        elif "yellow" in text and "red" not in text:
            ok, cls, quality = score >= 0.4 and yel >= 0.2 and 0.005 < frac < 0.25, 1, score * yel
        else:
            continue
        if ok and quality > best.get(cls, (0.0,))[0]:
            best[cls] = (quality, box)

    # a candidate that swallows another candidate is a body+lid union
    cases = [(s, b) for s, b in cases
             if not any(o is not b and inside(o, b) > 0.8 and area(o) < 0.8 * area(b) for _, o in cases)]
    holding = [(s, b) for s, b in cases
               if sum(inside(best[c][1], b) for c in (0, 1) if c in best) >= 0.5]
    pick: Optional[Tuple[float, Box]] = None
    if holding:
        pick = min(holding, key=lambda t: area(t[1]))
    else:
        wide = [(s, b) for s, b in cases if s >= 0.3 and (b[2] - b[0]) >= 0.95 * (b[3] - b[1])]
        if wide:
            lowest = max(wide, key=lambda t: t[1][3])
            # a lone candidate high in the frame with nothing below it is a lid
            if len(wide) > 1 or lowest[1][3] > 0.55 * h_img:
                pick = lowest
    if pick:
        best[2] = pick
    return best


def smooth_case(per_frame: List[Dict[int, Tuple[float, Box]]], radius: int = 4) -> None:
    """Median-filter the case box over +-radius frames, in place."""
    cases = [f[2][1] if 2 in f else None for f in per_frame]
    for k, f in enumerate(per_frame):
        window = [c for c in cases[max(0, k - radius):k + radius + 1] if c is not None]
        if len(window) >= 3:
            f[2] = (1.0, [float(v) for v in np.median(np.array(window), axis=0)])


def drop_teacher_misses(per_frame: List[Dict[int, Tuple[float, Box]]]) -> List[bool]:
    """Keep a frame only if it has the case and no box that vanishes between
    two neighbours that both show it."""
    keep = []
    for k, f in enumerate(per_frame):
        if 2 not in f:
            keep.append(False)
            continue
        nb = [set(per_frame[j]) for j in (k - 1, k + 1) if 0 <= j < len(per_frame)]
        miss = any(c not in f and len(nb) == 2 and all(c in n for n in nb) for c in (0, 1))
        keep.append(not miss)
    return keep


class Teacher:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(TEACHER)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(TEACHER).to(self.device).eval()

    def __call__(self, frame_bgr: np.ndarray) -> List[dict]:
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = self.processor(images=img, text=PROMPT, return_tensors="pt").to(self.device)
        with self.torch.no_grad(), self.torch.autocast(self.device, dtype=self.torch.float16, enabled=self.device == "cuda"):
            out = self.model(**inp)
        res = self.processor.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[img.size[::-1]])[0]
        return [{"text": t, "score": float(s), "xyxy": [float(v) for v in b.tolist()]}
                for s, t, b in zip(res["scores"], res["text_labels"], res["boxes"])]


def qa_sheet(frames: List[np.ndarray], labelled: List[Dict[int, Tuple[float, Box]]], numbers: List[int]) -> np.ndarray:
    colours = [(0, 0, 255), (0, 215, 255), (255, 255, 255)]
    tiles = []
    step = max(1, len(frames) // 12)
    for img, best, n in list(zip(frames, labelled, numbers))[::step][:12]:
        img = img.copy()
        for c, (_, b) in best.items():
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colours[c], 4)
            cv2.putText(img, CLASSES[c].split("_")[0], (int(b[0]), int(b[1]) + 28), 0, 1.0, colours[c], 3)
        cv2.putText(img, str(n), (10, 40), 0, 1.2, (0, 255, 0), 3)
        tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])


def label_clip(video: Path, out_dir: Path, teacher: Teacher, stride: int, max_frames: int) -> Dict[str, int]:
    clip = video.stem
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    frames, numbers, per_frame = [], [], []
    index = -1
    while True:
        ok, frame = cap.read()
        index += 1
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if index % stride:
            continue
        frames.append(frame)
        numbers.append(index)
        per_frame.append(verify_frame(frame, teacher(frame)))
    cap.release()

    smooth_case(per_frame)
    keep = drop_teacher_misses(per_frame)
    clip_dir = out_dir / clip
    clip_dir.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in CLASSES}
    kept_frames, kept_best, kept_numbers = [], [], []
    for frame, best, n, k in zip(frames, per_frame, numbers, keep):
        if not k:
            continue
        h, w = frame.shape[:2]
        lines = []
        for c, (_, b) in sorted(best.items()):
            lines.append(f"{c} {(b[0] + b[2]) / 2 / w:.5f} {(b[1] + b[3]) / 2 / h:.5f} "
                         f"{(b[2] - b[0]) / w:.5f} {(b[3] - b[1]) / h:.5f}")
            counts[CLASSES[c]] += 1
        stem = f"{clip}_{n:05d}"
        cv2.imwrite(str(clip_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (clip_dir / f"{stem}.txt").write_text("\n".join(lines))
        kept_frames.append(frame)
        kept_best.append(best)
        kept_numbers.append(n)
    if kept_frames:
        cv2.imwrite(str(out_dir / f"qa_{clip}.jpg"), qa_sheet(kept_frames, kept_best, kept_numbers))
    print(f"{clip}: {len(frames)} sampled, {len(kept_frames)} kept | "
          + " ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("dataset/yolo_boxes_real/labelled"))
    ap.add_argument("--stride", type=int, default=2, help="label every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N sampled frames (smoke test)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = Teacher()
    for video in args.video:
        label_clip(video, args.out, teacher, args.stride, args.max_frames)


def _selftest() -> None:
    """The verification rules on synthetic detections, no model needed."""
    img = np.full((720, 1280, 3), 120, np.uint8)
    img[400:500, 700:840] = (0, 0, 220)        # red patch (BGR), 140x100
    img[400:500, 880:1020] = (0, 220, 230)     # yellow patch
    dets = [
        {"text": "a red box", "score": 0.9, "xyxy": [700.0, 400.0, 840.0, 500.0]},
        {"text": "a yellow box", "score": 0.9, "xyxy": [880.0, 400.0, 1020.0, 500.0]},
        {"text": "a red box", "score": 0.9, "xyxy": [100.0, 100.0, 240.0, 200.0]},            # grey patch: not red
        {"text": "a metal storage case", "score": 0.5, "xyxy": [650.0, 380.0, 1060.0, 560.0]},  # body, holds both
        {"text": "a metal storage case", "score": 0.5, "xyxy": [650.0, 150.0, 1060.0, 560.0]},  # body+lid union
    ]
    best = verify_frame(img, dets)
    assert set(best) == {0, 1, 2}, best
    assert best[0][1][0] == 700.0, "the grey 'red box' must lose to the real red one"
    assert best[2][1][1] == 380.0, "the case BODY, not the body+lid union, must win"
    z = (1, [0] * 4)
    assert drop_teacher_misses([{0: z, 2: z}, {2: z}, {0: z, 2: z}]) == [True, False, True]
    print("pseudo_label_boxes selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
