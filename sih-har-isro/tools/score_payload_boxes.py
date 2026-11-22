"""Score red/yellow detections against the Blender twin's projected boxes."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.yolo_detector import YOLOBoxDetector


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def collect_predictions(detector, dataset, stride=1, start=0, stop=None):
    manifests = sorted(dataset.glob("take_*/boxes.json"))
    if not manifests:
        raise ValueError(f"No visibility-labelled takes in {dataset}. Export with --previews first.")
    frames = [(path.parent / f["image"], f["annotations"]) for path in manifests
              for f in json.loads(path.read_text())]
    for path, annotations in frames[start:stop:stride]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        predictions = sorted(detector.detect(image), key=lambda d: -d.confidence)
        yield annotations, predictions


def score_predictions(frames, thresholds):
    names = ("red_box", "yellow_box", "main_box")
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in names}
    for annotations, predictions in frames:
        for name in names:
            truth = []
            for a in annotations:
                if a["label"] == name:
                    x, y, w, h = a["bbox_xywh"]
                    truth.append((x, y, x+w, y+h))
            for d in predictions:
                if d.label != name or d.confidence < thresholds.get(name, 0):
                    continue
                overlaps = [iou(d.bbox, b) for b in truth]
                if overlaps and max(overlaps) >= 0.5:
                    truth.pop(overlaps.index(max(overlaps)))
                    counts[name]["tp"] += 1
                else:
                    counts[name]["fp"] += 1
            counts[name]["fn"] += len(truth)
    return counts



def score(detector, dataset, stride=1, start=0, stop=None):
    return score_predictions(collect_predictions(detector, dataset, stride, start, stop), {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Split directory from blender/export_depth_takes.py")
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, help="Override checkpoint class thresholds with one confidence value")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    model = YOLOBoxDetector(weights=args.weights, conf=args.conf)
    print(json.dumps(score(model, args.dataset, args.stride, args.start, args.stop), indent=2))
