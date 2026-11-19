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


def score(detector, dataset, stride=1, start=0, stop=None):
    manifest = json.loads((dataset / "annotations_ground_truth.json").read_text())
    names = {"red_payload": "red_box", "yellow_payload": "yellow_box"}
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in names.values()}
    for frame in manifest["frames"][start:stop:stride]:
        image = cv2.imread(str(dataset / frame["image"]))
        if image is None:
            raise FileNotFoundError(dataset / frame["image"])
        predictions = sorted(detector.detect(image), key=lambda d: -d.confidence)
        for raw, name in names.items():
            truth = []
            for a in frame["annotations"]:
                if a["label"] == raw:
                    x, y, w, h = a["bbox_xywh"]
                    truth.append((x, y, x+w, y+h))
            for d in predictions:
                if d.label != name:
                    continue
                overlaps = [iou(d.bbox, b) for b in truth]
                if overlaps and max(overlaps) >= 0.5:
                    truth.pop(overlaps.index(max(overlaps)))
                    counts[name]["tp"] += 1
                else:
                    counts[name]["fp"] += 1
            counts[name]["fn"] += len(truth)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/blender_twin")
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    model = YOLOBoxDetector(weights=args.weights, conf=args.conf)
    print(json.dumps(score(model, args.dataset, args.stride, args.start, args.stop), indent=2))
