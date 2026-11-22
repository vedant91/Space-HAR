"""Choose each YOLO confidence threshold by validation F1, and score a separate split."""
import argparse
import json
from pathlib import Path

from score_payload_boxes import collect_predictions, score_predictions
from pipeline.yolo_detector import CLASSES, YOLOBoxDetector


def calibrate(predictions):
    def quality(confidence, name):
        count = score_predictions(predictions, {name: confidence})[name]
        tp, fp, fn = (count[key] for key in ("tp", "fp", "fn"))
        return (2 * tp / max(2 * tp + fp + fn, 1), -fp, confidence)

    candidates = [i / 20 for i in range(1, 19)]
    return {name: max(candidates, key=lambda c: quality(c, name)) for name in CLASSES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    detector = YOLOBoxDetector(weights=args.weights, conf=.05)
    validation = list(collect_predictions(detector, args.validation))
    thresholds = calibrate(validation)
    report = dict(thresholds=thresholds, validation=score_predictions(validation, thresholds))
    if args.test:
        report["test"] = score_predictions(collect_predictions(detector, args.test), thresholds)
    if args.out:
        args.out.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
