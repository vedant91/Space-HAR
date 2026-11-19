"""
YOLO Box Detector - drop-in replacement for pipeline/hsv_detector.py.

Same interface (Detection dataclass, detect(), draw(), get_feature_dict()),
so tools/build_sequences.py, tools/evaluate_real.py, pipeline/har_pipeline.py
and simulation/space_sim.py need only swap which detector they construct -
no other line of calling code changes. This is exactly the contract
docs/HANDOFF_YOLO_MEDIAPIPE.md specified for whoever built this.

Trained on Blender's own ground truth (tools/export_yolo_labels.py converts
payload.json's exact bbox + visible flag into YOLO labels - no manual
labelling, and no synthetic-vs-real domain gap for boxes the way there is for
a human body: a red box looks like a red box whether it's rendered or real).

Weights: models/yolo_boxes.pt (see train/train_yolo_boxes.py). Missing
weights fail loudly at construction, not silently at the first detect() call -
a live pipeline running on a detector that never found anything would be a
much worse failure to debug than one that never started.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

CLASSES = ["red_box", "yellow_box", "main_box"]   # index == YOLO class id; must match tools/export_yolo_labels.py
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolo_boxes.pt"
DEFAULT_CONF = 0.25


@dataclass
class Detection:
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    centroid: Tuple[int, int]
    area: int
    mask: Optional[np.ndarray] = None
    rect: Optional[Tuple[float, float, float]] = None   # YOLO gives axis-aligned boxes only; always None here


class YOLOBoxDetector:
    """detect()/draw()/get_feature_dict() match HSVBoxDetector's signatures
    exactly (pipeline/hsv_detector.py). frame_width/frame_height are accepted
    for constructor-compatibility with every existing call site
    (`YOLOBoxDetector(frame_width=w, frame_height=h)`) but are only used for
    get_feature_dict()'s area ratio - YOLO reads the frame's own resolution
    directly, it never needs to be told what it is."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720,
                use_morphology: bool = True,   # accepted, unused: HSV-only knob, kept for signature compatibility
                weights: Optional[str] = None, conf: float = DEFAULT_CONF):
        self.fw = frame_width
        self.fh = frame_height
        self.frame_area = frame_width * frame_height
        self.conf = conf

        weights_path = Path(weights) if weights else DEFAULT_WEIGHTS
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO box weights missing: {weights_path} "
                f"(train with train/train_yolo_boxes.py, or pass weights=... explicitly)")

        import os
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never let ultralytics fetch anything: offline system
        os.environ.setdefault("YOLO_OFFLINE", "True")
        from ultralytics import YOLO   # type: ignore
        self.model = YOLO(str(weights_path))
        # Checkpoint class IDs depend on the dataset used for training.
        names = self.model.names
        aliases = {"main_box_open": "main_box", "main_box_closed": "main_box"}
        self.labels = {i: aliases.get(name, name) for i, name in names.items()
                       if name in CLASSES or name in aliases}
        if set(self.labels.values()) != set(CLASSES):
            raise ValueError(f"Expected payload classes {CLASSES} in {weights_path}, found {names}")
        logger.info("YOLOBoxDetector initialized (weights=%s, conf=%.2f)", weights_path, conf)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.labels:
                continue   # a weights file trained on a different class set must not mislabel silently
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            area = max(0, x2 - x1) * max(0, y2 - y1)
            detections.append(Detection(
                label=self.labels[cls_id],
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                centroid=((x1 + x2) // 2, (y1 + y2) // 2),
                area=area,
            ))
        return detections

    def get_feature_dict(self, frame_bgr: np.ndarray) -> dict:
        """Same shape as HSVBoxDetector.get_feature_dict() - see that
        docstring for the key meanings."""
        dets = self.detect(frame_bgr)
        result = {
            "red_visible": False, "yellow_visible": False, "main_visible": False,
            "red_centroid": None, "yellow_centroid": None, "main_centroid": None,
            "red_bbox": None, "yellow_bbox": None,
            "red_area_ratio": 0.0, "yellow_area_ratio": 0.0,
        }
        # Multiple proposals for one payload are common with distractors.
        # Fill the single-object feature slot with the strongest proposal last.
        for d in sorted(dets, key=lambda item: item.confidence):
            if d.label == "red_box":
                result["red_visible"] = True
                result["red_centroid"] = d.centroid
                result["red_bbox"] = d.bbox
                result["red_area_ratio"] = d.area / self.frame_area
            elif d.label == "yellow_box":
                result["yellow_visible"] = True
                result["yellow_centroid"] = d.centroid
                result["yellow_bbox"] = d.bbox
                result["yellow_area_ratio"] = d.area / self.frame_area
            elif d.label == "main_box":
                result["main_visible"] = True
                result["main_centroid"] = d.centroid
        return result

    def draw(self, frame_bgr: np.ndarray, detections: Optional[List[Detection]] = None) -> np.ndarray:
        import cv2
        if detections is None:
            detections = self.detect(frame_bgr)
        out = frame_bgr.copy()
        colors = {"red_box": (0, 0, 220), "yellow_box": (0, 215, 255), "main_box": (200, 200, 200)}
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.label, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{det.label} {det.confidence:.2f}", (x1, max(y1 - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    try:
        det = YOLOBoxDetector()
    except FileNotFoundError as e:
        print(f"weights not trained yet (expected before train/train_yolo_boxes.py finishes): {e}")
        raise SystemExit(0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert isinstance(dets, list)
    feats = det.get_feature_dict(frame)
    assert set(feats.keys()) == {"red_visible", "yellow_visible", "main_visible", "red_centroid",
                                 "yellow_centroid", "main_centroid", "red_bbox", "yellow_bbox",
                                 "red_area_ratio", "yellow_area_ratio"}
    out = det.draw(frame)
    assert out.shape == frame.shape
    print(f"yolo_detector smoke test passed (blank frame -> {len(dets)} detections, "
         f"feature dict + draw both work).")
