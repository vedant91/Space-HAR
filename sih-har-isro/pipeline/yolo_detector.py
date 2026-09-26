"""
YOLO Detector — Runtime Wrapper (opt-in, DETECTOR_BACKEND="yolo")
====================================================================
Drop-in alternative to HSVBoxDetector — same public contract (`.detect()`
returning List[Detection], `.draw()`), so har_pipeline.py needs zero changes
beyond picking which backend to construct (see DETECTOR_BACKEND).

Loads a checkpoint fine-tuned by train/train_yolo_detector.py on our own
synthetic renderer bboxes. The pretrained COCO base checkpoint this was
fine-tuned from is never used as-is at inference time — COCO has no
red_box/yellow_box/hand classes, so an un-fine-tuned model would detect
nothing meaningful for this task.

hsv_detector.py's own docstring already concludes HSV beats a trained
detector here (measured 1.000/1.000 precision/recall, near-zero CPU cost).
This backend exists to measure against that, not as the shipped default —
see config.DETECTOR_BACKEND (stays "hsv" unless explicitly flipped).

Usage mirrors HSVBoxDetector's contract:
    detector = YOLOBoxDetector()
    detections = detector.process(frame_bgr)   # or .detect(), same thing
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    DETECTION_CLASSES, YOLO_MODEL_PATH, YOLO_IMG_SIZE, YOLO_CONF_THRESHOLD,
)
from pipeline.hsv_detector import Detection, draw_detections

logger = logging.getLogger(__name__)


class YOLOBoxDetector:
    """Same public surface as HSVBoxDetector (.detect(), .draw()) — the two
    are interchangeable wherever har_pipeline.py holds a detector instance."""

    def __init__(self, model_path: str = YOLO_MODEL_PATH,
                 conf_threshold: float = YOLO_CONF_THRESHOLD,
                 imgsz: int = YOLO_IMG_SIZE):
        self._model = None
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        if Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path)
                logger.info("YOLO detector loaded: %s", model_path)
            except Exception as e:
                logger.warning("YOLO checkpoint load failed (%s): %s", model_path, e)
        else:
            logger.warning(
                "No trained YOLO checkpoint at %s — detections will be empty until "
                "`python train/train_yolo_detector.py` is run.", model_path)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        results = self._model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf_threshold,
            device="cpu", verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        dets = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_label = DETECTION_CLASSES.get(cls_id, str(cls_id))
            # main_box_open/main_box_closed are two separate TRAINING
            # classes (a genuinely learnable visual distinction — see
            # simulation/renderer.py's lid graphic) but ONE semantic
            # label at this API boundary, so every existing
            # `d.label == "main_box"` check downstream (rack_frame's
            # pick_rack_rect, anomaly_monitor's DRIFT_TRACKED_LABELS,
            # features.BOX_LABELS, ...) keeps working unchanged whether
            # HSV or YOLO is the active backend. Lid state itself is
            # still available via Detection.lid_state for anything that
            # wants it.
            if raw_label == "main_box_open":
                label, lid_state = "main_box", "open"
            elif raw_label == "main_box_closed":
                label, lid_state = "main_box", "closed"
            else:
                label, lid_state = raw_label, None
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            dets.append(Detection(label=label, bbox=(x1, y1, x2, y2),
                                  confidence=conf, centroid=centroid, area=area,
                                  lid_state=lid_state))
        return dets

    def draw(self, frame_bgr: np.ndarray,
             detections: Optional[List[Detection]] = None) -> np.ndarray:
        if detections is None:
            detections = self.detect(frame_bgr)
        return draw_detections(frame_bgr, detections)

    def close(self):
        """No persistent resources to release — kept for interface symmetry
        with HSVBoxDetector/PoseNetWrapper (har_pipeline.py may call .close())."""
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = YOLOBoxDetector()
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = d.detect(dummy)
    print(f"available={d.available} n_detections={len(dets)}")
    print("yolo_detector smoke test passed.")
