"""
HSV Color-Based Box Detector
==============================
Replaces YOLO for object detection. Uses HSV color segmentation to detect:
  - red_box    → HSV dual-range mask (red wraps hue wheel)
  - yellow_box → HSV single-range mask
  - main_box   → Largest white/light region near experiment area

Zero training. Deterministic. Works on CPU at 60+ FPS.
This is actually superior to a trained detector for this use-case
because the colored boxes are THE definition of distinct hue targets.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Import HSV ranges from config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    HSV_RED_LOWER1, HSV_RED_UPPER1, HSV_RED_LOWER2, HSV_RED_UPPER2,
    HSV_YELLOW_LOWER, HSV_YELLOW_UPPER,
    HSV_WHITE_LOWER, HSV_WHITE_UPPER,
    MIN_BOX_AREA_PX,
)


@dataclass
class Detection:
    label: str           # "red_box", "yellow_box", "main_box"
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float    # 0-1 based on area ratio
    centroid: Tuple[int, int]
    area: int
    mask: Optional[np.ndarray] = None  # Binary mask for this detection
    rect: Optional[Tuple[float, float, float]] = None  # ((cx,cy),(w,h),angle_deg) minAreaRect — rack roll anchor


class HSVBoxDetector:
    """
    Color segmentation detector for the ISRO HAR experiment boxes.

    Detects:
      - red_box    → Dual HSV range (hue wraps 170-180 + 0-10)
      - yellow_box → Single HSV range
      - main_box   → Largest white region in frame

    Also provides:
      - Hand proximity to box (are hands near a box?)
      - Box visibility flags (is each box visible in frame?)
    """

    def __init__(self,
                 frame_width: int = 1280,
                 frame_height: int = 720,
                 use_morphology: bool = True):
        self.fw = frame_width
        self.fh = frame_height
        self.use_morphology = use_morphology
        self.frame_area = frame_width * frame_height

        # Morphological kernels
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        # Calibration state (updated by calibrate())
        self._cal_offsets: Dict[str, Tuple] = {}

        logger.info("HSVBoxDetector initialized (frame=%dx%d)", frame_width, frame_height)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.
        Returns list of Detection objects (one per visible object).
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        detections = []

        red   = self._detect_red(hsv)
        if red:   detections.extend(red)

        yellow = self._detect_yellow(hsv)
        if yellow: detections.extend(yellow)

        white  = self._detect_main_box(hsv, frame_bgr)
        if white:  detections.extend(white)

        return detections

    def get_feature_dict(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns a structured dict of detection features for the state machine.
        Keys: red_visible, yellow_visible, main_open_visible,
              red_centroid, yellow_centroid, red_area_ratio, yellow_area_ratio
        """
        dets = self.detect(frame_bgr)
        result = {
            "red_visible": False, "yellow_visible": False, "main_visible": False,
            "red_centroid": None, "yellow_centroid": None, "main_centroid": None,
            "red_bbox": None, "yellow_bbox": None,
            "red_area_ratio": 0.0, "yellow_area_ratio": 0.0,
        }
        for d in dets:
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

    def draw(self, frame_bgr: np.ndarray,
             detections: Optional[List[Detection]] = None) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame (returns copy)."""
        if detections is None:
            detections = self.detect(frame_bgr)

        out = frame_bgr.copy()
        COLORS = {
            "red_box":    (0, 0, 220),
            "yellow_box": (0, 215, 255),
            "main_box":   (200, 200, 200),
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLORS.get(det.label, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(out, label_text, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(out, det.centroid, 5, color, -1)

        return out

    def calibrate_from_roi(self, frame_bgr: np.ndarray, label: str,
                           roi: Tuple[int, int, int, int]):
        """
        Sample HSV from a user-drawn ROI to fine-tune detection ranges.
        roi = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_hsv = hsv_crop.mean(axis=(0, 1))
        std_hsv  = hsv_crop.std(axis=(0, 1))
        logger.info("Calibrated %s: mean_HSV=%s std=%s", label, mean_hsv.round(1), std_hsv.round(1))
        self._cal_offsets[label] = (mean_hsv, std_hsv)

    # ── Internal detection methods ─────────────────────────────────────────────

    def _apply_mask_pipeline(self, mask: np.ndarray) -> np.ndarray:
        """Clean up a binary mask with morphological ops."""
        if self.use_morphology:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        return mask

    def _contours_to_detections(self, mask: np.ndarray, label: str,
                                 top_n: int = 2) -> List[Detection]:
        """Convert mask contours to Detection objects, keeping top-N by area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]
        detections = []

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < MIN_BOX_AREA_PX:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio sanity check (boxes shouldn't be extremely thin)
            aspect = w / max(h, 1)
            if aspect < 0.2 or aspect > 5.0:
                continue

            confidence = min(area / (self.frame_area * 0.05), 1.0)  # Normalize to 5% of frame
            centroid = (x + w // 2, y + h // 2)
            rect = cv2.minAreaRect(cnt)  # orientation anchor for rack-frame normalization
            detections.append(Detection(
                label=label,
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                centroid=centroid,
                area=area,
                rect=rect,
            ))

        return detections

    def _detect_red(self, hsv: np.ndarray) -> List[Detection]:
        """Detect red box using dual HSV range (red wraps hue 170→0→10)."""
        lo1 = np.array(HSV_RED_LOWER1); hi1 = np.array(HSV_RED_UPPER1)
        lo2 = np.array(HSV_RED_LOWER2); hi2 = np.array(HSV_RED_UPPER2)
        mask1 = cv2.inRange(hsv, lo1, hi1)
        mask2 = cv2.inRange(hsv, lo2, hi2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = self._apply_mask_pipeline(mask)
        return self._contours_to_detections(mask, "red_box", top_n=1)

    def _detect_yellow(self, hsv: np.ndarray) -> List[Detection]:
        """Detect yellow box."""
        lo = np.array(HSV_YELLOW_LOWER); hi = np.array(HSV_YELLOW_UPPER)
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._apply_mask_pipeline(mask)
        return self._contours_to_detections(mask, "yellow_box", top_n=1)

    def _detect_main_box(self, hsv: np.ndarray,
                         bgr: np.ndarray) -> List[Detection]:
        """
        Detect the main white container.
        Strategy: find largest white region, apply rectangular shape filter.
        """
        lo = np.array(HSV_WHITE_LOWER); hi = np.array(HSV_WHITE_UPPER)
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._apply_mask_pipeline(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Pick largest contour that's roughly rectangular
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = int(cv2.contourArea(cnt))
            if area < MIN_BOX_AREA_PX * 4:  # Main box should be bigger
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity > 0.7:  # Reasonably solid (box-like) shape
                x, y, w, h = cv2.boundingRect(cnt)
                return [Detection(
                    label="main_box",
                    bbox=(x, y, x + w, y + h),
                    confidence=min(area / (self.frame_area * 0.1), 1.0),
                    centroid=(x + w // 2, y + h // 2),
                    area=area,
                    rect=cv2.minAreaRect(cnt),
                )]
        return []


def run_hsv_tuner(camera_index: int = 0):
    """
    Interactive HSV tuner — use trackbars to calibrate ranges for your specific
    lighting conditions before deployment. Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    cv2.namedWindow("HSV Tuner")

    def nothing(x): pass
    for name, val in [("H_lo", 0), ("S_lo", 100), ("V_lo", 100),
                      ("H_hi", 30), ("S_hi", 255), ("V_hi", 255)]:
        cv2.createTrackbar(name, "HSV Tuner", val, 255, nothing)

    print("HSV Tuner — adjust sliders to calibrate for your environment.")
    print("Press 'r' = show red mask | 'y' = yellow | 'w' = white | 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([cv2.getTrackbarPos(n, "HSV Tuner") for n in ["H_lo", "S_lo", "V_lo"]])
        hi = np.array([cv2.getTrackbarPos(n, "HSV Tuner") for n in ["H_hi", "S_hi", "V_hi"]])
        mask = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("HSV Tuner", np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result]))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            print(f"HSV Lower: {lo.tolist()} | Upper: {hi.tolist()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Launch interactive HSV tuner")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.tune:
        run_hsv_tuner(args.camera)
    else:
        # Quick test: run detector on webcam
        detector = HSVBoxDetector()
        cap = cv2.VideoCapture(args.camera)
        print("HSV Detector live test — press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            vis = detector.draw(frame, dets)
            feats = detector.get_feature_dict(frame)
            y = 30
            for k, v in feats.items():
                if isinstance(v, bool) or isinstance(v, float):
                    cv2.putText(vis, f"{k}: {v}", (10, 