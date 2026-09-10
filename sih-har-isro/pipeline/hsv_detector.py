"""
HSV Color-Based Box + Hand Detector
=====================================
Replaces YOLO (or any other pretrained/open-source detector) for object
detection. Uses HSV color segmentation to detect:
  - red_box    → HSV dual-range mask (red wraps hue wheel)
  - yellow_box → HSV single-range mask
  - main_box   → Largest white/light region near experiment area
  - hand       → Skin-tone HSV band, confidence-boosted (not gated) by
                 frame-differencing motion — see _detect_hand()

Zero training, zero pretrained weights. Deterministic. Works on CPU at 60+
FPS. For the colored boxes this is actually superior to a trained detector
for this use-case because the colors ARE the definition of the targets.
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
    HSV_SKIN_LOWER, HSV_SKIN_UPPER, MIN_HAND_AREA_PX,
    HAND_MOTION_THRESHOLD, HAND_TOP_EXCLUDE_FRAC,
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
      - hand       → Skin-tone HSV band + motion confidence boost

    Also provides:
      - Box/hand visibility flags (get_feature_dict)
      - Live HSV recalibration from a user-drawn ROI (calibrate_from_roi)
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

        # Live HSV bounds — instance state (not module constants) so
        # calibrate_from_roi() can actually adjust what detect() uses.
        # Seeded from config; np.array once here instead of per-frame.
        self._red_lo1 = np.array(HSV_RED_LOWER1, dtype=np.float64)
        self._red_hi1 = np.array(HSV_RED_UPPER1, dtype=np.float64)
        self._red_lo2 = np.array(HSV_RED_LOWER2, dtype=np.float64)
        self._red_hi2 = np.array(HSV_RED_UPPER2, dtype=np.float64)
        self._yellow_lo = np.array(HSV_YELLOW_LOWER, dtype=np.float64)
        self._yellow_hi = np.array(HSV_YELLOW_UPPER, dtype=np.float64)
        self._white_lo = np.array(HSV_WHITE_LOWER, dtype=np.float64)
        self._white_hi = np.array(HSV_WHITE_UPPER, dtype=np.float64)
        self._skin_lo = np.array(HSV_SKIN_LOWER, dtype=np.float64)
        self._skin_hi = np.array(HSV_SKIN_UPPER, dtype=np.float64)

        # Motion state for hand detection (frame-differencing) — None until
        # the first frame has been seen.
        self._prev_gray: Optional[np.ndarray] = None

        # Calibration state (updated by calibrate_from_roi())
        self._cal_offsets: Dict[str, Dict[str, Tuple[float, float]]] = {}

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

        hand  = self._detect_hand(hsv, frame_bgr)
        if hand:  detections.extend(hand)

        return detections

    def get_feature_dict(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns a structured dict of detection features for the state machine.
        Keys: red_visible, yellow_visible, main_open_visible,
              red_centroid, yellow_centroid, red_area_ratio, yellow_area_ratio
        """
        return self.dets_to_feature_dict(self.detect(frame_bgr))

    def dets_to_feature_dict(self, dets: List[Detection]) -> dict:
        """Pure aggregation of an already-computed detection list into the
        same structured dict get_feature_dict() returns. Split out so callers
        that already have `dets` (e.g. data_generation/real_video_autolabel.py)
        don't have to call detect() a second time on the same frame — doing so
        would also double-advance HSVBoxDetector's internal motion baseline
        (see _detect_hand's self._prev_gray) for no benefit."""
        result = {
            "red_visible": False, "yellow_visible": False, "main_visible": False,
            "hand_visible": False,
            "red_centroid": None, "yellow_centroid": None, "main_centroid": None,
            "red_bbox": None, "yellow_bbox": None,
            "red_area_ratio": 0.0, "yellow_area_ratio": 0.0,
            "hand_centroids": [], "hand_count": 0,
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
            elif d.label == "hand":
                result["hand_visible"] = True
                result["hand_centroids"].append(d.centroid)
                result["hand_count"] += 1
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
            "hand":       (60, 220, 60),
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
                           roi: Tuple[int, int, int, int], k: float = 2.5):
        """
        Sample HSV from a user-drawn ROI and actually shift the live detection
        bounds for `label` to `mean +/- k*std` (clipped to the valid HSV
        range). `label` must be one of "red_box"/"yellow_box"/"main_box"/"hand".

        Hue is circular (OpenCV's 0-180 range wraps): a plain arithmetic mean
        would be wrong for red, whose true samples straddle the 0/180 seam
        (e.g. hues near 2 and 178 averaging to a bogus ~90). Hue is doubled to
        map the half-circle onto a full circle, averaged as a vector, then
        halved back — a proper circular mean/std.
        """
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("calibrate_from_roi: empty ROI for %s, ignoring", label)
            return

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        h, s, v = hsv_crop[:, 0], hsv_crop[:, 1], hsv_crop[:, 2]

        theta = np.deg2rad(h * 2.0)
        cos_m, sin_m = np.cos(theta).mean(), np.sin(theta).mean()
        mean_h = (np.rad2deg(np.arctan2(sin_m, cos_m)) / 2.0) % 180.0
        resultant = np.hypot(sin_m, cos_m)
        std_h = min(float(np.rad2deg(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-6)), 0.0))) / 2.0), 45.0)
        mean_s, std_s = float(s.mean()), float(s.std())
        mean_v, std_v = float(v.mean()), float(v.std())

        self._cal_offsets[label] = {"h": (mean_h, std_h), "s": (mean_s, std_s), "v": (mean_v, std_v)}

        s_lo, s_hi = max(0.0, mean_s - k * std_s), min(255.0, mean_s + k * std_s)
        v_lo, v_hi = max(0.0, mean_v - k * std_v), min(255.0, mean_v + k * std_v)
        h_half = max(k * std_h, 4.0)  # never collapse to a zero-width band

        if label == "red_box":
            lo_edge, hi_edge = (mean_h - h_half) % 180.0, (mean_h + h_half) % 180.0
            if lo_edge <= hi_edge:
                # Calibrated band doesn't straddle the seam — keep the
                # dual-range *shape* anyway (second range becomes a no-op).
                self._red_lo1 = np.array([lo_edge, s_lo, v_lo])
                self._red_hi1 = np.array([hi_edge, s_hi, v_hi])
                self._red_lo2 = np.array([180.0, s_lo, v_lo])
                self._red_hi2 = np.array([180.0, s_hi, v_hi])
            else:
                self._red_lo1 = np.array([0.0, s_lo, v_lo])
                self._red_hi1 = np.array([hi_edge, s_hi, v_hi])
                self._red_lo2 = np.array([lo_edge, s_lo, v_lo])
                self._red_hi2 = np.array([179.0, s_hi, v_hi])
        elif label == "yellow_box":
            self._yellow_lo = np.array([max(0.0, mean_h - h_half), s_lo, v_lo])
            self._yellow_hi = np.array([min(179.0, mean_h + h_half), s_hi, v_hi])
        elif label == "main_box":
            # The main box is detected by being light/low-saturation, not by
            # hue — keep hue span wide, recenter S/V only.
            self._white_lo = np.array([0.0, 0.0, v_lo])
            self._white_hi = np.array([180.0, max(10.0, s_hi), 255.0])
        elif label == "hand":
            # Skin tone: single hue band (doesn't wrap the seam the way the
            # dual-range red box does).
            self._skin_lo = np.array([max(0.0, mean_h - h_half), s_lo, v_lo])
            self._skin_hi = np.array([min(179.0, mean_h + h_half), s_hi, v_hi])
        else:
            logger.warning("calibrate_from_roi: unknown label '%s', ignoring", label)
            return

        logger.info("Calibrated %s: mean_H=%.1f std_H=%.1f mean_S=%.1f mean_V=%.1f — bounds updated",
                   label, mean_h, std_h, mean_s, mean_v)

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
        mask1 = cv2.inRange(hsv, self._red_lo1, self._red_hi1)
        mask2 = cv2.inRange(hsv, self._red_lo2, self._red_hi2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = self._apply_mask_pipeline(mask)
        return self._contours_to_detections(mask, "red_box", top_n=1)

    def _detect_yellow(self, hsv: np.ndarray) -> List[Detection]:
        """Detect yellow box."""
        mask = cv2.inRange(hsv, self._yellow_lo, self._yellow_hi)
        mask = self._apply_mask_pipeline(mask)
        return self._contours_to_detections(mask, "yellow_box", top_n=1)

    def _detect_main_box(self, hsv: np.ndarray,
                         bgr: np.ndarray) -> List[Detection]:
        """
        Detect the main white container.
        Strategy: find largest white region, apply rectangular shape filter.
        """
        mask = cv2.inRange(hsv, self._white_lo, self._white_hi)
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

    def _detect_hand(self, hsv: np.ndarray, bgr: np.ndarray) -> List[Detection]:
        """
        Detect hands/gloves via skin-tone HSV, boosted (not gated) by motion.

        Classical CV, no training: fills DETECTION_CLASSES' "hand" slot, which
        HSVBoxDetector previously never actually produced. Two deliberate
        design choices:
          - A spatial prior excludes the top HAND_TOP_EXCLUDE_FRAC of the
            frame (typically head/helmet) instead of requiring motion to
            disambiguate — hands routinely pause (e.g. during an "examine"
            step) and would otherwise disappear from detection while still.
          - Motion still *boosts* confidence (a moving skin blob is more
            likely a hand than a stationary one) without ever fully zeroing
            it out, so a paused hand stays detected at reduced confidence
            rather than vanishing.
        """
        h, w = hsv.shape[:2]
        skin_mask = cv2.inRange(hsv, self._skin_lo, self._skin_hi)
        top_cut = int(h * HAND_TOP_EXCLUDE_FRAC)
        skin_mask[:top_cut, :] = 0
        skin_mask = self._apply_mask_pipeline(skin_mask)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion_mask = (diff >= HAND_MOTION_THRESHOLD).astype(np.uint8) * 255
        else:
            motion_mask = np.zeros_like(gray)  # unknown on the very first frame
        self._prev_gray = gray

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        detections = []
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < MIN_HAND_AREA_PX:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect < 0.3 or aspect > 3.0:  # hands are less regular than boxes
                continue

            motion_frac = float(np.mean(motion_mask[y:y + ch, x:x + cw]) / 255.0)
            base_conf = min(area / (self.frame_area * 0.03), 1.0)
            confidence = float(np.clip(base_conf * (0.6 + 0.4 * motion_frac), 0.0, 1.0))

            detections.append(Detection(
                label="hand",
                bbox=(x, y, x + cw, y + ch),
                confidence=confidence,
                centroid=(x + cw // 2, y + ch // 2),
                area=area,
                rect=cv2.minAreaRect(cnt),
            ))
        return detections


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
                    cv2.putText(vis, f"{k}: {v}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y += 20
            cv2.imshow("HSV Detector", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
