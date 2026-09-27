"""
ArUco rack anchor - a more reliable rack-angle source for RackFrameNormalizer.

RackFrameNormalizer.normalize() (pipeline/rack_frame.py) already accepts any
((cx,cy),(w,h),angle_deg) rect and derives the rack's in-plane angle from it -
today that rect comes from the detected main_box's minAreaRect, so a fuzzy or
missing box detection directly degrades the rack angle. A printed marker's
corners are known exactly the instant it's detected, so fitting the SAME
minAreaRect shape to the marker's own corners is a drop-in, more reliable
replacement for that one input - RackFrameNormalizer itself does not change
at all, and nothing about its existing tests or behaviour is touched.

This intentionally does NOT do full solvePnP 6DOF pose: the rest of the
pipeline only ever consumes a single in-plane rack angle (see
rack_line_from_rect's docstring), so recovering translation/depth here would
be real work spent on something nothing downstream reads yet. If the HMR
side later wants a true 3D rack transform (e.g. to express attitude in rack
space rather than camera space), solvePnP with a calibrated camera matrix is
the next step - marker_corners() below already returns what that needs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# 6x6, 250-marker dictionary: small enough to print clearly at rack scale,
# large enough vocabulary that a stray real-world QR/barcode won't false-match.
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_ID = 0   # the one marker this project uses; a second could anchor a second rack face


def _detector():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(d, params)


_DETECTOR = _detector()


def marker_corners(frame_bgr: np.ndarray, marker_id: int = MARKER_ID) -> Optional[np.ndarray]:
    """(4, 2) pixel corners of the marker, in cv2.aruco's own order
    (top-left, top-right, bottom-right, bottom-left), or None if not found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == marker_id:
            return c.reshape(4, 2).astype(np.float32)
    return None


def rack_rect_from_marker(frame_bgr: np.ndarray, marker_id: int = MARKER_ID
                          ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """((cx,cy),(w,h),angle_deg) fitted to the marker's own corners - the
    exact shape RackFrameNormalizer.normalize()/rack_line_from_rect() already
    expect from a box's minAreaRect. None if the marker isn't visible this
    frame (caller falls back to the box-geometry rect, same as today).

    Deliberately NOT cv2.minAreaRect(corners): a marker is square, and
    minAreaRect's choice of which perpendicular edge to call "width" is only
    stable for a clearly elongated shape - on a near-perfect square it ties
    on sub-pixel noise and can flip 90 degrees frame to frame (caught by this
    module's own self-test failing until this was found). The marker's own
    corner order (cv2.aruco always returns top-left, top-right, bottom-right,
    bottom-left, regardless of rotation) gives an unambiguous, continuous
    angle instead: the top-left -> top-right vector is always the same
    physical edge. (w, h) is then reported as (side*1.5, side) - deliberately
    non-square - so rack_line_from_rect's w<h branch never fires and this
    angle is used exactly as measured, not "corrected" by a branch tuned for
    elongated boxes."""
    corners = marker_corners(frame_bgr, marker_id)
    if corners is None:
        return None
    cx, cy = corners.mean(axis=0)
    dx, dy = corners[1] - corners[0]   # top-right - top-left: the marker's own top edge
    angle_deg = -float(np.degrees(np.arctan2(dy, dx)))   # calibrated against a known-rotation self-test below
    side = float(np.linalg.norm((dx, dy)))
    return (float(cx), float(cy)), (side * 1.5, side), angle_deg


def generate_marker(marker_id: int = MARKER_ID, size_px: int = 600) -> np.ndarray:
    """A printable marker image. Print at real-world rack scale and mount
    flat on the rack face - tools/print_aruco_marker.py wraps this for CLI use."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.generateImageMarker(d, marker_id, size_px)


if __name__ == "__main__":
    # Round-trip: generate a marker, render it into a synthetic frame at a
    # known position/rotation, detect it back, and confirm the recovered
    # rect's angle and center match what was drawn - not just "detection
    # didn't crash", but "the geometry it hands to RackFrameNormalizer is
    # actually correct".
    marker = generate_marker(size_px=200)
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)   # white background: high contrast, reliable detection
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # Rotate on a canvas big enough that the corners survive: a 200px square
    # rotated to 45 deg needs ~283px of diagonal room, or warpAffine's fixed
    # output size clips it and no amount of interpolation tuning fixes that
    # (caught by this test itself failing until this padding was added).
    angle_deg = 27.0
    cx, cy = 640, 360
    canvas = 320
    padded = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    off = (canvas - 200) // 2
    padded[off:off + 200, off:off + 200] = marker_bgr
    m = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(padded, m, (canvas, canvas), borderValue=(255, 255, 255))
    x1, y1 = cx - canvas // 2, cy - canvas // 2
    frame[y1:y1 + canvas, x1:x1 + canvas] = rotated

    rect = rack_rect_from_marker(frame)
    assert rect is not None, "marker must be detected in a clean synthetic frame"
    (rcx, rcy), (w, h), a = rect
    assert abs(rcx - cx) < 5 and abs(rcy - cy) < 5, f"center off: got ({rcx},{rcy}), want ({cx},{cy})"
    assert w > h, "w must always exceed h so rack_line_from_rect's w<h branch never fires"
    assert h > 150, f"recovered marker size implausible: {h}"

    # No 90-degree ambiguity to tolerate here (unlike a plain minAreaRect on a
    # square): the corner-order-based angle is continuous and unambiguous by
    # construction, so this must match the drawn rotation directly, through
    # the real production function (rack_line_from_rect), not a hand-rolled
    # equivalent in the test.
    from pipeline.rack_frame import wrap90, RackFrameNormalizer
    recovered_line = RackFrameNormalizer().rack_line_from_rect(rect)
    expected_line = wrap90(angle_deg)
    diff = abs(wrap90(recovered_line - expected_line))
    assert diff < 3.0, f"recovered angle {recovered_line} vs drawn {expected_line}"

    assert marker_corners(np.full((720, 1280, 3), 255, dtype=np.uint8)) is None, \
        "a blank frame must report no marker, not a false positive"

    print("aruco_rack smoke test passed (marker generation, detection, rect geometry round-trip).")
