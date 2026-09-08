"""
Orientation-agnostic pose estimation by canonicalising the IMAGE, not the output.

The problem
-----------
MediaPipe Pose is trained on photographs of people who are the right way up.
In microgravity a crew member is routinely not, and the detector does not
merely get less accurate - it stops firing. Measured on rendered BAS footage
(`dataset/blender`, payload_a, 20 mid-protocol frames per take):

    crew roll     pose detected
    0 deg         19/20   (95%)
    90 deg         7/20   (35%)
    180 deg       12/20   (60%)

That is the failure the problem statement predicts in so many words:
"Standard 2D or ground-based 3D posture models fail because astronauts do not
have a fixed 'up' or 'down' orientation."

Why rack_frame.py alone cannot fix it
-------------------------------------
`pipeline/rack_frame.py` is good work and solves a real part of this: it
re-expresses landmarks in a rack-anchored frame so a rolled body maps to the
same canonical feature vector. But it operates on the landmarks MediaPipe
returns. When MediaPipe returns *nothing* - which is what happens 65% of the
time at 90 degrees - there is nothing to normalise. Feature normalisation
cannot recover a detection that never occurred.

The fix
-------
Rotate the frame so the crew member is approximately upright, run pose on
that, then map the landmarks back into original image coordinates. The
detector then sees the upright human it was trained on, and every consumer
downstream still receives coordinates in the original frame - so the 132-dim
contract, `rack_frame.py`, the LSTM and the state machine are all unchanged.

Two sources for the rotation angle, in order:

1.  The payload rack itself. `hsv_detector` already returns a `minAreaRect`
    for the main box every frame, and the rack is bolted to the module - so
    its roll in the image IS the camera-relative "up" of the workspace. This
    is free: the detection already runs.

2.  A short search, when the rack is not visible or its angle did not help.
    Only the frames that actually failed pay for this, and the result is
    cached as the working angle for subsequent frames, so a stable
    orientation costs one search and then nothing.

The search is what makes this robust rather than clever: it does not need the
rack heuristic to be right, only to be a good first guess.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import SKELETON_FEATURES  # noqa: E402
from pipeline.pose_backend import PoseBackend  # noqa: E402

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES

# Landmarks used to judge whether a candidate rotation produced a *plausible*
# body rather than merely some output: torso corners plus the arms that drive
# the protocol.
_SCORE_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)


def _rotation_matrix(width: int, height: int, degrees: float):
    """Rotation about the image centre, expanded so nothing is cropped.

    Cropping matters here: a limb that rotates out of the canvas is a limb
    MediaPipe cannot see, which would trade one detection failure for another.
    """
    import cv2

    centre = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += new_w * 0.5 - centre[0]
    matrix[1, 2] += new_h * 0.5 - centre[1]
    return matrix, new_w, new_h


def rack_roll_degrees(detections) -> Optional[float]:
    """Image-plane roll of the payload rack, from the HSV main-box rect.

    Returned in (-90, 90]: a rectangle defines a line, not a direction, so
    this is the rack's axis, not its polarity. The search below resolves the
    180-degree ambiguity by simply trying both.
    """
    from pipeline.rack_frame import pick_rack_rect, wrap90

    rect = pick_rack_rect(detections)
    if rect is None:
        return None
    (_cx, _cy), (w, h), angle = rect
    angle = float(angle)
    if w < h:
        angle -= 90.0
    return wrap90(angle)


class UprightPoseEstimator:
    """PoseBackend + image canonicalisation.

    Drop-in for PoseBackend where a rolled body is expected:

        est = UprightPoseEstimator(complexity=1, frame_width=w, frame_height=h)
        feats = est.process(frame_rgb, detections=hsv_dets)
    """

    def __init__(self,
                 complexity: int = 1,
                 min_det_conf: float = 0.3,
                 min_trk_conf: float = 0.3,
                 frame_width: int = 1280,
                 frame_height: int = 720,
                 downscale: int = 1,
                 search_angles: Sequence[float] = (0.0, 90.0, 180.0, 270.0,
                                                   45.0, 135.0, 225.0, 315.0),
                 model_path: Optional[str] = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.search_angles = tuple(search_angles)

        # static_image_mode=True is required, not incidental. Candidate
        # rotations of the SAME frame are independent hypotheses; feeding them
        # to a VIDEO-mode tracker would make each one contaminate the next
        # through the temporal filter, and the tracker's motion prior would be
        # nonsense across a 90-degree jump.
        self._backend = PoseBackend(
            complexity=complexity, min_det_conf=min_det_conf,
            min_trk_conf=min_trk_conf, downscale=downscale,
            frame_width=frame_width, frame_height=frame_height,
            model_path=model_path, static_image_mode=True,
        )
        self._working_angle = 0.0
        self.stats = {"frames": 0, "hit_first_try": 0, "searched": 0,
                      "failed": 0, "angle_changes": 0}

    @property
    def available(self) -> bool:
        return self._backend.available

    @property
    def backend(self) -> str:
        return self._backend.backend

    # ── internals ────────────────────────────────────────────────────────

    def _try_angle(self, frame_rgb: np.ndarray, degrees: float
                   ) -> Tuple[Optional[np.ndarray], float]:
        """Run pose on the frame rotated by `degrees`; return (feats, score)
        with feats already mapped back to the ORIGINAL image's normalised
        coordinates."""
        import cv2

        height, width = frame_rgb.shape[:2]
        if abs(degrees) < 1e-3:
            rotated = frame_rgb
            matrix = None
            new_w, new_h = width, height
        else:
            matrix, new_w, new_h = _rotation_matrix(width, height, degrees)
            rotated = cv2.warpAffine(frame_rgb, matrix, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        feats = self._backend.process(rotated)
        if not np.any(feats):
            return None, 0.0

        lm = feats.reshape(NUM_LANDMARKS, 4)
        score = float(np.mean([lm[i, 3] for i in _SCORE_LANDMARKS]))

        if matrix is not None:
            inverse = cv2.invertAffineTransform(matrix)
            pts = np.stack([lm[:, 0] * new_w, lm[:, 1] * new_h,
                            np.ones(NUM_LANDMARKS)], axis=0)
            back = inverse @ pts
            lm[:, 0] = back[0] / width
            lm[:, 1] = back[1] / height
            # z is a depth in units of the subject's own scale and is
            # unaffected by an in-plane rotation; visibility likewise.

        return lm.reshape(-1).astype(np.float32), score

    # ── public ───────────────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray,
                detections: Optional[Iterable] = None,
                min_score: float = 0.55) -> np.ndarray:
        """Full-resolution RGB frame -> (132,) features in ORIGINAL image
        coordinates.

        `detections` is the HSV detection list for this frame, if the caller
        already has it - it costs nothing to reuse and supplies the rack
        angle. Passing None just means the search does more work.
        """
        self.stats["frames"] += 1
        if not self._backend.available:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        # 1. The angle that worked last frame. Orientation changes slowly, so
        #    this is right almost always and costs a single inference.
        feats, score = self._try_angle(frame_rgb, self._working_angle)
        if feats is not None and score >= min_score:
            self.stats["hit_first_try"] += 1
            return feats

        best_feats, best_score, best_angle = feats, score, self._working_angle

        # 2. The rack's own roll, and its 180-degree flip.
        candidates: List[float] = []
        rack = rack_roll_degrees(detections) if detections is not None else None
        if rack is not None:
            candidates += [-rack, -rack + 180.0]

        # 3. A coarse sweep as the backstop.
        candidates += list(self.search_angles)

        self.stats["searched"] += 1
        for angle in candidates:
            angle = float(angle) % 360.0
            if abs(angle - self._working_angle) < 1e-3:
                continue
            feats, score = self._try_angle(frame_rgb, angle)
            if feats is not None and score > best_score:
                best_feats, best_score, best_angle = feats, score, angle
                if score >= min_score:
                    break

        if best_feats is None:
            self.stats["failed"] += 1
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        if abs(best_angle - self._working_angle) > 1e-3:
            self.stats["angle_changes"] += 1
            logger.debug("upright pose: working angle %.1f -> %.1f (score %.2f)",
                         self._working_angle, best_angle, best_score)
            self._working_angle = best_angle
        return best_feats

    def reset(self) -> None:
        """Forget the working angle. Call between takes/sessions."""
        self._working_angle = 0.0

    def close(self) -> None:
        self._backend.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    est = UprightPoseEstimator(complexity=0, frame_width=640, frame_height=360)
    out = est.process(np.zeros((360, 640, 3), dtype=np.uint8))
    assert out.shape == (FEATURE_DIM,) and np.isfinite(out).all()
    est.close()
    print("upright_pose smoke test passed.")
