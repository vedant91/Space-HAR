"""
Pose estimation backend — MediaPipe Tasks API, with a legacy Solutions fallback.

Why this file exists
--------------------
Every pose call in this project used to go through `mp.solutions.pose.Pose`.
That API — the "Solutions" API — was deprecated by Google and **removed
entirely in MediaPipe 1.0**. On a current install, `mediapipe.solutions`
does not exist, so `pipeline/har_pipeline.py` and
`data_generation/mediapipe_labeler.py` both fail at import with

    AttributeError: module 'mediapipe' has no attribute 'solutions'

i.e. the deployed pipeline could not run at all on any recent MediaPipe.
This module is the single place that knows how to talk to MediaPipe, so the
rest of the codebase never has to care which era of the API is installed:

  * MediaPipe >= 0.10 with Tasks available -> PoseLandmarker (preferred).
  * Older MediaPipe with `mp.solutions` -> the legacy path, unchanged.
  * Neither -> `available` is False and callers get zero features, exactly
    as before.

Both paths emit the identical 132-dim contract (33 landmarks x, y, z,
visibility), so `config.SKELETON_FEATURES`, the LSTM input size and the
state machine are all untouched.

Offline operation
-----------------
The Tasks API needs a `.task` model bundle on disk. That fits the problem
statement's "runs on offline standalone system" requirement better than the
old Solutions API did, because the weights are an explicit, versioned file
that ships with the application instead of being fetched into a hidden
package-internal cache. `models/mediapipe/pose_landmarker_full.task` is the
default; `tools/fetch_pose_model.py` downloads it once.

A note on `visibility`
----------------------
The Tasks API reports both `presence` and `visibility` per landmark, whereas
the Solutions API reported only `visibility`. The 132-dim contract has one
slot, so this returns `min(visibility, presence)` — a landmark that the model
believes is absent should not score highly just because, if it were present,
it would not be occluded.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import SKELETON_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mediapipe"
_MODEL_FILENAMES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

# MediaPipe's own mapping from model_complexity (Solutions) to bundle name.
_COMPLEXITY_TO_VARIANT = {0: "lite", 1: "full", 2: "heavy"}


def default_model_path(variant: str = "full") -> Optional[Path]:
    path = _DEFAULT_MODEL_DIR / _MODEL_FILENAMES.get(variant, _MODEL_FILENAMES["full"])
    return path if path.exists() else None


def _probe_backends() -> Tuple[bool, bool]:
    """(tasks_available, solutions_available) without importing heavy state."""
    try:
        import mediapipe as mp
    except ImportError:
        return False, False
    tasks = False
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        tasks = True
    except Exception:
        tasks = False
    return tasks, hasattr(mp, "solutions")


class PoseBackend:
    """Unified pose estimator.

    Usage mirrors the old wrapper closely:

        backend = PoseBackend(complexity=0, downscale=2)
        feats = backend.process(frame_rgb_full)     # (132,) float32
        backend.close()

    `process` takes a full-resolution RGB frame and handles its own
    downscaling, so callers keep the latency behaviour the pipeline was tuned
    for (config.MEDIAPIPE_DOWNSCALE).
    """

    def __init__(self,
                 complexity: int = 0,
                 min_det_conf: float = 0.3,
                 min_trk_conf: float = 0.3,
                 downscale: int = 2,
                 frame_width: int = 1280,
                 frame_height: int = 720,
                 model_path: Optional[str] = None,
                 prefer: str = "auto",
                 static_image_mode: bool = False):
        self.downscale = max(int(downscale), 1)
        self.small_w = max(int(frame_width) // self.downscale, 16)
        self.small_h = max(int(frame_height) // self.downscale, 16)
        self.backend = "none"
        self._impl = None
        self._timestamp_ms = 0
        self._static = static_image_mode

        tasks_ok, solutions_ok = _probe_backends()

        order: Sequence[str]
        if prefer == "tasks":
            order = ("tasks",)
        elif prefer == "solutions":
            order = ("solutions",)
        else:
            order = ("tasks", "solutions")

        for choice in order:
            if choice == "tasks" and tasks_ok:
                if self._init_tasks(complexity, min_det_conf, min_trk_conf, model_path):
                    break
            elif choice == "solutions" and solutions_ok:
                if self._init_solutions(complexity, min_det_conf, min_trk_conf):
                    break

        if self._impl is None:
            logger.warning(
                "No usable MediaPipe pose backend "
                "(tasks_available=%s, solutions_available=%s). Skeleton features "
                "will be zeros. Install mediapipe and run "
                "tools/fetch_pose_model.py.", tasks_ok, solutions_ok)

    # ── initialisation ───────────────────────────────────────────────────

    def _init_tasks(self, complexity, min_det_conf, min_trk_conf,
                    model_path) -> bool:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            variant = _COMPLEXITY_TO_VARIANT.get(int(complexity), "full")
            path = Path(model_path) if model_path else default_model_path(variant)
            if path is None or not Path(path).exists():
                # Fall back through the variants before giving up: any bundle
                # is better than no pose estimation at all.
                for alt in ("full", "lite", "heavy"):
                    candidate = default_model_path(alt)
                    if candidate is not None:
                        path = candidate
                        variant = alt
                        break
            if path is None:
                logger.warning("MediaPipe Tasks available but no .task bundle found "
                               "under %s — run tools/fetch_pose_model.py.",
                               _DEFAULT_MODEL_DIR)
                return False

            running_mode = (vision.RunningMode.IMAGE if self._static
                            else vision.RunningMode.VIDEO)
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=float(min_det_conf),
                min_pose_presence_confidence=float(min_det_conf),
                min_tracking_confidence=float(min_trk_conf),
                output_segmentation_masks=False,
            )
            self._impl = vision.PoseLandmarker.create_from_options(options)
            self._vision = vision
            self.backend = "tasks"
            self.model_variant = variant
            self.model_path = str(path)
            logger.info("MediaPipe Tasks PoseLandmarker loaded (%s, mode=%s, "
                        "downscale=%d)", variant,
                        "IMAGE" if self._static else "VIDEO", self.downscale)
            return True
        except Exception as e:
            logger.warning("MediaPipe Tasks backend failed to initialise: %s", e)
            self._impl = None
            return False

    def _init_solutions(self, complexity, min_det_conf, min_trk_conf) -> bool:
        try:
            import mediapipe as mp
            self._impl = mp.solutions.pose.Pose(
                static_image_mode=self._static,
                model_complexity=int(complexity),
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=float(min_det_conf),
                min_tracking_confidence=float(min_trk_conf),
            )
            self.backend = "solutions"
            self.model_variant = _COMPLEXITY_TO_VARIANT.get(int(complexity), "full")
            self.model_path = "(bundled with mediapipe.solutions)"
            logger.info("MediaPipe Solutions Pose loaded (legacy API, complexity=%d)",
                        complexity)
            return True
        except Exception as e:
            logger.warning("MediaPipe Solutions backend failed to initialise: %s", e)
            self._impl = None
            return False

    # ── inference ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._impl is not None

    def _downscale(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2
        if frame_rgb.shape[1] == self.small_w and frame_rgb.shape[0] == self.small_h:
            return frame_rgb
        return cv2.resize(frame_rgb, (self.small_w, self.small_h),
                          interpolation=cv2.INTER_LINEAR)

    def process(self, frame_rgb_full: np.ndarray,
                timestamp_ms: Optional[int] = None) -> np.ndarray:
        """Full-res RGB frame -> (132,) float32 feature vector."""
        if self._impl is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        try:
            small = self._downscale(frame_rgb_full)
            if self.backend == "tasks":
                return self._process_tasks(small, timestamp_ms)
            return self._process_solutions(small)
        except Exception as e:
            logger.warning("Pose inference failed (%s); returning zeros.", e)
            return np.zeros(FEATURE_DIM, dtype=np.float32)

    def _process_tasks(self, small_rgb: np.ndarray,
                       timestamp_ms: Optional[int]) -> np.ndarray:
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(small_rgb))
        if self._static:
            result = self._impl.detect(image)
        else:
            # VIDEO mode requires a strictly increasing timestamp; the tracker
            # uses it to decide whether to re-detect, so a repeated or
            # decreasing value makes it throw.
            if timestamp_ms is None:
                self._timestamp_ms += 33
                timestamp_ms = self._timestamp_ms
            else:
                self._timestamp_ms = max(self._timestamp_ms + 1, int(timestamp_ms))
                timestamp_ms = self._timestamp_ms
            result = self._impl.detect_for_video(image, timestamp_ms)

        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        landmark_sets = getattr(result, "pose_landmarks", None) or []
        if not landmark_sets:
            return feats.reshape(-1)
        for i, lm in enumerate(landmark_sets[0][:NUM_LANDMARKS]):
            feats[i, 0] = lm.x
            feats[i, 1] = lm.y
            feats[i, 2] = lm.z
            vis = getattr(lm, "visibility", None)
            pres = getattr(lm, "presence", None)
            if vis is None and pres is None:
                feats[i, 3] = 1.0
            else:
                vals = [v for v in (vis, pres) if v is not None]
                feats[i, 3] = float(min(vals))
        return feats.reshape(-1)

    def _process_solutions(self, small_rgb: np.ndarray) -> np.ndarray:
        buf = np.ascontiguousarray(small_rgb)
        buf.flags.writeable = False
        result = self._impl.process(buf)
        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        if result and getattr(result, "pose_landmarks", None):
            for i, lm in enumerate(result.pose_landmarks.landmark):
                if i >= NUM_LANDMARKS:
                    break
                feats[i, 0] = lm.x
                feats[i, 1] = lm.y
                feats[i, 2] = lm.z
                feats[i, 3] = lm.visibility
        return feats.reshape(-1)

    def close(self) -> None:
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:
                pass
            self._impl = None


def describe() -> dict:
    """Report which backends this installation can use — used by
    `main.py --mode status` so a missing model bundle is visible before a run
    rather than as zeroed features during one."""
    tasks_ok, solutions_ok = _probe_backends()
    bundles = {name: str(_DEFAULT_MODEL_DIR / fname)
               for name, fname in _MODEL_FILENAMES.items()
               if (_DEFAULT_MODEL_DIR / fname).exists()}
    try:
        import mediapipe as mp
        version = getattr(mp, "__version__", "unknown")
    except ImportError:
        version = None
    return {
        "mediapipe_version": version,
        "tasks_api": tasks_ok,
        "solutions_api": solutions_ok,
        "model_bundles": bundles,
        "usable": bool((tasks_ok and bundles) or solutions_ok),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json
    print(json.dumps(describe(), indent=2))
    backend = PoseBackend(complexity=0, frame_width=640, frame_height=360, downscale=1)
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    feats = backend.process(dummy)
    print(f"backend={backend.backend} available={backend.available} "
          f"shape={feats.shape} finite={bool(np.isfinite(feats).all())}")
    assert feats.shape == (FEATURE_DIM,)
    backend.close()
    print("pose_backend smoke test passed.")
