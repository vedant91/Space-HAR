"""
MediaPipe Auto-Labeler
=======================
Runs MediaPipe Holistic on a folder of videos/frames to auto-generate
skeleton sequence data for LSTM training. Zero manual annotation.

Usage:
    python data_generation/mediapipe_labeler.py --input dataset/synthetic_gemini/videos --output dataset/skeleton_sequences
"""

import os
import sys
import cv2
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.rack_frame import RackFrameNormalizer  # noqa: E402
from pipeline.hsv_detector import HSVBoxDetector  # noqa: E402
from pipeline.features import LAYOUT_DIMS, build_frame_features, stack_windows  # noqa: E402
from config.experiment_config import FEATURE_LAYOUT, SAMPLE_EVERY  # noqa: E402

# Pose goes through pipeline/pose_backend.py, not `mp.solutions` directly:
# that API was removed in MediaPipe 1.0 and this module would not import at
# all on a current install. See pose_backend.py for the full note.
from pipeline.pose_backend import PoseBackend, describe as describe_pose_backend  # noqa: E402

_POSE_INFO = describe_pose_backend()
MP_AVAILABLE = bool(_POSE_INFO.get("usable"))
if not MP_AVAILABLE:
    print(f"[WARNING] No usable MediaPipe pose backend: {_POSE_INFO}\n"
          "          pip install mediapipe && python tools/fetch_pose_model.py")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# MediaPipe constants
NUM_POSE_LANDMARKS = 33

# Feature vector length per frame: the deployed model layout (pose + object
# channels), built by pipeline/features.py exactly as the live pipeline does.
FEATURE_DIM = LAYOUT_DIMS[FEATURE_LAYOUT]


def extract_landmarks_from_frame(results) -> np.ndarray:
    """Convert MediaPipe Pose results to a flat feature vector (pose only, 132-dim)."""
    features = []

    # Pose landmarks only (no hands — faster inference)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0.0] * (NUM_POSE_LANDMARKS * 4))

    return np.array(features, dtype=np.float32)


def _frame_vector(frame_bgr: np.ndarray, pose, hsv_detector: HSVBoxDetector,
                  normalizer: Optional[RackFrameNormalizer], prev_dets):
    """One frame -> (model vector, this frame's detections), through the live
    loop's own perceive() (pose hinted with the previous frame's detections)
    and the shared builder."""
    from pipeline.har_pipeline import perceive
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    dets, raw = perceive(frame_bgr, rgb, hsv_detector, pose, prev_dets)
    return build_frame_features(raw, dets, (w, h), normalizer, FEATURE_LAYOUT), dets


def process_video(video_path: str, step_label: int, pose,
                  normalizer: Optional[RackFrameNormalizer] = None,
                  hsv_detector: Optional[HSVBoxDetector] = None) -> Optional[np.ndarray]:
    """
    Process a video file and return skeleton sequence array.
    Returns: shape (num_frames, FEATURE_DIM)
    If normalizer is given, features are re-expressed in the rack frame
    (matching RACK_FRAME_NORMALIZE at inference).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return None

    if normalizer is not None:
        normalizer.reset()  # each video is a new session: re-latch polarity
    pose.reset()            # ...and re-latch the upright working angle

    sequences, prev = [], None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vec, prev = _frame_vector(frame, pose, hsv_detector or HSVBoxDetector(), normalizer, prev)
        sequences.append(vec)

    cap.release()
    if not sequences:
        return None
    return np.array(sequences)


def process_frames_folder(folder: str, step_label: int, pose,
                          normalizer: Optional[RackFrameNormalizer] = None,
                          hsv_detector: Optional[HSVBoxDetector] = None) -> Optional[np.ndarray]:
    """Process a folder of image frames as a sequence."""
    frame_files = sorted(
        list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
    )
    if not frame_files:
        return None

    if normalizer is not None:
        normalizer.reset()  # each folder is a new session: re-latch polarity
    pose.reset()

    sequences, prev = [], None
    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        vec, prev = _frame_vector(frame, pose, hsv_detector or HSVBoxDetector(), normalizer, prev)
        sequences.append(vec)

    return np.array(sequences) if sequences else None


def run_labeling(input_path: str, output_dir: str, step_label_map: dict,
                 rack_normalize: bool = False):
    """
    Main labeling loop.
    
    step_label_map: {folder_or_file_pattern: step_id}
    Example: {"step_01": 1, "step_02": 2, ...}

    rack_normalize: re-express landmarks in the payload-rack frame so training
    data matches RACK_FRAME_NORMALIZE=True at inference.
    """
    if not MP_AVAILABLE:
        logger.error("mediapipe not installed.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_sequences = []
    all_labels = []
    metadata = []

    normalizer = None
    # Always on: object channels need the boxes, not just the rack anchor.
    hsv_detector = HSVBoxDetector()
    if rack_normalize:
        normalizer = RackFrameNormalizer()
        logger.info("Rack-frame normalization enabled for labeling.")

    # The live pose stage with config defaults (complexity, POSE_INPUT_WIDTH,
    # upright), so labelled data is built with the same pose as serving.
    from pipeline.har_pipeline import OptimizedMPWrapper
    pose = OptimizedMPWrapper()
    try:

        for step_tag, step_id in step_label_map.items():
            candidate = Path(input_path) / step_tag

            if candidate.is_dir():
                # Process as folder of frames
                logger.info("Processing folder: %s → Step %d", candidate, step_id)
                seq = process_frames_folder(str(candidate), step_id, pose,
                                            normalizer=normalizer,
                                            hsv_detector=hsv_detector)

            elif candidate.is_file() and candidate.suffix in {".mp4", ".avi", ".mov"}:
                # Process as video file
                logger.info("Processing video: %s → Step %d", candidate, step_id)
                seq = process_video(str(candidate), step_id, pose,
                                    normalizer=normalizer,
                                    hsv_detector=hsv_detector)

            else:
                # Try globbing videos in a step folder
                step_dir = Path(input_path) / f"step_{step_id:02d}"
                if not step_dir.exists():
                    logger.warning("No data found for step %d", step_id)
                    continue

                videos = list(step_dir.glob("*.mp4")) + list(step_dir.glob("*.avi"))
                all_step_seqs = []
                for vid in videos:
                    logger.info("  → %s", vid.name)
                    s = process_video(str(vid), step_id, pose,
                                      normalizer=normalizer,
                                      hsv_detector=hsv_detector)
                    if s is not None:
                        all_step_seqs.append(s)

                if not all_step_seqs:
                    continue
                seq = np.concatenate(all_step_seqs, axis=0)

            if seq is None or len(seq) == 0:
                continue

            # Create sliding windows of SEQUENCE_WINDOW frames.
            # NOTE for whoever wires this into training later: this output is
            # currently NOT consumed by train_lstm.py (only synthetic_pose.py's
            # output is, and that one now saves groups.npy for a leakage-safe
            # split — see its generate_dataset()). If this path is wired in,
            # give it the same treatment: a per-VIDEO group id (not per-step —
            # grouping by step_id here would put entire classes only in train
            # or only in val, since GroupShuffleSplit keeps whole groups
            # together and there's only one group per step in this function).
            # Shared windowing (pipeline/features.py): decimated to
            # SAMPLE_EVERY like the live buffer, 50%-overlap stride in samples.
            from config.experiment_config import SEQUENCE_WINDOW
            windows_arr, labels_arr = stack_windows(
                seq, np.full(len(seq), step_id), stride=SEQUENCE_WINDOW // 2,
                sample_every=SAMPLE_EVERY)

            if len(windows_arr):
                all_sequences.append(windows_arr)
                all_labels.append(labels_arr)
                metadata.append({"step_id": step_id, "windows": int(len(windows_arr))})
                logger.info("  → %d windows extracted (step %d)", len(windows_arr), step_id)
    finally:
        # The old `with mp_pose.Pose(...)` released the graph on any exit path.
        # The pose stage is not a context manager, so the close has to be explicit
        # or an exception mid-labelling leaks the TFLite interpreter.
        pose.close()

    if not all_sequences:
        logger.error("No sequences extracted. Check input data.")
        return

    # Save consolidated numpy arrays
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)

    np.save(str(out_path / "X_sequences.npy"), X)
    np.save(str(out_path / "y_labels.npy"), y)

    with open(str(out_path / "metadata.json"), "w") as f:
        json.dump({"total_windows": int(len(X)), "feature_dim": int(FEATURE_DIM),
                   "feature_layout": FEATURE_LAYOUT, "sample_every": SAMPLE_EVERY,
                   "rack_normalized": bool(rack_normalize),
                   "steps": metadata}, f, indent=2)

    logger.info("=" * 60)
    logger.info("Labeling complete: X=%s, y=%s", X.shape, y.shape)
    logger.info("Saved to: %s", output_dir)


def build_default_label_map() -> dict:
    """Build label map for the standard experiment folders."""
    from config.experiment_config import EXPERIMENT_STEPS
    return {f"step_{s['id']:02d}": s["id"] for s in EXPERIMENT_STEPS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Auto-Labeler")
    parser.add_argument("--input", default="dataset/synthetic_gemini/frames",
                        help="Input directory (contains step_XX sub-folders)")
    parser.add_argument("--output", default="dataset/skeleton_sequences",
                        help="Output directory for .npy sequence files")
    from config.experiment_config import RACK_FRAME_NORMALIZE
    parser.add_argument("--rack-normalize", dest="rack_normalize",
                        action="store_true",
                        help="Re-express pose in the payload-rack frame "
                             "(must match RACK_FRAME_NORMALIZE at inference)")
    parser.add_argument("--no-rack-normalize", dest="rack_normalize",
                        action="store_false")
    parser.set_defaults(rack_normalize=RACK_FRAME_NORMALIZE)
    args = parser.parse_args()

    label_map = build_default_label_map()
    run_labeling(args.input, args.output, label_map,
                 rack_normalize=args.rack_normalize)
