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

from pipeline.rack_frame import RackFrameNormalizer, pick_rack_rect  # noqa: E402
from pipeline.hsv_detector import HSVBoxDetector  # noqa: E402

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARNING] mediapipe not installed. Run: pip install mediapipe")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# MediaPipe constants
NUM_POSE_LANDMARKS = 33

# Feature vector length per frame:
# pose only: 33 × 4 (x,y,z,vis) = 132
# (Hands removed for faster inference — pose captures key motion)
FEATURE_DIM = NUM_POSE_LANDMARKS * 4  # 132


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


def _rack_rect_for_frame(frame_bgr: np.ndarray, hsv_detector: Optional[HSVBoxDetector]):
    """Pick the rack-anchor rect for a frame (None when no box is visible)."""
    if hsv_detector is None:
        return None
    return pick_rack_rect(hsv_detector.detect(frame_bgr))


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

    sequences = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        features = extract_landmarks_from_frame(results)
        if normalizer is not None:
            features = normalizer.normalize(
                features, rack_rect=_rack_rect_for_frame(frame, hsv_detector)
            )
        sequences.append(features)

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

    sequences = []
    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        features = extract_landmarks_from_frame(results)
        if normalizer is not None:
            features = normalizer.normalize(
                features, rack_rect=_rack_rect_for_frame(frame, hsv_detector)
            )
        sequences.append(features)

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

    # Use Pose only (not Holistic) for faster processing
    mp_pose = mp.solutions.pose
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_sequences = []
    all_labels = []
    metadata = []

    normalizer = None
    hsv_detector = None
    if rack_normalize:
        normalizer = RackFrameNormalizer()
        hsv_detector = HSVBoxDetector()
        logger.info("Rack-frame normalization enabled for labeling.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Fastest mode
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:

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

            # Create sliding windows of SEQUENCE_WINDOW frames
            from config.experiment_config import SEQUENCE_WINDOW
            windows = []
            for start in range(0, len(seq) - SEQUENCE_WINDOW + 1, SEQUENCE_WINDOW // 2):
                window = seq[start: start + SEQUENCE_WINDOW]
                if len(window) == SEQUENCE_WINDOW:
                    windows.append(window)

            if windows:
                windows_arr = np.array(windows)
                labels_arr = np.full(len(windows), step_id, dtype=np.int64)
                all_sequences.append(windows_arr)
                all_labels.append(labels_arr)
                metadata.append({"step_id": step_id, "windows": len(windows)})
                logger.info("  → %d windows extracted (step %d)", len(windows), step_id)

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
    parser.add_argument("--rack-normalize", action="store_true",
                        help="Re-express pose in the payload-rack frame "
                             "(must match RACK_FRAME_NORMALIZE=True at inference)")
    args = parser.parse_args()

    label_map = build_default_label_map()
    run_labeling(args.input, args.output, label_map,
                 rack_normalize=args.rack_normalize)
