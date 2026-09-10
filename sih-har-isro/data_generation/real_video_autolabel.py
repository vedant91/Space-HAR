"""
Real "Gravitational-Mimic" Video Auto-Labeler
================================================
Turns the real footage in `new video data/` (Earth-recorded, deliberately
slow/floaty handling that mimics microgravity object handling) into training
signal for both the custom PoseNet and the LSTM/CNN step classifiers —
without any manual annotation and without any pretrained/open-source model:

  1. Pseudo-POSE labels: HSVBoxDetector's classical-CV hand detector (skin
     tone + motion, see pipeline/hsv_detector.py) tracks up to two hand
     blobs per frame. Only the two WRIST joints are ever labeled — we
     deliberately do NOT guess shoulders/hips/elbows from anatomical
     proportion, because those coarse guesses would be fed straight into
     pipeline/rack_frame.py's rotation/scale/origin math (the most
     safety-relevant computation downstream) if they were wrong. Unmeasured
     joints are left invalid and contribute no fine-tune loss (see
     train/train_posenet.py's masked loss).

  2. Pseudo-STEP labels: a small rule engine (RuleBasedStepLabeler) reads
     HSVBoxDetector's box-visibility/position/hand-proximity features frame
     by frame and maps them onto the 8-step protocol using the same
     required_objects logic the real state machine (pipeline/state_machine.py)
     encodes — e.g. "a hand is near the red box and it's moving" looks like
     step 3/4, "the red box is stationary, away from the main box, and no
     hand is near it" looks like step 5 (placed). Ambiguous frames are
     labeled 0 (idle/unknown) rather than forced into a guess. This is a
     heuristic, not ground truth — see the printed/serialized report and
     dataset/real_pseudo/autolabel_report.json for exactly how much of each
     video landed in each step vs. unknown, so a human can spot-check before
     trusting it.

Output layout (REAL_PSEUDO_DIR, default dataset/real_pseudo/):
    <video_stem>/frames/frame_00000.jpg, ...
    <video_stem>/poses.npz          — pose (N,132) float32, valid (N,13) float32
    <video_stem>/step_labels.npy    — (N,) int64, 0=unknown/idle, 1..8=step id
    <video_stem>/meta.json          — per-video summary (fps, n_frames, label histogram)
    autolabel_report.json           — overall summary across all processed videos

Usage:
    python data_generation/real_video_autolabel.py
    python data_generation/real_video_autolabel.py --video-dir "/path/to/videos" --stride 2
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    FRAME_WIDTH, FRAME_HEIGHT, REAL_VIDEO_DIR, REAL_PSEUDO_DIR,
    EXPERIMENT_STEPS, POSE_JOINT_SLOTS,
)
from pipeline.hsv_detector import HSVBoxDetector, Detection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NUM_LANDMARKS = 33
L_WRIST, R_WRIST = 15, 16
JOINT_SLOTS = list(POSE_JOINT_SLOTS)
_WRIST_VALID_IDX = {L_WRIST: JOINT_SLOTS.index(L_WRIST), R_WRIST: JOINT_SLOTS.index(R_WRIST)}

STEP_NAMES = {s["id"]: s["name"] for s in EXPERIMENT_STEPS}
STEP_NAMES[0] = "Unknown/idle"

# Pixel-space heuristics, tuned for FRAME_WIDTH x FRAME_HEIGHT (1280x720).
_NEAR_PX = 150            # hand<->box "handling it" distance
_FAR_FROM_MAIN_PX = 170   # box<->main_box "moved away from container" distance
_STABLE_PX_PER_FRAME = 5  # centroid displacement below this counts as "stationary"
_SMOOTH_WINDOW = 11       # rolling majority-vote window (odd, ~0.3-0.5s at 24-32fps)


# ═══════════════════════════════════════════════════════════════
# Pseudo-pose (wrists only — see module docstring for why)
# ═══════════════════════════════════════════════════════════════

def hands_to_wrist_pose(dets: List[Detection], frame_w: int, frame_h: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Up to two hand Detections -> a full (132,) pose vector with only
    L_WRIST/R_WRIST slots filled, plus a (13,) valid mask aligned to
    config.POSE_JOINT_SLOTS (all-zero except the wrist(s) actually found).

    Left/right assignment is by image x-coordinate (smaller x = L_WRIST),
    matching data_generation/synthetic_pose.py's base pose convention (its
    L_WRIST sits left of R_WRIST before any rotation) — valid as long as the
    real footage is captured with the astronaut roughly upright/facing the
    camera, same assumption the rack-frame torso fallback already makes for
    "protocol start ~ upright".
    """
    pose = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    valid = np.zeros(len(JOINT_SLOTS), dtype=np.float32)
    hands = [d for d in dets if d.label == "hand"]
    hands.sort(key=lambda d: d.centroid[0])  # left-to-right by x

    slots = []
    if len(hands) == 1:
        slots = [L_WRIST] if hands[0].centroid[0] < frame_w / 2 else [R_WRIST]
    elif len(hands) >= 2:
        slots = [L_WRIST, R_WRIST]
        hands = hands[:2]

    for det, slot in zip(hands, slots):
        cx, cy = det.centroid
        pose[slot, 0] = float(np.clip(cx / frame_w, 0.0, 1.0))
        pose[slot, 1] = float(np.clip(cy / frame_h, 0.0, 1.0))
        pose[slot, 2] = 0.0  # no depth signal from 2D skin/motion tracking
        pose[slot, 3] = float(np.clip(det.confidence, 0.0, 1.0))
        valid[_WRIST_VALID_IDX[slot]] = 1.0

    return pose.reshape(-1), valid


# ═══════════════════════════════════════════════════════════════
# Pseudo-step rule engine
# ═══════════════════════════════════════════════════════════════

class RuleBasedStepLabeler:
    """
    Frame-by-frame heuristic step classifier from HSV box/hand features —
    the same "what's visible + who's near it" logic pipeline/state_machine.py
    encodes via EXPERIMENT_STEPS.required_objects, run backwards (features ->
    step guess) instead of forwards (step -> expected features). Stateful:
    tracks centroid velocity so it can tell "picking up / still moving" (early
    step of a pair) from "placed / stationary" (late step of a pair).
    """

    def __init__(self):
        self._prev_centroid: Dict[str, Optional[Tuple[float, float]]] = {"red_box": None, "yellow_box": None}
        self._reached_yellow_phase = False  # once true, never re-emit red-only steps (protocol is one-directional)

    @staticmethod
    def _dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _speed(self, label: str, centroid: Optional[Tuple[float, float]]) -> Optional[float]:
        prev = self._prev_centroid.get(label)
        self._prev_centroid[label] = centroid
        return self._dist(prev, centroid)

    def update(self, feats: dict) -> int:
        """Returns a raw (pre-smoothing) step guess in 0..8."""
        main_c = feats.get("main_centroid")
        red_c, yel_c = feats.get("red_centroid"), feats.get("yellow_centroid")
        hand_cs = feats.get("hand_centroids") or []

        red_speed = self._speed("red_box", red_c)
        yel_speed = self._speed("yellow_box", yel_c)

        def nearest_hand_dist(box_c):
            if box_c is None or not hand_cs:
                return None
            return min(self._dist(box_c, h) for h in hand_cs)

        d_hand_red = nearest_hand_dist(red_c)
        d_hand_yellow = nearest_hand_dist(yel_c)
        d_red_main = self._dist(red_c, main_c)
        d_yellow_main = self._dist(yel_c, main_c)

        red_visible, yellow_visible = feats.get("red_visible"), feats.get("yellow_visible")
        red_handled = red_visible and d_hand_red is not None and d_hand_red < _NEAR_PX
        yellow_handled = yellow_visible and d_hand_yellow is not None and d_hand_yellow < _NEAR_PX
        red_moving = (red_speed or 0) > _STABLE_PX_PER_FRAME
        yellow_moving = (yel_speed or 0) > _STABLE_PX_PER_FRAME

        # ── Phase 0: lid still closed (neither inner box visible yet) ──
        if not red_visible and not yellow_visible:
            hand_near_main = feats.get("main_visible") and d_hand_red is None and d_hand_yellow is None \
                and hand_cs and main_c and min(self._dist(main_c, h) for h in hand_cs) < _NEAR_PX
            return 2 if hand_near_main else 1

        # ── Red-box phase (protocol does red before yellow) ──
        if red_visible and not self._reached_yellow_phase:
            if red_handled and (red_moving or (d_red_main is not None and d_red_main < _FAR_FROM_MAIN_PX)):
                return 3  # picking up, still near the container
            if red_handled:
                return 4  # holding it away from the container — examining
            if not red_moving and d_red_main is not None and d_red_main >= _FAR_FROM_MAIN_PX \
                    and (d_hand_red is None or d_hand_red >= _NEAR_PX):
                return 5  # released, stationary, away from the container

        # ── Yellow-box phase ──
        if yellow_visible:
            if yellow_handled and (yellow_moving or (d_yellow_main is not None and d_yellow_main < _FAR_FROM_MAIN_PX)):
                self._reached_yellow_phase = True
                return 6
            if yellow_handled:
                self._reached_yellow_phase = True
                return 7
            if not yellow_moving and d_yellow_main is not None and d_yellow_main >= _FAR_FROM_MAIN_PX \
                    and (d_hand_yellow is None or d_hand_yellow >= _NEAR_PX):
                self._reached_yellow_phase = True
                return 8

        return 0  # doesn't clearly match any step's expected evidence


def smooth_labels(raw: np.ndarray, window: int = _SMOOTH_WINDOW) -> np.ndarray:
    """Centered rolling majority vote — same spirit as state_machine.py's
    STEP_CONFIRM_FRAMES debounce, applied offline."""
    n = len(raw)
    out = np.zeros(n, dtype=np.int64)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = raw[lo:hi]
        vals, counts = np.unique(window_vals, return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return out


# ═══════════════════════════════════════════════════════════════
# Per-video processing
# ═══════════════════════════════════════════════════════════════

def process_video(video_path: Path, out_dir: Path, stride: int = 1) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return {"video": str(video_path), "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    detector = HSVBoxDetector(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    labeler = RuleBasedStepLabeler()

    video_out = out_dir / video_path.stem
    frames_dir = video_out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    poses, valids, raw_labels = [], [], []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if (idx - 1) % stride != 0:
            continue
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        dets = detector.detect(frame)
        feats = detector.dets_to_feature_dict(dets)

        pose, valid = hands_to_wrist_pose(dets, FRAME_WIDTH, FRAME_HEIGHT)
        step_guess = labeler.update(feats)

        cv2.imwrite(str(frames_dir / f"frame_{saved:05d}.jpg"), frame,
                   [cv2.IMWRITE_JPEG_QUALITY, 92])
        poses.append(pose)
        valids.append(valid)
        raw_labels.append(step_guess)
        saved += 1
    cap.release()

    if saved == 0:
        return {"video": str(video_path), "error": "no frames decoded"}

    smoothed = smooth_labels(np.array(raw_labels, dtype=np.int64))
    np.savez(str(video_out / "poses.npz"),
             pose=np.stack(poses, axis=0).astype(np.float32),
             valid=np.stack(valids, axis=0).astype(np.float32))
    np.save(str(video_out / "step_labels.npy"), smoothed)

    hist = {int(k): int(v) for k, v in zip(*np.unique(smoothed, return_counts=True))}
    n_wrists = int(np.sum(np.stack(valids, axis=0).sum(axis=0)[list(_WRIST_VALID_IDX.values())]))
    meta = {
        "video": str(video_path),
        "fps": fps,
        "n_frames_source": idx,
        "n_frames_saved": saved,
        "stride": stride,
        "step_label_histogram": {STEP_NAMES.get(k, str(k)): v for k, v in hist.items()},
        "frac_unknown": hist.get(0, 0) / saved,
        "wrist_detections_total": n_wrists,
        "wrist_detection_rate": n_wrists / (saved * 2),
    }
    (video_out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("%s: %d frames, unknown=%.0f%%, wrist detect rate=%.0f%%, labels=%s",
               video_path.name, saved, 100 * meta["frac_unknown"],
               100 * meta["wrist_detection_rate"], meta["step_label_histogram"])
    return meta


def run_autolabel(video_dir: Optional[str] = None, out_dir: str = REAL_PSEUDO_DIR,
                  stride: int = 1) -> Dict:
    video_dir = Path(video_dir) if video_dir else Path(REAL_VIDEO_DIR)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov"))
                    + list(video_dir.glob("*.avi")))
    if not videos:
        logger.warning("No videos found under %s", video_dir)
        return {"videos": [], "n_videos": 0}

    results = []
    for v in videos:
        logger.info("Processing %s ...", v.name)
        results.append(process_video(v, out_path, stride=stride))

    report = {
        "source_dir": str(video_dir),
        "output_dir": str(out_path),
        "n_videos": len(videos),
        "videos": results,
        "note": (
            "Pseudo-labels are heuristic (classical CV rule engine + skin/motion "
            "hand tracking), not human ground truth. Pose labels only ever claim "
            "the two wrist joints — see this module's docstring for why. Review "
            "step_label_histogram / frac_unknown per video before trusting a "
            "video's labels heavily in training."
        ),
    }
    (out_path / "autolabel_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                     encoding="utf-8")
    logger.info("Auto-label report written to %s", out_path / "autolabel_report.json")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label real gravitational-mimic videos")
    parser.add_argument("--video-dir", type=str, default=None,
                        help=f"Defaults to config.REAL_VIDEO_DIR ({REAL_VIDEO_DIR})")
    parser.add_argument("--out-dir", type=str, default=REAL_PSEUDO_DIR)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()
    report = run_autolabel(args.video_dir, args.out_dir, stride=args.stride)
    print(json.dumps({k: v for k, v in report.items() if k != "videos"}, indent=2, default=str))
