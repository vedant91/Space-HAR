# ============================================================
# ISRO SIH 2026 — HAR System Configuration
# Experiment: Red/Yellow Box Sorting Protocol
# ============================================================

from dataclasses import dataclass, field
from typing import List, Dict
import json
import logging
import os
from pathlib import Path

# ── Experiment Step Definitions ──────────────────────────────
# Loaded from a procedure pack (YAML), not hardcoded — see
# config/procedure_pack.py and packs/box_sort_v1.yaml. This makes swapping
# the whole protocol a file swap, not a code change. Override with the
# HAR_PROCEDURE_PACK env var (main.py's --pack flag sets this before any
# config-dependent module is imported) — defaults to the box-sort protocol
# this project has always run. Fails loud (raises, no silent fallback) on a
# missing/malformed pack, deliberately: running the wrong protocol
# unnoticed is worse than crashing at startup.
from config.procedure_pack import load_pack, ProcedurePackError  # noqa: E402

_PACKS_DIR = Path(__file__).resolve().parent.parent / "packs"
_DEFAULT_PACK_PATH = _PACKS_DIR / "box_sort_v1.yaml"
_PACK_PATH = Path(os.environ.get("HAR_PROCEDURE_PACK", str(_DEFAULT_PACK_PATH)))
try:
    _ACTIVE_PACK = load_pack(str(_PACK_PATH))
except ProcedurePackError as _e:
    raise RuntimeError(f"Failed to load procedure pack '{_PACK_PATH}': {_e}") from _e

EXPERIMENT_STEPS = _ACTIVE_PACK.experiment_steps
PROCEDURE_PACK_ID = _ACTIVE_PACK.experiment_id
PROCEDURE_PACK_NAME = _ACTIVE_PACK.name
PROCEDURE_PACK_PATH = str(_PACK_PATH)
# ── YOLO Detection Classes ────────────────────────────────────
# main_box is split into main_box_open/main_box_closed here because that's
# a genuinely LEARNABLE visual distinction now that simulation/renderer.py
# draws a real lid graphic difference between the two states (previously
# both states rendered pixel-identical main_box pixels — splitting the
# label without splitting the pixels would have taught YOLO to guess from
# nearby red/yellow-box visibility instead of the box itself). HSV can't
# tell lid state from color, so it still only ever emits plain "main_box";
# YOLOBoxDetector.detect() normalizes its two lid classes back to
# label="main_box" + a separate lid_state field, so every existing
# `d.label == "main_box"` check downstream (rack_frame.pick_rack_rect,
# anomaly_monitor.DRIFT_TRACKED_LABELS, features.BOX_LABELS, ...) works
# unchanged with either detector backend. person is new: HANDOFF's
# concern about fine-tuning erasing COCO's pretrained person-detection
# doesn't apply here — we provide our OWN explicit synthetic person
# bboxes in this same training set, not relying on inherited behavior
# surviving fine-tuning.
DETECTION_CLASSES = {
    0: "hand",
    1: "red_box",
    2: "yellow_box",
    3: "main_box_open",
    4: "main_box_closed",
    5: "person",
}
CLASS_NAMES = list(DETECTION_CLASSES.values())
# Named DETECTION_NUM_CLASSES (not NUM_CLASSES) — this is the YOLO/HSV
# detection class count, unrelated to the step
# classifiers' class count (NUM_STEPS, 9). A generic "NUM_CLASSES" name here
# previously collided in meaning with train_cnn.py's own same-named local.
DETECTION_NUM_CLASSES = len(CLASS_NAMES)

# ── Camera & Video Settings ───────────────────────────────────
CAMERA_INDEX = 0          # Default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for max latency (was 2 for speed)

# ── Latency Optimization Settings ─────────────────────────────
# Pose settings are ONE value for training, scoring and the live loop
# (pipeline/har_pipeline.OptimizedMPWrapper is the only pose stage). Measured
# at 1280x720 on rendered frames: complexity 0 at 640 px 45.8 ms/frame vs
# complexity 1 at full res 60.6 ms, both detecting 60/60.
MEDIAPIPE_MODEL_COMPLEXITY = 0   # 0=fastest, 1=balanced, 2=accurate
MEDIAPIPE_MIN_DET_CONF = 0.3    # Lower = faster (fewer re-detections)
MEDIAPIPE_MIN_TRK_CONF = 0.3    # Lower = faster tracking
# Pose input is specified by pixel WIDTH, not a downscale factor, so a 900-px
# render and a 1280-px camera frame reach MediaPipe at the same size
# (height follows the frame's aspect ratio).
POSE_INPUT_WIDTH = 640
HSV_DOWNSCALE = 2               # Process HSV at FRAME_WIDTH/HSV_DOWNSCALE
USE_THREADED_INFERENCE = True    # Parallelize HSV + pose net in threads
LSTM_USE_ONNX = True            # Use ONNX Runtime for LSTM (faster than PyTorch)

# ── Custom Pose Model (replaces MediaPipe — trained from scratch, zero
# pretrained/third-party weights) ─────────────────────────────
# HARPoseNet (pipeline/pose_net.py) is a small heatmap-regression CNN
# supervised entirely by simulation/renderer.py's own deterministic pose
# ground truth (it draws the astronaut, so it knows every joint's exact
# pixel location) — no MediaPipe, no YOLO, no other pretrained detector
# anywhere in this pipeline. See train/train_posenet.py.
#
# Only the 13 joints that actually drive step classification (the arms +
# coarse posture — see data_generation/synthetic_pose.py's _STEP_WAYPOINTS,
# which only ever varies wrist position) are predicted; the remaining 20 of
# the 33 MediaPipe-style slots stay zero/invisible in the output feature
# vector. That keeps the 132-dim contract every downstream module
# (rack_frame.py, train_lstm.py, train_cnn.py's temporal buffer) already
# assumes, without claiming to detect landmarks nothing ever supervised.
POSE_JOINT_SLOTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
#                    nose  Lsh  Rsh  Lel  Rel  Lwr  Rwr  Lhip Rhip Lkn  Rkn  Lank Rank
POSE_NUM_JOINTS = len(POSE_JOINT_SLOTS)
POSE_INPUT_SIZE = 256       # model input, square RGB (architecture is a fixed stride-4
                            # ladder — must scale together with POSE_HEATMAP_SIZE, 4:1)
POSE_HEATMAP_SIZE = 64      # output heatmap resolution (stride 4). Bumped 48->64 to close
                            # the elbow/wrist PCK gap (see README.md "Current results") —
                            # quantization error scales with heatmap pixel size, and
                            # elbows/wrists move too fast for the old 48px grid to localize
                            # precisely even with soft-argmax decode.
POSE_HEATMAP_SIGMA = 2.0    # gaussian target sigma, in heatmap-pixel units (scaled 1.5->2.0
                            # with the 48->64 heatmap bump, same 64/48 ratio, so the target's
                            # size relative to torso — and thus training difficulty — is
                            # unchanged, not accidentally sharpened)
POSE_MIN_CONFIDENCE = 0.15  # heatmap peak below this -> visibility 0 (joint not claimed)
# Sub-pixel decode: a plain argmax on a 64x64 heatmap quantizes to ~1 heatmap
# pixel, which alone can burn most of the PCK@0.10*torso error budget (torso
# is only ~15-16 heatmap-px wide at this resolution/frame). Decode instead
# takes a local softmax-weighted centroid in a (2*radius+1) window around the
# argmax — cheap (13 tiny windows/frame) and measured ~40% relative PCK
# improvement over plain argmax on the same checkpoint, no retraining needed.
POSE_DECODE_RADIUS = 3      # scaled 2->3 with the heatmap bump (same relative window size)
POSE_DECODE_BETA = 20.0

POSENET_EPOCHS = 40              # bumped 24->40: more synth samples per epoch needs more
                                 # epochs to converge; early stopping (patience=6) still caps
                                 # wasted time if it plateaus sooner
POSENET_BATCH_SIZE = 32
POSENET_LR = 1e-3
POSENET_SYNTH_SAMPLES = 8000     # rendered synthetic frames for training (was 4000 — more
                                 # coverage of fast elbow/wrist motion)
POSENET_VAL_SAMPLES = 600        # held-out rendered frames for PCK eval
POSENET_FINETUNE_EPOCHS = 25     # real-video pseudo-label fine-tune epochs. Safe to run
                                 # longer than Stage-1-scale epoch counts: finetune_on_real's
                                 # _configure_finetune_trainable() freezes the entire shared
                                 # trunk (only heatmap_head + aux_head's last layer adapt —
                                 # ~0.4% of params), so this cannot regress other joints by
                                 # construction, and train_posenet.py's regression guard
                                 # double-checks synthetic PCK before accepting the result
                                 # regardless.
POSENET_FINETUNE_LR = 1e-3       # higher than Stage-1's LR is fine here — the blast radius
                                 # is a tiny, isolated readout layer, not the shared trunk
POSENET_PCK_THRESHOLD = 0.10     # "correct" = within 10% of torso length

# ── Orientation-Agnostic Pose (microgravity: no fixed 'up') ──────────────────
# Stage 1: re-express 2D pose in a payload-rack reference frame instead of the
# image/floor frame. Roll comes from the HSV main_box rect (fallback: torso),
# origin is the torso center, scale is torso length. Output stays 132-dim, so
# the LSTM contract is unchanged — retrain with the same flag for best results.
RACK_FRAME_NORMALIZE = True     # training must match: build_sequences.py --rack-normalize
RACK_ANGLE_EMA = 0.85           # Rack roll smoothing (0=raw, 1=frozen)
RACK_SCALE_EMA = 0.90           # Torso-scale smoothing

# ArUco rack marker (pipeline/rack_frame.py:estimate_rack_pose) — an
# alternative, more precise rotation source for the SAME Stage 1 transform
# above: a printed marker sheet on the rack corners gives a real solvePnP
# pose instead of inferring rack angle from the HSV main-box rect. Optional
# — RackFrameNormalizer still works with rack_rect (HSV) or no rack info at
# all (torso fallback) exactly as before when no marker is visible.
ARUCO_DICT_NAME = "DICT_4X4_50"   # 50 distinct 4x4-bit markers — plenty for 4 rack corners
ARUCO_MARKER_LENGTH_M = 0.05      # printed marker side length, meters (measure your printout)
ARUCO_MIN_MARKERS_FOR_POSE = 2    # solvePnP on coplanar points is pose-ambiguous with only 1
# Marker center positions in the RACK's own 3D frame (meters), coplanar
# (z=0) unless your rig mounts markers on different faces. PLACEHOLDER
# layout — replace with your physically measured positions before trusting
# the pose for anything beyond a smoke test (see estimate_rack_pose's
# docstring). ~30x20cm rectangle, markers at the 4 corners.
ARUCO_RACK_LAYOUT_M = {
    0: (-0.15, 0.10, 0.0),   # top-left
    1: (0.15, 0.10, 0.0),    # top-right
    2: (0.15, -0.10, 0.0),   # bottom-right
    3: (-0.15, -0.10, 0.0),  # bottom-left
}
# Uncalibrated default camera intrinsics (assumed ~65deg horizontal FOV
# pinhole). solvePnP's accuracy depends directly on real calibration —
# this is a usable-without-hardware default, NOT a substitute for running
# cv2.calibrateCamera on your actual camera before a real demo.
ARUCO_ASSUMED_FOV_DEG = 65.0

# Stage 0 (runs BEFORE pose estimation): canonicalise the IMAGE, not just the
# landmarks. MediaPipe is trained on upright people and does not degrade
# gracefully when the subject is rolled - it stops detecting entirely.
# Measured on dataset/blender (payload_a, 20 mid-protocol frames per take):
#
#     crew roll      pose detected        with UPRIGHT_POSE
#     0 deg          19/20  (95%)         19/20  (95%)
#     90 deg          7/20  (35%)         20/20 (100%)
#     135 deg         5/20  (25%)         20/20 (100%)
#     180 deg        12/20  (60%)         19/20  (95%)
#
# RACK_FRAME_NORMALIZE below cannot fix this on its own: it normalises the
# landmarks MediaPipe returns, and at 90 degrees there are none to normalise.
# pipeline/upright_pose.py rotates the frame upright (using the rack's own
# roll from the HSV main-box rect, with a coarse search as backstop), runs
# pose, and maps the landmarks back into original image coordinates - so the
# 132-dim contract and everything downstream are unchanged.
#
# Default True: it strictly dominates in the measurement above (no regression
# upright, large gain rolled), and orientation-agnostic operation is an
# explicit requirement of the problem statement. The working angle latches, so
# a stable orientation costs one extra inference and then nothing.
UPRIGHT_POSE = True
UPRIGHT_POSE_MIN_SCORE = 0.55   # mean visibility over torso+arms to accept an angle

# Stage 2: optional true 3D Human Mesh Recovery backend (GPU, SMPL-based).
# "none"/"mediapipe" = current 2D pose; "hmr2"/"wham" = attempt to load that
# package (pip install hmr2 / wham) and fall back to MediaPipe if unavailable.
HMR_BACKEND = "none"


# ── Streaming Settings ────────────────────────────────────────
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8554
ENABLE_STREAMING = False   # Push video over UDP to STREAM_HOST:STREAM_PORT (needs ffmpeg on PATH)
LOCAL_RECORDING_DIR = "recordings"

# ── Smart clip uplink (Brief §12 bandwidth thesis) ──────────────
# "stream"  — today's behavior unchanged: continuous full IP stream (SIH
#             requirement) + full local recording, no clip files.
# "clip"    — no continuous stream; only anomaly-triggered clips (pre-roll +
#             post-roll) are saved/would-be-uplinked. Full local recording
#             still runs (never lose the raw video for review).
# "both"    — full stream AND clips, for comparing actual vs. clip-only
#             bandwidth on the same run (what UplinkManager.bandwidth_summary
#             reports).
# Override with HAR_UPLINK_MODE or `main.py --uplink-mode`.
UPLINK_MODE = os.environ.get("HAR_UPLINK_MODE", "stream")
CLIP_PRE_ROLL_S = 3.0     # seconds of ring-buffered video kept before a trigger
CLIP_POST_ROLL_S = 3.0    # seconds captured after a trigger before the clip is flushed
CLIP_OUTPUT_DIR = "dataset/clips"

# ── Session metrics (Brief §13 — G5) ────────────────────────────
# time-to-detect / false-hold / abstain-quality report, written once per
# session at pipeline shutdown. See pipeline/session_metrics.py.
SESSION_METRICS_PATH = "logs/session_metrics.json"

# ── Model Paths ───────────────────────────────────────────────
MODEL_DIR = os.environ.get("HAR_MODEL_DIR", "models")  # override lets A/B experiments run without touching committed checkpoints
# Custom CNN activity classifier (trained from scratch)
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "activity_cnn.pt")
CNN_ONNX_PATH  = os.path.join(MODEL_DIR, "activity_cnn.onnx")
# LSTM over pose-net skeletons (trained from scratch)
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_classifier.pt")
LSTM_ONNX_PATH = os.path.join(MODEL_DIR, "lstm_classifier.onnx")  # ONNX for CPU inference
# Custom pose heatmap net (trained from scratch, replaces MediaPipe — see above)
POSENET_PATH = os.path.join(MODEL_DIR, "pose_net.pt")
POSENET_ONNX_PATH = os.path.join(MODEL_DIR, "pose_net.onnx")
# Optional YOLO detector (off by default — see DETECTOR_BACKEND below). Fine-tuned
# from a pretrained COCO checkpoint on our own synthetic box/hand bboxes — COCO
# itself has no red_box/yellow_box/hand classes, so the pretrained weights alone
# do nothing useful here; they're just a feature-extractor starting point.
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo_detector.pt")

# ── Real "gravitational mimic" video data ──────────────────────
# Real footage (Earth-recorded, deliberately slow/floaty motion miming
# microgravity handling) used as domain-adaptation / fine-tune data on top of
# the synthetic renderer data — see data_generation/real_video_autolabel.py
# and data_generation/build_real_dataset.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_VIDEO_DIR = _REPO_ROOT / "new video data"
REAL_PSEUDO_DIR = "dataset/real_pseudo"                    # per-frame pseudo pose/labels
REAL_SEQUENCES_DIR = "dataset/real_sequences"              # windowed (X,y,groups) from real videos
REAL_ANNOTATED_DIR = "dataset/annotated_real"              # real frames, step_XX/ layout for the CNN
COMBINED_SEQUENCES_DIR = "dataset/skeleton_sequences_combined"  # synthetic + real, for final LSTM train

# ── LSTM / Sequence Settings ──────────────────────────────────
SEQUENCE_WINDOW = 30        # Samples per LSTM input window
# The model sees samples on a fixed clock of period SAMPLE_EVERY / FPS
# (100 ms), so the 30-sample window spans 3.0 s instead of 1.0 s: a protocol
# step takes 4-6 s and a 1 s glimpse cannot tell arriving from leaving. The
# live loop runs ~15-22 fps, never 30, so frames are timestamped and
# resampled onto that grid (pipeline/features.SampleWindow); on rendered
# frames at exactly i/30 s this is identical to keeping every 3rd frame.
# HSV, pose and the rack EMA still run on every frame
# (PROCESS_EVERY_N_FRAMES stays 1).
SAMPLE_EVERY = 3
# A gap longer than this between consecutive frames clears the window and
# warms up again, rather than inventing motion across it.
MAX_FRAME_GAP_S = 0.5
SKELETON_FEATURES = 33 * 4  # MediaPipe: 33 landmarks × (x, y, z, visibility) — the POSE contract
# Object channels appended after the pose: 3 boxes (red, yellow, main) x
# (cx, cy, visible) + left/right wrist distance to the nearest visible box.
OBJECT_FEATURES = 3 * 3 + 2
MODEL_FEATURE_DIM = SKELETON_FEATURES + OBJECT_FEATURES   # the MODEL input contract (143)
# The layout the live pipeline builds and expects its model to be trained on.
# Names and dims live in pipeline/features.py ("pose132" is the legacy
# pose-only layout of every checkpoint written before layouts were recorded).
FEATURE_LAYOUT = "pose132+obj11"
# Fraction of a window that must share its majority step label to be used for
# training or scoring. One value for both, or they disagree about what a
# window is (build_sequences used 0.75 while evaluate_real used 0.6).
WINDOW_PURITY = 0.75
NUM_STEPS = len(EXPERIMENT_STEPS) + 1  # +1 for "idle/unknown"

# ── State Machine Thresholds ──────────────────────────────────
# Counted in SAMPLES, not camera frames: the pipeline feeds the state machine
# once per sample on the 100 ms clock. 5 samples = 0.5 s, the same wall-clock
# debounce the old 15 frames @ 30 fps gave. Worst-case deviation-alert
# latency: 3.0 s window turnover + 0.5 s debounce + up to one camera frame
# before the grid sample is emitted (~67 ms at 15 fps) + that frame's
# processing (~66 ms) = ~3.6 s (was ~1.5 s with a 1 s window).
STEP_CONFIRM_FRAMES = 5     # Consecutive samples needed to confirm a step
STEP_CONFIDENCE_THRESHOLD = 0.65

# ── CNN Ensemble (optional, off by default) ────────────────────
# The CNN (frame-level classifier) is trained but was previously never used
# for inference — pure dead weight. When enabled, its prediction is fused
# with the LSTM's (see pipeline/har_pipeline.py's _fuse_predictions): if both
# are confident but DISAGREE, the pipeline does not guess — it fires an
# "uncertain, please confirm" alert instead, per the PS's own stated
# principle (ask rather than silently pass or fail).
# Default False: the shipped CNN checkpoint was trained before its
# train/val-split leakage fix (see train/train_cnn.py) — retrain it clean
# before trusting it in production, same off-until-retrained convention as
# RACK_FRAME_NORMALIZE.
CNN_ENSEMBLE_ENABLED = False
CNN_CONFIDENCE_THRESHOLD = 0.60

# ── Object Detector Backend (optional YOLO, off by default) ────
# hsv_detector.py's own docstring already concludes HSV is "actually superior
# to a trained detector for this use-case because the colors ARE the
# definition of the targets" — measured 1.000/1.000 precision/recall (see
# README). YOLO exists here as an opt-in alternative to measure against, not
# a default replacement. Same "free, exact, renderer-driven ground truth"
# trick as HARPoseNet — COCO has no red_box/yellow_box/hand classes, so the
# pretrained checkpoint is only ever a fine-tune starting point, never used
# as-is at inference time.
DETECTOR_BACKEND = "hsv"             # "hsv" (default) or "yolo"
YOLO_BASE_CHECKPOINT = "yolo11n.pt"  # pretrained COCO nano checkpoint — smallest
                                     # variant available, chosen for CPU latency
                                     # (not the yolov9c requested — see train_yolo_detector.py)
YOLO_IMG_SIZE = 384                  # detector input size — CPU latency budget, not accuracy
YOLO_SYNTH_SAMPLES = 3000
YOLO_VAL_SAMPLES = 400
YOLO_EPOCHS = 30
YOLO_BATCH_SIZE = 16
YOLO_CONF_THRESHOLD = 0.25

# ── HSV Color Ranges for Box Detection (no training needed) ───
# Format: (H_min, S_min, V_min), (H_max, S_max, V_max)
HSV_RED_LOWER1  = (0,   120, 70)    # Red wraps around hue wheel
HSV_RED_UPPER1  = (10,  255, 255)
HSV_RED_LOWER2  = (170, 120, 70)
HSV_RED_UPPER2  = (180, 255, 255)
HSV_YELLOW_LOWER = (20, 100, 100)
HSV_YELLOW_UPPER = (35, 255, 255)
HSV_WHITE_LOWER  = (0,   0,  200)   # Main box (white container)
HSV_WHITE_UPPER  = (180, 30, 255)
MIN_BOX_AREA_PX  = 800              # Minimum contour area to consider a detection
# Hand detection (fills the "hand" DETECTION_CLASSES slot, previously never
# actually produced by HSVBoxDetector — see _detect_hand). Broad skin-tone
# band intersected with motion (frame-differencing), not a learned model;
# calibrate_from_roi("hand", ...) narrows this for a specific skin tone/light.
HSV_SKIN_LOWER = (0, 25, 60)
HSV_SKIN_UPPER = (25, 150, 255)
MIN_HAND_AREA_PX = 500              # Hands can be smaller/farther than boxes
HAND_MOTION_THRESHOLD = 18          # Frame-diff gray-level threshold for "moving"
HAND_TOP_EXCLUDE_FRAC = 0.15        # Exclude top of frame (head/helmet skin tone)

# ── Voice Alert Settings ──────────────────────────────────────
VOICE_ENABLED = True
VOICE_MODEL = "en_US-lessac-medium"  # Piper TTS model name
PIPER_BINARY = "piper"               # Assumes piper is in PATH

# ── Log Settings ──────────────────────────────────────────────
LOG_DIR = "logs"
LOG_FILENAME_FORMAT = "experiment_log_%Y%m%d_%H%M%S.txt"

# ── Gemini Synthetic Data Generation ─────────────────────────
GEMINI_MODEL = "veo-3.1-fast-generate-preview"  # Free tier available model
SYNTHETIC_OUTPUT_DIR = "dataset/synthetic_gemini"
FRAMES_EXTRACT_PER_VIDEO = 100  # Frames to extract per generated clip

# ── Augmentation Settings ─────────────────────────────────────
AUG_OUTPUT_DIR = "dataset/augmented"
AUGMENTATION_FACTOR = 8     # Multiply dataset 8x via augmentation

# ── Training Settings — RTX 3050 6GB Optimized ────────────────
# Custom CNN (from scratch)
CNN_EPOCHS         = 60
CNN_BATCH_SIZE     = 32       # Fits comfortably in 6GB VRAM with fp16
CNN_IMG_SIZE       = 224      # Input resolution for custom CNN
CNN_LEARNING_RATE  = 1e-3
CNN_WEIGHT_DECAY   = 1e-4
CNN_DROPOUT        = 0.4
CNN_NUM_FRAMES_IN  = 8        # Temporal stack: 8 consecutive frames per sample

# LSTM over skeletons (from scratch)
LSTM_EPOCHS        = 80
LSTM_BATCH_SIZE    = 64       # RTX 3050 can handle 64 easily for LSTM
LSTM_LEARNING_RATE = 1e-3
LSTM_HIDDEN_SIZE   = 256      # Larger hidden size — more VRAM available
LSTM_NUM_LAYERS    = 3
LSTM_DROPOUT       = 0.5
TRAIN_VAL_SPLIT    = 0.8
USE_AMP            = True     # Mixed precision (fp16) — halves VRAM, 2x faster

# ── End-to-end loop gates (space-sim must meet these) ─────────
# E2E_LSTM_MIN_VAL_ACC and E2E_MIN_ORACLE_STEP_ACC are calibrated for
# end_to_end_loop.py's OWN internal LSTM (trained on clean ground-truth pose
# vectors, by design — see that file's module docstring). They will correctly
# read as failing against `main.py --mode train`'s production LSTM, which is
# deliberately trained on noisier, realistic PoseNet features instead — for
# that model, clean ground-truth input (what the oracle check feeds it) is
# itself out-of-distribution. See README.md's "Current results" section; the
# metric that actually matters for the production model is the real (non-
# oracle) camera->PoseNet->LSTM accuracy reported there, not these two gates.
E2E_LSTM_MIN_VAL_ACC = 0.90
E2E_LSTM_MIN_TEST_ACC = 0.88
E2E_HSV_MIN_RECALL = 0.85
E2E_MAX_MEAN_LATENCY_MS = 80.0
E2E_MAX_P95_LATENCY_MS = 130.0
E2E_MIN_ORACLE_STEP_ACC = 0.85
E2E_MIN_POSENET_PCK = 0.55    # PCK@0.10*torso on held-out synthetic renders. Set as a
# regression floor below the actually-achieved value (Stage 1: 0.665; +Stage 2 real
# fine-tune: 0.649 — a ~2% synthetic-PCK trade accepted in exchange for real-video wrist
# adaptation, see train/train_posenet.py's regression guard), not as an untested
# aspirational target. Per-joint PCK is uneven: nose/hips/knees/ankles score 0.70-0.99
# (they barely move — an easy target), while elbows/wrists — the joints that actually
# drive step classification, see data_generation/synthetic_pose.py's _STEP_WAYPOINTS —
# score only 0.22-0.40 at this heatmap resolution/training budget. The aggregate PCK
# above is a real, disclosed, CPU-training-budget limitation, not a bug — see README.md.

# ── Tuned runtime overrides ────────────────────────────────────
# Written by end_to_end_loop.py's auto-fix loop when a gate failure implies a
# different runtime threshold (e.g. STEP_CONFIRM_FRAMES) would pass. Applied
# last, here, so it can override any constant defined above.
#
# Previously that loop only monkeypatched pipeline.state_machine's already-
# imported module attribute — correct for the rest of that one process (the
# state machine re-reads its module global on every ExperimentStateMachine()
# call), but it evaporated the instant the process exited: a "passed" e2e
# run's config was not the config that actually ships. Loading a small file
# here makes any future process (including a real main.py --mode pipeline
# run) see the same tuned value.
_TUNED_OVERRIDES_PATH = Path(__file__).parent / "tuned_overrides.json"
if _TUNED_OVERRIDES_PATH.exists():
    try:
        globals().update(json.loads(_TUNED_OVERRIDES_PATH.read_text(encoding="utf-8")))
    except Exception as _e:
        logging.getLogger(__name__).warning("Failed to load tuned_overrides.json: %s", _e)
