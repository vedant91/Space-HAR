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
EXPERIMENT_STEPS = [
    {
        "id": 1,
        "name": "Approach Main Box",
        "description": "Astronaut approaches and positions hands near the main container",
        "duration_hint_sec": 5,
        "required_objects": ["main_box"],
        "voice_cue": "Step 1: Approach the main box.",
    },
    {
        "id": 2,
        "name": "Open Main Box",
        "description": "Astronaut opens the lid of the main container",
        "duration_hint_sec": 4,
        "required_objects": ["main_box"],
        "voice_cue": "Step 2: Open the main box.",
    },
    {
        "id": 3,
        "name": "Pick Red Box",
        "description": "Astronaut picks up the red inner box with both hands",
        "duration_hint_sec": 5,
        "required_objects": ["red_box", "hand"],
        "voice_cue": "Step 3: Pick up the red box.",
    },
    {
        "id": 4,
        "name": "Examine Red Box",
        "description": "Astronaut examines and verifies the red box",
        "duration_hint_sec": 6,
        "required_objects": ["red_box", "hand"],
        "voice_cue": "Step 4: Examine the red box.",
    },
    {
        "id": 5,
        "name": "Place Red Box",
        "description": "Astronaut places the red box in the designated zone",
        "duration_hint_sec": 4,
        "required_objects": ["red_box"],
        "voice_cue": "Step 5: Place the red box in the designated zone.",
    },
    {
        "id": 6,
        "name": "Pick Yellow Box",
        "description": "Astronaut picks up the yellow inner box",
        "duration_hint_sec": 5,
        "required_objects": ["yellow_box", "hand"],
        "voice_cue": "Step 6: Pick up the yellow box.",
    },
    {
        "id": 7,
        "name": "Examine Yellow Box",
        "description": "Astronaut examines and verifies the yellow box",
        "duration_hint_sec": 6,
        "required_objects": ["yellow_box", "hand"],
        "voice_cue": "Step 7: Examine the yellow box.",
    },
    {
        "id": 8,
        "name": "Place Yellow Box",
        "description": "Astronaut places the yellow box in the designated zone",
        "duration_hint_sec": 4,
        "required_objects": ["yellow_box"],
        "voice_cue": "Step 8: Place the yellow box in the designated zone.",
    },
]
# ── YOLO Detection Classes ────────────────────────────────────
DETECTION_CLASSES = {
    0: "hand",
    1: "red_box",
    2: "yellow_box",
    3: "main_box",
}
CLASS_NAMES = list(DETECTION_CLASSES.values())
# Named DETECTION_NUM_CLASSES (not NUM_CLASSES) — this is the HSV detection
# class count (4: hand/red_box/yellow_box/main_box), unrelated to the step
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
USE_THREADED_INFERENCE = True    # Parallelize HSV + MediaPipe in threads
LSTM_USE_ONNX = True            # Use ONNX Runtime for LSTM (faster than PyTorch)

# ── Orientation-Agnostic Pose (microgravity: no fixed 'up') ──────────────────
# Stage 1: re-express 2D pose in a payload-rack reference frame instead of the
# image/floor frame. Roll comes from the HSV main_box rect (fallback: torso),
# origin is the torso center, scale is torso length. Output stays 132-dim, so
# the LSTM contract is unchanged — retrain with the same flag for best results.
RACK_FRAME_NORMALIZE = True     # training must match: build_sequences.py --rack-normalize
RACK_ANGLE_EMA = 0.85           # Rack roll smoothing (0=raw, 1=frozen)
RACK_SCALE_EMA = 0.90           # Torso-scale smoothing
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

# Box detector: "hsv" = colour-threshold (pipeline/hsv_detector.py, zero
# training, ~0.12-0.15 recall on yellow); "yolo" = trained detector
# (pipeline/yolo_detector.py, models/yolo_boxes.pt from
# train/train_yolo_boxes.py). Same Detection interface either way - every
# caller constructs whichever this says and nothing else changes.
DETECTOR_BACKEND = "yolo"


# ── Streaming Settings ────────────────────────────────────────
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8554
ENABLE_STREAMING = False   # Push video over UDP to STREAM_HOST:STREAM_PORT (needs ffmpeg on PATH)
LOCAL_RECORDING_DIR = "recordings"

# ── Model Paths ───────────────────────────────────────────────
MODEL_DIR = "models"
# Custom CNN activity classifier (trained from scratch)
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "activity_cnn.pt")
CNN_ONNX_PATH  = os.path.join(MODEL_DIR, "activity_cnn.onnx")
# LSTM over MediaPipe skeletons (trained from scratch)
# lstm_classifier.pt is the stale pre-object-features checkpoint (132-dim,
# val_acc=1.0 from a leaky split - this is the "shipped model is at chance"
# one). lstm_blender_core.pt is the honest by-take-split retrain on the
# current 143-dim pose+object contract (val_acc=0.73).
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_blender_core.pt")
LSTM_ONNX_PATH = os.path.join(MODEL_DIR, "lstm_classifier.onnx")  # ONNX for CPU inference
# DETECTOR_BACKEND below picks HSV vs the trained YOLOv8 box detector.

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

# Shannon entropy (nats) of the LSTM's full softmax, gating feed_prediction()
# alongside the scalar confidence above. A top-1 confidence can clear 0.65
# while the REST of the distribution is still spread across several other
# classes (e.g. 0.65 on the true step, remaining 0.35 spread over the other
# 8 classes) rather than concentrated on one clear runner-up - that frame is
# noise wearing a confident number, not a real signal, and the confidence
# check alone cannot tell the two apart. Entropy can: worked example at
# NUM_STEPS=9, top1=0.65 -> spread over 1 other class: H=0.65 nats (a
# decisive two-way call, should pass); spread uniformly over the other 8:
# H=1.38 nats (should not). Max possible is ln(9)=2.197. 1.4 sits just above
# that second case, so it rejects "confident but shapeless" without also
# rejecting ordinary two-way confusion. Optional at the call site
# (pipeline/state_machine.py feed_prediction's `entropy` kwarg defaults to
# None, which skips this gate) so callers that only have a scalar confidence
# - tools/evaluate_real.py, simulation/space_sim.py, tools/showcase_overlay.py
# - are unaffected.
STEP_ENTROPY_MAX = 1.4

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

# ── Voice Alert Settings ──────────────────────────────────────
VOICE_ENABLED = True
VOICE_REMINDER_INTERVAL_SEC = 20.0
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
E2E_LSTM_MIN_VAL_ACC = 0.90
E2E_LSTM_MIN_TEST_ACC = 0.88
E2E_HSV_MIN_RECALL = 0.85
E2E_MAX_MEAN_LATENCY_MS = 80.0
E2E_MAX_P95_LATENCY_MS = 130.0
E2E_MIN_ORACLE_STEP_ACC = 0.85

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
