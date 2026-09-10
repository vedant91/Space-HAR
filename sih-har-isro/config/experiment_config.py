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
POSE_INPUT_SIZE = 192       # model input, square RGB
POSE_HEATMAP_SIZE = 48      # output heatmap resolution (stride 4)
POSE_HEATMAP_SIGMA = 1.5    # gaussian target sigma, in heatmap-pixel units
POSE_MIN_CONFIDENCE = 0.15  # heatmap peak below this -> visibility 0 (joint not claimed)
# Sub-pixel decode: a plain argmax on a 48x48 heatmap quantizes to ~1 heatmap
# pixel, which alone can burn most of the PCK@0.10*torso error budget (torso
# is only ~11-12 heatmap-px wide at this resolution/frame). Decode instead
# takes a local softmax-weighted centroid in a (2*radius+1) window around the
# argmax — cheap (13 tiny windows/frame) and measured ~40% relative PCK
# improvement over plain argmax on the same checkpoint, no retraining needed.
POSE_DECODE_RADIUS = 2
POSE_DECODE_BETA = 20.0

POSENET_EPOCHS = 24
POSENET_BATCH_SIZE = 32
POSENET_LR = 1e-3
POSENET_SYNTH_SAMPLES = 4000     # rendered synthetic frames for training
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
RACK_FRAME_NORMALIZE = False    # Enable after retraining data with --rack-normalize
RACK_ANGLE_EMA = 0.85           # Rack roll smoothing (0=raw, 1=frozen)
RACK_SCALE_EMA = 0.90           # Torso-scale smoothing
# Stage 2 (retired): a true 3D Human Mesh Recovery backend (SMPL-based) would
# itself be a third-party pretrained model (HMR 2.0 / WHAM), which conflicts
# with this project's "no open-source/pretrained model" requirement for
# movement detection. pipeline/hmr_backend.py is kept only as an inert shim
# (always reports unavailable) so old configs/imports referencing it don't
# break. The custom PoseNet's own z-head is this project's depth signal.
HMR_BACKEND = "none"


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
# LSTM over pose-net skeletons (trained from scratch)
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_classifier.pt")
LSTM_ONNX_PATH = os.path.join(MODEL_DIR, "lstm_classifier.onnx")  # ONNX for CPU inference
# Custom pose heatmap net (trained from scratch, replaces MediaPipe — see above)
POSENET_PATH = os.path.join(MODEL_DIR, "pose_net.pt")
POSENET_ONNX_PATH = os.path.join(MODEL_DIR, "pose_net.onnx")
# No YOLOv8 — object detection uses HSV color segmentation (no training needed)

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
SEQUENCE_WINDOW = 30        # Frames for LSTM input (1 sec @ 30fps)
SKELETON_FEATURES = 33 * 4  # MediaPipe: 33 landmarks × (x, y, z, visibility)
NUM_STEPS = len(EXPERIMENT_STEPS) + 1  # +1 for "idle/unknown"

# ── State Machine Thresholds ──────────────────────────────────
STEP_CONFIRM_FRAMES = 15    # Consecutive frames needed to confirm a step
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
