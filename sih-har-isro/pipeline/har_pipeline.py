"""
Main HAR Pipeline — Optimized Inference Engine (Tier 1)
========================================================
Integrates all components into a single real-time processing loop:
  HSVDetector + MediaPipe Pose + LSTM + StateMachine + VoiceAlert + Logger

Latency optimizations applied:
  1. MediaPipe model_complexity=0 (fastest mode)
  2. Frame downscaling for MediaPipe & HSV (640×360)
  3. Threaded parallel execution (HSV + MediaPipe run simultaneously)
  4. Pre-allocated numpy buffers (no hot-loop allocations)
  5. LSTM ONNX Runtime inference (faster than PyTorch)

Target: <15ms per frame processing (60+ FPS)

Usage:
    python pipeline/har_pipeline.py --camera 0
    python pipeline/har_pipeline.py --video path/to/video.mp4
"""

import sys
import cv2
import time
import queue
import threading
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Deque

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    PROCESS_EVERY_N_FRAMES, SEQUENCE_WINDOW,
    LSTM_PATH, CNN_MODEL_PATH, CNN_ONNX_PATH, LSTM_ONNX_PATH, LSTM_USE_ONNX,
    VOICE_ENABLED, STREAM_HOST, STREAM_PORT, ENABLE_STREAMING, LOCAL_RECORDING_DIR,
    STEP_CONFIDENCE_THRESHOLD, SKELETON_FEATURES,
    MEDIAPIPE_MODEL_COMPLEXITY, MEDIAPIPE_MIN_DET_CONF, MEDIAPIPE_MIN_TRK_CONF,
    POSE_INPUT_WIDTH, HSV_DOWNSCALE, USE_THREADED_INFERENCE,
    RACK_FRAME_NORMALIZE, RACK_ANGLE_EMA, RACK_SCALE_EMA, HMR_BACKEND,
    CNN_ENSEMBLE_ENABLED, CNN_CONFIDENCE_THRESHOLD, CNN_NUM_FRAMES_IN, CNN_IMG_SIZE,
    UPRIGHT_POSE, UPRIGHT_POSE_MIN_SCORE, SAMPLE_EVERY, FEATURE_LAYOUT,
)
from pipeline.rack_frame import RackFrameNormalizer
from pipeline.features import (
    SampleWindow, LAYOUT_DIMS, build_frame_features, contract_of,
)
from pipeline.state_machine import ExperimentStateMachine, StepRecord
from pipeline.voice_alert import VoiceAlertSystem
from pipeline.logger import ExperimentLogger
from pipeline.hsv_detector import HSVBoxDetector, Detection
from pipeline.stream_sender import NetworkStreamer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Import optional dependencies ──────────────────────────────────────────────
# Pose estimation goes through pipeline/pose_backend.py rather than touching
# mediapipe directly. `mp.solutions.pose` — what this file used to call — was
# removed in MediaPipe 1.0, so the direct import made the whole pipeline
# unimportable on any current install. The backend picks the Tasks API when
# available and falls back to the legacy Solutions API on older versions.
from pipeline.pose_backend import PoseBackend, describe as describe_pose_backend

_POSE_INFO = describe_pose_backend()
MP_AVAILABLE = bool(_POSE_INFO.get("usable"))
if not MP_AVAILABLE:
    logger.warning("No usable MediaPipe pose backend (%s). Skeleton features disabled.",
                   _POSE_INFO)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed. Using PyTorch for CNN inference.")


# Feature dimension: MediaPipe Pose only (faster than Holistic)
# Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
# Hands removed for speed — pose alone captures the key motion
# Single source of truth is config.SKELETON_FEATURES.
POSE_FEATURE_DIM = SKELETON_FEATURES


class ThreadedInference:
    """
    Runs MediaPipe and HSV detection in parallel threads.
    The main thread waits for both to complete before combining results.
    """

    def __init__(self):
        self._mp_result = None
        self._hsv_result = None

    def run_parallel(self, frame_bgr: np.ndarray, frame_rgb_full: np.ndarray,
                     hsv_detector, mp_wrapper, pose_hint: Optional[list] = None):
        """
        Run HSV detection (this frame) and MediaPipe Pose (this frame, hinted
        with `pose_hint`, the PREVIOUS frame's detections) in parallel.
        Returns: (detections, skeleton_features)
        """
        self._mp_result = None
        self._hsv_result = None

        def _run_hsv():
            try:
                self._hsv_result = hsv_detector.detect(frame_bgr)
            except Exception as e:
                logger.warning("HSV thread error: %s", e)
                self._hsv_result = []

        def _run_mediapipe():
            try:
                if mp_wrapper is not None:
                    self._mp_result = mp_wrapper.process(frame_rgb_full, detections=pose_hint)
                else:
                    self._mp_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
            except Exception as e:
                logger.warning("MediaPipe thread error: %s", e)
                self._mp_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        t1 = threading.Thread(target=_run_hsv, daemon=True)
        t2 = threading.Thread(target=_run_mediapipe, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Fallback if either thread failed
        if self._hsv_result is None:
            self._hsv_result = []
        if self._mp_result is None:
            self._mp_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        return self._hsv_result, self._mp_result


def to_frame_size(frame_bgr: np.ndarray) -> np.ndarray:
    """Every source - camera, video file, rendered PNG - is brought to
    FRAME_WIDTH x FRAME_HEIGHT before perception, live and offline alike: the
    HSV detector's area thresholds are in pixels, so a 900-px render and a
    1280-px camera frame must reach it at the same size."""
    if frame_bgr.shape[1] != FRAME_WIDTH or frame_bgr.shape[0] != FRAME_HEIGHT:
        return cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))
    return frame_bgr


def perceive(frame_bgr: np.ndarray, frame_rgb: np.ndarray, hsv_detector, mp_wrapper,
             prev_detections: Optional[list], threaded: Optional[ThreadedInference] = None,
             timings: Optional[dict] = None):
    """One frame -> (this frame's HSV detections, raw 132-dim pose).

    The ONE perception step for the live loop (run() and process_frame()),
    tools/build_sequences.py and tools/evaluate_real.py. Pose is hinted with
    the PREVIOUS frame's detections (None on a take's first frame, so every
    path runs the upright coarse angle search once): that is what lets HSV and
    pose run in parallel live, and because training and scoring do the same,
    threading changes latency but never the features. `threaded` only changes
    how the work is scheduled, not its result. `timings` (sequential only)
    receives hsv_ms / pose_ms.
    """
    if threaded is not None:
        return threaded.run_parallel(frame_bgr, frame_rgb, hsv_detector, mp_wrapper,
                                     pose_hint=prev_detections)
    t0 = time.perf_counter()
    dets = hsv_detector.detect(frame_bgr)
    t1 = time.perf_counter()
    pose = (mp_wrapper.process(frame_rgb, detections=prev_detections)
            if mp_wrapper is not None else np.zeros(POSE_FEATURE_DIM, dtype=np.float32))
    if timings is not None:
        timings["hsv_ms"] = (t1 - t0) * 1000.0
        timings["pose_ms"] = (time.perf_counter() - t1) * 1000.0
    return dets, pose


class OptimizedMPWrapper:
    """
    The pose stage: training (tools/build_sequences.py), scoring
    (tools/evaluate_real.py) and the live pipeline all construct this with
    config defaults, so pose runs identically everywhere - same complexity,
    same confidences, same upright search, same input size.

    The input is resized to POSE_INPUT_WIDTH pixels wide (height by the
    frame's aspect) BEFORE the pose estimator sees it, so a 900-px render and
    a 1280-px camera frame reach MediaPipe at the same pixel size. Landmarks
    are image-normalised, so this does not change their meaning.

    Kept as a named class (rather than using PoseBackend directly) because
    the underlying MediaPipe API has changed once already. `process()`
    returns the 132-dim feature vector directly.
    """

    def __init__(self, complexity: int = MEDIAPIPE_MODEL_COMPLEXITY,
                 min_det_conf: float = MEDIAPIPE_MIN_DET_CONF,
                 min_trk_conf: float = MEDIAPIPE_MIN_TRK_CONF,
                 frame_wh: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
                 static_image_mode: bool = False,
                 upright: Optional[bool] = None):
        w, h = frame_wh
        self.small_w = int(POSE_INPUT_WIDTH)
        self.small_h = max(int(round(h * POSE_INPUT_WIDTH / float(w))), 16)
        self.upright = UPRIGHT_POSE if upright is None else upright

        if self.upright:
            # Canonicalise the frame before pose. See config.UPRIGHT_POSE for
            # the measurement that justifies this being on by default.
            from pipeline.upright_pose import UprightPoseEstimator
            self._backend = UprightPoseEstimator(
                complexity=complexity,
                min_det_conf=min_det_conf,
                min_trk_conf=min_trk_conf,
                downscale=1,
                frame_width=self.small_w,
                frame_height=self.small_h,
            )
        else:
            self._backend = PoseBackend(
                complexity=complexity,
                min_det_conf=min_det_conf,
                min_trk_conf=min_trk_conf,
                downscale=1,
                frame_width=self.small_w,
                frame_height=self.small_h,
                static_image_mode=static_image_mode,
            )
        if self._backend.available:
            logger.info(
                "Pose backend '%s' ready (upright=%s, complexity=%d, det=%.1f, "
                "trk=%.1f, input=%dx%d)", self._backend.backend, self.upright,
                complexity, min_det_conf, min_trk_conf, self.small_w, self.small_h)

    @property
    def available(self) -> bool:
        return self._backend.available

    @property
    def backend_name(self) -> str:
        return self._backend.backend

    def process(self, frame_rgb_full: np.ndarray,
                detections: Optional[list] = None) -> np.ndarray:
        """RGB frame (any size) -> (132,) float32 features.

        `detections` is the upright estimator's rack-roll hint. Every caller
        goes through perceive(), which passes the PREVIOUS frame's HSV output
        here (None on the first frame: the coarse angle search then does the
        work and the result latches).
        """
        if frame_rgb_full.shape[1] != self.small_w or frame_rgb_full.shape[0] != self.small_h:
            frame_rgb_full = cv2.resize(frame_rgb_full, (self.small_w, self.small_h),
                                        interpolation=cv2.INTER_LINEAR)
        if self.upright:
            return self._backend.process(
                frame_rgb_full, detections=detections,
                min_score=UPRIGHT_POSE_MIN_SCORE)
        return self._backend.process(frame_rgb_full)

    def reset(self):
        if hasattr(self._backend, "reset"):
            self._backend.reset()

    def close(self):
        self._backend.close()


class HARPipeline:
    """
    Optimized real-time HAR pipeline for ISRO experiment monitoring.

    Processing per frame:
      1. HSV color detection  → box visibility flags
      2. MediaPipe Pose       → skeleton feature vector (132-dim, pose only)
      3. Append to LSTM buffer → classify step when buffer full
      4. Feed prediction to State Machine
      5. State Machine fires callbacks → Voice + Log + GUI

    Optimizations:
      - MediaPipe Pose (not Holistic) — 2-3× faster
      - model_complexity=0 — fastest mode
      - Frame downscaling for MediaPipe — 4× fewer pixels
      - Threaded parallel execution — HSV + MediaPipe run simultaneously
      - Pre-allocated numpy buffers — zero hot-loop allocations
      - ONNX LSTM inference — faster than PyTorch
    """

    def __init__(self,
                 source: int | str = 0,
                 enable_streaming: Optional[bool] = None,
                 stream_host: Optional[str] = None,
                 stream_port: Optional[int] = None,
                 gui_queue: Optional[queue.Queue] = None,
                 headless: bool = False,
                 enable_voice: Optional[bool] = None,
                 enable_recording: bool = True,
                 enable_cnn_ensemble: Optional[bool] = None,
                 use_threaded: Optional[bool] = None,
                 lstm_path: Optional[str] = None):
        """`lstm_path`: serve this .pt checkpoint (PyTorch) instead of the
        promoted models/lstm_classifier.{onnx,pt}. Either way the pipeline
        builds exactly the inputs the loaded model records it was trained on
        (layout, sample rate, rack normalisation), names it on the HUD banner
        and in a WARNING, and refuses to start only when a model's input dim
        contradicts its own recorded layout."""

        self.source         = source
        self.gui_queue      = gui_queue
        self.headless       = headless
        self.enable_recording = enable_recording and not headless
        self.enable_streaming = ENABLE_STREAMING if enable_streaming is None else enable_streaming
        self.stream_host    = stream_host or STREAM_HOST
        self.stream_port    = stream_port or STREAM_PORT
        self.streamer: Optional[NetworkStreamer] = None
        self.enable_cnn_ensemble = (CNN_ENSEMBLE_ENABLED if enable_cnn_ensemble is None
                                    else enable_cnn_ensemble)
        self.lstm_path      = lstm_path
        # The model input contract. Config until _load_models() adopts the
        # loaded checkpoint's recorded contract.
        self.feature_layout = FEATURE_LAYOUT
        self.sample_every   = SAMPLE_EVERY
        self.model_banner   = ""
        self.frame_idx      = 0
        self._prev_dets     = None   # previous frame's HSV output: the pose hint
        self._file_fps      = None   # set by run() for file sources
        self._running       = False
        self._last_pred     = (0, 0.0)

        voice_on = VOICE_ENABLED if enable_voice is None else enable_voice
        if headless and enable_voice is None:
            voice_on = False

        # ── Sub-systems ───────────────────────────────────────
        self.state_machine  = ExperimentStateMachine()
        self.voice          = VoiceAlertSystem(enabled=voice_on)
        self.exp_logger     = ExperimentLogger()
        self.hsv_detector   = HSVBoxDetector(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )

        # ── Orientation-agnostic pose (no fixed 'up' in microgravity) ──
        # Stage 1: rack-anchored reference frame (CPU, always available).
        self.rack_normalizer = None
        if RACK_FRAME_NORMALIZE:
            self.rack_normalizer = RackFrameNormalizer(
                angle_ema=RACK_ANGLE_EMA,
                scale_ema=RACK_SCALE_EMA,
            )
            logger.info("Rack-frame normalization ENABLED (pose is rack-relative).")
        # Stage 2: optional 3D HMR backend; silently falls back to MediaPipe.
        self.hmr = None
        if HMR_BACKEND not in ("none", "mediapipe", ""):
            from pipeline.hmr_backend import HMRBackend
            self.hmr = HMRBackend(backend=HMR_BACKEND)

        # ── Models ────────────────────────────────────────────
        self.mp_wrapper     = None
        self.lstm_model     = None
        self.lstm_ort_sess  = None  # ONNX Runtime session for LSTM
        self.cnn_session    = None  # ONNX Runtime session for CNN
        self._load_models()

        # ── Threaded inference ─────────────────────────────────
        threaded_flag = USE_THREADED_INFERENCE if use_threaded is None else use_threaded
        if headless and use_threaded is None:
            threaded_flag = False
        self.threaded = ThreadedInference() if threaded_flag else None

        # ── Sequence buffer (for LSTM) ─────────────────────────
        # Resampling window (pipeline/features.py): timestamped frames in,
        # samples out on a fixed sample_every/FPS clock. Built after
        # _load_models() so it uses the loaded model's contract.
        self.skeleton_buffer = SampleWindow(SEQUENCE_WINDOW, self.sample_every, FPS)
        # ── Frame-stack buffer (for the optional CNN ensemble signal) ──
        # Only populated when enable_cnn_ensemble — zero cost otherwise.
        self.frame_stack_buffer: Deque[np.ndarray] = deque(maxlen=CNN_NUM_FRAMES_IN)

        # ── Pre-allocated buffers for zero-copy ─────────────────
        self._frame_rgb_full = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self._frame_hsv_small = np.zeros(
            (FRAME_HEIGHT // HSV_DOWNSCALE, FRAME_WIDTH // HSV_DOWNSCALE, 3), dtype=np.uint8
        )

        # ── Bind state machine callbacks ───────────────────────
        self._bind_callbacks()

        # ── Video writer ───────────────────────────────────────
        self.video_writer: Optional[cv2.VideoWriter] = None
        Path(LOCAL_RECORDING_DIR).mkdir(parents=True, exist_ok=True)

        # ── Performance tracking ───────────────────────────────
        self._times = deque(maxlen=60)

        logger.info("HARPipeline ready (headless=%s, voice=%s).", headless, voice_on)

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models(self):
        """Load LSTM and CNN models. Gracefully skip if not trained yet."""
        # MediaPipe Pose: config defaults, the same pose stage training uses.
        if MP_AVAILABLE:
            self.mp_wrapper = OptimizedMPWrapper()

        self._lstm_label_map = {}
        self.model_banner = "MODEL none - no step classifier loaded"
        if self.lstm_path:
            self._load_explicit_lstm(self.lstm_path)
        else:
            self._load_promoted_lstm()
        self._load_cnn()

    def _adopt_contract(self, name: str, meta: Optional[dict], input_dim: Optional[int]):
        """Build exactly the inputs this model was trained on: its recorded
        feature_layout, sample_every and rack_normalized (no fields = legacy
        pose132, every frame; unrecorded rack = config). Refuse only a model
        whose actual input dim contradicts its own recorded layout - then
        there is no way to know which of the two is true."""
        from pipeline.hmr_backend import check_pose_source
        # The pose source actually running (a requested HMR that failed to
        # load means MediaPipe is), against the one the model was trained on.
        check_pose_source(name, meta or {}, self.hmr.backend_name if self.hmr else "none")
        contract = contract_of(meta or {})
        layout = contract["feature_layout"]
        want = LAYOUT_DIMS.get(layout)
        if want is None or (input_dim is not None and int(input_dim) != want):
            raise RuntimeError(
                f"{name}: model takes {input_dim}-dim frames but records layout "
                f"{layout!r} ({want}-dim). Refusing to start rather than guess.")
        self.feature_layout = layout
        self.sample_every = contract["sample_every"]
        rack = contract["rack_normalized"]
        if rack is not None:
            if rack and self.rack_normalizer is None:
                self.rack_normalizer = RackFrameNormalizer(angle_ema=RACK_ANGLE_EMA,
                                                           scale_ema=RACK_SCALE_EMA)
            elif not rack:
                self.rack_normalizer = None
        rack_txt = ("on" if self.rack_normalizer is not None else "off") + (
            "" if rack is not None else " (unrecorded, from config)")
        self.model_banner = (
            f"MODEL {Path(name).name} | {layout} ({want}-d) | "
            f"1 sample / {1000.0 * self.sample_every / FPS:.0f} ms | rack {rack_txt}")
        if layout != FEATURE_LAYOUT or self.sample_every != SAMPLE_EVERY:
            self.model_banner += " | NOT THE DEPLOYED CONTRACT"
        logger.warning("Serving %s", self.model_banner)

    def _load_explicit_lstm(self, path: str):
        """Serve a chosen .pt (PyTorch) under the contract it records."""
        from train.train_lstm import HARLSTMClassifier
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._adopt_contract(path, ckpt, ckpt["feature_dim"])
        self.lstm_model = HARLSTMClassifier(
            feature_dim=ckpt["feature_dim"], hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"], num_classes=ckpt["num_classes"])
        self.lstm_model.load_state_dict(ckpt["model_state"])
        self.lstm_model.eval()
        self._lstm_label_map = {int(k): int(v) for k, v in ckpt.get("idx_to_label", {}).items()}

    def _load_promoted_lstm(self):
        """Load models/lstm_classifier.{onnx,pt} and serve it under the
        contract its .pt records - loudly (banner + WARNING), so whichever
        model is running is never a surprise."""
        # LSTM — prefer ONNX, fallback to PyTorch
        if LSTM_USE_ONNX and ORT_AVAILABLE and Path(LSTM_ONNX_PATH).exists():
            try:
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.lstm_ort_sess = ort.InferenceSession(
                    str(LSTM_ONNX_PATH),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("LSTM ONNX session loaded: %s", LSTM_ONNX_PATH)
            except Exception as e:
                logger.warning("LSTM ONNX load failed: %s", e)
        onnx_dim = None
        if self.lstm_ort_sess is not None:
            onnx_dim = self.lstm_ort_sess.get_inputs()[0].shape[-1]
            onnx_dim = onnx_dim if isinstance(onnx_dim, int) else None

        # Label map + recorded contract live in the .pt beside the ONNX.
        ckpt = None
        if Path(LSTM_PATH).exists():
            try:
                ckpt = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
                raw_map = ckpt.get("idx_to_label", {})
                self._lstm_label_map = {int(k): int(v) for k, v in raw_map.items()}
            except Exception as e:
                logger.warning("LSTM checkpoint %s unreadable: %s", LSTM_PATH, e)

        if ckpt is not None:
            self._adopt_contract(LSTM_PATH, ckpt, ckpt.get("feature_dim"))
            if onnx_dim is not None and onnx_dim != LAYOUT_DIMS[self.feature_layout]:
                raise RuntimeError(
                    f"{LSTM_ONNX_PATH} takes {onnx_dim}-dim frames but {LSTM_PATH} records "
                    f"layout {self.feature_layout!r} ({LAYOUT_DIMS[self.feature_layout]}-dim): "
                    f"the pair was not promoted together.")
        elif self.lstm_ort_sess is not None:
            # No .pt: nothing records a contract, so it can only be legacy.
            self._adopt_contract(LSTM_ONNX_PATH, None, onnx_dim)

        if self.lstm_ort_sess is None and ckpt is not None:
            try:
                from train.train_lstm import HARLSTMClassifier
                self.lstm_model = HARLSTMClassifier(
                    feature_dim  = ckpt["feature_dim"],
                    hidden_size  = ckpt["hidden_size"],
                    num_layers   = ckpt["num_layers"],
                    num_classes  = ckpt["num_classes"],
                )
                self.lstm_model.load_state_dict(ckpt["model_state"])
                self.lstm_model.eval()
                logger.info("LSTM loaded (PyTorch): val_acc=%.3f", ckpt.get("val_acc", 0))
            except Exception as e:
                logger.warning("LSTM load failed: %s", e)
        elif self.lstm_ort_sess is None:
            logger.warning("LSTM not found at %s — train first.", LSTM_PATH)

    def _load_cnn(self):
        # CNN (ONNX preferred for speed) — only loaded eagerly when the
        # ensemble is actually enabled; otherwise this used to load a session
        # that nothing ever called (pure startup memory/latency for zero
        # effect on any prediction).
        self._cnn_idx_to_step = {}
        if self.enable_cnn_ensemble:
            if ORT_AVAILABLE and Path(CNN_ONNX_PATH).exists():
                try:
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = 4
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.cnn_session = ort.InferenceSession(
                        CNN_ONNX_PATH,
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    logger.info("CNN ONNX session loaded: %s", CNN_ONNX_PATH)
                except Exception as e:
                    logger.warning("CNN ONNX load failed: %s", e)
            elif Path(CNN_MODEL_PATH).exists():
                logger.info("CNN .pt found but ONNX preferred. Export with train_cnn._export_onnx()")

            # Class-index -> step-id map, needed regardless of ONNX/PyTorch —
            # only the .pt checkpoint carries train_cnn.py's label_map.
            if Path(CNN_MODEL_PATH).exists():
                try:
                    ckpt = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=False)
                    raw_map = ckpt.get("label_map", {})  # {step_id: class_idx}
                    self._cnn_idx_to_step = {int(v): int(k) for k, v in raw_map.items()}
                except Exception as e:
                    logger.warning("CNN label map load failed: %s", e)

            if self.cnn_session is None:
                logger.warning("CNN ensemble enabled but no usable CNN model found/loaded — "
                               "ensemble will behave as LSTM-only until a CNN is trained.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _bind_callbacks(self):
        sm = self.state_machine

        def on_step_started(rec: StepRecord):
            self.exp_logger.log_step_start(rec.step_id, rec.name, rec.confidence)

        def on_step_completed(rec: StepRecord):
            self.exp_logger.log_step_complete(
                rec.step_id, rec.name, rec.confidence, rec.duration_sec or 0.0,
                recovered=rec.recovered,
            )
            if not rec.recovered:
                self.voice.alert_step_complete(rec.step_id)
            # current_step advances only after completion, so it is the
            # instruction the astronaut should hear next.
            if sm.current_step and not rec.recovered:
                self.voice.alert_step_next(sm.current_step["id"], sm.current_step["name"])
            if self.gui_queue:
                self.gui_queue.put_nowait(("step_complete", rec))

        def on_step_skipped(expected_id: int, observed_id: int):
            from config.experiment_config import EXPERIMENT_STEPS
            step = next((s for s in EXPERIMENT_STEPS if s["id"] == expected_id), None)
            name = step["name"] if step else f"Step {expected_id}"
            self.exp_logger.log_step_skipped(expected_id, name, observed_id)
            self.voice.alert_skip(expected_id)
            if self.gui_queue:
                self.gui_queue.put_nowait(("step_skipped", expected_id))

        def on_out_of_sequence(expected_id: int, observed_id: int):
            self.exp_logger.log_out_of_sequence(expected_id, observed_id)
            self.voice.alert_out_of_sequence(expected_id)
            if self.gui_queue:
                self.gui_queue.put_nowait(("out_of_sequence", expected_id))

        def on_step_recovered(rec: StepRecord, observed_id: int | None):
            self.exp_logger.log_alert(
                f"Recovery confirmed: step {rec.step_id} corrected after observed step {observed_id}."
            )
            self.voice.alert_step_recovered(rec.step_id, sm.current_step)
            if self.gui_queue:
                self.gui_queue.put_nowait(("step_recovered", rec.step_id))

        def on_experiment_complete():
            summary = self.state_machine.get_status_summary()
            completed = sum(1 for s in summary["steps"] if s["status"] == "COMPLETED")
            # StepStatus.SKIPPED is never actually assigned (holds are always
            # resolved by correction, not abandonment) — this stays for a
            # possible future policy change. recovery_events is the real
            # signal for "did this run need a hold/correction."
            skipped   = sum(1 for s in summary["steps"] if s["status"] == "SKIPPED")
            self.exp_logger.log_experiment_complete(
                summary["elapsed_sec"], completed, skipped,
                recovery_events=summary["recovery"]["events"],
            )
            self.voice.alert_experiment_complete()
            if self.gui_queue:
                self.gui_queue.put_nowait(("experiment_complete", summary))

        sm.on_step_started        = on_step_started
        sm.on_step_completed      = on_step_completed
        sm.on_step_skipped        = on_step_skipped
        sm.on_out_of_sequence     = on_out_of_sequence
        sm.on_step_recovered      = on_step_recovered
        sm.on_experiment_complete = on_experiment_complete

    # ── Per-Frame Processing (Optimized) ─────────────────────────────────────

    def _run_lstm_inference(self) -> tuple[int, float]:
        """
        Run LSTM on current skeleton buffer.
        Returns (predicted_step_id, confidence).
        Prefers ONNX Runtime for speed.
        """
        if not self.skeleton_buffer.full:
            return 0, 0.0

        seq = self.skeleton_buffer.array()                   # (T, features)
        x = seq[np.newaxis, ...].astype(np.float32)          # (1, T, features)

        # ── ONNX Runtime path (faster) ──────────────────────
        if self.lstm_ort_sess is not None:
            input_name = self.lstm_ort_sess.get_inputs()[0].name
            logits = self.lstm_ort_sess.run(None, {input_name: x})[0]
            probs = self._softmax(logits[0])
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            step_id = self._lstm_label_map.get(cls_idx, cls_idx + 1)
            if isinstance(step_id, str):
                step_id = int(step_id)
            return int(step_id), conf

        # ── PyTorch fallback ─────────────────────────────────
        if self.lstm_model is None:
            return 0, 0.0

        x_t = torch.from_numpy(x)
        with torch.no_grad():
            logits = self.lstm_model(x_t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, cls_idx = probs.max(0)
            conf = conf.item()
            cls_idx = cls_idx.item()

        step_id = self._lstm_label_map.get(cls_idx, cls_idx + 1)
        if isinstance(step_id, str):
            step_id = int(step_id)
        return int(step_id), conf

    @staticmethod
    def _softmax(x):
        """NumPy softmax for ONNX path."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── Optional CNN ensemble signal (off by default, see config) ───────────

    def _append_cnn_frame(self, frame_rgb_full: np.ndarray):
        """Resize + buffer one frame for the CNN's temporal stack. Cheap;
        only called when enable_cnn_ensemble."""
        small = cv2.resize(frame_rgb_full, (CNN_IMG_SIZE, CNN_IMG_SIZE))
        self.frame_stack_buffer.append(small)

    def _run_cnn_inference(self) -> tuple[int, float]:
        """
        Run the CNN on the current frame-stack buffer. Mirrors
        FrameStackDataset's exact (non-augmented) preprocessing in
        train/train_cnn.py, since that's the distribution the model actually
        learned — resize, RGB, stack-as-channels, ImageNet-ish normalize.
        Returns (predicted_step_id, confidence); (0, 0.0) if unavailable.
        """
        if self.cnn_session is None or len(self.frame_stack_buffer) < CNN_NUM_FRAMES_IN:
            return 0, 0.0

        frames = list(self.frame_stack_buffer)  # each (CNN_IMG_SIZE, CNN_IMG_SIZE, 3) RGB uint8
        stacked = np.concatenate(frames, axis=-1)               # (H, W, N*3)
        stacked = stacked.transpose(2, 0, 1).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406] * CNN_NUM_FRAMES_IN, dtype=np.float32).reshape(-1, 1, 1)
        std  = np.array([0.229, 0.224, 0.225] * CNN_NUM_FRAMES_IN, dtype=np.float32).reshape(-1, 1, 1)
        stacked = (stacked - mean) / (std + 1e-7)

        x = stacked[np.newaxis, ...].astype(np.float32)  # (1, N*3, H, W)
        input_name = self.cnn_session.get_inputs()[0].name
        logits = self.cnn_session.run(None, {input_name: x})[0]
        probs = self._softmax(logits[0])
        cls_idx = int(np.argmax(probs))
        conf = float(probs[cls_idx])
        step_id = self._cnn_idx_to_step.get(cls_idx, cls_idx + 1)
        return int(step_id), conf

    def _fuse_predictions(self, lstm_step: int, lstm_conf: float,
                          cnn_step: int, cnn_conf: float) -> tuple[int, float, bool]:
        """
        Combine the LSTM's temporal-sequence prediction with the CNN's
        frame-level prediction. Returns (step_id, confidence, uncertain).

        When uncertain is True the caller must NOT feed this into the state
        machine — this is the concrete implementation of the PS explainer's
        own stated principle: "if the system isn't confident about what
        it's seeing, it shouldn't silently pass or fail the step ... it
        should ask the astronaut for a quick confirmation instead." Two
        independently-confident models disagreeing is exactly that
        situation, not a case to average or coin-flip through.
        """
        lstm_ok = lstm_step > 0 and lstm_conf >= STEP_CONFIDENCE_THRESHOLD
        cnn_ok  = cnn_step > 0 and cnn_conf >= CNN_CONFIDENCE_THRESHOLD

        if lstm_ok and cnn_ok:
            if lstm_step == cnn_step:
                return lstm_step, max(lstm_conf, cnn_conf), False
            return 0, 0.0, True  # both confident, but disagree
        if lstm_ok:
            return lstm_step, lstm_conf, False
        if cnn_ok:
            return cnn_step, cnn_conf, False
        return 0, 0.0, False  # neither confident — idle, same as today

    def _handle_uncertain(self, lstm_step: int, lstm_conf: float,
                          cnn_step: int, cnn_conf: float):
        """LSTM and CNN are each individually confident but disagree — ask
        the astronaut to confirm rather than feed a guess to the state
        machine."""
        expected = self.state_machine.expected_step_id
        self.voice.alert_uncertain(expected)
        self.exp_logger.log_uncertain(expected, lstm_step, cnn_step, lstm_conf, cnn_conf)
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("uncertain", {
                    "expected_step_id": expected,
                    "lstm_step": lstm_step, "lstm_confidence": lstm_conf,
                    "cnn_step": cnn_step, "cnn_confidence": cnn_conf,
                }))
            except queue.Full:
                pass

    def _predict_and_feed_state_machine(self) -> tuple[int, float]:
        """
        Run LSTM inference (always) and, when enable_cnn_ensemble, fuse it
        with a CNN frame-level prediction before feeding the state machine.
        Shared by process_frame() and run() so the fusion/uncertainty policy
        lives in exactly one place. Returns (step_id, confidence) for
        HUD/GUI display — when uncertain, this is the LSTM's own raw result
        (for display continuity only; the state machine was not fed).
        """
        lstm_step, lstm_conf = self._run_lstm_inference()

        if not self.enable_cnn_ensemble or len(self.frame_stack_buffer) < CNN_NUM_FRAMES_IN:
            self.state_machine.feed_prediction(lstm_step, lstm_conf)
            return lstm_step, lstm_conf

        cnn_step, cnn_conf = self._run_cnn_inference()
        fused_step, fused_conf, uncertain = self._fuse_predictions(
            lstm_step, lstm_conf, cnn_step, cnn_conf
        )
        if uncertain:
            self._handle_uncertain(lstm_step, lstm_conf, cnn_step, cnn_conf)
            return lstm_step, lstm_conf
        self.state_machine.feed_prediction(fused_step, fused_conf)
        return fused_step, fused_conf

    def _ingest(self, features: np.ndarray, t: float, label=None) -> int:
        """Offer one timestamped frame's model features to the window; classify
        and feed the state machine once per SAMPLE it emits (possibly 0, 1 or
        several per frame). This is the single sampling gate for BOTH run()
        and process_frame(), and a sample is the unit STEP_CONFIRM_FRAMES is
        expressed in. `label` (sim/eval only) rides along with the sample.
        Returns the number of samples emitted."""
        def on_sample():
            if self.skeleton_buffer.full:
                self._last_pred = self._predict_and_feed_state_machine()
        resets = self.skeleton_buffer.gap_resets
        n = self.skeleton_buffer.push(features, t, label, on_sample)
        if self.skeleton_buffer.gap_resets != resets:
            logger.warning("Frame gap > %.1f s: model window cleared, warming up again.",
                           self.skeleton_buffer.max_gap)
        return n

    def _perceive(self, frame: np.ndarray, timings: Optional[dict] = None):
        """HSV + pose for the current frame via the shared perceive()."""
        dets, pose = perceive(frame, self._frame_rgb_full, self.hsv_detector,
                              self.mp_wrapper, self._prev_dets, self.threaded, timings)
        self._prev_dets = dets
        return dets, pose

    def _frame_features(self, skel: np.ndarray, detections: list) -> np.ndarray:
        return build_frame_features(skel, detections, (FRAME_WIDTH, FRAME_HEIGHT),
                                    self.rack_normalizer, self.feature_layout)

    def _annotate_frame(self, frame: np.ndarray,
                        detections: list,
                        pred_step: int,
                        confidence: float) -> np.ndarray:
        """Draw HUD overlay on frame with space physics simulation."""
        vis = frame.copy()

        # ── SPACE SIMULATION VISUAL PHYSICS ──
        # Add a subtle cyan/blue tint to simulate ISS module lighting
        tint = np.full_like(vis, (50, 20, 0), dtype=np.uint8) # BGR (Blueish)
        cv2.addWeighted(vis, 0.85, tint, 0.15, 0, vis)
        
        # Simulate microgravity drifting particles (dust/debris)
        if not hasattr(self, '_particles'):
            self._particles = np.random.rand(40, 5) 
            self._particles[:, 0] *= frame.shape[1]
            self._particles[:, 1] *= frame.shape[0]
            self._particles[:, 2] = (self._particles[:, 2] - 0.5) * 1.5  # vx
            self._particles[:, 3] = (self._particles[:, 3] - 0.5) * 1.5  # vy
            self._particles[:, 4] = self._particles[:, 4] * 2 + 1        # size

        # Update and draw particles in zero-g
        for p in self._particles:
            p[0] = (p[0] + p[2]) % frame.shape[1]
            p[1] = (p[1] + p[3]) % frame.shape[0]
            # Draw particle with slight glow
            cv2.circle(vis, (int(p[0]), int(p[1])), int(p[4]), (255, 230, 200), -1)
            
        # Draw floating zero-g tracking grid
        grid_offset = int((time.time() * 15) % 100)
        for i in range(0, frame.shape[0], 100):
            cv2.line(vis, (0, i + grid_offset), (frame.shape[1], i + grid_offset), (50, 30, 0), 1)
        for i in range(0, frame.shape[1], 100):
            cv2.line(vis, (i + grid_offset, 0), (i + grid_offset, frame.shape[0]), (50, 30, 0), 1)
        # ──────────────────────────────────────

        # Draw HSV detections
        vis = self.hsv_detector.draw(vis, detections)

        # Status overlay
        sm_status = self.state_machine.get_status_summary()
        current = sm_status.get("current_step")
        nxt     = sm_status.get("next_step")

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (420, 120), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        step_name = current["name"] if current else "COMPLETE"
        cv2.putText(vis, f"Current: Step {current['id'] if current else '-'}: {step_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
        next_name = nxt["name"] if nxt else "—"
        cv2.putText(vis, f"Next:    Step {nxt['id'] if nxt else '-'}: {next_name}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(vis, f"Predict: Step {pred_step} ({confidence:.2f})",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(vis, f"Elapsed: {sm_status['elapsed_sec']:.1f}s",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        if (self.rack_normalizer is not None
                and self.rack_normalizer.rack_angle_deg is not None):
            cv2.putText(vis, f"Rack frame: {self.rack_normalizer.rack_angle_deg:.1f} deg",
                        (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1)

        # Which model is running, and on what inputs - always on screen.
        if self.model_banner:
            warn = "NOT THE DEPLOYED" in self.model_banner or "none" in self.model_banner
            y = frame.shape[0] - 12
            cv2.rectangle(vis, (0, y - 20), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
            cv2.putText(vis, self.model_banner, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 190, 255) if warn else (200, 200, 200), 1)

        # FPS + latency
        if len(self._times) > 1:
            avg_ms = np.mean(list(self._times)) * 1000
            fps_actual = 1000.0 / max(avg_ms, 0.1)
            cv2.putText(vis, f"ISRO HAR | {fps_actual:.0f} FPS | {avg_ms:.1f}ms",
                        (frame.shape[1] - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

        return vis

    # ── Headless / simulation API ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray,
                      injected_skel: Optional[np.ndarray] = None,
                      annotate: bool = False,
                      timestamp: Optional[float] = None,
                      label: Optional[int] = None) -> dict:
        """
        Process a single BGR frame. Always runs HSV + MediaPipe for real latency.
        If injected_skel is provided it is used for LSTM (oracle pose) so accuracy
        can be measured independently of MediaPipe on synthetic figures.

        `timestamp` is the frame's capture time in seconds; when absent the
        frame is assumed to sit on the nominal FPS clock ((n-1)/FPS), which
        keeps sim/showcase runs deterministic. `label` (ground-truth step id,
        sim/eval only) travels with the model window's samples so a caller can
        score exactly what the classifier saw (skeleton_buffer.labels()).
        """
        t0 = time.perf_counter()

        if frame is None or frame.size == 0:
            return {
                "pred_step": 0, "confidence": 0.0, "detections": [], "vis": None,
                "sampled": False,
                "timings_ms": {"hsv": 0.0, "mediapipe": 0.0, "lstm": 0.0, "total": 0.0},
            }

        frame = to_frame_size(frame)

        self.frame_idx += 1
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_full)

        t_perc = time.perf_counter()
        timings = {}
        if self.hmr is not None and self.hmr.available:
            # Stage 2: root-relative 3D from the HMR mesh (same 132-dim contract).
            detections = self.hsv_detector.detect(frame)
            timings["hsv_ms"] = (time.perf_counter() - t_perc) * 1000.0
            skel_mp, _ = self.hmr.get_pose_features(self._frame_rgb_full)
            self._prev_dets = detections
        else:
            detections, skel_mp = self._perceive(frame, timings)
        perc_ms = (time.perf_counter() - t_perc) * 1000.0
        hsv_ms = timings.get("hsv_ms", 0.0)          # threaded: not separable
        mp_ms = timings.get("pose_ms", perc_ms - hsv_ms)

        skel = injected_skel if injected_skel is not None else skel_mp
        feats = self._frame_features(skel, detections)
        if self.enable_cnn_ensemble:
            self._append_cnn_frame(self._frame_rgb_full)

        t_lstm = time.perf_counter()
        t = timestamp if timestamp is not None else (self.frame_idx - 1) / float(FPS)
        n_samples = self._ingest(feats, t, label)
        pred_step, pred_conf = self._last_pred
        lstm_ms = (time.perf_counter() - t_lstm) * 1000.0

        vis = self._annotate_frame(frame, detections, pred_step, pred_conf) if annotate else None
        total_ms = (time.perf_counter() - t0) * 1000.0
        self._times.append(total_ms / 1000.0)

        return {
            "pred_step": int(pred_step),
            "confidence": float(pred_conf),
            "detections": detections,
            "vis": vis,
            "sampled": n_samples > 0,   # this frame emitted >= 1 model sample
            "samples": int(n_samples),
            "timings_ms": {
                "hsv": float(hsv_ms),
                "mediapipe": float(mp_ms),
                "lstm": float(lstm_ms),
                "total": float(total_ms),
            },
        }

    def reset_runtime(self):
        """Clear buffers and state machine for a new simulation trial."""
        self.frame_idx = 0
        self._last_pred = (0, 0.0)
        self._prev_dets = None            # first frame of a trial gets no pose hint
        self.skeleton_buffer.clear()
        self.frame_stack_buffer.clear()
        self._times.clear()
        if self.mp_wrapper is not None:
            self.mp_wrapper.reset()       # re-latch the upright working angle
        if self.rack_normalizer is not None:
            self.rack_normalizer.reset()  # re-latch rack polarity for the new trial
        self.state_machine.reset()
        self._bind_callbacks()

    def _push_gui_status(self):
        """Push a full state-machine status snapshot to the GUI, alongside the
        discrete step/alert events already pushed from the state-machine
        callbacks. Lets the dashboard render the checklist/current/next step
        without re-deriving state from individual events.

        Also carries live recording/streaming health, since both can be true
        one moment and false the next (a died mid-run) — a footer string set
        once at GUI construction can't reflect that.
        """
        if not self.gui_queue:
            return
        summary = self.state_machine.get_status_summary()
        summary["recording_active"] = self.video_writer is not None
        summary["streaming_active"] = self.streamer is not None and self.streamer.available
        summary["stream_target"] = (f"{self.stream_host}:{self.stream_port}"
                                    if self.enable_streaming else None)
        summary["model_banner"] = self.model_banner
        try:
            self.gui_queue.put_nowait(("status", summary))
        except queue.Full:
            pass

    def close(self):
        self._running = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.streamer is not None:
            self.streamer.close()
            self.streamer = None
        if self.mp_wrapper is not None:
            self.mp_wrapper.close()
            self.mp_wrapper = None

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        # Frame timestamps: a camera frame is stamped with its capture time; a
        # video file with its own timeline (i / file fps), since decode speed
        # is not the time the scene was captured at.
        self._file_fps = None
        if isinstance(self.source, str):
            self._file_fps = cap.get(cv2.CAP_PROP_FPS)
            if not self._file_fps or self._file_fps <= 0:
                # e.g. an image sequence reports -1; every frame would look
                # like a >0.5 s gap and the window would never fill.
                logger.warning("Source %s reports fps=%s; assuming %d fps.",
                               self.source, self._file_fps, FPS)
                self._file_fps = float(FPS)

        # Video writer (local storage)
        if self.enable_recording:
            ts  = time.strftime("%Y%m%d_%H%M%S")
            vw_path = str(Path(LOCAL_RECORDING_DIR) / f"experiment_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(vw_path, fourcc, FPS,
                                                (FRAME_WIDTH, FRAME_HEIGHT))
            logger.info("Recording to: %s", vw_path)

        # Network streamer (push to a specific IP, alongside local storage)
        if self.enable_streaming:
            self.streamer = NetworkStreamer(self.stream_host, self.stream_port,
                                            FRAME_WIDTH, FRAME_HEIGHT, FPS)
            if not self.streamer.available:
                self.streamer = None

        self.voice.alert_experiment_start()
        # "At the start ... the model should suggest the next step to be
        # performed" — alert_experiment_start() only ever named step 1 by
        # number, never by action. Announce it properly, same as every later
        # step gets via alert_step_next() after completion.
        first_step = self.state_machine.current_step
        if first_step:
            self.voice.alert_step_next(first_step["id"], first_step["name"])
        self.exp_logger.log_experiment_start()
        self._running = True

        while self._running:
            ret, frame = cap.read()
            t_capture = time.monotonic()
            if not ret:
                logger.info("Video source exhausted.")
                break

            # A camera/video source isn't guaranteed to deliver exactly
            # FRAME_WIDTH x FRAME_HEIGHT (cap.set() above is a no-op for file
            # sources and not guaranteed for webcams). The network stream is a
            # fixed-size raw-video pipe (stream_sender.py), so a mismatched
            # frame desyncs it. Normalize once, here, before anything writes
            # or converts the frame — process_frame() already does this;
            # run() didn't.
            frame = to_frame_size(frame)

            self.frame_idx += 1
            t_start = time.time()

            # ── Always write raw frame (local) + push to network stream ──
            if self.video_writer:
                self.video_writer.write(frame)
            if self.streamer and not self.streamer.available:
                # The sender died mid-run (e.g. ffmpeg exited) — drop the
                # reference so the GUI status stops claiming it's active.
                self.streamer = None
            if self.streamer:
                self.streamer.write(frame)

            # ── Process every N frames for speed ────────────
            if self.frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                # Still show last frame for smooth display
                if self.gui_queue:
                    try:
                        self.gui_queue.put_nowait(("frame", frame))
                    except queue.Full:
                        pass
                    self._push_gui_status()
                if not self.headless and self.gui_queue is None:
                    cv2.imshow("ISRO HAR Monitor", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # ── Convert to RGB (shared buffer) ──────────────
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_full)

            # ── Parallel or sequential inference (same result either way) ──
            detections, skel = self._perceive(frame)

            feats = self._frame_features(skel, detections)
            if self.enable_cnn_ensemble:
                self._append_cnn_frame(self._frame_rgb_full)

            # ── LSTM (+ optional CNN ensemble) inference ─────
            t = ((self.frame_idx - 1) / self._file_fps) if self._file_fps else t_capture
            self._ingest(feats, t)
            pred_step, pred_conf = self._last_pred

            # ── Build annotated frame & push to GUI ──────────
            vis = self._annotate_frame(frame, detections, pred_step, pred_conf)

            if self.gui_queue:
                try:
                    self.gui_queue.put_nowait(("frame", vis))
                except queue.Full:
                    pass
                self._push_gui_status()

            # ── Timing ───────────────────────────────────────
            t_elapsed = time.time() - t_start
            self._times.append(t_elapsed)

            if not self.headless and self.gui_queue is None:
                cv2.imshow("ISRO HAR Monitor", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        # Cleanup
        self._running = False
        cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.streamer:
            self.streamer.close()
            self.streamer = None
        if self.mp_wrapper:
            self.mp_wrapper.close()
        if not self.headless and self.gui_queue is None:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Log: %s", self.exp_logger.get_log_path())

    def stop(self):
        self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISRO HAR Inference Pipeline (Optimized)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--video",  type=str, default=None,
                        help="Path to video file (uses camera if not set)")
    parser.add_argument("--stream", action="store_true",
                        help="Also push video to STREAM_HOST:STREAM_PORT via ffmpeg")
    parser.add_argument("--stream-host", type=str, default=None)
    parser.add_argument("--stream-port", type=int, default=None)
    parser.add_argument("--cnn-ensemble", action="store_true",
                        help="Fuse CNN + LSTM predictions (retrain CNN first — see train/train_cnn.py)")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    pipeline = HARPipeline(source=source,
                           enable_streaming=True if args.stream else None,
                           stream_host=args.stream_host, stream_port=args.stream_port,
                           enable_cnn_ensemble=True if args.cnn_ensemble else None)
    pipeline.run()
