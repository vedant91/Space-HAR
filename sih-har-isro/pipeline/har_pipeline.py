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
    VOICE_ENABLED, STREAM_PORT, LOCAL_RECORDING_DIR,
    STEP_CONFIDENCE_THRESHOLD,
    MEDIAPIPE_MODEL_COMPLEXITY, MEDIAPIPE_MIN_DET_CONF, MEDIAPIPE_MIN_TRK_CONF,
    MEDIAPIPE_DOWNSCALE, HSV_DOWNSCALE, USE_THREADED_INFERENCE,
)
from pipeline.state_machine import ExperimentStateMachine, StepRecord
from pipeline.voice_alert import VoiceAlertSystem
from pipeline.logger import ExperimentLogger
from pipeline.hsv_detector import HSVBoxDetector, Detection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Import optional dependencies ──────────────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("mediapipe not installed. Skeleton features disabled.")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed. Using PyTorch for CNN inference.")


# Feature dimension: MediaPipe Pose only (faster than Holistic)
# Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
# Hands removed for speed — pose alone captures the key motion
POSE_FEATURE_DIM = 33 * 4  # 132


class ThreadedInference:
    """
    Runs MediaPipe and HSV detection in parallel threads.
    The main thread waits for both to complete before combining results.
    """

    def __init__(self):
        self._mp_result = None
        self._hsv_result = None

    def run_parallel(self, frame_bgr: np.ndarray, frame_rgb_full: np.ndarray,
                     hsv_detector, mp_wrapper):
        """
        Run HSV detection and MediaPipe Pose in parallel threads.
        mp_wrapper is an OptimizedMPWrapper instance.
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
                    results = mp_wrapper.process(frame_rgb_full)
                    features = self._extract_pose_features(results)
                    self._mp_result = features
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

    @staticmethod
    def _extract_pose_features(results) -> np.ndarray:
        """Extract pose-only features (132-dim). Much faster than Holistic."""
        features = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        if results and results.pose_landmarks:
            idx = 0
            for lm in results.pose_landmarks.landmark:
                features[idx] = lm.x
                features[idx + 1] = lm.y
                features[idx + 2] = lm.z
                features[idx + 3] = lm.visibility
                idx += 4
        return features


class OptimizedMPWrapper:
    """
    Wrapper around MediaPipe Pose with pre-allocated buffers
    and configurable downscaling.
    """

    def __init__(self, complexity: int = 0,
                 min_det_conf: float = 0.3,
                 min_trk_conf: float = 0.3,
                 downscale: int = 2):
        self.downscale = downscale
        self.small_w = FRAME_WIDTH // downscale
        self.small_h = FRAME_HEIGHT // downscale

        # Pre-allocate the small RGB buffer
        self.frame_rgb_small = np.zeros((self.small_h, self.small_w, 3), dtype=np.uint8)
        self.frame_flags = self.frame_rgb_small  # alias for writeable flag access

        if MP_AVAILABLE:
            mp_pose = mp.solutions.pose
            self.holistic = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,       # 0 = fastest
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_det_conf,
                min_tracking_confidence=min_trk_conf,
            )
            logger.info(
                "MediaPipe Pose loaded (complexity=%d, det=%.1f, trk=%.1f, downscale=%d)",
                complexity, min_det_conf, min_trk_conf, downscale
            )
        else:
            self.holistic = None

    def process(self, frame_rgb_full: np.ndarray):
        """Downscale + process. Returns pose landmarks result."""
        if self.holistic is None:
            return None

        # Downscale for speed
        cv2.resize(frame_rgb_full, (self.small_w, self.small_h),
                   dst=self.frame_rgb_small, interpolation=cv2.INTER_LINEAR)
        self.frame_rgb_small.flags.writeable = False
        results = self.holistic.process(self.frame_rgb_small)
        self.frame_rgb_small.flags.writeable = True
        return results

    def close(self):
        if self.holistic:
            self.holistic.close()


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
                 enable_streaming: bool = False,
                 gui_queue: Optional[queue.Queue] = None,
                 headless: bool = False,
                 enable_voice: Optional[bool] = None,
                 enable_recording: bool = True,
                 use_threaded: Optional[bool] = None):

        self.source         = source
        self.gui_queue      = gui_queue
        self.headless       = headless
        self.enable_recording = enable_recording and not headless
        self.frame_idx      = 0
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
        self.skeleton_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_WINDOW)

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
        # MediaPipe Pose (optimized)
        if MP_AVAILABLE:
            self.mp_wrapper = OptimizedMPWrapper(
                complexity=MEDIAPIPE_MODEL_COMPLEXITY,
                min_det_conf=MEDIAPIPE_MIN_DET_CONF,
                min_trk_conf=MEDIAPIPE_MIN_TRK_CONF,
                downscale=MEDIAPIPE_DOWNSCALE,
            )

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

        # Load label map from .pt checkpoint (needed for both ONNX and PyTorch paths)
        self._lstm_label_map = {}
        if Path(LSTM_PATH).exists():
            try:
                ckpt = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
                raw_map = ckpt.get("idx_to_label", {})
                self._lstm_label_map = {int(k): int(v) for k, v in raw_map.items()}
            except Exception:
                pass

        if self.lstm_ort_sess is None and Path(LSTM_PATH).exists():
            try:
                ckpt = torch.load(LSTM_PATH, map_location="cpu")
                from train.train_lstm import HARLSTMClassifier
                self.lstm_model = HARLSTMClassifier(
                    feature_dim  = ckpt["feature_dim"],
                    hidden_size  = ckpt["hidden_size"],
                    num_layers   = ckpt["num_layers"],
                    num_classes  = ckpt["num_classes"],
                )
                self.lstm_model.load_state_dict(ckpt["model_state"])
                self.lstm_model.eval()
                raw_map = ckpt.get("idx_to_label", {})
                self._lstm_label_map = {int(k): int(v) for k, v in raw_map.items()}
                logger.info("LSTM loaded (PyTorch): val_acc=%.3f", ckpt.get("val_acc", 0))
            except Exception as e:
                logger.warning("LSTM load failed: %s", e)
        else:
            if self.lstm_ort_sess is None:
                logger.warning("LSTM not found at %s — train first.", LSTM_PATH)

        # CNN (ONNX preferred for speed)
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

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _bind_callbacks(self):
        sm = self.state_machine

        def on_step_started(rec: StepRecord):
            self.exp_logger.log_step_start(rec.step_id, rec.name, rec.confidence)

        def on_step_completed(rec: StepRecord):
            self.exp_logger.log_step_complete(
                rec.step_id, rec.name, rec.confidence, rec.duration_sec or 0.0
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
            skipped   = sum(1 for s in summary["steps"] if s["status"] == "SKIPPED")
            self.exp_logger.log_experiment_complete(summary["elapsed_sec"], completed, skipped)
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

    def _extract_skeleton_features_optimized(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Run optimized MediaPipe Pose and return 132-dim feature vector.
        Uses pre-allocated buffers and downscaled frames.
        """
        if self.mp_wrapper is None:
            return np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        results = self.mp_wrapper.process(frame_rgb)

        features = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        if results and results.pose_landmarks:
            idx = 0
            for lm in results.pose_landmarks.landmark:
                features[idx] = lm.x
                features[idx + 1] = lm.y
                features[idx + 2] = lm.z
                features[idx + 3] = lm.visibility
                idx += 4

        return features

    def _run_lstm_inference(self) -> tuple[int, float]:
        """
        Run LSTM on current skeleton buffer.
        Returns (predicted_step_id, confidence).
        Prefers ONNX Runtime for speed.
        """
        if len(self.skeleton_buffer) < SEQUENCE_WINDOW:
            return 0, 0.0

        seq = np.stack(list(self.skeleton_buffer), axis=0)  # (T, features)
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
                      annotate: bool = False) -> dict:
        """
        Process a single BGR frame. Always runs HSV + MediaPipe for real latency.
        If injected_skel is provided it is used for LSTM (oracle pose) so accuracy
        can be measured independently of MediaPipe on synthetic figures.
        """
        t0 = time.perf_counter()

        if frame is None or frame.size == 0:
            return {
                "pred_step": 0, "confidence": 0.0, "detections": [], "vis": None,
                "timings_ms": {"hsv": 0.0, "mediapipe": 0.0, "lstm": 0.0, "total": 0.0},
            }

        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        self.frame_idx += 1
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_full)

        t_hsv = time.perf_counter()
        detections = self.hsv_detector.detect(frame)
        hsv_ms = (time.perf_counter() - t_hsv) * 1000.0

        t_mp = time.perf_counter()
        skel_mp = self._extract_skeleton_features_optimized(self._frame_rgb_full)
        mp_ms = (time.perf_counter() - t_mp) * 1000.0

        skel = injected_skel if injected_skel is not None else skel_mp
        self.skeleton_buffer.append(np.asarray(skel, dtype=np.float32).reshape(-1))

        t_lstm = time.perf_counter()
        pred_step, pred_conf = self._last_pred
        if len(self.skeleton_buffer) >= SEQUENCE_WINDOW:
            pred_step, pred_conf = self._run_lstm_inference()
            self._last_pred = (pred_step, pred_conf)
            self.state_machine.feed_prediction(pred_step, pred_conf)
        lstm_ms = (time.perf_counter() - t_lstm) * 1000.0

        vis = self._annotate_frame(frame, detections, pred_step, pred_conf) if annotate else None
        total_ms = (time.perf_counter() - t0) * 1000.0
        self._times.append(total_ms / 1000.0)

        return {
            "pred_step": int(pred_step),
            "confidence": float(pred_conf),
            "detections": detections,
            "vis": vis,
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
        self.skeleton_buffer.clear()
        self._times.clear()
        self.state_machine.reset()
        self._bind_callbacks()

    def close(self):
        self._running = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
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

        # Video writer
        if self.enable_recording:
            ts  = time.strftime("%Y%m%d_%H%M%S")
            vw_path = str(Path(LOCAL_RECORDING_DIR) / f"experiment_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(vw_path, fourcc, FPS,
                                                (FRAME_WIDTH, FRAME_HEIGHT))
            logger.info("Recording to: %s", vw_path)
        self.voice.alert_experiment_start()
        self.exp_logger.log_experiment_start()
        self._running = True

        pred_step, pred_conf = 0, 0.0   # Last prediction (persisted between frames)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video source exhausted.")
                break

            self.frame_idx += 1
            t_start = time.time()

            # ── Always write raw frame ──────────────────────
            if self.video_writer:
                self.video_writer.write(frame)

            # ── Process every N frames for speed ────────────
            if self.frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                # Still show last frame for smooth display
                if self.gui_queue:
                    try:
                        self.gui_queue.put_nowait(("frame", frame))
                    except queue.Full:
                        pass
                if not self.headless:
                    cv2.imshow("ISRO HAR Monitor", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # ── Convert to RGB (shared buffer) ──────────────
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_full)

            # ── Parallel or sequential inference ─────────────
            if self.threaded:
                detections, skel = self.threaded.run_parallel(
                    frame, self._frame_rgb_full,
                    self.hsv_detector, self.mp_wrapper,
                )
            else:
                detections = self.hsv_detector.detect(frame)
                skel = self._extract_skeleton_features_optimized(self._frame_rgb_full)

            self.skeleton_buffer.append(skel)

            # ── LSTM Inference (when buffer full) ────────────
            if len(self.skeleton_buffer) >= SEQUENCE_WINDOW:
                pred_step, pred_conf = self._run_lstm_inference()
                self.state_machine.feed_prediction(pred_step, pred_conf)

            # ── Build annotated frame & push to GUI ──────────
            vis = self._annotate_frame(frame, detections, pred_step, pred_conf)

            if self.gui_queue:
                try:
                    self.gui_queue.put_nowait(("frame", vis))
                except queue.Full:
                    pass

            # ── Timing ───────────────────────────────────────
            t_elapsed = time.time() - t_start
            self._times.append(t_elapsed)

            if not self.headless:
                cv2.imshow("ISRO HAR Monitor", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        # Cleanup
        self._running = False
        cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.mp_wrapper:
            self.mp_wrapper.close()
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Log: %s", self.exp_logger.get_log_path())

    def stop(self):
        self._running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISRO HAR Inference Pipeline (Optimized)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--video",  type=str, default=None,
                        help="Path to video file (uses camera if not set)")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    pipeline = HARPipeline(source=source)
    pipeline.run()
