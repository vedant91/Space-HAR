"""
Main HAR Pipeline — Optimized Inference Engine (Tier 1)
========================================================
Integrates all components into a single real-time processing loop:
  HSVDetector + custom PoseNet + LSTM + StateMachine + VoiceAlert + Logger

No open-source/pretrained model is used anywhere in this pipeline: object
detection is classical-CV HSV segmentation (pipeline/hsv_detector.py) and
pose/movement tracking is HARPoseNet (pipeline/pose_net.py, train/
train_posenet.py) — a small CNN trained entirely from scratch on this
project's own synthetic renderer ground truth plus classical-CV pseudo-
labels on the real "gravitational mimic" footage. Both classifiers (LSTM,
CNN) are likewise trained from scratch.

Latency optimizations applied:
  1. Small (~1.2M param) pose CNN, ONNX Runtime inference
  2. Threaded parallel execution (HSV + pose net run simultaneously)
  3. Pre-allocated numpy buffers (no hot-loop allocations)
  4. LSTM ONNX Runtime inference (faster than PyTorch)

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
    STEP_CONFIDENCE_THRESHOLD, SKELETON_FEATURES, HSV_DOWNSCALE, USE_THREADED_INFERENCE,
    RACK_FRAME_NORMALIZE, RACK_ANGLE_EMA, RACK_SCALE_EMA, HMR_BACKEND,
    CNN_ENSEMBLE_ENABLED, CNN_CONFIDENCE_THRESHOLD, CNN_NUM_FRAMES_IN, CNN_IMG_SIZE,
    UPLINK_MODE, CLIP_PRE_ROLL_S, CLIP_POST_ROLL_S, CLIP_OUTPUT_DIR,
)
from pipeline.rack_frame import RackFrameNormalizer, pick_rack_rect
from pipeline.state_machine import ExperimentStateMachine, StepRecord
from pipeline.voice_alert import VoiceAlertSystem
from pipeline.logger import ExperimentLogger
from pipeline.hsv_detector import HSVBoxDetector, Detection
from pipeline.anomaly_monitor import AnomalyMonitor
from pipeline.clip_uplink import ClipUplinkManager
from pipeline.stream_sender import NetworkStreamer
from pipeline.pose_net import PoseNetWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Import optional dependencies ──────────────────────────────────────────────
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed. Using PyTorch for CNN inference.")


# Feature dimension: custom PoseNet only tracks 13 of the 33 MediaPipe-style
# slots (see config.POSE_JOINT_SLOTS) but the vector stays 33 x 4 = 132 so
# every downstream module (rack_frame, LSTM, CNN ensemble) is unaffected.
# Single source of truth is config.SKELETON_FEATURES.
POSE_FEATURE_DIM = SKELETON_FEATURES


class ThreadedInference:
    """
    Runs pose-net inference and HSV detection in parallel threads.
    The main thread waits for both to complete before combining results.
    """

    def __init__(self):
        self._pose_result = None
        self._hsv_result = None
        # Per-branch wall time (each thread times only its own work). These
        # overlap in real wall-clock time (that's the point of threading them)
        # but still tell the GUI/Pipeline-Internals view which branch is the
        # more expensive one to optimize.
        self.last_hsv_ms = 0.0
        self.last_pose_ms = 0.0

    def run_parallel(self, frame_bgr: np.ndarray, frame_rgb_full: np.ndarray,
                     hsv_detector, pose_wrapper):
        """
        Run HSV detection and the custom PoseNet in parallel threads.
        pose_wrapper is a pipeline.pose_net.PoseNetWrapper instance.
        Returns: (detections, skeleton_features)
        """
        self._pose_result = None
        self._hsv_result = None

        def _run_hsv():
            t0 = time.perf_counter()
            try:
                self._hsv_result = hsv_detector.detect(frame_bgr)
            except Exception as e:
                logger.warning("HSV thread error: %s", e)
                self._hsv_result = []
            self.last_hsv_ms = (time.perf_counter() - t0) * 1000.0

        def _run_pose():
            t0 = time.perf_counter()
            try:
                if pose_wrapper is not None:
                    self._pose_result = pose_wrapper.process(frame_rgb_full)
                else:
                    self._pose_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
            except Exception as e:
                logger.warning("PoseNet thread error: %s", e)
                self._pose_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
            self.last_pose_ms = (time.perf_counter() - t0) * 1000.0

        t1 = threading.Thread(target=_run_hsv, daemon=True)
        t2 = threading.Thread(target=_run_pose, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Fallback if either thread failed
        if self._hsv_result is None:
            self._hsv_result = []
        if self._pose_result is None:
            self._pose_result = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)

        return self._hsv_result, self._pose_result


class HARPipeline:
    """
    Optimized real-time HAR pipeline for ISRO experiment monitoring.

    Processing per frame:
      1. HSV color detection  → box + hand visibility flags
      2. Custom PoseNet       → skeleton feature vector (132-dim, 13 joints tracked)
      3. Append to LSTM buffer → classify step when buffer full
      4. Feed prediction to State Machine
      5. State Machine fires callbacks → Voice + Log + GUI

    Optimizations:
      - HARPoseNet: ~1.2M params, ONNX Runtime — fast on CPU
      - Threaded parallel execution — HSV + PoseNet run simultaneously
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
                 enable_race: bool = False,
                 earth_delay_s: Optional[float] = None,
                 uplink_mode: Optional[str] = None):

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
        self.frame_idx      = 0
        self._running       = False
        self._last_pred     = (0, 0.0)

        # ── Latency-race demo (display/logging only — see its module
        # docstring for the "cannot affect FSM authority" guarantee) ──
        self.enable_race = enable_race
        self.earth_channel: Optional["EarthDelayChannel"] = None
        if enable_race:
            from pipeline.earth_delay_channel import EarthDelayChannel, DEFAULT_EARTH_DELAY_S
            delay = DEFAULT_EARTH_DELAY_S if earth_delay_s is None else earth_delay_s
            self.earth_channel = EarthDelayChannel(delay_s=delay, on_deliver=self._on_earth_delivered)

        # ── Smart clip uplink (Brief §12 — ring-buffer + anomaly-clip mode,
        # additive to the full IP stream, not a replacement — see
        # pipeline/clip_uplink.py's module docstring) ──
        self.uplink_mode = (UPLINK_MODE if uplink_mode is None else uplink_mode).lower()
        self.clip_uplink: Optional[ClipUplinkManager] = None
        if self.uplink_mode in ("clip", "both"):
            self.clip_uplink = ClipUplinkManager(
                frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, fps=FPS,
                pre_roll_s=CLIP_PRE_ROLL_S, post_roll_s=CLIP_POST_ROLL_S,
                output_dir=CLIP_OUTPUT_DIR,
            )
        if self.uplink_mode == "clip":
            # Clip-only mode means no continuous stream — the whole point of
            # the bandwidth thesis. Full local recording is untouched (never
            # lose the raw video for post-mission review).
            self.enable_streaming = False

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
        self.anomaly_monitor = AnomalyMonitor(FRAME_WIDTH, FRAME_HEIGHT)

        # ── Orientation-agnostic pose (no fixed 'up' in microgravity) ──
        # Stage 1: rack-anchored reference frame (CPU, always available).
        self.rack_normalizer = None
        if RACK_FRAME_NORMALIZE:
            self.rack_normalizer = RackFrameNormalizer(
                angle_ema=RACK_ANGLE_EMA,
                scale_ema=RACK_SCALE_EMA,
            )
            logger.info("Rack-frame normalization ENABLED (pose is rack-relative).")
        # Stage 2: retired — see pipeline/hmr_backend.py's module docstring.
        # Always reports unavailable; kept only so old configs/imports don't break.
        self.hmr = None
        if HMR_BACKEND not in ("none", "posenet", "mediapipe", ""):
            from pipeline.hmr_backend import HMRBackend
            self.hmr = HMRBackend(backend=HMR_BACKEND)

        # ── Models ────────────────────────────────────────────
        self.pose_wrapper   = None
        self.lstm_model     = None
        self.lstm_ort_sess  = None  # ONNX Runtime session for LSTM
        self.cnn_session    = None  # ONNX Runtime session for CNN
        self._load_models()

        # ── Threaded inference ─────────────────────────────────
        threaded_flag = USE_THREADED_INFERENCE if use_threaded is None else use_threaded
        if headless and use_threaded is None:
            threaded_flag = False
        self.threaded = ThreadedInference() if threaded_flag else None

        # One-time model/backend summary for the GUI's Pipeline Internals tab
        # (pushed here, before the dashboard's event loop even starts —
        # queue.Queue buffers it until _drain_queue picks it up).
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("model_info", {
                    "pose_backend": self.pose_wrapper.backend if self.pose_wrapper else "none",
                    "pose_available": bool(self.pose_wrapper and self.pose_wrapper.available),
                    "lstm_backend": ("onnx" if self.lstm_ort_sess is not None
                                    else "pytorch" if self.lstm_model is not None else "none"),
                    "cnn_backend": "onnx" if self.cnn_session is not None else "disabled",
                    "cnn_ensemble_enabled": self.enable_cnn_ensemble,
                    "rack_normalize": self.rack_normalizer is not None,
                    "threaded": self.threaded is not None,
                }))
            except queue.Full:
                pass

        # ── Sequence buffer (for LSTM) ─────────────────────────
        self.skeleton_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_WINDOW)
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
        """Load pose/LSTM/CNN models. Gracefully skip if not trained yet."""
        # Custom pose net (trained from scratch — see pipeline/pose_net.py)
        self.pose_wrapper = PoseNetWrapper()
        if not self.pose_wrapper.available:
            logger.warning("PoseNet not trained yet (models/pose_net.onnx|.pt missing) — "
                           "pose features will be all-zero until train/train_posenet.py is run.")

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
                # weights_only=False: PyTorch 2.6+ defaults this to True, which
                # rejects our checkpoint's numpy-typed label-map values
                # ("Unsupported global: numpy._core.multiarray.scalar") and
                # silently falls through to logger.warning below — leaving
                # self.lstm_model None, i.e. every LSTM prediction is (0, 0.0)
                # forever. These are self-produced checkpoints (train_lstm.py),
                # not third-party weights, so trusting them here is safe —
                # matches the other torch.load call just above.
                ckpt = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
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
            self._clear_dwell_hold_if_any(rec.step_id)
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
            self._fire_race("step_skipped", {"expected_id": expected_id, "observed_id": observed_id,
                                             "message": f"Step {expected_id} skipped (observed {observed_id})"})

        def on_out_of_sequence(expected_id: int, observed_id: int):
            self.exp_logger.log_out_of_sequence(expected_id, observed_id)
            self.voice.alert_out_of_sequence(expected_id)
            if self.gui_queue:
                self.gui_queue.put_nowait(("out_of_sequence", expected_id))
            self._fire_race("out_of_sequence", {"expected_id": expected_id, "observed_id": observed_id,
                                                "message": f"Out of sequence: expected {expected_id}, "
                                                          f"observed {observed_id}"})

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

    # ── Latency-race demo (display/logging only) ────────────────────────────

    def _fire_race(self, kind: str, payload: dict):
        """Record that the LOCAL system just alerted on `kind`, and (only
        when race mode is on) schedule the delayed 'Earth found out' replay.
        Pushing the local event to the GUI happens unconditionally so the
        Latency Race tab can show it even before the Earth copy arrives.

        Also the single choke point for every anomaly-ish event in this
        pipeline (step_skipped, out_of_sequence, uncertain, typed A4/A5/A7 —
        see the four call sites), so it's the natural place to trigger the
        smart clip uplink too, independent of whether race mode is on."""
        self._maybe_clip_trigger(kind, payload.get("message", kind))
        if not self.enable_race:
            return
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("race_local", {"kind": kind, **payload}))
            except queue.Full:
                pass
        if self.earth_channel is not None:
            self.earth_channel.fire(kind, payload)

    def _on_earth_delivered(self, ev):
        """EarthDelayChannel's on_deliver callback — fires on a background
        Timer thread after `earth_delay_s`. Display/logging only; never
        touches state_machine."""
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("race_earth", {
                    "kind": ev.kind, **ev.payload,
                    "delay_s": round(ev.earth_deliver_time - ev.local_fire_time, 2),
                }))
            except queue.Full:
                pass

    # ── Typed anomalies (A4 forbidden zone / A5 dwell / A7 occlusion) ────────

    def _run_anomaly_checks(self, detections: list) -> None:
        """Run pipeline.anomaly_monitor's per-frame checks and reconcile the
        results with the state machine's external_holds. See
        AnomalyMonitor.step()'s docstring for the (code, active, severity,
        message, extra) tuple shape."""
        sm = self.state_machine
        events = self.anomaly_monitor.step(detections, sm.current_step,
                                           self._current_step_status())
        for code, active, severity, message, extra in events:
            self._sync_anomaly_event(code, active, severity, message, extra)

    def _current_step_status(self) -> Optional[str]:
        sm = self.state_machine
        if sm.current_step is None:
            return None
        step_id = sm.current_step["id"]
        for rec in sm.step_records:
            if rec.step_id == step_id:
                return rec.status.name
        return None

    def _sync_anomaly_event(self, code: str, active: bool, severity: str,
                            message: str, extra: dict) -> None:
        if severity == "cleared":
            self.state_machine.clear_hold(code)
            self.exp_logger.log_anomaly(code, "cleared", message, **extra)
            if self.gui_queue:
                try:
                    self.gui_queue.put_nowait(("anomaly_cleared", {"code": code, "message": message}))
                except queue.Full:
                    pass
            return

        # "hold" and "abstain" both stop the FSM from acting (see
        # ExperimentStateMachine.force_hold's docstring); "soft" is an
        # alert-only warning shot before that.
        self.exp_logger.log_anomaly(code, severity, message, **extra)
        self.voice.alert_anomaly(code, message, priority=(severity != "soft"))
        if active and severity in ("hold", "abstain"):
            self.state_machine.force_hold(code, message)
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("anomaly", {
                    "code": code, "severity": severity, "message": message, **extra}))
            except queue.Full:
                pass
        self._fire_race(f"anomaly_{code}", {"code": code, "severity": severity, "message": message})

    def _clear_dwell_hold_if_any(self, step_id: int) -> None:
        cleared = self.anomaly_monitor.clear_dwell_hold(step_id)
        if cleared:
            self._sync_anomaly_event(*cleared)

    # ── Smart clip uplink (Brief §12) ────────────────────────────────────────

    def _feed_clip_uplink(self, frame: np.ndarray) -> None:
        """Call once per processed frame (both run() and process_frame()).
        No-op when uplink_mode is plain "stream" (self.clip_uplink is None —
        zero cost)."""
        if self.clip_uplink is None:
            return
        finished = self.clip_uplink.feed(frame)
        if finished is not None:
            self._on_clip_saved(finished)

    def _maybe_clip_trigger(self, kind: str, message: str) -> None:
        """Called from _fire_race — the same choke point every anomaly-ish
        event in this pipeline already flows through (step_skipped,
        out_of_sequence, uncertain, typed A4/A5/A7). No-op in "stream" mode."""
        if self.clip_uplink is None:
            return
        self.clip_uplink.trigger(kind, message)

    def _on_clip_saved(self, rec) -> None:
        self.exp_logger.log_clip_saved(rec.code, rec.path, rec.frames, rec.duration_s,
                                       rec.size_bytes)
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(("clip_saved", {
                    "code": rec.code, "message": rec.message, "path": rec.path,
                    "frames": rec.frames, "duration_s": rec.duration_s,
                    "size_bytes": rec.size_bytes,
                }))
            except queue.Full:
                pass

    # ── Per-Frame Processing (Optimized) ─────────────────────────────────────

    def _extract_skeleton_features_optimized(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Run the custom PoseNet and return the 132-dim feature vector."""
        if self.pose_wrapper is None:
            return np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        return self.pose_wrapper.process(frame_rgb)

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
        self._fire_race("uncertain", {"expected_id": expected, "lstm_step": lstm_step,
                                      "cnn_step": cnn_step,
                                      "message": f"Model disagreement near step {expected} "
                                                f"(LSTM={lstm_step}, CNN={cnn_step})"})

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
        Process a single BGR frame. Always runs HSV + PoseNet for real latency.
        If injected_skel is provided it is used for LSTM (oracle pose) so accuracy
        can be measured independently of the pose net on synthetic figures.
        """
        t0 = time.perf_counter()

        if frame is None or frame.size == 0:
            return {
                "pred_step": 0, "confidence": 0.0, "detections": [], "vis": None,
                "timings_ms": {"hsv": 0.0, "pose": 0.0, "lstm": 0.0, "total": 0.0},
            }

        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        self.frame_idx += 1
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._frame_rgb_full)

        t_hsv = time.perf_counter()
        detections = self.hsv_detector.detect(frame)
        hsv_ms = (time.perf_counter() - t_hsv) * 1000.0

        t_pose = time.perf_counter()
        if self.hmr is not None and self.hmr.available:
            # Stage 2: root-relative 3D from the HMR mesh (same 132-dim contract).
            skel_pose, _ = self.hmr.get_pose_features(self._frame_rgb_full)
        else:
            skel_pose = self._extract_skeleton_features_optimized(self._frame_rgb_full)
        pose_ms = (time.perf_counter() - t_pose) * 1000.0

        skel = injected_skel if injected_skel is not None else skel_pose
        if self.rack_normalizer is not None:
            skel = self.rack_normalizer.normalize(
                skel, rack_rect=pick_rack_rect(detections)
            )
        self.skeleton_buffer.append(np.asarray(skel, dtype=np.float32).reshape(-1))
        if self.enable_cnn_ensemble:
            self._append_cnn_frame(self._frame_rgb_full)

        self._run_anomaly_checks(detections)
        self._feed_clip_uplink(frame)

        t_lstm = time.perf_counter()
        pred_step, pred_conf = self._last_pred
        if len(self.skeleton_buffer) >= SEQUENCE_WINDOW:
            pred_step, pred_conf = self._predict_and_feed_state_machine()
            self._last_pred = (pred_step, pred_conf)
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
                "pose": float(pose_ms),
                "lstm": float(lstm_ms),
                "total": float(total_ms),
            },
        }

    def reset_runtime(self):
        """Clear buffers and state machine for a new simulation trial."""
        self.frame_idx = 0
        self._last_pred = (0, 0.0)
        self.skeleton_buffer.clear()
        self.frame_stack_buffer.clear()
        self._times.clear()
        if self.rack_normalizer is not None:
            self.rack_normalizer.reset()  # re-latch rack polarity for the new trial
        self.anomaly_monitor.reset()
        if self.clip_uplink is not None:
            self.clip_uplink.reset()
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
        summary["uplink_mode"] = self.uplink_mode
        if self.clip_uplink is not None:
            summary["clip_uplink"] = self.clip_uplink.summary()
        try:
            self.gui_queue.put_nowait(("status", summary))
        except queue.Full:
            pass

    def close(self):
        self._running = False
        if self.clip_uplink is not None:
            finished = self.clip_uplink.close()  # flush a still-pending clip
            if finished is not None:
                self._on_clip_saved(finished)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.streamer is not None:
            self.streamer.close()
            self.streamer = None
        if self.pose_wrapper is not None:
            self.pose_wrapper.close()
            self.pose_wrapper = None

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

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

        pred_step, pred_conf = 0, 0.0   # Last prediction (persisted between frames)

        while self._running:
            ret, frame = cap.read()
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
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

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

            # ── Parallel or sequential inference ─────────────
            if self.threaded:
                detections, skel = self.threaded.run_parallel(
                    frame, self._frame_rgb_full,
                    self.hsv_detector, self.pose_wrapper,
                )
                hsv_ms, pose_ms = self.threaded.last_hsv_ms, self.threaded.last_pose_ms
            else:
                t_hsv = time.perf_counter()
                detections = self.hsv_detector.detect(frame)
                hsv_ms = (time.perf_counter() - t_hsv) * 1000.0
                t_pose = time.perf_counter()
                skel = self._extract_skeleton_features_optimized(self._frame_rgb_full)
                pose_ms = (time.perf_counter() - t_pose) * 1000.0

            if self.rack_normalizer is not None:
                skel = self.rack_normalizer.normalize(
                    skel, rack_rect=pick_rack_rect(detections)
                )
            self.skeleton_buffer.append(skel)
            if self.enable_cnn_ensemble:
                self._append_cnn_frame(self._frame_rgb_full)

            self._run_anomaly_checks(detections)
            self._feed_clip_uplink(frame)

            # ── LSTM (+ optional CNN ensemble) inference ─────
            t_lstm = time.perf_counter()
            if len(self.skeleton_buffer) >= SEQUENCE_WINDOW:
                pred_step, pred_conf = self._predict_and_feed_state_machine()
            lstm_ms = (time.perf_counter() - t_lstm) * 1000.0

            # ── Build annotated frame & push to GUI ──────────
            vis = self._annotate_frame(frame, detections, pred_step, pred_conf)
            total_ms = (time.time() - t_start) * 1000.0

            if self.gui_queue:
                try:
                    self.gui_queue.put_nowait(("frame", vis))
                except queue.Full:
                    pass
                try:
                    self.gui_queue.put_nowait(("timings", {
                        "hsv": hsv_ms, "pose": pose_ms, "lstm": lstm_ms, "total": total_ms,
                    }))
                    self.gui_queue.put_nowait(("detections", [
                        {"label": d.label, "confidence": round(float(d.confidence), 3),
                         "bbox": d.bbox, "centroid": d.centroid}
                        for d in detections
                    ]))
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
        if self.clip_uplink is not None:
            finished = self.clip_uplink.close()  # flush a still-pending clip
            if finished is not None:
                self._on_clip_saved(finished)
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.streamer:
            self.streamer.close()
            self.streamer = None
        if self.pose_wrapper:
            self.pose_wrapper.close()
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
