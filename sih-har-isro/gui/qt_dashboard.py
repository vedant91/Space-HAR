"""
PyQt6 Dashboard — live monitoring GUI for the ISRO HAR pipeline
=================================================================
Consumes the gui_queue events emitted by pipeline.har_pipeline.HARPipeline:
  ("frame", frame_bgr)            - latest (annotated) video frame
  ("status", state_summary)       - ExperimentStateMachine.get_status_summary()
  ("step_complete", rec)
  ("step_skipped", expected_id)
  ("out_of_sequence", expected_id)
  ("step_recovered", step_id)
  ("experiment_complete", summary)
  ("uncertain", {"expected_step_id", "lstm_step", "lstm_confidence", "cnn_step", "cnn_confidence"})

Renders live video, a step checklist colored by status, current/next step,
elapsed time, and a transient alert banner for skip/out-of-sequence/recovery
events. Local-recording and network-stream targets are shown in the footer
so an operator can see at a glance whether both required outputs (local
file + IP stream) are active.

This module is optional — `gui/dashboard.py` imports it lazily and falls
back to a headless queue-drain if PyQt6 isn't installed.
"""

from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import EXPERIMENT_STEPS, STREAM_HOST, STREAM_PORT

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QVBoxLayout, QWidget,
)

STATUS_COLORS = {
    "PENDING":     "#3a3a3a",
    "IN_PROGRESS": "#c98a12",
    "COMPLETED":   "#1f8a3b",
    "SKIPPED":     "#a83232",
    "ERROR":       "#a83232",
}


class Dashboard(QMainWindow):
    def __init__(self, gui_queue: "queue.Queue", enable_streaming: bool = False,
                 stream_host: str = STREAM_HOST, stream_port: int = STREAM_PORT):
        super().__init__()
        self.gui_queue = gui_queue
        self._on_close = None
        self.setWindowTitle("ISRO HAR — Experiment Monitor")
        self.resize(1180, 640)

        self.video_label = QLabel("Waiting for video…")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#888;")
        self.video_label.setMinimumSize(800, 450)

        self.current_label = QLabel("Current: —")
        self.next_label = QLabel("Next: —")
        self.elapsed_label = QLabel("Elapsed: 0.0s")
        for lbl in (self.current_label, self.next_label, self.elapsed_label):
            lbl.setStyleSheet("font-size:14px;")

        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet("font-size:14px; padding:6px; border-radius:4px;")
        self.alert_label.setVisible(False)

        self.step_list = QListWidget()
        self.step_items = {}
        for step in EXPERIMENT_STEPS:
            item = QListWidgetItem()
            self.step_list.addItem(item)
            self.step_items[step["id"]] = item
        self._recolor_steps({})

        # Placeholder text until the first ("status", ...) event arrives and
        # _on_status() replaces it with live recording/streaming health —
        # a footer set once here and never updated couldn't reflect a stream
        # that dies mid-run.
        self._stream_target_text = (f"udp://{stream_host}:{stream_port}"
                                    if enable_streaming else None)
        self.footer_label = QLabel("Local recording: —   |   Network stream: —")
        self.footer_label.setStyleSheet("color:#999; font-size:11px;")

        left = QVBoxLayout()
        left.addWidget(self.video_label, 1)
        left.addWidget(self.footer_label)

        right = QVBoxLayout()
        right.addWidget(self.current_label)
        right.addWidget(self.next_label)
        right.addWidget(self.elapsed_label)
        right.addWidget(self.alert_label)
        right.addWidget(self.step_list, 1)

        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        right_w.setMaximumWidth(320)

        root = QHBoxLayout()
        root.addWidget(left_w, 1)
        root.addWidget(right_w)
        central = QWidget(); central.setLayout(root)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._drain_queue)
        self.timer.start(33)

    def set_on_close(self, callback):
        """Callback invoked once when the window is closed, e.g. to stop the
        pipeline's capture loop cleanly instead of leaving it running."""
        self._on_close = callback

    def closeEvent(self, event):
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        super().closeEvent(event)

    def _drain_queue(self):
        # Bounded per tick so a burst of frames can't stall the Qt event loop.
        # The get *and* the dispatch both live in one try/except: previously
        # only get_nowait() was guarded, so an exception raised inside any of
        # the dispatch handlers (e.g. a malformed payload) was fully
        # uncaught — PyQt6's default policy for an exception escaping a
        # slot/timer callback is to abort the whole application, not just
        # misrender one frame.
        for _ in range(10):
            try:
                kind, payload = self.gui_queue.get_nowait()
                if kind == "frame":
                    self._on_frame(payload)
                elif kind == "status":
                    self._on_status(payload)
                elif kind == "step_skipped":
                    self._show_alert(f"⚠ Step {payload} skipped — awaiting correction", "#a83232")
                elif kind == "out_of_sequence":
                    self._show_alert(f"⚠ Out of sequence — expected step {payload}", "#a83232")
                elif kind == "step_recovered":
                    self._show_alert(f"Step {payload} recovered", "#1f8a3b")
                elif kind == "experiment_complete":
                    self._show_alert("Experiment complete", "#1f8a3b")
                elif kind == "uncertain":
                    self._show_alert(
                        f"⚠ Uncertain near step {payload.get('expected_step_id')} — "
                        f"LSTM says {payload.get('lstm_step')}, CNN says {payload.get('cnn_step')}. "
                        f"Please confirm.", "#c98a12")
            except queue.Empty:
                return
            except Exception:
                # The bad event is already consumed from the queue — log and
                # keep draining the rest rather than losing the whole tick.
                logger.exception("Error handling gui_queue event")

    def _on_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _on_status(self, summary: dict):
        cur = summary.get("current_step")
        nxt = summary.get("next_step")
        self.current_label.setText(
            f"Current: Step {cur['id']}: {cur['name']}" if cur else "Current: COMPLETE")
        self.next_label.setText(
            f"Next: Step {nxt['id']}: {nxt['name']}" if nxt else "Next: —")
        self.elapsed_label.setText(f"Elapsed: {summary.get('elapsed_sec', 0.0):.1f}s")

        recovery = summary.get("recovery") or {}
        if recovery.get("active"):
            self._show_alert(
                f"⚠ HOLD — expected step {recovery['expected_step_id']}, "
                f"observed step {recovery['observed_step_id']}", "#a83232", sticky=True)
        elif self.alert_label.property("sticky"):
            self._clear_alert()

        self._recolor_steps({s["id"]: s["status"] for s in summary.get("steps", [])})

        rec_text = "Local recording: active" if summary.get("recording_active") else "Local recording: off"
        if summary.get("streaming_active"):
            stream_text = f"Network stream: {summary.get('stream_target') or self._stream_target_text}"
        elif self._stream_target_text:
            stream_text = "Network stream: down"  # was configured but isn't alive right now
        else:
            stream_text = "Network stream: disabled"
        self.footer_label.setText(f"{rec_text}   |   {stream_text}")

    def _recolor_steps(self, status_by_id: dict):
        for step_id, item in self.step_items.items():
            status = status_by_id.get(step_id, "PENDING")
            item.setBackground(QColor(STATUS_COLORS.get(status, "#3a3a3a")))
            item.setForeground(QColor("white"))
            item.setText(f"{step_id}. {self._name_of(step_id)}  [{status}]")

    @staticmethod
    def _name_of(step_id: int) -> str:
        for s in EXPERIMENT_STEPS:
            if s["id"] == step_id:
                return s["name"]
        return "?"

    def _show_alert(self, text: str, color: str, sticky: bool = False):
        self.alert_label.setText(text)
        self.alert_label.setStyleSheet(
            f"font-size:14px; padding:6px; border-radius:4px; background:{color}; color:white;")
        self.alert_label.setProperty("sticky", sticky)
        self.alert_label.setVisible(True)

    def _clear_alert(self):
        self.alert_label.setVisible(False)
        self.alert_label.setProperty("sticky", False)


def launch_qt_dashboard(gui_queue: "queue.Queue", on_close=None,
                        enable_streaming: bool = False,
                        stream_host: str = STREAM_HOST, stream_port: int = STREAM_PORT):
    """Blocking call — runs the Qt event loop. Must be called from the main
    thread (Qt/Cocoa requires the GUI event loop own the main thread); run
    the pipeline's capture loop in a background thread instead — see
    main.py's run_pipeline()."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = Dashboard(gui_queue, enable_streaming=enable_streaming,
                       stream_host=stream_host, stream_port=stream_port)
    if on_close:
        window.set_on_close(on_close)
    window.show()
    app.exec()
