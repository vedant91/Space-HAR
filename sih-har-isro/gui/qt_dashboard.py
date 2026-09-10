"""
PyQt6 Dashboard — detailed live monitoring + training console
=================================================================
Consumes the gui_queue events emitted by pipeline.har_pipeline.HARPipeline:
  ("frame", frame_bgr)            - latest (annotated) video frame
  ("status", state_summary)       - ExperimentStateMachine.get_status_summary()
  ("timings", {"hsv","pose","lstm","total"})  - per-frame latency breakdown (ms)
  ("detections", [{"label","confidence","bbox","centroid"}, ...])
  ("model_info", {...})           - loaded model backends, pushed once at startup
  ("step_complete", rec)
  ("step_skipped", expected_id)
  ("out_of_sequence", expected_id)
  ("step_recovered", step_id)
  ("experiment_complete", summary)
  ("uncertain", {"expected_step_id", "lstm_step", "lstm_confidence", "cnn_step", "cnn_confidence"})

Six tabs:
  1. Live Monitor       — video feed, step checklist, alerts, recording/stream footer
  2. Pipeline Internals — latency breakdown + rolling FPS chart, backend info
  3. Detections         — live HSV+hand detection table
  4. Model & Dataset     — checkpoint metrics, dataset sizes (synthetic vs real)
  5. Training Console    — launch train/e2e/autolabel/posenet as a live subprocess
  6. Logs                — tail the newest experiment log

This module is optional — `gui/dashboard.py` imports it lazily and falls
back to a headless queue-drain if PyQt6 isn't installed.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    EXPERIMENT_STEPS, STREAM_HOST, STREAM_PORT, REAL_VIDEO_DIR, REAL_PSEUDO_DIR,
    REAL_ANNOTATED_DIR, LOG_DIR,
)

from PyQt6.QtCore import Qt, QTimer, QProcess
from PyQt6.QtGui import QColor, QImage, QPixmap, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton, QSizePolicy,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib not available — Pipeline Internals tab will show text-only stats.")

STATUS_COLORS = {
    "PENDING":     "#3a3a3a",
    "IN_PROGRESS": "#c98a12",
    "COMPLETED":   "#1f8a3b",
    "SKIPPED":     "#a83232",
    "ERROR":       "#a83232",
}
MONO_FONT = QFont("Menlo, Consolas, monospace")
_PROJECT_ROOT = Path(__file__).parent.parent


def _fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


class Dashboard(QMainWindow):
    def __init__(self, gui_queue: "queue.Queue", enable_streaming: bool = False,
                 stream_host: str = STREAM_HOST, stream_port: int = STREAM_PORT):
        super().__init__()
        self.gui_queue = gui_queue
        self._on_close = None
        self.setWindowTitle("ISRO HAR — Experiment Monitor")
        self.resize(1400, 820)

        self._stream_target_text = (f"udp://{stream_host}:{stream_port}"
                                    if enable_streaming else None)

        # Rolling history for the Pipeline Internals charts.
        self._latency_history = deque(maxlen=150)   # total_ms per processed frame
        self._breakdown_history = {"hsv": deque(maxlen=150), "pose": deque(maxlen=150),
                                   "lstm": deque(maxlen=150)}
        self._last_timings = {"hsv": 0.0, "pose": 0.0, "lstm": 0.0, "total": 0.0}
        self._model_info = {}
        self._training_proc: Optional[QProcess] = None
        self._log_file_pos = 0

        self._race_events: list = []  # [{"kind","local_t","earth_t","delay_s","message"}]

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Live Monitor")
        self.tabs.addTab(self._build_internals_tab(), "Pipeline Internals")
        self.tabs.addTab(self._build_detections_tab(), "Detections")
        self.tabs.addTab(self._build_race_tab(), "Latency Race")
        self.tabs.addTab(self._build_model_tab(), "Model && Dataset")
        self.tabs.addTab(self._build_training_tab(), "Training Console")
        self.tabs.addTab(self._build_logs_tab(), "Logs")
        self.setCentralWidget(self.tabs)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._drain_queue)
        self.timer.start(33)

        self._chart_timer = QTimer(self)
        self._chart_timer.timeout.connect(self._refresh_charts)
        self._chart_timer.start(500)

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._refresh_log_tail)
        self._log_timer.start(1000)

        self._refresh_model_dataset_info()

    def set_on_close(self, callback):
        """Callback invoked once when the window is closed, e.g. to stop the
        pipeline's capture loop cleanly instead of leaving it running."""
        self._on_close = callback

    def closeEvent(self, event):
        if self._training_proc is not None and self._training_proc.state() != QProcess.ProcessState.NotRunning:
            self._training_proc.kill()
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        super().closeEvent(event)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Live Monitor
    # ══════════════════════════════════════════════════════════════════════

    def _build_live_tab(self) -> QWidget:
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
        right.addWidget(QLabel("<b>Step checklist</b>"))
        right.addWidget(self.step_list, 1)

        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        right_w.setMaximumWidth(340)

        root = QHBoxLayout()
        root.addWidget(left_w, 1)
        root.addWidget(right_w)
        w = QWidget(); w.setLayout(root)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Pipeline Internals
    # ══════════════════════════════════════════════════════════════════════

    def _build_internals_tab(self) -> QWidget:
        root = QVBoxLayout()

        info_box = QGroupBox("Model backends")
        info_layout = QVBoxLayout()
        self.backend_label = QLabel("Waiting for pipeline to start…")
        self.backend_label.setFont(MONO_FONT)
        info_layout.addWidget(self.backend_label)
        info_box.setLayout(info_layout)
        root.addWidget(info_box)

        stats_box = QGroupBox("Live performance")
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: —")
        self.latency_label = QLabel("Total latency: —")
        self.breakdown_label = QLabel("HSV: —   Pose: —   LSTM: —")
        for lbl in (self.fps_label, self.latency_label, self.breakdown_label):
            lbl.setFont(MONO_FONT)
            stats_layout.addWidget(lbl)
        stats_box.setLayout(stats_layout)
        root.addWidget(stats_box)

        if MPL_AVAILABLE:
            self._fig = Figure(figsize=(8, 4), tight_layout=True)
            self._ax_bar = self._fig.add_subplot(1, 2, 1)
            self._ax_line = self._fig.add_subplot(1, 2, 2)
            self._canvas = FigureCanvas(self._fig)
            root.addWidget(self._canvas, 1)
        else:
            root.addWidget(QLabel("matplotlib not installed — install it for latency charts "
                                  "(pip install matplotlib)."), 1)

        w = QWidget(); w.setLayout(root)
        return w

    def _refresh_charts(self):
        t = self._last_timings
        self.fps_label.setText(f"FPS: {1000.0 / max(t['total'], 0.01):.1f}")
        self.latency_label.setText(f"Total latency: {t['total']:.1f} ms")
        self.breakdown_label.setText(
            f"HSV: {t['hsv']:.1f}ms   Pose: {t['pose']:.1f}ms   LSTM: {t['lstm']:.1f}ms")

        if not MPL_AVAILABLE or not self._latency_history:
            return
        self._ax_bar.clear()
        stages = ["hsv", "pose", "lstm"]
        vals = [t[s] for s in stages]
        colors = ["#4fa3d1", "#7fbf7f", "#d1a34f"]
        self._ax_bar.bar(stages, vals, color=colors)
        self._ax_bar.set_title("Latest per-stage latency (ms)")
        self._ax_bar.set_ylabel("ms")

        self._ax_line.clear()
        self._ax_line.plot(list(self._latency_history), color="#7fbf7f", linewidth=1.2)
        self._ax_line.set_title(f"Total latency, last {len(self._latency_history)} frames")
        self._ax_line.set_ylabel("ms")
        self._ax_line.axhline(y=sum(self._latency_history) / len(self._latency_history),
                              color="#888", linestyle="--", linewidth=0.8)
        self._canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Detections
    # ══════════════════════════════════════════════════════════════════════

    def _build_detections_tab(self) -> QWidget:
        root = QVBoxLayout()
        self.detection_summary_label = QLabel("No detections yet.")
        root.addWidget(self.detection_summary_label)

        self.detection_table = QTableWidget(0, 4)
        self.detection_table.setHorizontalHeaderLabels(["Label", "Confidence", "BBox (x1,y1,x2,y2)", "Centroid"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.detection_table, 1)

        w = QWidget(); w.setLayout(root)
        return w

    def _on_detections(self, dets: list):
        labels_present = {d["label"] for d in dets}
        parts = []
        for name in ("main_box", "red_box", "yellow_box", "hand"):
            n = sum(1 for d in dets if d["label"] == name)
            parts.append(f"{name}: {'✓×' + str(n) if n else '—'}")
        self.detection_summary_label.setText("   ".join(parts))

        self.detection_table.setRowCount(len(dets))
        for row, d in enumerate(dets):
            self.detection_table.setItem(row, 0, QTableWidgetItem(str(d["label"])))
            self.detection_table.setItem(row, 1, QTableWidgetItem(f"{d['confidence']:.2f}"))
            self.detection_table.setItem(row, 2, QTableWidgetItem(str(d["bbox"])))
            self.detection_table.setItem(row, 3, QTableWidgetItem(str(d["centroid"])))

    # ══════════════════════════════════════════════════════════════════════
    # Tab — Latency Race
    # ══════════════════════════════════════════════════════════════════════

    def _build_race_tab(self) -> QWidget:
        root = QVBoxLayout()
        root.addWidget(QLabel(
            "<b>Onboard alert vs. simulated delayed-Earth alert</b> — same anomaly, two "
            "channels. The Earth column is a display-only replay (see pipeline/"
            "earth_delay_channel.py); it never affects what the onboard system does."))

        self.race_banner = QLabel("Waiting for a race run (python main.py --mode race)...")
        self.race_banner.setStyleSheet(
            "font-size:15px; font-weight:bold; padding:8px; border-radius:4px; "
            "background:#1f3a1f; color:#9be89b;")
        root.addWidget(self.race_banner)

        cols = QHBoxLayout()
        local_box = QVBoxLayout()
        local_box.addWidget(QLabel("<b>LOCAL (onboard) — fires immediately</b>"))
        self.race_local_list = QListWidget()
        local_box.addWidget(self.race_local_list, 1)
        local_w = QWidget(); local_w.setLayout(local_box)

        earth_box = QVBoxLayout()
        earth_box.addWidget(QLabel("<b>EARTH (simulated delay) — arrives late</b>"))
        self.race_earth_list = QListWidget()
        earth_box.addWidget(self.race_earth_list, 1)
        earth_w = QWidget(); earth_w.setLayout(earth_box)

        cols.addWidget(local_w)
        cols.addWidget(earth_w)
        cols_w = QWidget(); cols_w.setLayout(cols)
        root.addWidget(cols_w, 1)

        w = QWidget(); w.setLayout(root)
        return w

    def _on_race_local(self, payload: dict):
        t = time.time()
        msg = payload.get("message", payload.get("kind", "anomaly"))
        item = QListWidgetItem(f"[{time.strftime('%H:%M:%S', time.localtime(t))}] {msg}")
        item.setForeground(QColor("#9be89b"))
        self.race_local_list.addItem(item)
        self.race_local_list.scrollToBottom()
        self._race_events.append({"kind": payload.get("kind"), "message": msg,
                                  "local_wall_time": t, "earth_wall_time": None})
        self.race_banner.setText(f"LOCAL has fired {self.race_local_list.count()} alert(s) — "
                                 f"Earth has {self.race_earth_list.count()} pending/delivered so far.")

    def _on_race_earth(self, payload: dict):
        t = time.time()
        msg = payload.get("message", payload.get("kind", "anomaly"))
        delay = payload.get("delay_s")
        item = QListWidgetItem(f"[{time.strftime('%H:%M:%S', time.localtime(t))}] "
                               f"(+{delay:.1f}s late) {msg}" if delay is not None else msg)
        item.setForeground(QColor("#e89b9b"))
        self.race_earth_list.addItem(item)
        self.race_earth_list.scrollToBottom()
        for ev in reversed(self._race_events):
            if ev["kind"] == payload.get("kind") and ev["earth_wall_time"] is None:
                ev["earth_wall_time"] = t
                break
        self.race_banner.setText(
            f"LOCAL WINS by {delay:.1f}s on the last event — onboard already handled it while "
            f"Earth was still finding out." if delay is not None else self.race_banner.text())

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Model & Dataset info
    # ══════════════════════════════════════════════════════════════════════

    def _build_model_tab(self) -> QWidget:
        root = QVBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_model_dataset_info)
        root.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setFont(MONO_FONT)
        root.addWidget(self.model_info_text, 1)

        w = QWidget(); w.setLayout(root)
        return w

    def _refresh_model_dataset_info(self):
        try:
            from config.experiment_config import (
                PROCEDURE_PACK_ID, PROCEDURE_PACK_NAME, PROCEDURE_PACK_PATH, EXPERIMENT_STEPS,
            )
            lines = [f"=== Procedure pack: {PROCEDURE_PACK_ID} — \"{PROCEDURE_PACK_NAME}\" ===", ""]
            lines.append(f"Source: {PROCEDURE_PACK_PATH}")
            for s in EXPERIMENT_STEPS:
                lines.append(f"  {s['id']}. {s['name']}  (needs: {', '.join(s['required_objects']) or '—'})")
            lines += ["", "=== Models (all trained from scratch — no pretrained/open-source weights) ===", ""]
        except Exception as e:
            lines = [f"(failed to read active procedure pack: {e})", "",
                    "=== Models (all trained from scratch — no pretrained/open-source weights) ===", ""]

        def _ckpt_line(name, path, fields):
            p = _PROJECT_ROOT / path
            if not p.exists():
                lines.append(f"{name}: NOT TRAINED YET ({path})")
                return
            try:
                import torch
                ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
                info = "  ".join(f"{f}={ckpt.get(f)}" for f in fields if f in ckpt)
                lines.append(f"{name}: {info}")
            except Exception as e:
                lines.append(f"{name}: (failed to read checkpoint: {e})")

        _ckpt_line("PoseNet", "models/pose_net.pt",
                  ["architecture", "stage", "val_pck", "real_finetune_val_loss", "epoch"])
        _ckpt_line("CNN", "models/activity_cnn.pt",
                  ["architecture", "val_acc", "epoch", "in_channels", "num_classes"])
        _ckpt_line("LSTM", "models/lstm_classifier.pt",
                  ["hidden_size", "num_layers", "num_classes", "val_acc", "epoch"])

        lines += ["", "=== Datasets ===", ""]

        def _meta_line(name, path):
            p = _PROJECT_ROOT / path
            if not p.exists():
                lines.append(f"{name}: (none) — {path}")
                return
            try:
                meta = json.loads(p.read_text())
                lines.append(f"{name}: total_windows={meta.get('total_windows')} "
                             f"source={meta.get('source', meta.get('posenet_backend', ''))}")
            except Exception as e:
                lines.append(f"{name}: (failed to read metadata: {e})")

        _meta_line("Synthetic sequences", "dataset/skeleton_sequences/metadata.json")
        _meta_line("Real sequences", "dataset/real_sequences/metadata.json")
        _meta_line("Combined sequences", "dataset/skeleton_sequences_combined/metadata.json")

        def _count_frames(path):
            p = _PROJECT_ROOT / path
            if not p.exists():
                return 0
            return sum(1 for _ in p.rglob("*.jpg"))

        lines.append(f"Synthetic annotated frames: {_count_frames('dataset/annotated')}")
        lines.append(f"Real annotated frames:      {_count_frames(REAL_ANNOTATED_DIR)}")

        lines += ["", "=== Real 'gravitational mimic' source videos ===", ""]
        real_dir = Path(REAL_VIDEO_DIR)
        if real_dir.exists():
            videos = sorted(list(real_dir.glob("*.mp4")) + list(real_dir.glob("*.mov")))
            if videos:
                for v in videos:
                    lines.append(f"  {v.name}")
            else:
                lines.append("  (no video files found)")
        else:
            lines.append(f"  (directory not found: {real_dir})")

        report_path = Path(REAL_PSEUDO_DIR) / "autolabel_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                lines += ["", "=== Real-video pseudo-label summary ==="]
                lines.append(f"  {report.get('note', '')}")
                for v in report.get("videos", []):
                    if "error" in v:
                        lines.append(f"  {Path(v['video']).name}: ERROR — {v['error']}")
                        continue
                    lines.append(f"  {Path(v['video']).name}: frames={v.get('n_frames_saved')} "
                                 f"unknown={v.get('frac_unknown', 0):.0%} "
                                 f"wrist_rate={v.get('wrist_detection_rate', 0):.0%}")
            except Exception:
                pass

        self.model_info_text.setPlainText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Training Console
    # ══════════════════════════════════════════════════════════════════════

    def _build_training_tab(self) -> QWidget:
        root = QVBoxLayout()

        btn_row = QHBoxLayout()
        actions = [
            ("Train All (PoseNet+CNN+LSTM)", ["--mode", "train"]),
            ("Train PoseNet only", ["--mode", "posenet"]),
            ("PoseNet + real fine-tune", ["--mode", "posenet", "--finetune-real"]),
            ("Auto-label real videos", ["--mode", "autolabel"]),
            ("Run E2E Loop", ["--mode", "e2e"]),
        ]
        for label, cmd_args in actions:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked, a=cmd_args: self._run_training_command(a))
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        stop_row = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training_command)
        self.training_status_label = QLabel("Idle.")
        stop_row.addWidget(self.stop_btn)
        stop_row.addWidget(self.training_status_label, 1)
        root.addLayout(stop_row)

        self.training_output = QTextEdit()
        self.training_output.setReadOnly(True)
        self.training_output.setFont(MONO_FONT)
        self.training_output.setStyleSheet("background:#111; color:#ddd;")
        root.addWidget(self.training_output, 1)

        w = QWidget(); w.setLayout(root)
        return w

    def _run_training_command(self, cmd_args: list):
        if self._training_proc is not None and self._training_proc.state() != QProcess.ProcessState.NotRunning:
            self.training_output.append("[dashboard] A training command is already running — stop it first.\n")
            return
        self.training_output.append(f"[dashboard] $ {sys.executable} main.py {' '.join(cmd_args)}\n")
        proc = QProcess(self)
        proc.setWorkingDirectory(str(_PROJECT_ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_training_output(proc))
        proc.finished.connect(self._on_training_finished)
        proc.start(sys.executable, ["main.py", *cmd_args])
        self._training_proc = proc
        self.stop_btn.setEnabled(True)
        self.training_status_label.setText(f"Running: {' '.join(cmd_args)}")

    def _on_training_output(self, proc: QProcess):
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.training_output.moveCursor(QTextCursor.MoveOperation.End)
        self.training_output.insertPlainText(data)
        self.training_output.moveCursor(QTextCursor.MoveOperation.End)

    def _on_training_finished(self, exit_code, _exit_status):
        self.training_output.append(f"\n[dashboard] process finished, exit_code={exit_code}\n")
        self.training_status_label.setText(f"Idle (last run exit_code={exit_code}).")
        self.stop_btn.setEnabled(False)
        self._refresh_model_dataset_info()

    def _stop_training_command(self):
        if self._training_proc is not None:
            self._training_proc.kill()
            self.training_output.append("\n[dashboard] killed by user.\n")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 6 — Logs
    # ══════════════════════════════════════════════════════════════════════

    def _build_logs_tab(self) -> QWidget:
        root = QVBoxLayout()
        row = QHBoxLayout()
        self.log_combo = QComboBox()
        refresh_btn = QPushButton("Refresh file list")
        refresh_btn.clicked.connect(self._refresh_log_list)
        row.addWidget(QLabel("Log file:"))
        row.addWidget(self.log_combo, 1)
        row.addWidget(refresh_btn)
        root.addLayout(row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(MONO_FONT)
        root.addWidget(self.log_text, 1)

        w = QWidget(); w.setLayout(root)
        self._refresh_log_list()
        return w

    def _refresh_log_list(self):
        log_dir = _PROJECT_ROOT / LOG_DIR
        current = self.log_combo.currentText()
        self.log_combo.clear()
        if not log_dir.exists():
            return
        files = sorted(log_dir.glob("experiment_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            self.log_combo.addItem(f.name)
        if current:
            idx = self.log_combo.findText(current)
            if idx >= 0:
                self.log_combo.setCurrentIndex(idx)
        self._log_file_pos = 0
        self.log_text.clear()

    def _refresh_log_tail(self):
        name = self.log_combo.currentText()
        if not name:
            return
        path = _PROJECT_ROOT / LOG_DIR / name
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_file_pos)
                chunk = f.read()
                self._log_file_pos = f.tell()
            if chunk:
                self.log_text.moveCursor(QTextCursor.MoveOperation.End)
                self.log_text.insertPlainText(chunk)
                self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Shared: gui_queue draining
    # ══════════════════════════════════════════════════════════════════════

    def _drain_queue(self):
        # Bounded per tick so a burst of frames can't stall the Qt event loop.
        for _ in range(10):
            try:
                kind, payload = self.gui_queue.get_nowait()
                if kind == "frame":
                    self._on_frame(payload)
                elif kind == "status":
                    self._on_status(payload)
                elif kind == "timings":
                    self._on_timings(payload)
                elif kind == "detections":
                    self._on_detections(payload)
                elif kind == "model_info":
                    self._on_model_info(payload)
                elif kind == "race_local":
                    self._on_race_local(payload)
                elif kind == "race_earth":
                    self._on_race_earth(payload)
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

    def _on_timings(self, timings: dict):
        self._last_timings = timings
        self._latency_history.append(timings.get("total", 0.0))
        for k in ("hsv", "pose", "lstm"):
            self._breakdown_history[k].append(timings.get(k, 0.0))

    def _on_model_info(self, info: dict):
        self._model_info = info
        lines = [
            f"Pose model:  {info.get('pose_backend', '?')}  (available={info.get('pose_available')})",
            f"LSTM model:  {info.get('lstm_backend', '?')}",
            f"CNN model:   {info.get('cnn_backend', '?')}  (ensemble_enabled={info.get('cnn_ensemble_enabled')})",
            f"Rack-frame normalize: {info.get('rack_normalize')}    Threaded inference: {info.get('threaded')}",
        ]
        self.backend_label.setText("\n".join(lines))

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
