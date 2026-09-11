"""
Voice Alert System
==================
Fully offline TTS using Piper TTS.
Falls back to pyttsx3 if Piper is not installed.

Piper setup:
    1. Download from: https://github.com/rhasspy/piper/releases
    2. Download voice: en_US-lessac-medium.onnx
    3. Place piper binary in PATH

Usage:
    python pipeline/voice_alert.py --text "Step 3 skipped"
"""

import queue
import threading
import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceAlertSystem:
    """
    Thread-safe, queued voice alert system.
    Alerts are spoken in a background thread so they never block the main pipeline.
    """

    # Alert message templates
    ALERT_STEP_NEXT = "Step {step_id}: {step_name}. Please proceed."
    ALERT_STEP_COMPLETE = "Step {step_id} complete. Well done."
    ALERT_STEP_SKIP = "Warning. Step {expected_id} was skipped. Please complete the missed step before continuing."
    ALERT_STEP_RECOVERED = "Step {step_id} is now correct. Next, step {next_id}: {next_name}."
    ALERT_OUT_OF_SEQUENCE = "Warning. Out of sequence action detected. Please return to step {expected_id}."
    ALERT_EXPERIMENT_COMPLETE = "Experiment complete. All steps have been successfully executed."
    ALERT_EXPERIMENT_START = "Experiment started. Please proceed with step 1."
    ALERT_UNCERTAIN = "Uncertain about the current action near step {step_id}. Please confirm manually."

    def __init__(self, voice_model: str = "en_US-lessac-medium",
                 piper_binary: str = "piper", enabled: bool = True):
        self.enabled = enabled
        self.voice_model = voice_model
        self.piper_binary = piper_binary
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._lock = threading.Lock()
        self._speaking = False
        self._pyttsx3_engine = None  # created once in _detect_backend, reused for every alert
        # Retained even when audio is disabled.  This makes the simulator able
        # to verify the exact recovery guidance without playing sound.
        self.history: list[str] = []

        # Start background speaker thread
        self._thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._thread.start()

        # Detect available TTS backend
        self._backend = self._detect_backend()
        logger.info("Voice system initialized. Backend: %s", self._backend)

    def _detect_backend(self) -> str:
        """Detect which TTS backend is available."""
        # Try Piper first
        try:
            result = subprocess.run(
                [self.piper_binary, "--version"],
                capture_output=True, timeout=3
            )
            if result.returncode == 0:
                return "piper"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try pyttsx3 — create the engine once here and reuse it for every
        # alert. Re-running pyttsx3.init()/stop() per call is a known cause of
        # hangs/"run loop already started" errors on the Windows SAPI5 driver
        # in a long-running process; all speech already happens serially on
        # this single _speaker_loop thread, so one persistent engine is safe.
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 160)
            self._pyttsx3_engine.setProperty("volume", 1.0)
            return "pyttsx3"
        except Exception:
            self._pyttsx3_engine = None

        logger.warning("No TTS backend found. Voice alerts disabled.")
        return "none"

    def speak(self, text: str, priority: bool = False):
        """Queue a text-to-speech message."""
        with self._lock:
            self.history.append(text)
            del self.history[:-100]
        if not self.enabled or self._backend == "none":
            logger.info("[VOICE SUPPRESSED] %s", text)
            return

        try:
            if priority:
                # Clear queue and insert at front
                with self._lock:
                    while not self._queue.empty():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            break
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning("Voice queue full. Dropping: %s", text[:50])

    def _speaker_loop(self):
        """Background thread that speaks queued messages."""
        while True:
            try:
                text = self._queue.get(timeout=1.0)
                self._speaking = True
                self._speak_now(text)
                self._speaking = False
            except queue.Empty:
                continue

    def _speak_now(self, text: str):
        """Actually speak the text using the detected backend."""
        try:
            if self._backend == "piper":
                self._speak_piper(text)
            elif self._backend == "pyttsx3":
                self._speak_pyttsx3(text)
        except Exception as e:
            logger.error("TTS error: %s", e)

    def _speak_piper(self, text: str):
        """Speak using Piper TTS (high quality, fully offline)."""
        model_path = f"{self.voice_model}.onnx"

        cmd = [
            self.piper_binary,
            "--model", model_path,
            "--output-raw",
        ]

        # Pipe text to piper, capture raw 16-bit mono 22050Hz PCM.
        proc1 = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        audio_data, _ = proc1.communicate(text.encode())
        self._play_pcm(audio_data)

    def _play_pcm(self, audio_data: bytes):
        """Play raw 16-bit mono 22050Hz PCM via ffplay.

        Replaces the previous Windows(PowerShell SoundPlayer)/Linux(aplay)
        fork — that fork had no macOS branch at all (fell through to the
        Linux path, `aplay` missing there just silently dropped every alert),
        and the Windows path wrote headerless raw PCM into a `.wav`-suffixed
        temp file that `SoundPlayer` expects a real RIFF header for. ffmpeg
        is already a hard dependency (video streaming), and `ffplay` ships
        alongside it on every platform this project targets, so one command
        replaces all three previous paths.
        """
        proc2 = subprocess.Popen(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet",
             "-f", "s16le", "-ar", "22050", "-ac", "1", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc2.communicate(audio_data)

    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3 (cross-platform fallback). Reuses the single
        engine created once in _detect_backend — see its comment for why."""
        self._pyttsx3_engine.say(text)
        self._pyttsx3_engine.runAndWait()

    # ── Alert convenience methods ──────────────────────────────

    def alert_step_next(self, step_id: int, step_name: str):
        text = self.ALERT_STEP_NEXT.format(step_id=step_id, step_name=step_name)
        self.speak(text)

    def alert_step_complete(self, step_id: int):
        text = self.ALERT_STEP_COMPLETE.format(step_id=step_id)
        self.speak(text)

    def alert_skip(self, expected_id: int):
        text = self.ALERT_STEP_SKIP.format(expected_id=expected_id)
        self.speak(text, priority=True)

    def alert_step_recovered(self, step_id: int, next_step: Optional[dict]):
        if next_step:
            text = self.ALERT_STEP_RECOVERED.format(
                step_id=step_id, next_id=next_step["id"], next_name=next_step["name"]
            )
        else:
            text = f"Step {step_id} is now correct. The experiment is complete."
        self.speak(text, priority=True)

    def alert_out_of_sequence(self, expected_id: int):
        text = self.ALERT_OUT_OF_SEQUENCE.format(expected_id=expected_id)
        self.speak(text, priority=True)

    def alert_experiment_complete(self):
        self.speak(self.ALERT_EXPERIMENT_COMPLETE, priority=True)

    def alert_experiment_start(self):
        self.speak(self.ALERT_EXPERIMENT_START)

    def alert_uncertain(self, step_id: int):
        """LSTM/CNN ensemble disagreement — ask for confirmation rather than
        silently guessing, per the PS's own confidence principle."""
        text = self.ALERT_UNCERTAIN.format(step_id=step_id)
        self.speak(text, priority=True)

    def alert_anomaly(self, code: str, message: str, priority: bool = True):
        """Typed anomaly (A4 forbidden zone, A5 dwell/timeout, A7 occlusion —
        see pipeline/anomaly_monitor.py). Spoken text intentionally doesn't
        include the raw code (not useful to a crew member mid-task); the
        code still reaches the structured log via ExperimentLogger.log_anomaly."""
        self.speak(f"Attention. {message}", priority=priority)

    @property
    def is_speaking(self) -> bool:
        return self._speaking


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Voice alert system initialized successfully.")
    args = parser.parse_args()

    tts = VoiceAlertSystem()
    tts.speak(args.text)
    time.sleep(5)  # Wait for speech to complete
