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

import os
import queue
import threading
import logging
import subprocess
import tempfile
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

    def __init__(self, voice_model: str = "en_US-lessac-medium",
                 piper_binary: str = "piper", enabled: bool = True):
        self.enabled = enabled
        self.voice_model = voice_model
        self.piper_binary = piper_binary
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._lock = threading.Lock()
        self._speaking = False
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

        # Try pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            return "pyttsx3"
        except Exception:
            pass

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
        config_path = f"{self.voice_model}.onnx.json"

        cmd = [
            self.piper_binary,
            "--model", model_path,
            "--output-raw",
        ]

        # Pipe text to piper, pipe output to aplay/ffplay
        proc1 = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        audio_data, _ = proc1.communicate(text.encode())

        # Play audio
        if os.name == "nt":  # Windows — use PowerShell
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name
            subprocess.run(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()"],
                capture_output=True
            )
            os.unlink(tmp_path)
        else:
            proc2 = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            proc2.communicate(audio_data)

    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3 (cross-platform fallback)."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)  # Slower for clarity
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

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
