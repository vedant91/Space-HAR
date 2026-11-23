"""Offline spoken guidance for the BAS procedure."""

from collections import deque
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from typing import Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.experiment_config import PIPER_BINARY, VOICE_MODEL

logger = logging.getLogger(__name__)


class VoiceAlertSystem:
    """Speak in a worker thread and keep the latest instruction ahead of stale cues."""

    def __init__(self, voice_model: str = VOICE_MODEL,
                 piper_binary: str = PIPER_BINARY, enabled: bool = True):
        self.enabled = enabled
        self.voice_model = Path(voice_model)
        if self.voice_model.suffix != ".onnx":
            self.voice_model = self.voice_model.with_suffix(".onnx")
        self.piper_binary = piper_binary
        self._pending = deque(maxlen=10)
        self._condition = threading.Condition()
        self._closed = False
        self._speaking = False
        self.history: list[str] = []
        self.spoken = 0  # Count of submitted cues, including when playback is disabled.
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._speaker_loop, daemon=True)
            self._thread.start()

    def _backends(self):
        player = shutil.which("ffplay")
        piper = shutil.which(self.piper_binary)
        if (player and piper and self.voice_model.is_file()
                and self.voice_model.with_suffix(".onnx.json").is_file()):
            yield "piper", lambda text: self._speak_piper(piper, player, text)

        if sys.platform == "darwin":
            native = shutil.which("say")
        else:
            native = shutil.which("espeak-ng") or shutil.which("espeak")
        if native:
            yield Path(native).name, lambda text: subprocess.run(
                [native, "-r" if sys.platform == "darwin" else "-s", "165", text],
                check=True, timeout=15, capture_output=True,
            )

        # Some speech drivers need initialization and playback on the same thread.
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception as exc:
            logger.debug("Could not initialize pyttsx3: %s", exc)
        else:
            def speak(text):
                engine.say(text)
                engine.runAndWait()
            yield "pyttsx3", speak

    def _speak_piper(self, piper, player, text):
        with TemporaryDirectory() as directory:
            audio = str(Path(directory) / "guidance.wav")
            subprocess.run([piper, "--model", str(self.voice_model), "--output_file", audio],
                           input=text + "\n", text=True, check=True, timeout=15, capture_output=True)
            subprocess.run([player, "-autoexit", "-nodisp", "-loglevel", "error", audio],
                           check=True, timeout=15, capture_output=True)

    def speak(self, text: str, priority: bool = False):
        """Record a cue and queue it for playback. Priority cues replace pending guidance."""
        with self._condition:
            if self._closed:
                return
            self.history.append(text)
            del self.history[:-100]
            self.spoken += 1
            if self.enabled:
                if priority:
                    self._pending.clear()
                self._pending.append(text)
                self._condition.notify()
        logger.info("Voice cue: %s", text)

    def _speaker_loop(self):
        backends = self._backends()
        backend = next(backends, None)
        while backend is not None:
            name, play = backend
            logger.info("Voice playback ready via %s", name)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if not self._pending:
                        return
                    text = self._pending.popleft()
                    self._speaking = True
                try:
                    play(text)
                except Exception as exc:
                    logger.warning("Voice playback via %s failed: %s", name, exc)
                    with self._condition:
                        # Retry this cue unless a newer instruction is waiting.
                        if not self._pending:
                            self._pending.append(text)
                    break
                finally:
                    with self._condition:
                        self._speaking = False
            backend = next(backends, None)
        logger.error("Offline speech is unavailable. Guidance is still recorded in the log.")

    def discard_pending(self):
        """Remove cues from the preceding run when a trial resets."""
        with self._condition:
            self._pending.clear()

    def close(self, timeout: float = 15):
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Voice playback was still active at shutdown")

    @property
    def is_speaking(self) -> bool:
        with self._condition:
            return self._speaking

    @staticmethod
    def _cue(step: dict) -> str:
        return step.get("voice_cue") or f"Step {step['id']}. {step['name']}."

    def alert_experiment_start(self, first_step: Optional[dict] = None):
        message = "Experiment started."
        if first_step:
            message += f" {self._cue(first_step)}"
        self.speak(message, priority=True)

    def alert_step_next(self, step_id: int, step_name: str):
        self.speak(self._cue({"id": step_id, "name": step_name}), priority=True)

    def alert_step_complete(self, step_id: int, next_step: Optional[dict] = None):
        message = f"Step {step_id} confirmed."
        if next_step:
            message += f" Next, {self._cue(next_step)}"
        self.speak(message, priority=True)

    def alert_skip(self, expected_id: int, expected_step: Optional[dict] = None):
        instruction = self._cue(expected_step) if expected_step else f"Return to step {expected_id}."
        self.speak(f"Pause. Step {expected_id} has not been confirmed. {instruction}", priority=True)

    def alert_out_of_sequence(self, expected_id: int, expected_step: Optional[dict] = None):
        instruction = self._cue(expected_step) if expected_step else f"Return to step {expected_id}."
        self.speak(f"Pause. Action out of sequence. {instruction}", priority=True)

    def alert_step_recovered(self, step_id: int, next_step: Optional[dict]):
        instruction = f"Next, {self._cue(next_step)}" if next_step else "The experiment is complete."
        self.speak(f"Step {step_id} is now correct. {instruction}", priority=True)

    def alert_uncertain(self, step_id: int, expected_step: Optional[dict] = None):
        instruction = self._cue(expected_step) if expected_step else f"Check step {step_id}."
        self.speak(f"I could not confirm the action. {instruction}")

    def alert_experiment_complete(self):
        self.speak("Experiment complete. All steps confirmed.", priority=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Voice guidance is ready. Step 1. Approach the main box.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    voice = VoiceAlertSystem()
    voice.speak(args.text)
    voice.close()
