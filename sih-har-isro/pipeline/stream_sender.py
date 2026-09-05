"""
Network Video Streaming — push frames to a specific IP over UDP
=================================================================
Wraps an `ffmpeg` subprocess: raw BGR frames go in over stdin, an
H.264/MPEG-TS stream comes out over UDP to STREAM_HOST:STREAM_PORT.

Chosen over a GStreamer cv2.VideoWriter pipeline because the pip
opencv-python wheel isn't built with GStreamer support. Spawning an
external encoder binary mirrors the pattern already used for offline
TTS in voice_alert.py (piper/aplay subprocesses): if the binary isn't
available, log a warning and disable the feature — never crash the
capture loop.

Receive the stream with:
    ffplay udp://<host>:<port>
    vlc udp://@:<port>          (VLC syntax when receiving on the same host)
"""

import logging
import queue
import shutil
import subprocess
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NetworkStreamer:
    """Streams BGR frames to a network destination via an ffmpeg subprocess."""

    def __init__(self, host: str, port: int, width: int, height: int, fps: int = 30):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.available = False

        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found on PATH — network video streaming disabled.")
            return

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-f", "mpegts", f"udp://{host}:{port}",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("Failed to launch ffmpeg for streaming: %s", e)
            self._proc = None
            return

        self.available = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        logger.info("Streaming video to udp://%s:%s", host, port)

    def _send_loop(self):
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError, ValueError) as e:
                # ValueError covers "I/O operation on closed file" — reachable
                # if close() closes stdin concurrently with a write that's
                # still in flight (see close()'s is_alive() check below).
                logger.warning("Streaming pipe broke (%s) — disabling network stream.", e)
                self.available = False
                return

    def write(self, frame_bgr: np.ndarray):
        """Queue a frame for streaming. Never blocks the capture loop — under
        backpressure the oldest queued frame is dropped in favor of the newest."""
        if not self.available:
            return
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame_bgr)
            except queue.Empty:
                pass

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # The send thread is still blocked inside stdin.write() —
                # touching stdin concurrently from here is exactly what
                # produces the "I/O operation on closed file" race. Kill the
                # process outright instead of trying to close/wait on it.
                logger.warning("Streaming sender thread did not stop in time — killing ffmpeg.")
                if self._proc:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self.available = False
                return
        if self._proc:
            try:
                self._proc.stdin.close()  # EOF — ffmpeg flushes remaining frames and exits on its own
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3.0)
            except Exception:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
        self.available = False
