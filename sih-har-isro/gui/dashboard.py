"""Minimal GUI dashboard.

The full PyQt dashboard is optional. This module keeps `main.py --mode pipeline`
from crashing when PyQt is missing, by draining the GUI queue in a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import time

logger = logging.getLogger(__name__)


def launch_dashboard(gui_queue: "queue.Queue"):
    """Drain pipeline GUI events. Prefer PyQt if installed, else no-op consumer."""
    try:
        from gui.qt_dashboard import launch_qt_dashboard  # optional richer UI
        launch_qt_dashboard(gui_queue)
        return
    except ImportError:
        logger.info("PyQt dashboard not present — running queue drain (headless GUI).")

    while True:
        try:
            gui_queue.get(timeout=1.0)
        except queue.Empty:
            time.sleep(0.05)
        except Exception:
            break
