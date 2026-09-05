#!/usr/bin/env python3
"""Launcher — real loop lives in sih-har-isro/end_to_end_loop.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "sih-har-isro" / "end_to_end_loop.py"),
               run_name="__main__")
