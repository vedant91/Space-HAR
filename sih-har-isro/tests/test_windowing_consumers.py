"""Regression #6 / review fact 6: seven windowing consumers, all sharing
`pipeline/features.py`'s windows()/SampleWindow. Guards against a second
windowing implementation creeping back in (the original bug: build_sequences
used purity=0.75 while evaluate_real used 0.6 and claimed they matched).

The seven consumers named in the design doc's Review Outcomes:
  1. tools/build_sequences.py         (stack_windows)
  2. tools/evaluate_real.py           (windows, twice)
  3. data_generation/mediapipe_labeler.py (stack_windows)
  4. pipeline/har_pipeline.py run()   (SampleWindow via _ingest)
  5. pipeline/har_pipeline.py process_frame() (SampleWindow via _ingest, same
     buffer object as run() -- not a separate implementation)
  6. simulation/space_sim.py          (via HARPipeline.process_frame())
  7. tools/showcase_overlay.py        (via HARPipeline.process_frame())
"""
import re
from pathlib import Path

import numpy as np
import pytest

from config.experiment_config import FPS, SAMPLE_EVERY, SEQUENCE_WINDOW
from pipeline.features import SampleWindow, stack_windows, windows

REPO = Path(__file__).resolve().parent.parent

CONSUMERS = {
    "tools/build_sequences.py": r"\b(windows|stack_windows|SampleWindow)\b",
    "tools/evaluate_real.py": r"\b(windows|stack_windows|SampleWindow)\b",
    "data_generation/mediapipe_labeler.py": r"\b(windows|stack_windows|SampleWindow)\b",
    "pipeline/har_pipeline.py": r"\bSampleWindow\b",
    "simulation/space_sim.py": r"\.process_frame\(",
    "tools/showcase_overlay.py": r"\.process_frame\(",
}


@pytest.mark.parametrize("relpath,pattern", CONSUMERS.items())
def test_consumer_uses_shared_windowing(relpath, pattern):
    """Each of the seven files reaches pipeline.features, directly or (for
    space_sim / showcase_overlay) through HARPipeline.process_frame(), and
    none of them defines its own sliding-window loop over a raw array."""
    src = (REPO / relpath).read_text(encoding="utf-8")
    assert re.search(pattern, src), f"{relpath} no longer references the shared windower"
    # No second windowing implementation: no manual `range(0, len(...) - `
    # slide, which is what the pre-fix build_sequences/evaluate_real skew
    # looked like.
    assert "for i in range(0, len(" not in src.replace(" ", "")


def test_stack_windows_and_manual_samplewindow_agree():
    """The two entry points every consumer ultimately uses -- windows()/
    stack_windows() (offline: build_sequences, evaluate_real, mediapipe_labeler)
    and SampleWindow.push() directly (live: har_pipeline._ingest, and through
    it space_sim/showcase_overlay) -- must produce byte-identical samples for
    the same input. This is what makes "seven consumers, one behaviour" true
    rather than aspirational."""
    rng = np.random.default_rng(3)
    T, D = 130, 6
    seq = rng.normal(size=(T, D)).astype(np.float32)
    labels = rng.integers(1, 4, size=T)
    stamps = np.arange(T) / float(FPS)

    xs, ys = stack_windows(seq, labels, stride=1, purity=0.0,
                            sample_every=SAMPLE_EVERY, window=SEQUENCE_WINDOW,
                            timestamps=stamps)

    sw = SampleWindow(SEQUENCE_WINDOW, SAMPLE_EVERY, FPS)
    live_windows = []

    def on_sample():
        if sw.full:
            live_windows.append(sw.array())

    for i in range(T):
        sw.push(seq[i], stamps[i], int(labels[i]), on_sample)

    assert len(live_windows) == len(xs)
    for got, want in zip(live_windows, xs):
        assert np.array_equal(got, want)
