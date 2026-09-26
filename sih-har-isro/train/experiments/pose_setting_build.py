"""Build sequences with the pose stage's complexity / input width overridden
for THIS process only (config/experiment_config.py is untouched).

Everything else is tools/build_sequences.build() unchanged: same HSV, rack
normaliser, feature builder, windowing, split. Cache is off so nothing is
written under the dataset directory.

    python train/experiments/pose_setting_build.py --complexity 1 --width 1280 \
        --out D:/SIH-2026/.tmp/ml/seq_c1_w1280
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

ap = argparse.ArgumentParser()
ap.add_argument("--complexity", type=int, required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default=str(ROOT / "dataset" / "blender_core"))
args = ap.parse_args()

import pipeline.har_pipeline as hp  # noqa: E402
import build_sequences as bs  # noqa: E402

# OptimizedMPWrapper reads POSE_INPUT_WIDTH at call time from its module and
# binds complexity as a default argument at def time; patch both. The two
# names in build_sequences only feed metadata.json, so it records the truth.
hp.POSE_INPUT_WIDTH = args.width
d = list(hp.OptimizedMPWrapper.__init__.__defaults__)
d[0] = args.complexity
hp.OptimizedMPWrapper.__init__.__defaults__ = tuple(d)
bs.POSE_INPUT_WIDTH = args.width
bs.MEDIAPIPE_MODEL_COMPLEXITY = args.complexity

# Keep the raw per-take pose so its error vs ground truth can be measured.
_orig = bs._frame_inputs


def _saving(dataset_dir, take_dir, cam_dir, *a, **k):
    import numpy as np
    out = _orig(dataset_dir, take_dir, cam_dir, *a, **k)
    raw = Path(args.out + "_rawpose")
    raw.mkdir(parents=True, exist_ok=True)
    np.save(raw / f"{take_dir.name}.npy", out[0])
    return out


bs._frame_inputs = _saving

bs.build(args.dataset, args.out, "mediapipe", stride=1,
         holdout_tags=["orient_180", "orient_-60", "yaw_90", "pitch_60"],
         rack_normalize=True, exclude_tags=["skip", "recover", "heldout_view"],
         use_cache=False)
