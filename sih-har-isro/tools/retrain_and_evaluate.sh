#!/usr/bin/env bash
# Retrain the step classifier on rendered video and score it honestly.
#
# The whole point of this script is that the four stages below are the ONLY
# supported way to produce an accuracy claim for this project. The previous
# path (end_to_end_loop.py) trained and tested on the same closed-form
# generator, so its numbers could not fail.
#
#   1  build_sequences   rendered frames -> MediaPipe -> 30-frame windows,
#                        grouped BY TAKE so a held-out crew orientation is
#                        genuinely unseen
#   2  train_lstm        GroupShuffleSplit over those take groups
#   3  evaluate_real     camera -> MediaPipe -> LSTM vs Blender ground truth
#   4  compare           new numbers against the shipped-model baseline
#
# Note stage 1 uses --pose-source mediapipe, NOT the ground-truth pose.
# Training on truth and serving on MediaPipe output is a train/serve skew that
# no amount of held-out splitting would reveal, and MediaPipe's errors on a
# bulky pressure suit are large and systematic enough that the model is better
# off learning to work with them.
#
# Usage:
#   tools/retrain_and_evaluate.sh [dataset_dir] [holdout_tags]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-./.venv312/Scripts/python.exe}"
DATASET="${1:-dataset/blender}"
HOLDOUT="${2:-orient_180,orient_-45}"
SEQ_DIR="dataset/sequences_mp"

command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "python not found at $PY"; exit 1; }

echo "══════════════════════════════════════════════════════════════"
echo " 1/4  Building sequences from rendered frames (MediaPipe pose)"
echo "══════════════════════════════════════════════════════════════"
"$PY" tools/build_sequences.py \
    --dataset "$DATASET" \
    --out "$SEQ_DIR" \
    --pose-source mediapipe \
    --complexity 1 \
    --stride 2 \
    --holdout-tags "$HOLDOUT"

echo
echo "══════════════════════════════════════════════════════════════"
echo " 2/4  Training LSTM (group split by take)"
echo "══════════════════════════════════════════════════════════════"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, ".")
from train.train_lstm import train_model
acc = train_model(data_dir="${SEQ_DIR}",
                  output_path="models/lstm_classifier_real.pt",
                  epochs=80)
print(f"RETRAINED_VAL_ACC {acc:.4f}")
PYEOF

echo
echo "══════════════════════════════════════════════════════════════"
echo " 3/4  Scoring against rendered video with independent truth"
echo "══════════════════════════════════════════════════════════════"
"$PY" tools/evaluate_real.py \
    --dataset "$DATASET" \
    --model models/lstm_classifier_real.pt \
    --out logs/real_eval_retrained.json \
    --label "retrained on rendered video (MediaPipe pose, take-grouped split)"

echo
echo "══════════════════════════════════════════════════════════════"
echo " 4/4  Baseline comparison"
echo "══════════════════════════════════════════════════════════════"
"$PY" - <<'PYEOF'
import json, pathlib

def load(p):
    path = pathlib.Path(p)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

rows = [
    ("shipped (synthetic_pose)", load("logs/real_eval_shipped_smoke.json")),
    ("retrained (rendered)",     load("logs/real_eval_retrained.json")),
]

keys = [
    ("step_acc_real",         "step acc (camera->MediaPipe->LSTM)"),
    ("step_acc_oracle",       "step acc (ground-truth pose)"),
    ("perception_cost",       "perception cost (oracle - real)"),
    ("key_landmark_mean_px",  "MediaPipe key-joint error (px)"),
    ("hsv_red_recall",        "HSV red recall"),
    ("hsv_yellow_recall",     "HSV yellow recall"),
]

width = max(len(label) for _, label in keys) + 2
header = "metric".ljust(width) + "".join(name.rjust(26) for name, _ in rows)
print(header)
print("-" * len(header))
for key, label in keys:
    line = label.ljust(width)
    for _, report in rows:
        value = (report or {}).get("aggregate", {}).get(key)
        line += ("n/a" if value is None else f"{value:.4f}").rjust(26)
    print(line)

print()
print("A model that generalises shows step_acc_real well above zero AND a")
print("small perception_cost. A large perception_cost means the classifier is")
print("fine but MediaPipe is the bottleneck on this figure - which is a")
print("different fix (better pose bundle, rack-frame normalisation, or a")
print("suit-appropriate pose model) than more LSTM training.")
PYEOF

echo
echo "Done. Reports: logs/real_eval_retrained.json"
