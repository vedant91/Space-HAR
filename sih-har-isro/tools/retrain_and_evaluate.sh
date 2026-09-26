#!/usr/bin/env bash
# Retrain the step classifier on rendered video and score it honestly.
#
# The whole point of this script is that the four stages below are the ONLY
# supported way to produce an accuracy claim for this project. The previous
# path (end_to_end_loop.py) trained and tested on the same closed-form
# generator, so its numbers could not fail.
#
#   0  check_dataset     refuse to start on an empty dataset or a holdout tag
#                        that matches no take (an empty test set scores
#                        nothing and looks like a result)
#   1  build_sequences   rendered frames -> MediaPipe -> 30-frame windows,
#                        grouped BY TAKE so a held-out crew orientation is
#                        genuinely unseen
#   2  train_lstm        GroupShuffleSplit over those take groups
#   3  evaluate_real     camera -> MediaPipe -> LSTM vs Blender ground truth,
#                        scored separately on train, holdout and error takes
#   4  compare           holdout column against the published baseline,
#                        scored on the SAME holdout takes
#
# Note stage 1 uses --pose-source mediapipe, NOT the ground-truth pose.
# Training on truth and serving on MediaPipe output is a train/serve skew that
# no amount of held-out splitting would reveal, and MediaPipe's errors on a
# bulky pressure suit are large and systematic enough that the model is better
# off learning to work with them.
#
# Stages 1 and 3 both pass --rack-normalize for the same reason: config sets
# RACK_FRAME_NORMALIZE = True and the live pipeline normalises every frame, so
# a model trained or scored without it is measured on features it never serves.
#
# There is deliberately no pooled headline. A single accuracy over train and
# holdout takes together (the old logs/real_eval_retrained.json 0.4032) mostly
# measures how well the model memorised the takes it trained on. Quote the
# holdout column, and say how many takes it was trained on.
#
# This script does not touch the model the demo runs. Promote explicitly:
#   python tools/promote_model.py <checkpoint.pt>
#
# Usage:
#   tools/retrain_and_evaluate.sh <dataset_dir> [holdout_tags]
#   DRY_RUN=1 tools/retrain_and_evaluate.sh <dataset_dir> [holdout_tags]
#       runs only the stage-0 gate - check the tags before a long run

set -euo pipefail

usage() { sed -n '/^# Usage:/,/^$/s/^#\( \|$\)//p' "${BASH_SOURCE[0]}"; }

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") echo "error: dataset_dir is required (there is no default: a default" \
             "once pointed at a dataset that no longer existed)" >&2
        usage >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
DATASET="${1%/}"
HOLDOUT="${2:-orient_180,orient_-60,yaw_90,pitch_60}"
ERROR_TAGS="skip,recover"
# Left out of training entirely: error sequences are for the state machine,
# heldout_view is an unseen-camera test. Passed explicitly so stage 1 and the
# train-tag list below cannot drift apart.
EXCLUDE="$ERROR_TAGS,heldout_view"
BASELINE="${BASELINE:-models/lstm_final_rack.pt}"   # published 0.201 held-out

# Everything a run writes is named after its dataset, so retraining on one
# dataset never overwrites another's sequences, model or reports.
NAME="$(basename "$DATASET")"
SEQ_DIR="dataset/seq_${NAME}"
MODEL="models/lstm_${NAME}.pt"
LOG="logs/real_eval_${NAME}"

command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "python not found at $PY"; exit 1; }

echo "══════════════════════════════════════════════════════════════"
echo " 0/4  Dataset gate"
echo "══════════════════════════════════════════════════════════════"
echo "dataset : $DATASET"
echo "holdout : $HOLDOUT"
echo "excluded: $EXCLUDE"
"$PY" tools/check_dataset.py "$DATASET" --holdout "$HOLDOUT"

# Train tags = every tag present, minus holdout and excluded.
TRAIN_TAGS="$("$PY" - "$DATASET" "$HOLDOUT,$EXCLUDE" <<'PYEOF'
import json, pathlib, sys
skip = set(sys.argv[2].split(","))
tags = {json.loads(m.read_text(encoding="utf-8")).get("tag", "nominal")
        for m in pathlib.Path(sys.argv[1]).glob("take_*/meta.json")}
print(",".join(sorted(tags - skip)))
PYEOF
)"
[ -n "$TRAIN_TAGS" ] || { echo "no take is left for training after holdout/exclude" >&2; exit 1; }
echo "train   : $TRAIN_TAGS"
echo "outputs : $SEQ_DIR, $MODEL, ${LOG}_*.json"
[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN=1: gate passed, stopping here."; exit 0; }

echo
echo "══════════════════════════════════════════════════════════════"
echo " 1/4  Building sequences from rendered frames (MediaPipe pose)"
echo "══════════════════════════════════════════════════════════════"
"$PY" tools/build_sequences.py \
    --dataset "$DATASET" \
    --out "$SEQ_DIR" \
    --pose-source mediapipe \
    --stride 1 \
    --rack-normalize \
    --holdout-tags "$HOLDOUT" \
    --exclude-tags "$EXCLUDE"

echo
echo "══════════════════════════════════════════════════════════════"
echo " 2/4  Training LSTM (group split by take)"
echo "══════════════════════════════════════════════════════════════"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, ".")
from train.train_lstm import train_model
acc = train_model(data_dir="${SEQ_DIR}",
                  output_path="${MODEL}",
                  epochs=80)
print(f"RETRAINED_VAL_ACC {acc:.4f}  (take-grouped val split of TRAIN takes; not a headline)")
PYEOF

echo
echo "══════════════════════════════════════════════════════════════"
echo " 3/4  Scoring against rendered video with independent truth"
echo "══════════════════════════════════════════════════════════════"
score() {  # score <model> <tags> <out.json> <label>
    "$PY" tools/evaluate_real.py \
        --dataset "$DATASET" \
        --model "$1" \
        --tags "$2" \
        --rack-normalize \
        --out "$3" \
        --label "$4"
}
score "$MODEL" "$TRAIN_TAGS" "${LOG}_train.json"   "$NAME train takes ($TRAIN_TAGS)"
score "$MODEL" "$HOLDOUT"    "${LOG}_holdout.json" "$NAME holdout takes ($HOLDOUT)"
score "$MODEL" "$ERROR_TAGS" "${LOG}_error.json"   "$NAME error sequences ($ERROR_TAGS)"

echo
echo "══════════════════════════════════════════════════════════════"
echo " 4/4  Baseline on the same holdout takes ($BASELINE)"
echo "══════════════════════════════════════════════════════════════"
# The published 0.201 was measured on a different holdout, so it is not
# comparable to this run's holdout column. Re-score the baseline here.
if [ -f "$BASELINE" ]; then
    score "$BASELINE" "$HOLDOUT" "${LOG}_baseline_holdout.json" "baseline $BASELINE on $NAME holdout"
else
    echo "[!] no baseline at $BASELINE - baseline column will read n/a"
fi

echo
"$PY" - "$LOG" "$HOLDOUT" <<'PYEOF'
import json, pathlib, sys

log, holdout = sys.argv[1], sys.argv[2]

def load(p):
    path = pathlib.Path(p)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

cols = [
    ("train",          load(f"{log}_train.json")),
    ("HOLDOUT",        load(f"{log}_holdout.json")),
    ("skip,recover",   load(f"{log}_error.json")),
    ("baseline@hold",  load(f"{log}_baseline_holdout.json")),
]

keys = [
    ("step_acc_real",         "step acc (camera->MediaPipe->LSTM)"),
    ("step_acc_oracle",       "step acc (ground-truth pose)"),
    ("perception_cost",       "perception cost (oracle - real)"),
    ("n_takes",               "takes scored"),
    ("step_windows_scored",   "windows scored"),
    ("key_landmark_mean_px",  "MediaPipe key-joint error (px)"),
    ("hsv_red_recall",        "HSV red recall"),
    ("hsv_yellow_recall",     "HSV yellow recall"),
]

width = max(len(label) for _, label in keys) + 2
header = "metric".ljust(width) + "".join(name.rjust(16) for name, _ in cols)
print(header)
print("-" * len(header))
for key, label in keys:
    line = label.ljust(width)
    for _, report in cols:
        value = (report or {}).get("aggregate", {}).get(key)
        text = "n/a" if value is None else (str(value) if isinstance(value, int) else f"{value:.4f}")
        line += text.rjust(16)
    print(line)

print()
print(f"Quote the HOLDOUT column ({holdout}), and the number of train takes.")
print("Chance is 0.125 (8 steps). The train column only says the model fit")
print("what it saw; it is not evidence of generalisation.")
print()
print("perception_cost > 0: the classifier is fine but MediaPipe is the")
print("bottleneck - a pose fix (better bundle, suit-appropriate model), not")
print("more LSTM training.")
print("perception_cost < 0: the model does WORSE on perfect pose than on")
print("MediaPipe output, i.e. it keys on pose-estimator artefacts rather than")
print("body configuration. More data of the same kind will not fix that.")
PYEOF

echo
echo "Done. Reports: ${LOG}_{train,holdout,error,baseline_holdout}.json"
echo "To put this model in the demo: python tools/promote_model.py $MODEL"
