#!/usr/bin/env bash
# Supervise the Blender dataset render across GPU-driver crashes.
#
# EEVEE on an integrated GPU is not reliable over a multi-hour unattended
# render: this machine's Intel UHD 620 driver takes an
# EXCEPTION_ACCESS_VIOLATION inside ig9icd64.dll after a few hundred frames of
# sustained work. That is a driver fault, not a scene fault - the same frames
# render correctly on relaunch.
#
# render_dataset.py is resumable at both take and frame granularity, so the
# correct response is simply to restart it. This loop does that until the run
# reports DATASET_OK or the attempt budget is exhausted, and it makes forward
# progress a hard requirement: if an attempt renders no new frames at all,
# something is genuinely wrong and looping further would just spin.
#
# Usage:
#   tools/render_dataset_resilient.sh <out_dir> <plan> [width] [height] [samples]

set -uo pipefail

BLENDER="${BLENDER:-/c/Program Files/Blender Foundation/Blender 5.2/blender.exe}"
OUT="${1:-dataset/blender}"
PLAN="${2:-study}"
WIDTH="${3:-900}"
HEIGHT="${4:-506}"
SAMPLES="${5:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$OUT"

count_frames() { find "$OUT" -name '*.png' -type f 2>/dev/null | wc -l; }

prev=$(count_frames)
echo "[supervisor] starting: plan=$PLAN out=$OUT ${WIDTH}x${HEIGHT} samples=$SAMPLES"
echo "[supervisor] frames already present: $prev"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[supervisor] attempt $attempt/$MAX_ATTEMPTS ..."
  "$BLENDER" --background --factory-startup \
      --python blender/render_dataset.py -- \
      --out "$(cd "$OUT" && pwd)" --plan "$PLAN" \
      --width "$WIDTH" --height "$HEIGHT" --samples "$SAMPLES" \
      2>&1 | grep -E "PLAN|TAKE|DATASET_OK|payload_[a-z]+ [0-9]+/|Error|EXCEPTION" || true

  if [ -f "$OUT/index.json" ]; then
    echo "[supervisor] DONE - index.json written after $attempt attempt(s)."
    exit 0
  fi

  now=$(count_frames)
  echo "[supervisor] frames: $prev -> $now"
  if [ "$now" -le "$prev" ]; then
    echo "[supervisor] no forward progress this attempt; aborting so the real"
    echo "[supervisor] failure is visible instead of being hidden by a retry loop."
    exit 1
  fi
  prev="$now"
  sleep 5
done

echo "[supervisor] exhausted $MAX_ATTEMPTS attempts without completing."
exit 1
