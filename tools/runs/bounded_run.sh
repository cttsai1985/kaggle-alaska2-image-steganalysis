#!/usr/bin/env bash
# One bounded GPU run: train (+ evaluate at end) under a hard wall-clock cap, then verify in a fresh process.
#   tools/runs/bounded_run.sh CONFIG RUN_ID CAP_S [extra --set KEY=VALUE ...]
# Writes everything under $DATA/runs/RUN_ID/ (outside Git): train.log, gpu_samples.csv, wall.json,
# verify.log. The cap is enforced by `timeout` (SIGTERM -> manifest status "interrupted"), then SIGKILL
# 60 s later. Never touches other processes; the caller runs the launch gate first.
set -uo pipefail
CONFIG=$1; RUN_ID=$2; CAP_S=$3; shift 3
DATA=${ALASKA2_DATA:-$HOME/data/kaggle/alaska2-image-steganalysis}
RUN_DIR=$DATA/runs/$RUN_ID
STAGE=$DATA/runs/_staging/$RUN_ID   # train refuses an existing run dir, so side files start here
mkdir -p "$STAGE"
SETS=()
for kv in "$@"; do SETS+=(--set "$kv"); done

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 \
  > "$STAGE/gpu_samples.csv" 2>/dev/null &
MON=$!
t0=$(date +%s.%N)
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
timeout --kill-after=60 "$CAP_S" uv run python -m alaska2 train --config "$CONFIG" --run-id "$RUN_ID" "${SETS[@]}" \
  > "$STAGE/train.log" 2>&1
rc=$?
t1=$(date +%s.%N)
verify_rc=skipped
PRED=$(ls "$RUN_DIR"/predictions/valid_epoch*[0-9].parquet 2>/dev/null | tail -1)
if [ "$rc" = 0 ] && [ -n "$PRED" ]; then
  uv run python -m alaska2 verify --predictions "$PRED" > "$STAGE/verify.log" 2>&1
  verify_rc=$?
fi
t2=$(date +%s.%N)
kill "$MON" 2>/dev/null
mkdir -p "$RUN_DIR"
cat > "$STAGE/wall.json" <<EOF
{"run_id": "$RUN_ID", "config": "$CONFIG", "cap_s": $CAP_S, "started_at": "$start_utc",
 "ended_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "train_exit": $rc,
 "train_eval_wall_s": $(echo "$t1 - $t0" | bc), "verify_exit": "$verify_rc",
 "verify_wall_s": $(echo "$t2 - $t1" | bc), "timed_out": $([ "$rc" = 124 ] || [ "$rc" = 137 ] && echo true || echo false)}
EOF
mv "$STAGE"/* "$RUN_DIR"/ && rmdir "$STAGE"
echo "exit=$rc verify=$verify_rc run_dir=$RUN_DIR"
