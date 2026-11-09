#!/usr/bin/env bash
# Evaluate/verify the predeclared incomplete-run fallback without training.
# retained_gpu.sh RUNTIME_CHECKOUT DATA_ROOT OUTPUT_ROOT RUN_ID [CAP_S]
set -euo pipefail
if [[ "${1:-}" == --inside ]]; then
  ckpt="/inputs/runs/$ALASKA2_RUN_ID/checkpoints/last.pt"
  "$ALASKA2_PYTHON" -m tools.readiness.launch_gate --data-dir /inputs --label retained-checkpoint --cpu-free-threads-min 4 --out /output/launch_gate.json
  "$ALASKA2_PYTHON" -m alaska2 evaluate --checkpoint "$ckpt" --out-dir /output/predictions --data-root /inputs --device cuda --num-workers 2 --eval-batch-size 64
  # Each command starts a fresh Python process and reconstructs predictions from the checkpoint alone.
  for split in valid test; do
    if [[ "$split" == test ]]; then
      "$ALASKA2_PYTHON" -m alaska2 predict --checkpoint "$ckpt" --out-dir /output/predictions --data-root /inputs --device cuda --num-workers 2 --eval-batch-size 64
    fi
    mapfile -t predictions < <(find /output/predictions -maxdepth 1 -name "${split}_epoch*.parquet" -type f)
    [[ ${#predictions[@]} == 1 ]] || { echo "expected one $split prediction artifact" >&2; exit 2; }
    "$ALASKA2_PYTHON" -m alaska2 verify --predictions "${predictions[0]}"
  done
  exit 0
fi
[[ $# -ge 4 && $# -le 5 ]] || { echo 'usage: retained_gpu.sh RUNTIME_CHECKOUT DATA_ROOT OUTPUT_ROOT RUN_ID [CAP_S]' >&2; exit 2; }
runtime=$(realpath "$1")
data=$(realpath "$2")
mkdir -p "$3"
output=$(realpath "$3")
run_id=$4
cap=${5:-1800}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$cap" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$runtime/.venv/bin/python" ]] || { echo 'runtime checkout needs its locked .venv' >&2; exit 2; }
exec 9>"$output/.execution.lock"
flock -n 9 || { echo 'another execution owns this output directory' >&2; exit 2; }
cidfile="$output/.container-$$.cid"
cleanup() {
  if [[ -s "$cidfile" ]]; then
    cid=$(cat "$cidfile")
    if [[ "$cid" =~ ^[a-f0-9]{64}$ ]]; then
      docker rm --force "$cid" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
uv_python=${ALASKA2_UV_PYTHON_ROOT:-$HOME/.local/share/uv/python}
# No new image build or downloads; retain the matching locked runtime as read-only mounts.
timeout --kill-after=60 "$cap" docker run --rm --init --cidfile "$cidfile" --gpus all --network none --cpus 6 --memory 24g --shm-size 2g --pids-limit 512 \
  --user "$(id -u):$(id -g)" --env PYTHONPATH=/workspace/src:/workspace --env OMP_NUM_THREADS=2 \
  --env MKL_NUM_THREADS=2 --env POLARS_MAX_THREADS=4 --env "ALASKA2_PYTHON=$runtime/.venv/bin/python" \
  --env "ALASKA2_RUN_ID=$run_id" --workdir /workspace \
  --mount "type=bind,source=$uv_python,target=$uv_python,readonly" \
  --mount "type=bind,source=$runtime,target=$runtime,readonly" \
  --mount "type=bind,source=$repo,target=/workspace,readonly" \
  --mount "type=bind,source=$data,target=/inputs,readonly" \
  --mount "type=bind,source=$output,target=/output" \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nice -n 19 bash /workspace/tools/runs/retained_gpu.sh --inside
