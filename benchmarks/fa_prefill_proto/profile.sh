#!/usr/bin/env bash
# Lead-only GPU window. Compile beforehand; no service/process management here.
set -euo pipefail
: "${FA_GPU:?Set FA_GPU=0 for the GPU granted by lead}"
: "${FA_OUT:?Set FA_OUT to a new result directory}"
: "${FA_BUILD:?Set FA_BUILD to the verified five-arm build directory}"
[[ "$FA_GPU" == 0 && "${FA_RUN_GPU:-0}" == 1 ]] || {
  printf '%s\n' 'Require FA_GPU=0 and FA_RUN_GPU=1 after lead grants the window.' >&2
  exit 2
}
fa_dir=$(cd -- "$(dirname -- "$0")" && pwd)
fa_python=${FA_PYTHON:-/data/venvs/1cat-main-integp7pr-p8/bin/python}
fa_fence=/home/cookys/projects/llm-playground/scripts/fence.sh
export TMPDIR=/data/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "${FA_IN_FENCE:-0}" != 1 ]]; then
  export FA_IN_FENCE=1
  exec "$fa_fence" --mem 24G --cpu 8 --name astra-fa-gpu -- bash "$0"
fi
mkdir -p /data/tmp
mkdir "$FA_OUT"
df -B1 /tmp /data/tmp > "$FA_OUT/df-before.txt"
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
  "$fa_python" "$fa_dir/probe.py" --build "$FA_BUILD" --describe --check-space \
  > "$FA_OUT/preflight.json"
# No profiler/counter permissions required. Runtime occupancy API is read-only.
CUDA_VISIBLE_DEVICES="$FA_GPU" TRITON_INTERPRET=1 \
  nice -n 19 taskset -c 0-26:2 \
  "$fa_python" -u "$fa_dir/probe.py" --build "$FA_BUILD" --out "$FA_OUT" \
  --run-on-gpu > "$FA_OUT/probe.log" 2>&1
df -B1 /tmp /data/tmp > "$FA_OUT/df-after.txt"
printf '%s\n' "$FA_OUT/results.json" "$FA_OUT/results.md"
