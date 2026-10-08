#!/usr/bin/env bash
# Lead window only. Production down/restore belongs to the external wrapper.
# CPU plan: M37_OUT=/data/bench/m37b bash m37-parking.sh
# Execute: M37_GO=1 M37_OUT=/data/bench/m37b bash m37-parking.sh --execute
set -euo pipefail
parking_wt=${M37_WORKTREE:-/data/src/1cat-wt-astra-host-parking}
parking_py=${M37_PYTHON:-/data/venvs/1cat-main-integp7pr-p8/bin/python}
parking_repo=${M37_REPO:-/home/cookys/projects/llm-playground}
parking_out=${M37_OUT:-/data/bench/m37b}
parking_execute=0
parking_extra=()
while (($#)); do
  case "$1" in
    --execute) parking_execute=1; shift ;;
    --out)
      [[ $# -ge 2 && -n $2 ]] || { echo '--out requires a path' >&2; exit 2; }
      parking_out=$2; shift 2 ;;
    --out=*) parking_out=${1#--out=}; shift ;;
    *) parking_extra+=("$1"); shift ;;
  esac
done
[[ -n $parking_out ]] || { echo 'Output path must not be empty' >&2; exit 2; }
parking_args=(--python "$parking_py" --worktree "$parking_wt" --repo "$parking_repo" "${parking_extra[@]}" --out "$parking_out")
if [[ $parking_execute == 0 ]]; then
  exec env CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 "$parking_py" "$parking_wt/benchmarks/host_parking_window.py" "${parking_args[@]}"
fi
[[ ${M37_GO:-0} == 1 ]] || { echo 'M37_GO=1 from lead GPU wrapper is required' >&2; exit 2; }
[[ ! -e $parking_out && ! -L $parking_out ]] || { echo "Refuse to overwrite existing $parking_out" >&2; exit 2; }
# Driver handles SIGTERM/ALRM by reaping only its own isolated child sessions.
# Fence may lower 48G to preserve 16G system reserve; actual cap is logged.
exec "$parking_repo/scripts/fence.sh" --name m37-parking --mem 48G --cpu 16 --high 100 -- \
  timeout --signal=TERM --kill-after=180 10800 \
  env TMPDIR=/data/tmp "$parking_py" "$parking_wt/benchmarks/host_parking_window.py" \
  --execute "${parking_args[@]}"
