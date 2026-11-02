#!/usr/bin/env bash
# Build sm70_lmhead_lt.so (host code only; headers + libcublasLt.so.12 from the serving venv's nvidia wheels). Through scripts/fence.sh. usage: build.sh [OUTDIR]  (default: this dir)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"; OUT="${1:-$HERE}"
SP=/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/nvidia
FENCE=/home/cookys/projects/llm-playground/scripts/fence.sh
mkdir -p "$OUT"
"$FENCE" --mem 8G --cpu 4 --name b3-lmhead-build -- g++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra -I"$SP/cublas/include" -I"$SP/cuda_runtime/include" -I"$SP/cuda_nvcc/include" -I"$SP/cuda_cccl/include" "$HERE/sm70_lmhead_lt.cpp" \
  -o "$OUT/sm70_lmhead_lt.so" -L"$SP/cublas/lib" -Wl,--no-as-needed -l:libcublasLt.so.12 -Wl,-rpath,"$SP/cublas/lib" -Wl,-soname,sm70_lmhead_lt.so -Wl,--build-id=none
sha256sum "$OUT/sm70_lmhead_lt.so"
