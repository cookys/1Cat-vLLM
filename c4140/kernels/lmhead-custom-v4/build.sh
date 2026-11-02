#!/usr/bin/env bash
# Build sm70_lmhead_custom_v4.so (nvcc 12.9, sm_70, -lineinfo, ptxas -v logged). No GPU (CUDA_VISIBLE_DEVICES empty). Through scripts/fence.sh. usage: build.sh [OUTDIR]  (default: this dir)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"; OUT="${1:-$HERE}"
FENCE=/home/cookys/projects/llm-playground/scripts/fence.sh
NVCC=/usr/local/cuda-12.9/bin/nvcc
mkdir -p "$OUT" /data/tmp
export CUDA_VISIBLE_DEVICES= TMPDIR=/data/tmp
"$FENCE" --mem 24G --cpu 8 --name lmcu4-build -- "$NVCC" -ccbin /usr/bin/g++-14 -std=c++17 -O3 -arch=sm_70 -lineinfo -Xptxas -v --shared -Xcompiler -fPIC -Xlinker --build-id=none \
  -o "$OUT/sm70_lmhead_custom_v4.so" "$HERE/sm70_lmhead_custom_v4.cu" 2> "$OUT/ptxas-v.log"
cat "$OUT/ptxas-v.log"
sha256sum "$OUT/sm70_lmhead_custom_v4.so"
