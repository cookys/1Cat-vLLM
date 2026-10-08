#!/usr/bin/env bash
# CPU-only. Invoke through scripts/fence.sh; no build/install into a venv.
set -euo pipefail
fa_dir=$(cd -- "$(dirname -- "$0")" && pwd)
fa_build=${FA_BUILD:?Set FA_BUILD to a new output directory}
fa_python=${FA_PYTHON:-/data/venvs/1cat-main-integp7pr-p8/bin/python}
fa_cuda=${FA_CUDA:-/usr/local/cuda-12.9}
fa_cxx=${FA_CXX:-/usr/bin/g++-14}
export CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 TMPDIR=/data/tmp
mkdir -p /data/tmp
mkdir "$fa_build"
"$fa_python" "$fa_dir/generate.py" --out "$fa_build"
"$fa_cuda/bin/nvcc" --version > "$fa_build/nvcc-version.txt"
for fa_variant in 0 1 2 1_free 2_free; do
  "$fa_cuda/bin/nvcc" -ccbin "$fa_cxx" -shared -Xcompiler=-fPIC -O3 -std=c++17 \
    -gencode arch=compute_70,code=sm_70 --use_fast_math \
    --expt-relaxed-constexpr --expt-extended-lambda \
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
    -U__CUDA_NO_HALF2_OPERATORS__ --ptxas-options=-v \
    "$fa_build/proto${fa_variant}.cu" -o "$fa_build/proto${fa_variant}.so" \
    > "$fa_build/proto${fa_variant}.build.log" 2>&1
  "$fa_cuda/bin/cuobjdump" --dump-resource-usage "$fa_build/proto${fa_variant}.so" \
    > "$fa_build/proto${fa_variant}.resources.txt" 2>&1
  "$fa_cuda/bin/cuobjdump" --dump-sass "$fa_build/proto${fa_variant}.so" \
    > "$fa_build/proto${fa_variant}.sass" 2>&1
done
"$fa_python" "$fa_dir/resources.py" "$fa_build" > "$fa_build/manifest.json"
