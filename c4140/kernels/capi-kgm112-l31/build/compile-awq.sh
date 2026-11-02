#!/bin/bash
# usage: compile-awq.sh <variant> <W include root>; nvcc argv = l31-fuse-v2 build.sh, include roots -> $W
set -euo pipefail
VAR=$1; W=$2
B=/data/bench/explore/pool-serve-v1/build
D=/data/src/1cat-wt-p075-develop
V=/data/venvs/1cat-main-integp7pr-p8-gvB-integ
T=$V/lib/python3.12/site-packages/torch
SRC=/data/bench/explore/l31-fuse-v2/pack/build-receipt/awq_sm70_gemm.cu
O=$B/out/$VAR; mkdir -p $O
export TMPDIR=$B/tmp CUDA_VISIBLE_DEVICES=
echo "compile start $(date -Is)"
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-14 \
  -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_NVSHMEM -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS \
  -I$W/csrc -I$D/.deps/cutlass-src/include -I$W/csrc/sm70_turbomind/lmdeploy -I$D/.deps/cutlass-src/tools/util/include -I$W/csrc/sm70_turbomind/ops \
  -isystem /home/cookys/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/include/python3.12 \
  -isystem $T/include -isystem $T/include/torch/csrc/api/include -isystem /usr/local/cuda/include \
  -DONNX_NAMESPACE=onnx_c2 \
  -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl \
  --expt-relaxed-constexpr --expt-extended-lambda -O2 -g -DNDEBUG -std=c++20 -Xcompiler=-fPIC --expt-relaxed-constexpr -DENABLE_FP8 --threads=1 -DENABLE_SM70_TURBOMIND=1 \
  -gencode arch=compute_70,code=sm_70 -Xptxas -v \
  -x cu -c $SRC -o $O/awq_sm70_gemm.cu.o 2> $O/awq_sm70_gemm.ptxas.log
echo "compile done $(date -Is)"
