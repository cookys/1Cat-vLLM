#!/bin/bash
# usage: compile-tu.sh <outdir under build/> ; compiles ONE object sm70_884_4.cu from /data/src/kgm112 with the develop-build flags (+ -Xptxas -v).
# Run under: CUDA_VISIBLE_DEVICES= scripts/fence.sh --mem 48G --cpu 24 -- env MAX_JOBS=12 bash compile-tu.sh proof
set -e
OUT=$1
W=${W:?}
B=$W/build
D=/data/src/1cat-wt-p075-develop        # read-only: cutlass .deps only
V=/data/venvs/1cat-main-integp7pr-p8-gvB-integ
T=$V/lib/python3.12/site-packages/torch
SRC=${SRC:-csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu}
mkdir -p "$B/$OUT"
export TMPDIR=$B/$OUT; 
echo "start $(date -Is)"
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++-14 \
  -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_NVSHMEM -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS \
  -I$W/csrc -I$D/.deps/cutlass-src/include -I$W/csrc/sm70_turbomind/lmdeploy -I$D/.deps/cutlass-src/tools/util/include \
  -isystem /home/cookys/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/include/python3.12 \
  -isystem $T/include -isystem $T/include/torch/csrc/api/include -isystem /usr/local/cuda/include \
  -DONNX_NAMESPACE=onnx_c2 \
  -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl \
  --expt-relaxed-constexpr --expt-extended-lambda -O2 -g -DNDEBUG -std=c++20 -Xcompiler=-fPIC --expt-relaxed-constexpr -DENABLE_FP8 --threads=1 -DENABLE_SM70_TURBOMIND=1 \
  -gencode arch=compute_70,code=sm_70 -Xptxas -v \
  -x cu -c "$W/$SRC" -o "$B/$OUT/$(basename $SRC).o" 2> "$B/$OUT/$(basename $SRC).ptxas.log"
echo "done $(date -Is) rc=$?"
