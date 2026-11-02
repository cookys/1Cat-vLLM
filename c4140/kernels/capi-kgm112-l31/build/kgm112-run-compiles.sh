#!/bin/bash
# sequential fenced compiles; logs in each build dir
KP=csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm
R=/home/cookys/projects/llm-playground
cd $R
t0=$(date +%s)
CUDA_VISIBLE_DEVICES= W=/data/src/kgm112-v13 SRC=$KP/kernel/sm70_884_4.cu $R/scripts/fence.sh --mem 48G --cpu 24 --name kgm13-884 -- env MAX_JOBS=12 W=/data/src/kgm112-v13 SRC=$KP/kernel/sm70_884_4.cu bash /data/src/kgm112-v13/build/compile-tu.sh o884 > /data/src/kgm112-v13/build/o884.run.log 2>&1; echo "884 rc=$? secs=$(( $(date +%s)-t0 ))"
t0=$(date +%s)
$R/scripts/fence.sh --mem 48G --cpu 24 --name kgm13-gemm -- env MAX_JOBS=12 W=/data/src/kgm112-v13 SRC=$KP/gemm.cu bash /data/src/kgm112-v13/build/compile-tu.sh ogemm > /data/src/kgm112-v13/build/ogemm.run.log 2>&1; echo "kgm13 gemm rc=$? secs=$(( $(date +%s)-t0 ))"
t0=$(date +%s)
$R/scripts/fence.sh --mem 48G --cpu 24 --name live13-gemm -- env MAX_JOBS=12 W=/data/src/kgm112-v13-live SRC=$KP/gemm.cu bash /data/src/kgm112-v13/build/compile-tu.sh ogemm > /data/src/kgm112-v13-live/build/ogemm.run.log 2>&1; echo "live13 gemm rc=$? secs=$(( $(date +%s)-t0 ))"
