#!/bin/bash
R=/home/cookys/projects/llm-playground; B=/data/bench/explore/pool-serve-v1/build
cd $R
run() { t0=$(date +%s); CUDA_VISIBLE_DEVICES= $R/scripts/fence.sh --mem 24G --cpu 6 --name poolc-$1 -- bash $B/compile-awq.sh $1 $2 > $B/out/$1.run.log 2>&1; echo "$1 rc=$? secs=$(( $(date +%s)-t0 ))" >> $B/out/run-compiles.log; }
run noD2E /data/src/kgm112-v13 &
run D2E /data/bench/explore/d2-empty-v1/work/src &
wait; echo ALLDONE >> $B/out/run-compiles.log
