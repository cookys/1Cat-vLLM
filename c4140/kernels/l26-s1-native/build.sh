#!/usr/bin/env bash
# usage: build.sh <arm> <tag>  -> build/<tag>/ ; fenced, CPU only
set -u
B=/data/bench/explore/l25-lut-fold-v1/build
cd /home/cookys/projects/llm-playground
mkdir -p $B/$2
TMPDIR=$B/tmp MAX_JOBS=8 CUDA_VISIBLE_DEVICES= scripts/fence.sh --mem 48G --cpu 8 -- python3 $B/build_arm.py $1 $B/$2 > $B/build-$2.log 2>&1
echo rc=$? > $B/DONE-$2
