#!/usr/bin/env bash
# EXPLORATORY launch (copy of /data/bench/s3-runner/w2/launch-w2-v3.sh; spec=dflash2, MAX_SEQS=20). serve.sh up self-fences (80G/28cpu) and backgrounds.
set -euo pipefail
GUARD_GMU=${GMU:-0.92}
for arg in "$@"; do
  case "$arg" in --gpu-memory-utilization|--gpu-memory-utilization=*|-gmu|-gmu=*)
    echo 'Use GMU env; duplicate CLI gmu would detach the guard policy from the launch.' >&2; exit 2;; esac
done
/data/venvs/1cat-main-integp7pr-p8-gvB-integ/bin/python -B -c 'import sys; sys.path.insert(0,"/data/bench/explore/preset-guard-v2"); from preset_guard_v2 import parse_gmu; parse_gmu(sys.argv[1])' "$GUARD_GMU"
R=/home/cookys/projects/llm-playground; STAGE=/data/bench/s2-impl/c1fix/stage-FIX; VENV=/data/venvs/1cat-main-integp7pr-p8-gvB-integ
PATH_V=/home/cookys/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
exec setsid -w env -i PYTHONPATH="$STAGE:/data/bench/explore/preset-guard-v2" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/tmp PATH="$PATH_V" HOME=/home/cookys USER=cookys LOGNAME=cookys \
  VLLM_TEAM_PRESET_GUARD=1 VLLM_TEAM_PRESET_GMU="$GUARD_GMU" \
  LANG=en_US.UTF-8 TERM=dumb XDG_RUNTIME_DIR=/run/user/1001 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus \
  C4140_PRINT_ARGV="${C4140_PRINT_ARGV:-0}" C4140_PORT=8021 C4140_HOST=127.0.0.1 C4140_VENV="$VENV" C4140_SPEC=dflash2 C4140_MAX_SEQS=20 C4140_KV_DTYPE=fp8_e4m3 \
  C4140_WRAP="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$STAGE:/data/bench/explore/preset-guard-v2" \
  VLLM_KV_CACHE_LAYOUT=NHD VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY=1 VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_EXTRA_PAGES=2048,4096 \
  VLLM_FLASH_V100_NVFP4_FIRST_CHUNK_FP16=1 VLLM_FLASH_V100_NVFP4_PREFIX_DECODE_ROWS=1 VLLM_FLASH_V100_GROUPED_VERIFY_PER_REQUEST_FALLBACK=1 \
  VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY=1 VLLM_USE_V2_MODEL_RUNNER=1 \
  "$R/scripts/c4140-1cat-serve.sh" up --middleware preset_guard_v2.PresetGuard --kv-cache-dtype nvfp4 --block-size 4096 --mamba-block-size 4096 --enable-prompt-tokens-details \
  --enable-cumem-allocator --spec-hot-switch-elastic "$@" --gpu-memory-utilization "$GUARD_GMU"
