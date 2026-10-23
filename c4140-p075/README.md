# c4140-p075 develop branch

Base: 120128680 (python tree of venv 1cat-main-integp7pr-p8-gvB-integ).
Commits: stage-active-v1f python snapshot, S1 verify attention (source + routing),
K-GEMM P1b whole-M tiles, P5-B threads 256, this directory.

## Launch env (required)
- VLLM_SM70_NVFP4_WHOLE_M_TILES=1
- VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M=160
- TEAM_ARM=S1 (FLASH_V100_VERIFY_ARM=S1; A1B is not loaded, its .so has no recovered source)

K7a is not included; merge it after K7a v1b PASS.

## Not in git
- Built .so files (ignored). Expected: flash_attn_v100/_s1_native/EXPECTED-SO.txt.
- vllm/_version.py (generated, gitignored); listed in STAGE-MANIFEST.txt.
- csrc: 120128680 adds KV_CACHE_DTYPE_NVFP4 = 3 in sm70_grouped_long/kernel/fp8_kv_utils.cuh
  versus d30469863; no .cu uses it, so a d30469863-based .so is equivalent.
