# BUILD-RECEIPT — d1 s1_verify dequant SASS diet (3 arms, one recipe)

Branch `c4140-d1-sass-diet` in /data/src/1cat-wt-fable-d1-diet, HEAD `59d8df8757459af8f969cb6af5378de177d3d309` (parent 2400dade58 = c4140-p075-develop-k7a). Not pushed.

Macro `S1_DEQUANT_DIET`: 0 = BASE (original code path, compiled), 1 = CS (Diet C sign-bit pre-align word decode, scale function untouched), 2 = CSEP (CS + per-byte `scale_u8_to_half2` HMUL2 scale path). Diet B was dropped by the DESIGN 09:1x amendment. Code source: agy live-TU arms (`agy-sass-diet-wave2-livetu-20261024`).

Recipe: `build_arm.py <arm>` = the repo's `_s1_native/source/build-receipt/build.py` argv with `-DS1_DEQUANT_DIET=<n>` appended, same live translation unit path, same common objects, same g++-14 link. The three compile argv differ only in `-o` and the macro (checked by diff). The .so is NOT bitwise reproducible (memory s1-so-not-bitwise-reproducible), so BASE is the macro-0 rebuild from this recipe, not the pinned fe35ca27 copy.

| arm | macro | .so sha256 | stage |
|---|---|---|---|
| BASE | 0 | `a123870683f45711790ae2bc98593195f77cb74c2d5fa87bb09dab8030859110` | `/data/bench/explore/stage-d1-diet-base-59d8df8` |
| CS | 1 | `3aa2a9d20b8bfb182e813d79331275ab2e4c6277b44b15980a827b560f94de6b` | `/data/bench/explore/stage-d1-diet-cs-59d8df8` |
| CSEP | 2 | `89095e273b91fa30d7569ffd1a88a31b1a9ed6f1769001efd5d1116408fd918b` | `/data/bench/explore/stage-d1-diet-csep-59d8df8` |

Stage = `cp -al` of `/data/bench/explore/stage-p075-develop-k7a-2400dad` with only `flash_attn_v100/_s1_native/<so>` replaced by a private (link count 1) copy; every other file is the parent's hardlink. Live `_s1_native` in the parent: sha256 prefix fe35ca27.

## Static counts (cuobjdump SASS of functions containing `flash_attention_s1_verify_partial_kernel`; ptxas -v)

| arm | SASS | regs | stack B | spill st/ld B | LDL/STL in HMMA span |
|---|---|---|---|---|---|
| LIVE fe35ca27 | 2688 | - | - | - | - |
| BASE | 2688 | 128 | 8 | 8/4 | 0 |
| CS | 2480 | 128 | 8 | 8/4 | 0 |
| CSEP | 2440 | 128 | 8 | 8/4 | 0 |

SASS delta vs BASE: CS -208, CSEP -248 (agy live-TU: -208 / -248, identical). BASE equals the live .so count (2688).

## argv (verbatim files in this directory: `<arm>/compile.argv.json`, `<arm>/link.argv.json`, `<arm>/ptxas.log`, `<arm>/build-receipt.json`)

```
/usr/local/cuda-12.9/bin/nvcc -c /data/src/1cat-wt-fable-d1-diet/flash-attention-v100/flash_attn_v100/_s1_native/source/kernel/flash_decode_paged.cu -o /data/bench/explore/d1-diet-build/csep/flash_decode_paged.o -O3 -std=c++17 -gencode arch=compute_70,code=sm_70 --use_fast_math --ptxas-options=-v --expt-relaxed-constexpr --expt-extended-lambda --compiler-options=-fPIC -ccbin /usr/bin/gcc-14 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=flash_attn_v100_cuda -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -DS1_DEQUANT_DIET=2 -I/data/src/1cat-wt-fable-d1-diet/flash-attention-v100/flash_attn_v100/_s1_native/source/kernel -I/data/src/1cat-wt-fable-d1-diet/flash-attention-v100/flash_attn_v100/_s1_native/source/include -I/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/torch/include -I/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/home/cookys/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12
```

```
/usr/bin/g++-14 -shared /data/bench/explore/d1-diet-build/csep/flash_decode_paged.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/fused_mha_api.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/fused_mha_forward.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/fused_mha_forward_paged.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/fp8_kv_bridge.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/flash_decode_turboquant.o /data/bench/explore/pv-hmma/harness/native-build-base/temp/kernel/fused_mha_backward.o -L/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/torch/lib -L/usr/local/cuda-12.9/lib64 -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda -o /data/bench/explore/d1-diet-build/csep/lib/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so
```

(CSEP shown; CS/BASE differ only in `-o` path and the macro value 1 / 0.)

## Edited headers (sha256 prefix16)

- `fp8_kv_utils.cuh` c6a98382ed1aef55
- `a_on_loader.cuh` d9ef2c3717ec14a6
- `s1_verify.cuh` 3b2c0bf8f4dcaac8
- translation unit `flash_decode_paged.cu` 11f1c66656734c57 (unedited)

Caveat: the six common `.o` (fused_mha_*, fp8_kv_bridge, flash_decode_turboquant, fused_mha_backward) were built earlier from the unedited headers and are reused unchanged by all three arms; only `flash_decode_paged.o` (which holds the s1 verify kernel) is compiled per arm.
