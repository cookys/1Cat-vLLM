# A+A1B: A loader + explicit shared WMMA layout

CPU-only package; GPU not executed. Existing verify-v2, A1, A1b and serving stages remain unchanged. This is a new combination arm, not evidence that the two speedups add or multiply.

**Lineage.** Winning A is the original `verify-v2` native `fe2eafc79db9beee` selected with `FLASH_V100_VERIFY_ARM=A`; BASE is deployed Proto B `75b4dafd1c435929`. `origin.json` pins the source dependencies. `a1_verify.cuh` is derived from winning `pvb_a.cuh`, preserving A's paired V(t)/K(t+1) raw loads, branchless E4M3 conversion, page-base handling and warp-uniform rescale==1 skip. It does not contain A2's explicit raw-load PTX. Only shared layout/reader/writer substitutions are added; reversing `layout-transform.json` makes the kernel and paired loader byte-identical to A.

**Both writers must move.** The initial K loader uses `a1_loader.cuh`. A's separate paired loader has four `raw_store` uint4 writes with a hard-coded row stride264, so it must also use the new stride256 and common XOR mask; swapping only WMMA readers would corrupt values. `a_on_loader.cuh` changes those four destination indices only, keeping A's raw load order, dequant operations and lane assignment. Q/KV strides264→256 and P40→64 leave total dynamic shared size87552B; M48/N32,512threads,80splits and arithmetic order remain A's.

**CPU evidence.**

- `cpu-layout.json`: 60,416 compiler-probed WMMA fragment half slots and270,336 initial-loader tail slots (valid_rows0…32); actual constexpr C++ mask/offset compiled for host checking.
- `cpu-on-A.json`: actual old/new paired-loader C++ functions, address-only conversion/intrinsic shims, 3,267 combinations (V/Knext valid0…32, offsets0/32/4064 with next offset wrapping4096→0),53,526,528 half-slot comparisons and output canaries. kv=0 leaves Knext untouched. Linear-reader counterexample fails. Shims test addresses/tails, not GPU FP arithmetic; conversion expressions are byte-identical to A by inverse-source contract.
- `cpu-dispatch.json`: exercises the actual bench select/fwd functions with three distinct native modules/output/workspaces and checks flags: BASE unset, A=A, combination=A1.
- `import-check.json`: all three .so modules import, Torch CUDA uninitialized. `native-inspection.json` and sass/ contain the real per-build resources/instructions; combiner equality is checked.

**Bank-model limit.** A's paired Phase-A producer retains `r=tid>>3,u=tid&7`. Its STS128 CPU model is4-way before and after this layout change; the initial K producer's4-row remap remains1-way. Do not quote the independent A1/A1b producer result as covering A's paired producer. All models still need NCU validation. No thread reordering or extra loader optimization is mixed into this additivity experiment.

## Run after the existing independent A1/A1b arms (lead owns GPU window)

First `python verify-package.py`. No server changes. Short L33/65/4097 first; then B4/B7 ×L131065/253953,3rounds×20iterations. This verify API has fixed80splits, not nospec P256/P1024.

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /home/cookys/projects/llm-playground/scripts/fence.sh --mem 16G --cpu 8 -- /data/venvs/1cat-main-integp7pr-p8-gvB-integ/bin/python /data/bench/explore/pv-hmma/a1b-on-A/bench.py --execute --batch 4 --length 253953 --rounds 3 --iters 20 --out /data/bench/explore/pv-hmma/out-a1b-on-A/b4-L253953
```

The bench runs BASE / A / A+A1B on the same Q/K/V and metadata. Each module path and .so hash is recorded. The combined .so retains its internal A1 env/kernel name; the benchmark's external label is A+A1B. All arms require finite output and exact byte equality (including signed zero) to BASE before timing. CUDA-graph replay must also equal eager output. Outputs include per-arm dpart vs BASE and `dpart_combo_vs_A`, so the incremental result cannot be confused with A's own gain.

For bounded NCU use `--profile-once --arm A+A1B` (or A/BASE), NVTX PVHV_NCU, partial+combine kernels only, under lead's launch/time limits. Compare shared load/store wavefronts, short/long scoreboard, barrier and partition_ms. No speed claim is inferred from the static instruction table. Without `--execute`, only a CPU plan is printed.
