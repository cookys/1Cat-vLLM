# Prefill FA negative-screen prototypes (2026-10-08)

**Stopped at the CPU resource gate. Neither candidate is approved for a GPU
window or serving use.** These are isolated translation units, not a production
dispatch change. No installed library or serving Python file is modified.

The experiment started from `de01359b8033144458ac7a495d73b592a1c5f6a0`.
`generate.py` pins the complete SHA256 of
`flash-attention-v100/kernel/fused_mha_forward_paged.cu`, extracts the BM32/D256
phase body, and emits independent P0/P1/P2 shared libraries. P0 preserves that
body apart from the namespace and kernel symbol. A standalone P0 still needs
GPU byte comparisons against the installed entry point before any future use
as a numerical reference.

## Decision and evidence

Fixed resource gate: sm_70, 512 threads, at most 64 registers, at most 48 KiB
shared storage, **zero spill stores/loads and zero SASS LDL/STL**. The gate also
checks a resource-only bound of two resident CTAs/SM. Runtime occupancy,
correctness, and speed are separate GPU gates, which were not run.

| Variant | Registers | Shared bytes | Stack bytes | Spill store/load bytes | SASS LDL+STL | Decision |
|---|---:|---:|---:|---:|---:|---|
| P0, extracted baseline | 64 | 41,936 | 0 | 0 / 0 | 0 | CPU resource pass only |
| P1-r1, own-warp scratch / padded score rows | 64 | 43,984 | 8 | 4 / 4 | 2 | Reject |
| P1-r2, one permitted lifetime adjustment | 64 | 44,000 | 8 | 4 / 4 | 2 | Reject; stop P1 |
| P2, 16 QK warps | 64 | 41,936 | 8 | 8 / 8 | 4 | Reject; stop P2 |

Compiler: CUDA 12.9.86, g++-14, sm_70, `-O3 --use_fast_math`; complete flags are
in `build.sh`. Baseline and candidates use the same flags and launch bounds.
PTXAS reports each candidate's spill count on line 5 of its `protoN.build.log`;
line 6 reports registers and shared bytes.

Preserved artifacts on the test host:

- `/data/bench/astra-fa-proto/build/`: first pass P0/P1-r1/P2, including generated
  source, libraries, compiler reports, SASS, and SHA256 manifest.
- `/data/bench/astra-fa-proto/build-r2/`: final P1-r2 plus unchanged P0/P2
  artifacts; `manifest.json` records both candidates as `RESOURCE_VETO`.
- `/data/bench/astra-fa-proto/cpu-tests.log`: 14 CPU tests passed.

P1-r1 spills the persistent N-block bound (`proto1.sass:899,5555`). The single
permitted adjustment moves it to a shared `volatile int`, published by the
existing initial block barrier. P1-r2 instead contains a spill at SASS PC 0x1c10
and reload at 0x50c0 (`build-r2/proto1.sass:919,2605`): the spill remains 4 bytes.
This is a compiler resource observation, not measured latency or spill traffic.

The current generator emits P1-r2. The rejected first-pass source and hashes
remain in the first artifact directory. No third allocation attempt, altered
launch bound, GPU script, GPU timing, or serving overlay is part of this result.

## Intended changes, not established GPU correctness

P1 changes only the physical score leading dimension to 144 floats and scratch
ownership to an individual warp. Each warp's 32-by-16 strip contains its two
PV accumulators. A vector transaction maps each half warp's 32 words onto
distinct banks. A warp barrier replaces the paired-warp scratch barrier; all
block barriers remain. QK still uses eight warps and the same K reduction order.
The resource revision adds one read-only shared scalar after initialization.

P2 assigns one 16-by-16 QK panel to each of 16 warps without splitting K.
Each warp spills one PV accumulator into its own score quadrant and retains the
other in registers. Softmax and PV bodies remain unchanged. Mapping tests do
not replace GPU racecheck/synccheck or output/LSE byte comparison.

The preregistered speed gate, never reached, required X126 (M4032/N200704)
unprofiled ABBA time at least 10% lower, no M96 regression, byte-identical output
and LSE, and two resident CTAs/SM. Neither speedup nor E1 has been demonstrated.
The rejection applies to these candidates under these resource constraints,
not every possible FA rewrite.

## CPU reproduction

Use a fresh artifact directory so previous reports are not overwritten. Run on
the build host through its CPU/memory fence; these commands hide all GPUs:

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
  /data/venvs/1cat-main-integp7pr-p8/bin/python \
  benchmarks/fa_prefill_proto/test_cpu.py

CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 FA_BUILD=/data/bench/astra-fa-proto/rebuild \
  nice -n 19 taskset -c 0-26:2 \
  /home/cookys/projects/llm-playground/scripts/fence.sh --mem 12G --cpu 4 \
  --name astra-fa-rebuild -- benchmarks/fa_prefill_proto/build.sh
```

Compilation success is not gate success: inspect `manifest.json`. The test file
uses only the Python standard library, imports no torch, and never loads a CUDA
library. It tests pinned extraction, layout ownership, K order, and fail-closed
resource screening; it makes no claim about GPU outputs.
