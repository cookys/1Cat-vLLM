# C4140 PLE batched prefill temporary storage

Base: `c4140-p069@f93c5ae32`. Candidate branch: `p069-astra-ple-memory`.
Opt-in: `VLLM_QWEN4EXP_PLE_PREFILL_LOW_MEMORY=1`; default `0`.
No C++/CUDA changes, model weights, KV geometry, batching or scheduler changes.

## Problem and scope

The C4140 c=3 crash had scheduler lengths `prefill=[8080,107], spec=[5]`.
The PLE convolution operates on 10240 FP16 channels, replicated across TP.
The two prefills are packed to `[2,8080,10240]`: each padded tensor costs
315.625 MiB per rank. The first OOM requested 316 MiB in SiLU. A later
exception requested 222 MiB; it is not the shape of the first failing step.

The original implementation can keep the packed tensor, concatenated history,
convolution output, SiLU output and contiguous transposed output alive together,
as well as a separately allocated unpadded output. V2 memory profiling skips
attention metadata and instead runs PLE's flattened fallback; it misses this
batched path and its padding overhead.

Production mitigation is per-request prefill threshold 1616 and GPU utilization
0.92. Lead reports c=1..4 survive, but single-request prefill is about 17% slower
and three of six 16K text parity cases diverge late. Utilization 0.92 alone
still OOMs at c=3. This patch attempts to recover the original chunk schedule.

## Chain13 and conservative revision

The first version, `401f950bd`, survived c=1..4 at 16K, c=4 at 30K and mixed
2x30K+2x4K at utilization 0.92, threshold 0, p3 on. Lead reports 5157/4896
tok/s single-request 16K/64K prefill, recovering the threshold's throughput
loss. Sampled peak memory.used was 32372 MiB/rank (500 ms sampling, not the
allocator's true peak). However, seven of twelve text-parity cases differ from
`c11-P3-nolp`, including three 1K cases. Same-server replay is identical. The
first version therefore failed the E1 gate despite CPU byte equality.

The final output stride alone is not an established explanation: advanced
indexing produces a contiguous tensor and the caller stacks/copies it into an
existing opaque-op output. The inner short-conv runs inside the registered
custom op, so the outer compiled stack frame is not evidence of Inductor
fusing the internal SiLU. The revision nevertheless restores these contracts
to eliminate unnecessary differences.

Another hypothesis is library convolution selection. The first version freed
packed_tokens and omitted the output allocation **before** F.conv1d, increasing
workspace headroom. The serving interpreter is PyTorch 2.10.0+cu128, git
449b1768410104d3ed79d3bcfe4ba1d65c7f22c0. Its [cuDNN plan selection source](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cudnn/Conv_v8.cpp)
tries configurations and continues after workspace allocation failures. This
supports a possible mechanism, not a measured algorithm change in chain13.
Fresh CPU-only interpreter defaults are benchmark=False/deterministic=False;
these are not a readback of the running workers. GPU operator/plan diagnosis
is still needed. deterministic=True alone would not force the old baseline's
algorithm or prove identical outputs across workspace conditions.

## Changes and exactness boundary

Only the opted-in batched-prefill method changes:

1. Preserve the legacy output allocation and keep packed_tokens and history
   alive through the convolution call. Do not change its local live set.
2. After the **same convolution**, gather a copy of the small state tail and
   release history and packed_tokens. Keep state writeback after output.
3. Apply the legacy **out-of-place** SiLU, then make the transposed result
   contiguous in a separate statement. This releases the original convolution
   result before creating the transpose copy, avoiding three live padded slabs.
4. Preserve the legacy masking, advanced indexing and copy into empty_like(x_p),
   including dense non-contiguous output strides. Release the last padded slab
   after copying the selected output.

Convolution inputs, strides, dtype, weights, groups, dilation, padding width and
batch shape are unchanged. The state-tail gather is a copy, not an alias that
would retain history. Valid/NULL/empty-row state masks, speculative tail bytes
and state writeback order are unchanged. Inputs and weights are never mutated.
All tensor work remains on the current stream, with no tensor-to-host scalar
reads or additional synchronization. PyTorch manages storage reuse on that
stream, including during graph capture. No new persistent buffers are added.

Expected major storage peak changes from approximately `5*P*L*C*2 + S*C*2`
to `3*P*L*C*2 + S*C*2 + small_state`, excluding caller-owned inputs,
convolution-library workspaces and unrelated activations. The [8080,107] crash
shape saves about **631.25 MiB/rank** (two padded slabs), rather than the first
version's estimated 1.08 GiB. The local convolution entry now matches legacy
memory liveness. This is a storage-liveness model, **not a measured CUDA peak**;
the conservative revision must repeat c=4/mixed survival at 0.92.

The intent remains E1, with the legacy arithmetic, layout and convolution-entry
live tensors restored. CPU byte equality, operator/layout and local storage
liveness are verified. This does not fix the library algorithm explicitly, or
reproduce the global caching allocator's history; GPU parity remains required.
The cause of the first version's divergence is **not yet localized**. No claim
that restoring these contracts has already resolved numerical parity or that
chain13's survival automatically transfers to this higher-memory revision.

Route hit: `Qwen4Exp PLE low-memory batched prefill enabled (legacy convolution
live set, SiLU and output layout).` once per worker.
The env accepts only `0` and `1`; an unset variable preserves the legacy path.

## CPU validation

Run standalone files to avoid repository-wide CUDA fixtures:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /data/src/1cat-wt-astra/.venv/bin/python -B \
  tests/models/qwen4_exp/test_ple_prefill_low_memory_cpu.py

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /data/venvs/1cat-m589/bin/python -B \
  tests/models/qwen4_exp/test_ple_prefill_env_import_cpu.py
```

The first executes the actual production methods extracted with AST, substituting
only the layer shell, metadata and env getter. It compares legacy/opt-in output
and full state bytes, checks an independent history-tail oracle, conv input/layout
identity, NULL/empty rows, zero history, strided inputs/state, mixed state dtype,
special values, the failing token lengths at smaller channel count and full
10240-channel width at short lengths. A weak-reference storage ledger counts
backing storage once for views and rejects tensor scalar reads; it verifies a
material peak reduction. It does not measure allocator-reserved memory.
The revision also compares live tensor storage at convolution entry and the
SiLU/masking/indexing ATen operators and input strides. These checks plus output
stride equality fail against `401f950bd` and pass against the revision. Eleven
CPU tests pass, including the existing 24 shape/dtype/kernel/dilation cases.
Logical peaks for P=1/2/4 are respectively 837610 -> 576186,
1496480 -> 973560 and 2811668 -> 1765756 bytes at reduced channel counts.

The second imports this checkout's full `vllm.envs` with real `env_var` metadata,
checks the default, metadata, strict parsing and preservation of the p3 registry,
and asserts CUDA has not been initialized. It borrows Python dependencies from
the existing interpreter read-only and disables bytecode writes.

## GPU acceptance (lead only)

Sync this candidate's Python files using the established venv workflow; do not
overwrite other production changes. First repeat the real-import smoke through
the selected worktree/venv. Keep E4M3, UVA, P flags and the same p3 mode as the
comparison. Explicitly override the production threshold back to zero:

```sh
# Merge this assignment into any existing C4140_WRAP prefix.
C4140_WRAP='env VLLM_QWEN4EXP_PLE_PREFILL_LOW_MEMORY=1' \
  scripts/c4140-1cat-flashnext-serve.sh up \
  --long-prefill-token-threshold 0 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 --max-num-seqs 4
```

This is a recipe, not executed by Astra. Lead first runs c=1..4 cold concurrent
prompts plus uneven tail/mixed-decode cases and survival checks. Record peak
allocated/reserved/free memory on every rank, KV IDs and the route log. Run
16K/64K prefill-probe and text parity vs `c11-P3-nolp`, with same-server replay
as the determinism control. Only then consider 0.94 with the same tests.
A c=4 equal-length test alone is insufficient: the failure came from uneven
chunks, so retain long+short arrivals and a live spec-decode request.

Rollback: unset the env or set `0`; return to threshold1616 + utilization0.92
when restoring the serving mitigation. Default-off behavior is not a cure for
the original OOM without that mitigation.

## Next stage: real-route memory profiling

No model-runner profiling changes are included in this first patch. After the
GPU gate above, profile the opted-in route rather than assuming the saved bytes
are available for KV. V2's `profile_run(skip_attn=True)` bypasses PLE metadata;
merely changing dummy request lengths does not exercise this code.

A follow-up should run a layer-specific probe with real PLE metadata and bounded
dummy conv-state storage before KV sizing, including P=1..maxseq, uneven lengths,
mixed decode and nonzero initial states. It must account for caller activations
and library workspace, restore profiling state, avoid publishing prefix hashes,
and not run attention against unallocated real KV. Consider a conservative
analytic reserve if the probe cannot reproduce the caller's full live set.
No graph-reserve reduction or GPU utilization increase is justified by CPU
storage accounting alone.
