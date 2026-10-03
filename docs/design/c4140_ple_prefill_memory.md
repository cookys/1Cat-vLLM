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

## Changes and exactness boundary

Only the opted-in batched-prefill method changes:

1. Release the packing tensor once concatenated history owns its values.
2. After the **same convolution**, gather a copy of the small state tail and
   release history. Keep state writeback at its original place, after output.
3. Apply the same ATen SiLU in place to the freshly allocated convolution result.
   The result has no reader requiring its pre-activation values.
4. Keep transpose as a view. Masking and advanced indexing read the same elements;
   indexing creates the contiguous unpadded output directly. Remove its previous
   empty allocation/copy and release the padded convolution result after indexing.

Convolution inputs, strides, dtype, weights, groups, dilation, padding width and
batch shape are unchanged. The state-tail gather is a copy, not an alias that
would retain history. Valid/NULL/empty-row state masks, speculative tail bytes
and state writeback order are unchanged. Inputs and weights are never mutated.
All tensor work remains on the current stream, with no tensor-to-host scalar
reads or additional synchronization. PyTorch manages storage reuse on that
stream, including during graph capture. No new persistent buffers are added.

Expected major storage peak changes from approximately `5*P*L*C*2 + S*C*2`
to `2*P*L*C*2 + small_state`, excluding caller-owned inputs, convolution-library
workspaces and unrelated activations. The exact crash shape therefore saves
roughly 1.08 GiB of those logical temporaries. Four uneven prefills can save
more. This is a storage-liveness model, **not a measured CUDA peak**.

The intent is E1: copying/view changes preserve bits, and in-place SiLU uses the
same operation on the same values. CPU byte equality is verified. CUDA/cuDNN
parity is still required: available workspace can affect convolution algorithm
selection, and CPU results cannot certify CUDA SiLU or compiled execution.
Strided advanced-index reads may cost more than indexing a pre-transposed buffer;
prefill throughput must be measured. No claim that the OOM is resolved yet.

Route hit: `Qwen4Exp PLE low-memory batched prefill enabled.` once per worker.
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
