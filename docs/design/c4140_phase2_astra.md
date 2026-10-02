# C4140 plan 069 Phase 2: metadata, drafts and upstream flags

Research snapshot: 2026-10-03. Author: GPT-6-Astra, tmux `%3` (`0:3.0`).
Worktree: `/data/src/1cat-wt-astra`, branch `p069-astra`, base `dbbc5f13e`
(`c4140-m589`). No service, GPU workload, installed venv, CUDA or C++ source
was changed by this work. CUDA source was read to check dispatch contracts.

## Decision and evidence boundaries

First combine the PLE metadata patch with Fable's sampler patches, then try
the independent, zero-extra-weight flags in section D. Preserve the 262144
context limit, E4M3 KV calibration, sampling parameters and memory settings.
The largest remaining uncertainty is how much eager CPU preparation can
overlap GPU execution once the four explicit stream waits disappear.

Exactness labels used below:

- **E1:** same effective values/token IDs and persistent state bits by design;
  bounded CPU/upstream tests support this, but local GPU parity is still needed.
- **E2:** same ideal rejection-sampling output distribution; fixed-seed token
  IDs need not match. Numerical implementation still needs validation.
- **E3:** output arithmetic can change, or equivalence is unproved; candidate
  only, not a quality-neutral production recommendation.

Timing ranges are forecasts, **not measured C4140 improvements**. They are
non-additive because host work, kernels and communication overlap. Sources:

- `/data/profiles/ANALYSIS-2026-10-03.md`: cell **E**, including intrusive node
  traces; pre-target 2.188 ms (2.111 ms idle), sampling 3.445 ms (2.434 ms idle),
  drafts about 6 ms, next preparation 0.330 ms. Its 7-ms target-graph hole is
  a profiling artifact, not a recoverable optimization budget.
- `/data/profiles/G-SYNC-CALLSITES-2026-10-03.md`: cell **G**, native callchains
  only, no Python file/line frames. Four main-thread stream waits per round.
- G profiler-free round time: 27.8--29.6 ms. G already enables grouped MTP5
  NVFP4 MoE and exact gated RMSNorm; do not count their gains again.
- [Upstream MTP4 research record](sm70_flash_next_mtp4_batch_gdn.md): bounded
  component tests and full-model runs on FP16 KV/32K. Those are supporting
  evidence, not proof for our E4M3/262K/mixed-request deployment.

## A. Metadata patch and remaining host preparation

### Delivered patch

Commit `aaa498b56607c3475b1b314673cec248b1006515` changes only
`vllm/v1/attention/backends/short_conv_attn.py` and its standalone CPU test.

At the base revision, PLE uploads CPU int64 request indices at lines 335/336
and uploads the same speculative indices again at line 403 (the expression
ends at 404). A one-request pure speculative batch has one spec index and
**zero non-spec indices**, so the two observed 8-byte copies most plausibly
come from 335 and 403. Native frames cannot prove that attribution.

The patch uses the existing `async_tensor_h2d` pinned/nonblocking transport,
reuses the uploaded spec indices for accepted-token selection, and reuses
non-spec indices for computed-token selection. It keeps a different-device
fallback. Mixed-batch group assignments use `index_fill_` instead of tensor
indexed scalar assignments. Stable request ordering, block-table selection,
accepted counts and CUDA graph output buffers remain the same.

**E1 reasoning:** only integer/bool transport and reuse change. Source values
are captured before upload, reads stay ordered on the current CUDA stream,
and PyTorch's pinned allocator tracks the asynchronous copy lifetime. No
float computation, RNG, recurrent-state ownership or graph padding changes.
Mixed batches with prefills, noncontiguous spec rows and `is_prefilling` are
covered by the CPU contracts. Cross-stream consumers would still require
the existing stream dependency; this patch does not invent one.

Standalone test, from the worktree:

```bash
OMP_NUM_THREADS=1 .venv/bin/python tests/v1/attention/test_ple_metadata_async_cpu.py
.venv/bin/pre-commit run ruff-check --files \
  vllm/v1/attention/backends/short_conv_attn.py \
  tests/v1/attention/test_ple_metadata_async_cpu.py
.venv/bin/pre-commit run ruff-format --files \
  vllm/v1/attention/backends/short_conv_attn.py \
  tests/v1/attention/test_ple_metadata_async_cpu.py
```

Result: four unittest methods pass, including 36 batch/cache/graph combinations
plus transport and fallback cases. Tests execute AST-extracted real builder
and block-table code with CPU torch; they do **not** validate CUDA stream
timing or a native vLLM import. CPU torch is isolated in this worktree's `.venv`.

**Budget:** G's two waits have medians 9.5/7.2 us, plus their pageable copies.
Standalone direct gain is plausibly 0--0.1 ms. In combination with Fable,
removing the serialization may expose useful overlap with drafts; old E's
2.111-ms pre-target idle is an upper bound on that stage's potential, not
a predicted saving. Use 0--0.8 ms as a low-confidence planning range for A's
marginal combined benefit, and measure it. Do not count the same gap in A
and Fable's sampling improvement.

### Other four pageable copies

G round 10, same process and main thread, shows this order and geometry:

| Bytes | Likely source | Subsequent work |
| ---: | --- | --- |
| 4 | `model_runner.prepare_inputs`: `idx_mapping_np`, int32[1] | expand request indices |
| 8 | same: `cu_num_logits_np`, int32[2] | `_expand_idx_mapping_kernel` |
| 20 | same: padded `query_start_loc_np`, int32[max_num_reqs+1] | `_prepare_pos_seq_lens_kernel` |
| 1 | `attn_utils.compute_common_gdn_attn_metadata`: speculative bool mask | common GDN state metadata |

This is source/shape/order attribution, not Python-stack proof. The first
three call `buffer_utils.async_copy_to_gpu`, which intentionally uses a
pageable source and `non_blocking=True`; its comment records prior driver
contention from explicit pinning at high concurrency. The GDN comparison
creates a fresh ordinary CPU tensor before `.to(non_blocking=True)`.

The four copies cost roughly 0.06--0.08 ms of API time in this trace. Do not
globally flip `async_copy_to_gpu` based on a one-request benchmark. If they
become a blocker after A+Fable, stage only these small metadata arrays via
allocator-managed pinned tensors, or a pool with explicit safe reuse.
Using a fixed round-robin buffer without proving completion is unsafe:
the CPU can overwrite bytes while a previous GPU copy is still pending.
The pinned design is E1 if it preserves values, shape, copy order, target
buffer pointers and source lifetime. Fusing its four copies would also need
stable offsets and consistent padding; forecast direct savings at most
the observed 0.06--0.08 ms, with any avoided queue-drain measured separately.

**Trace acceptance:** require zero explicit main-thread stream waits after
the combined patch, but also check copy API durations and GPU idle. CUDA
permits pageable `cudaMemcpyAsync` to wait while staging data; zero explicit
`cudaStreamSynchronize` calls alone does not prove asynchronous execution.
See [NVIDIA's synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

### Graph/async preparation boundaries

`model_runner.prepare_inputs` already prepares positions/sequence lengths
on device; block-table state uses UVA/staged writes. Common GDN classification
is already shared across groups in `attn_utils`; group-specific state indices
must remain separate. Do not hoist accepted-state selection ahead of rejection,
or reuse a metadata object whose captured tensors are updated each round.

A later optimization can split CPU request classification from GPU-dependent
state selection, precompute invariant token-index templates by graph shape,
and capture the device preparation sequence with stable buffers. It must
retain align-mode rollback, mixed-prefill initialization, null state slots,
per-group block ownership and updated request mapping. This is E1 only with
those invariants; current evidence cannot assign an independent saving beyond
the measured pre-target/next-preparation budget. No speculative graph patch
is included without a CUDA replay test.

## B. Draft steps and LM head

The active V2 path is `worker/gpu/spec_decode/eagle/speculator.py`, selected
by native MTP. `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_*`, including
`FUSED_PROPOSAL`, is consumed by the legacy proposer/static vocabulary code,
not this V2 path. That legacy feature also has model eligibility restrictions
aimed at Qwen3.6-27B. Enabling it here has **0-ms expected benefit**.

`VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=1` is already set. It maps target graph
widths to draft request counts; it does not fuse four drafts into one graph.
The loop updates positions, slot mappings and attention metadata for every
draft. Four draft steps depend on earlier proposals and recurrent state.

Old E inter-draft gaps are about 0.10/0.07/0.07 ms. An outer graph that
unrolls the recurrence and device metadata could remove some of this host
launch work, while retaining the same per-step operations and RNG coordinates.
Forecast 0--0.3 ms from those gaps; do not multiply graph API duration by four
and treat it as exposed idle. Graph pool ownership, changing batch shapes,
prefill handoff and shared metadata buffers require dedicated replay tests.

### Ready A/B: local argmax without changing LM-head arithmetic

```text
VLLM_SM70_LM_HEAD_TOP1=0
VLLM_SM70_LM_HEAD_TOP1_TC=0
VLLM_SM70_TOP1_CUSTOM_AR=0
--speculative-config '{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":true}'
```

Use the same explicit env settings in the control, with
`use_local_argmax_reduction=false`. Clear any diagnostic
`VLLM_SM70_SYNC_TOP1_ALLGATHER_STEPS` setting. The extra speculative JSON can
be passed to the existing serve script; it follows the script's default JSON.

`LogitsProcessor.get_top_tokens` then uses the same quant method to compute
local logits, masks shard padding, finds each shard's maximum and gathers only
value/index pairs. **E1 under the current finite-logit, scale=1, no-softcap,
contiguous base-vocabulary contract:** local first-index ties followed by
first-rank ties equal global first-index argmax. Float32 exactly represents
the current vocabulary indices (<2^24). Validate padded shards, ties and token
IDs locally; the proof does not claim arbitrary NaN/added-vocabulary behavior.

No LM-head GEMM is removed. Four smaller gathers suggest **0--0.15 ms/round**,
with about 0.05--0.1 ms a reasonable component budget. Native raw/TC top1
operators are a separate E3 experiment because their accumulation/rounding
can differ. Do not silently enable them in this communication-only A/B.

## C. Acceptance and MTP depth

The default V2 draft method is **greedy**, even though the target request uses
temperature 1, top-k 20 and top-p 0.95. The implemented probabilistic alternative
is `draft_sample_method="probabilistic"`; it uses target temperature but ignores
draft top-k/top-p, and stores the processed proposal logits for rejection.
`VLLM_SM70_MTP_PROB_DRAFT_*` flags are legacy-proposer flags and do not change
this V2 proposal. Toggling them is a no-op here.

For target distribution p and actual proposal q, exact rejection gives

```text
accepted mass for token x = min(p(x), q(x))
Z = sum_x min(p(x), q(x))
residual distribution = max(p-q, 0) / (1-Z)
total output mass = min(p,q) + max(p-q,0) = p
```

The Z=1 case always accepts and needs no residual. Greedy proposal is the
one-hot special case. Changing q's temperature or truncation is **E2**, only
if acceptance and residual sampling use that same normalized q (including
its support); changing proposal generation while storing old logits is wrong.
No quality-neutral claim follows from increasing measured acceptance alone.
Finite-precision residual computation and model batch invariance still matter.

### V2 proposal/rejection wiring audit

Fable independently checked the base revision on 2026-10-03. The source
wiring satisfies the E2 prerequisites above; no code change was needed.

| Contract | Source evidence at the base revision |
| --- | --- |
| Store the distribution actually sampled | `sample/gumbel.py:124` applies temperature once, stores FP32 logits before noise, then samples the same values with `APPLY_TEMPERATURE=False`. Vocabulary and valid-request masks agree. |
| Keep request/step alignment | `eagle/speculator.py` writes `[req_state_idx, current_draft_step, :]`; prefill initializes step 0 and the decode loop sets steps 1..K-1. Rejection block statistics use expanded request/local-position indices; acceptance step i checks the draft at input position i+1. The bonus column does not read q. |
| Use the same q in acceptance and residual | `spec_decode/rejection_sampler_utils.py` enables `HAS_DRAFT_LOGITS` when the buffer exists, computes its full-vocabulary logsumexp, and retains the rejected step's value. Resampling reads that same row and computes `log(max(p-q,0))`; bonus sampling uses p directly. Temperature-zero requests follow the greedy branch. |
| Preserve buffer lifetime | Current rejection reads and subsequent proposal writes are ordered on the main stream. Async host scheduling does not write this buffer. Moving either operation to another stream would require an explicit dependency before reuse. |
| Share the Gumbel precision setting | Astra additionally checked `eagle/speculator.py:92` and `model_runner.py:286`: both take `model_config.use_fp64_gumbel`. The runner passes that sampler to `RejectionSampler`, whose call at `rejection_sampler.py:135` forwards the same setting. |

Paths in the table are relative to `vllm/v1/worker/gpu/`. This closes the
source-level setting/alignment questions, not GPU numerical validation.
Proposal and target may use different filters without invalidating rejection;
adding draft truncation can help or hurt acceptance depending on the resulting
q. Any such change must affect both stored logits and sampled logits.
Finite PRNG resolution, Gumbel transforms, FP32 normalization and residual
rounding remain outside the ideal-distribution proof. Sharing the FP64 flag
does not turn an E2 experiment into bit-exact E1 output.

Ready E2 experiment:

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
```

Do not combine this with local argmax; V2 rejects that combination. Persistent
float32 proposal logits alone cost roughly 15.16 MiB/rank at max_reqs=4,
depth=4 and vocab=248320; transient softmax/rejection costs are additional.
Acceptance could improve or regress; no defensible positive-ms estimate is
available without a run. Fixed-seed outputs can change, so this belongs after
E1 work and requires distribution/quality review, not automatic deployment.

### Depth 3/4/5: measure tokens divided by round time

Let `s_i=P(first i drafts accepted)` and `L_d=1+sum(i=1..d, s_i)`, including
the bonus token. Throughput is `1000*L_d/T_d` for round time in milliseconds.
Report both L and T, acceptance by position, EOS truncation and target shape.

For 4 to 3, saving `delta=T4-T3` helps iff `delta/T4 > s4/L4`.
At the approximate 1K `L4=4.6`, monotonic survival probabilities imply
`0.6 <= s4 <= 0.9`: a 28.5-ms round must save about **3.7--5.6 ms** to win.
Removing one approximately 1.3-ms draft alone is insufficient. At 64K
`L4=3.9`, `s4` can be anywhere from 0 to 0.725; aggregate acceptance is
insufficient to select depth 3. These estimates came from short traces and
are not a substitute for long per-position acceptance measurements.

For 4 to 5, the extra survival probability must satisfy
`s5 > L4*(T5-T4)/T4`. An illustrative extra 1.3 ms needs s5 above about
0.21 at 1K or 0.18 at 64K. **Do not assume the cost is 1.3 ms:**
`nvfp4_sm70_moe._use_qwen38_qpn_mtp5_decode` requires exactly `(5,2560)`;
MTP3 and MTP5 can lose G's M5 grouped route. Other HC/router/shared paths
admit M5/M10. Capture padding and actual dispatch, rather than just nominal
depth, determine the discontinuity. Depth changes are E2 at the algorithmic
level but can also alter target floating-point results through kernel shapes;
they are not E1 and require the same quality review as a changed proposal.

Ready CLI candidates change only `num_speculative_tokens` to 3 or 5. Keep
all other settings fixed and inspect actual graph widths/routes before
interpreting a speed result. No additional kernel implementation is proposed.

## D. Upstream status and independent flags

Observed during this investigation: main had advanced from the brief's
`9295280f7` to `3a147164833e58df2f80203cd169cbd71cea41e1`.
[PR #796](https://github.com/1CatAI/1Cat-vLLM/pull/796) was open/draft at
head `4ef74800459dcdb2827e4f806be91a7b8358c684`. It promotes defaults and
widens the generic batch contract; it is not needed to enable the independent
flags already present at our base. Its GPU admission remains bounded.
[PR #689](https://github.com/1CatAI/1Cat-vLLM/pull/689) was open at head
`93ae0e2f416a523b1d9579575e35d44af5c14e3a`: it adds EXL3 packed PLE support,
not the metadata fix above. Avoid taking its whole `ple_layer.py` diff over
our exact-pin patch for an unmeasured NVFP4 speed benefit. Latest main still
contained the blocking PLE index uploads when inspected.

All flags in the table are prefixed **`VLLM_SM70_`**. Savings apply only when
the guarded route actually runs. E1 means original effective arithmetic is
preserved by the stated contract, not that local full-model parity has passed.

| Priority | Set suffix to 1 | Exactness and route | Extra weight/rank | Forecast ms/round |
| --- | --- | --- | ---: | ---: |
| 1 | `FUSED_SIGMOID_MIXED_QKV` | E1: same BV32/four-warp recurrence, FP32 state, FP16 output; removes Q/K/V split/concat; bounded 36-layer and integration parity | 0 | 0.25--0.50 |
| 2 | `MTP_MOE_FP16_EXACT` | E1: draft FP16 MoE M1/M5, E512/H2560/I160/top10; preserves split boundaries; upstream fixed-fixture gain 0.216 ms | 0 | 0.15--0.30 |
| 3 | `MTP_PLE_CONV` | E1 effective tokens/state: same rollback, four ordered taps, FP16 conv boundary/SiLU; one request, MTP4, M5/M10 padding; padded zero sign unpromised | 0 | 0.10--0.15 at c1; fallback otherwise |
| 4 | `MTP_ROUTER_TOP16` | E1: lossless half keys, same top10 and eight-warp norm, M5/M10; exhaustive half-payload/tie screens | 0 | 0.04--0.08 |
| 5 | `QSA_MTP_TOPK` | E1: same score/index lexicographic order; M5/M10, score columns <=9216; compact branch only <=2304 visible blocks | 0 | 0--0.15; likely fallback at long context |
| 6 | `MTP_ROUTER_BATCH` | E1: four K640 partials, FP16 partial boundary, original FP32 order; M5/M10 | 122.5 MiB | 0.15--0.30 |
| 7 | `MTP_SHARED_BATCH` | E1: eight K320 FP16 partials, ordered FP32 sum, FP16 SiLU/multiply; M5/M10 | 76.5625 MiB | 0.20--0.40 |
| 8 | `MTP_HC_BATCH`, then `MTP_HC_COOPERATIVE`, then `MTP_HC_FULL_UNROLL` | E1 bounded native TP4 checks: twenty K512 FP16 partials and same FP32 order; cooperative transport preserves bits; M5/M10 | about 336.875 MiB incl draft | 0.30--0.80 together; unroll incremental gain uncertain |
| Defer | `QWEN38_GDN_INPUT_BATCH` | E1 bounded M2..16 projection checks; same numerical contract | 725.625 MiB | 0.60--0.85, but capacity cost |
| Hold | `QWEN38_BATCH_FASTPATH` | E3 for newly admitted generic MTP dense paths: reduced FP16 reduction can change results; base excludes generic MTP | varies | no qualified estimate |
| Already G | `NVFP4_MOE_GROUPED_MTP5`, `RMSNORM_GATED_EXACT` | Preserve present settings | 0 | 0 incremental |

Upstream component savings are the basis of these ranges; no addition of all
rows is a valid endpoint forecast. Native dispatch availability in the installed
extension must be checked by the serving owner; a silent fallback is not a
negative performance finding. Some paths are M5/M10 only, so c3/c4 can fall
back even if c1/c2 qualify. E4M3 KV does not itself exclude the checked dense
runtime contracts, but it does change memory and end-to-end validation needs.

**Capacity:** G logs report KV 2.89 GiB and 356,962 tokens. GDN packing alone
consumes approximately 24.5% of that byte pool; naive proportional scaling
leaves about 269K tokens, only about 7K over 262144 before other changes.
Actual hybrid allocation, graph reservation and block granularity must be
measured, so this estimate is not a guarantee of fitting. Router+shared cost
about 199 MiB; adding HC makes about 536 MiB. Test those sequentially with
reported KV capacity and the intended longest request/concurrency. Do not
lower the context ceiling to make a candidate pass.

On PR #796, recovering packed GDN memory requires explicitly setting both
`VLLM_SM70_QWEN38_GDN_INPUT_BATCH=0` and
`VLLM_SM70_QWEN38_BATCH_FASTPATH=0` because the admission can use either flag.
This is another reason to evaluate individual flags on the current base.

## A/B list for Claude

These are instructions for the service owner; none were executed by Astra.

1. Four-cell factorial: G; A=`aaa498b56`; Fable=`93aaabdc5` + `ca959f993`
   (+ tests `6a2a121d3`) with `VLLM_SM70_TOPK_TOPP_BRANCHFREE=1`; A+Fable.
   Keep default/explicit mode identical in both cells containing Fable.
   Fable owns sampling parity and mode 2; do not infer its result from A's test.
2. On the best E1-qualified cell, enable table D priorities 1--5 individually,
   then combine only passing improvements. Restart under the same settings
   for every import-time env flag. Confirm route hits and record memory.
3. B local-argmax comparison with raw/TC top1 and custom reduction disabled
   in both arms. If it passes, test its combination with the selected D flags.
4. D priorities 6--8, one packed family at a time, with capacity checks. Keep
   generic batch and packed GDN off. Treat HC cooperative/unroll as nested
   comparisons, not three independent additive gains.
5. E2 candidates separately: probabilistic MTP4; greedy MTP3; greedy MTP5.
   Report tokens/round and round ms, plus per-position acceptance. Do not
   silently adopt a changed seeded trajectory under an E1 label.

For each E1 cell use `/data/bench/ab/parity.py`: same fixed messages, seed,
temperature/top-k/top-p, token IDs and logprobs. First repeat the same server
to establish its own determinism floor, then compare baseline/candidate.
`bench-agentic-shapes.py` uses salted prompts, so its speed JSON cannot alone
establish token parity. Include mixed prefill/decode and c1/c4 in validation.
For numerical kernel candidates, unchanged accepted outputs alone is weaker
than intermediate/state parity; upstream bounded tests supply supporting
evidence, not a universal proof.

Measure profiler-free throughput first; use short matched traces separately.
Record main-thread stream waits, pageable memcpy API times, GPU idle,
actual route/graph widths and acceptance. Do not infer host overhead from the
17.7--19-ms S3 wait: most of it is target GPU work. CPU scheduling/TP rank
stalls are a separate issue; no affinity changes were performed here.

## E. E4M3 QSA: applicability of the SGLang scratch-fusion idea

Analysis only. Primary code: `vllm/models/qwen4_exp/nvidia/ops/qsa.py`,
`csrc/qsa_lexicographic_topk.cuh` and
`vllm/models/deepseek_v4/common/ops/fp8_software.py`.

**Current c1 decode does not have the reported gather/dequantize/cast scratch
chain.** At default settings the XQA page4 gate requires at least 64 rows,
or more than 16 rows for uint8 E4M3 KV. Target M5 and draft M1 use the Triton
sparse split-K route; c4/M20 can reach page4 when the other guards qualify.

- `_qsa_sparse_paged_gqa_splitk_kernel` directly translates selected logical
  indices through the page table, masked-loads raw K/V bytes, software-decodes
  E4M3 into FP16 and performs the attention dot/online softmax. Gather, decode
  and attention are already fused. Invalid request/token/page lanes load zero
  and contribute zero probability; there is no stale dequantized-KV scratch
  row to zero-fill as in the reported SGLang implementation.
- Split-K partial output/LSE are FP32 and sized to active query shapes, then
  merged separately. Removing the merge requires a different cross-CTA
  reduction or partitioning, which can change numerical order. This is not
  a redundant full-capacity dtype conversion.
- `_qsa_mqa_paged_kernel` is the **compressed-key indexer**, with its own FP16
  cache, not the E4M3 full K/V dequantizer. It reads paged keys directly and
  limits scoring to visible blocks. Its +0.97 ms from 1K to 64K is consistent
  with more score work, not a full scratch conversion accidentally repeated.
- `_qsa_xqa_page4_block_table` constructs/sorts compact **page indices**.
  `_qsa_xqa_page4_physical_kv` uses `view`/`as_strided` to reinterpret four-token
  pages, without materializing converted K/V. Native attention receives raw
  bytes and scales. Workspace uses power-of-two allocation but active-row
  slices; the E4M3 FP32 intermediate is attention reduction output, not a
  converted full-capacity KV cache.
- A separate prefill indexer gather exists for large-row cuBLAS scoring;
  that route is not this M5 decode workload and does not establish the
  SGLang-style decode opportunity.

**SM70 feasibility:** this source uses uint8 plus integer/bitcast software
E4M3 decode, so it does not require native SM70 FP8 conversion support.
Any future Triton fusion must reuse those exact bit patterns and masking.
The current kernel applies K scale after QK and V scale after normalization;
moving scales into FP16 K/V scratch changes rounding and is not E1.
Preserve invalid-page masking, null requests, signed zeros/NaNs, scale
placement, FP32 LSE accumulation and merge order in any parity argument.

Expected saving from directly transplanting the **specific** SGLang
gather+decode+scale+scratch fusion: **0 ms at current c1**, because that
redundant chain is absent. The old E sparse attention kernel is about
1.116/1.135 ms at 1K/64K, while indexer and lexicographic top-k grow about
0.97/0.65 ms. D's exact MTP selector is a bounded experiment; its long-column
fallback prevents assuming it recovers the whole 0.65 ms. A page-index
construction fusion might save launches at c4/prefill, but requires a trace
of that route before assigning a budget. No QSA patch is included.
