# 27B NVFP4 KV: Python routing and hybrid allocation (Q1.23)

Status: Python/allocator integrated on Fable P0/P1/P3 tip `b9779191a`
(P0 `548ca4cea`, XQA `13a5f3294`, P0 test fixes `ccbd81616`, common base
`d30469863`). Native version1 admits XQA only; version>=2 also admits the
prefill bridge and profiling. **GPU integration, serving, quality and capacity
are unvalidated.** CPU tests do not establish native correctness.
Enable only by explicitly selecting `--kv-cache-dtype nvfp4`. Existing FP8/FP16
configurations retain their numerical paths. NVFP4 itself changes outputs (E3).

## Reader/writer contract

`FLASH_ATTN_V100` advertises NVFP4 and returns
`[blocks, 2, page_tokens, kv_heads, 144]`, uint8, head256 only. Each side is a
data slab followed by a scale slab, **not** 144 interleaved bytes per token.
The reused plan-071 writer stores even elements in the low nibble. Its Python
arguments are the **layer dequantization multipliers**, e.g. `layer._k_scale`,
not their inverses: `_store_nvfp4_kernel` takes `1/layer_scale` internally.
Both default to 1.0 when absent from the 27B checkpoint. Device scalar buffers
avoid host scalar allocation during graph capture. The nonunit 0.5/2.0 test
protects the convention even though default 1.0 hides an inversion error.

Construction requires `flash_attn_nvfp4_kv_available(min_version=1)` plus a
callable XQA export. Prefill separately requires `min_version=2` and a callable
`nvfp4_paged_kv_to_fp16`. Version1 still runs XQA but rejects prefill/profile;
a version2 native binary with a missing Python bridge export is a package
mismatch and fails at construction. A dtype string cannot admit an old .so.

The backend accepts causal full decoder attention, FP16 Q/head256, without
ALiBi, softcap, sinks, anchored windows, SW or tree masks:

* q1/request: XQA, GQA ratio4/6/8.
* q2--8/request: per-row XQA with persistent smallq metadata, including contexts
  below the old FP8 scalar crossover. Capture without persistent metadata raises.
* More than8 query rows: version>=2, eager per-request bridge for the entire
  live cached prefix (including the just-written current chunk), followed by the
  existing FP16 attention. It never bypasses KV quantization with raw K/V.
* Scalar and both native grouped readers remain forbidden. XQA flag/shape
  rejection fails explicitly rather than falling back to an FP8/scalar reader.

K/V native views are exactly `cache[:,0]` / `cache[:,1]`, uint8
`[blocks,page,H,144]`, stride `(2*page*H*144,H*144,144,1)`, 4-byte aligned.
The validator inspects metadata only. HND (even its singleton-head stride),
padded physical target pages and misaligned slices fail. Use
`VLLM_KV_CACHE_LAYOUT=NHD`. FP16 draft padding below is independent. XQA
partials are FP32, supplied by Fable's wrapper; partitions256/512/1024.

P3 bridge signature matches FP8: `(key_cache, value_cache, block_table,
seq_lens, key_out, value_out, k_scale, v_scale)`. It folds the layer scales into
FP16 output; downstream paged attention uses dtype `auto`, scales1.0.
Output workspace shape is `[blocks,784,H,256]`, never144. Native P3 writes live
rows plus exact +0 through `round_up(seq_len,16)`, leaving later rows untouched.
Its compiled-only/GPU-unvalidated arithmetic contract permits up to1 fp16 ulp
against direct reference dequantization for nonunit scales; unit scales are
intended bit-exact. Native GPU tests must verify this.

Async CPU seq_lens shadows can be upper bounds. Smallq fallback uses the
GPU seq_lens; the eager bridge copies those authoritative lengths to CPU once
per layer to construct exact dense slices/cu_seqlens, including mixed batches.
This adds a host synchronization point per prefill attention layer; measure it
in the GPU window. Graph decode remains on persistent device metadata.

Profile_run reserves the whole max-context FP16 bridge **before** KV sizing,
shared across attention layers on the same device/stream, with output-head256
in the workspace cache key. Input capacity rounds up to a full physical page
because native requires output capacity to cover the sliced input block table.
At262144 tokens/TP4/H=1 this is approximately256.5--258.8 MiB/rank. Existing
native prefill workspace profiling is also called. Failure to allocate, a
short input table, or runtime growth beyond the profiled envelope fails
explicitly; it cannot silently steal KV budget after startup.

## DFlash2 allocator

The SM70 DFlash model loader defaults the drafter to `auto` (FP16) for an
NVFP4 target, as it already did for FP8. This matters before allocation:
simply changing the global target dtype otherwise constructs a head128
NVFP4 drafter, unsupported by this first reader.

The old unifier enlarges smaller pages by integer multiples. Target NVFP4
page 838656 bytes versus FP16 SW draft page 2981888 bytes is not divisible.
Instead, for the narrow combination NVFP4 FullAttention + align-mode Mamba
+ FP16 SlidingWindow:

1. Keep target token/page geometry and Mamba checkpoint grid.
2. Select the largest multiple-of-16 **divisor of the target block** whose
   FP16 draft payload fits the target page.
3. Pad each draft physical page to target page bytes. Keep window=2048,
   dtype, head dimensions and Mamba state shapes unchanged.
4. Pass per-spec FP16 dtype to worker cache shaping, even though global
   target cache dtype is NVFP4. V1 and V2 both preserve physical page stride.

| Target block/page | Draft block | Draft useful bytes/page | Draft physical bytes/page |
|---|---:|---:|---:|
| 2912 / 838656 B | 416 | 425984 | 838656 |
| 4096 / 1179648 B | 1024 | 1048576 | 1179648 |

The divisor is intentional. Draft816 would fit 835584 bytes but inflate
LCM scheduler alignment to 148512 tokens; draft819 is not kernel-aligned.
The chosen sizes keep LCM equal to the target block and use GCD hashes
416/1024. These are allocation/layout changes, not a new arithmetic method.
Changing checkpoint/chunk boundaries can still change floating-point results;
no cross-configuration E1 claim is made.

`VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16=1` cannot virtually split these padded
draft pages; workers reject it. Use the default 0 for this layout. Multiple
context-parallel ranks are unvalidated (this work uses DCP=PCP=1). Other draft
formats or state modes do not enter the allocator's special case.

## Conditional capacity, not a GPU admission guarantee

`benchmarks/sm70_nvfp4_27b_capacity.py` calls real grouping, page unification,
allocation and block-size resolution. Arithmetic for platform alignment is
checked against actual `Platform._align_hybrid_block_size` in a CPU test,
with only model-registry/context dependencies substituted.

Fixed budgets from the existing FP8 recipes: nospec 19.36 GiB/rank,
MTP4 18.65, DFlash2 16.53. State geometry, TP4 and async in-flight budget
8192 tokens are held fixed. "Reserve" deducts the implemented version2 bridge
from those old budgets. Neither column is a GPU admission result. A newly
measured NVFP4 Available KV budget already includes this bridge and must not
have it deducted twice. JSON labels the calculation as allocator-only and the
reader as GPU-unvalidated; `bridge_reserve_applied` records the chosen budget.

| Mode | KV / CLI block | Actual main block | Pool IDs before/after reserve | IDs/full 262144-token request | Full requests before/after reserve |
|---|---|---:|---:|---:|---:|
| nospec | FP8 / 2048 | 2048 | 1239 / n.a. | 134 | 9 / n.a. |
| nospec | NVFP4 / 2048 | 4096 | 1101 / 1087 | 70 | 15 / 15 |
| nospec | NVFP4 / 16 | 2784 | 1620 / 1599 | 101 | 16 / 15 |
| MTP4 | FP8 / 2048 | 2048 | 1123 / n.a. | 146 | 7 / n.a. |
| MTP4 | NVFP4 / 2048 | 4096 | 998 / 985 | 82 | 12 / 12 |
| MTP4 | NVFP4 / 16 | 2864 | 1428 / 1408 | 110 | 12 / 12 |
| DFlash2 | FP8 / 2048 | 2048 | 1057 / n.a. | 188 | 5 / n.a. |
| DFlash2 | NVFP4 / 2048 | 4096 | 1880 / 1852 | 193 | 9 / 9 |
| DFlash2 | NVFP4 / 16 | 2912 | 2645 / 2605 | 262 | 10 / 9 |

All rows fit through the real allocator; they are not results of admitting
that many users on a GPU. A larger max-num-seqs changes profiling/graphs;
watermarks, retained cached prefixes and live scheduler state reduce headroom.
Refit with measured NVFP4 Available KV memory and verify concurrent serving.

**CLI trap:** the 27B launcher supplies both `--block-size 2048` and
`--mamba-block-size 2048`. NVFP4 alone raises main block to4096, making the
explicit state grid invalid. Override BOTH at the tail of the command:

* First comparison (all three modes): `--block-size 4096 --mamba-block-size 4096`.
* Minimum-page alternative: `--block-size 16 --mamba-block-size 2784` (nospec),
  `2864` (MTP4), or `2912` (DFlash2). Test prefix/chunk behavior separately.

## CPU verification

Run from this checkout, using a serving-compatible Python with pytest. On the
shared host pytest dependencies were copied to `/tmp` without modifying a venv.

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 env PYTHONPATH="$PWD:/tmp/p072-astra-q123/test-deps" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /data/venvs/1cat-main/bin/python -m pytest --noconftest -q tests/v1/attention/test_nvfp4_v100_cpu.py tests/v1/core/test_nvfp4_hybrid_cpu.py
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 env PYTHONPATH="$PWD" /data/venvs/1cat-main/bin/python benchmarks/sm70_nvfp4_27b_capacity.py --mode dflash7 --kv nvfp4 --cli-block 2048 --budget-gib 16.53 --reserve-bridge
```

Tests cover real writers (unit/nonunit scales, strided inputs, invalid slots,
untouched page padding), old-extension rejection, packed reader dispatch,
persistent smallq pointers, blocked grouped routes, version1 rejection/version2 bridge/profile dispatch, scale-once and exact-length
contracts, NHD/alignment guards,
real platform alignment/unifier/pool, and real V1/V2 NHD/HND cache views.

Existing plan-071 reference/store tests run on the Triton interpreter.
Fable's `ccbd81616` marks two `packed_members` integration tests skipped on
this base; the dense backend's real worker view tests run independently.
Two legacy-store delegation cases install their fake callback with
`raising=False`, since CUDA-hidden environments omit the optional FA export.
That small CPU-fixture change does not modify the production fallback.

The legacy V100 policy suite needs `current_platform.device_type='cpu'` in the
test driver because CUDA hidden otherwise infers no device. This only sets CPU
test configuration, not a CUDA device. All actual GPU tests skip.

CPU results (2026-10-06): 86 new contracts, 158 imported format/store checks,
193 legacy V100 policy tests, four selected allocator tests and two DFlash
config tests = **443 distinct passing CPU tests**. Skipped: two unavailable
packed-members cases, seven P1 GPU tests, nine P3 GPU tests and one legacy GPU
test. No CUDA context initialized. CPU attention fixtures also fail before any
attempted CUDA initialization; the bridge pipeline test substitutes the
CUDA-specific cu_seqlens helper and native ops with CPU spies.
Ruff and `git diff --check` pass. Fresh-process env/backend/V1/V2/draft imports
and old-extension rejection pass. Logs are `/tmp/p072-astra-q123/p3-*.log`.

## GPU window for the lead (not executed here)

Use an isolated checkout/venv with the P3 rebuilt native extension and matching
Python package. A Python-only overlay on an old .so must fail the capability
gate. Do not patch an active serving environment.

**Native/backend component gate first:**

1. Run both `tests/kernels/attention/test_sm70_flash_v100_nvfp4_kv.py` (seven)
   and `test_sm70_flash_v100_nvfp4_bridge.py` (nine). Add production page sizes
   2784/2864/2912/4096 and bridge output-page784 (native tests initially use the
   same input/output page size), boundary lengths1/15/16/17, permuted block maps,
   unwritten NaN sentinels and nonunit scales. The Python pipeline must send
   side views with the required stride, FP32 partials, and bridge head256.
2. Reader graph capture/replay with persistent metadata: alternate c1/c2/c4,
   slots and seq lengths; same-format eager/replay comparison. Negative gates:
   old .so, v1 prefill/profile, disabled XQA, scalar/grouped, tree masks, HND,
   misalignment and bridge growth beyond its profiled envelope.
3. Mixed decode/prefill with deliberately loose CPU length hints: verify
   actual GPU lengths govern masks, empty requests are skipped, output padding
   is unchanged and bridge scales are applied once. Measure eager D2H overhead.

**Then full serving (Python route implemented, GPU validation pending):**

1. Startup smoke: nospec, MTP4, DFlash2 (`kv_cache_dtype:"auto"` for draft),
   NVFP4 target + full prefix caching/align, TP4. Explicit paired block/grid
   settings above. Check capability INFO and padded draft-page INFO. Record
   pool bytes/IDs and graph/bridge memory; verify no grouped reader launches.
2. Graph replay: alternate c1/c2/c4, accepted draft lengths, seq/block boundary
   crossings, padding, and request slot reuse. Native graph warmup must leave
   no stale cache state. Compare eager/replay at the same cache format.
3. Per-serving-mode quality: repeated 12-case greedy/sampled 1K/16K runs,
   numeric logprob probe at longer context, then SWE34. NVFP4 vs FP16/FP8 is
   **E3**; report drift and task scores, not a bit-exact promise. Pair loader
   instances and measure same-format repeatability first.
4. Cold16K/64K/253K and warm append1K/4K; DFlash2 c1/c4/c8 plus capacity edge
   c9/c10 at256K, nospec c15/c16, MTP4 c12/c13. Record actual admitted counts,
   cache retention/preemption, TTFT, errors, GPU peak, round_ms. The capacity
   edge is a test target, not a guaranteed successful operating point.

Use the existing `scripts/eval/serving-parity.py`, `kv-logprob-probe.py`
(`capture --backend vllm --server-kv-dtype nvfp4`, FP16/E4M3/repeats as separate
cells), `prefill-probe.py` and `concurrency-curve.py` from llm-playground.
These clients require the lead's GPU window and were not run here.
