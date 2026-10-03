# C4140 Phase 2: proving the selected SM70 routes

Source: `dbbc5f13e` plus Astra's metadata/startup changes. This is a read-only
audit; no route instrumentation or GPU workload was added. Flag prefixes below
are `VLLM_SM70_`. Keep source, native extension, env and graph-shape identity
with every cell. Flags are independent opt-ins on this base; enabling the
entire PR #796 changes more than these choices.

An `info_once` line proves that the Python/native dispatch was reached at
least once, often during warmup/capture. It does **not** prove that a measured
request replayed that graph. Check a closed steady decode round, correct TP
process, and actual M5/M10 graph variant. Kernel-family names that appear in
both branches need the extra evidence stated below. Absence of an info-once
line from a partial/truncated log is inconclusive.

## Flag-by-flag evidence

| Flag suffix | Existing log or selected call | Evidence in the measured route / caveat |
| --- | --- | --- |
| `FUSED_SIGMOID_MIXED_QKV` | **`SM70 mixed-QKV fused GDN target-verification route hit.`** | `qwen_gdn_linear_attn.py` logs this inside `use_sm70_mixed_qkv_verify`, after the mixed-loader call. Require target verify rows 2..16, pure spec, FP32 SSM and the same shape's replay. The Triton name `fused_sigmoid_gating_delta_rule_update_kernel` alone is insufficient: both old and new paths use it with `MIXED_QKV=False/True`. Verify Q/K/V contiguous-copy/concat kernels disappear. The separate **decode** mixed-QKV log is not proof of the MTP verifier route. |
| `MTP_MOE_FP16_EXACT` | **`Using exact SM70 FP16 MTP MoE projections (M1/M5, E512, H2560, I160).`** | Native kernels contain `mtp_moe_fp16_tile_kernel` and, for the M1 W13 specialization, `mtp_moe_fp16_m1_w13_kernel`. Require these in draft windows replacing the corresponding Triton projections; keep `MTP_MOE_TUNED_CONFIG=1`, supported BM2/N128/K64 configuration and matching native op. |
| `MTP_PLE_CONV` | **`SM70 MTP4 PLE rollback/conv/SiLU/state fusion enabled.`** | Native `ple_conv<...>` from `qwen38_ple_spec_sm70.cu`; one fused rollback/conv/SiLU/commit call replaces the old sequence. Gate requires one real request, query length 5, H10240, M5/M10 padded shape, dilation 3 and 13-position state. c2/c4 are expected fallbacks, even if startup logged a c1 hit. |
| `MTP_ROUTER_TOP16` | **`SM70 Qwen3.8 E512/K10 router top-k path enabled for M=5.`** (or M=10), plus the rank's flag value `1` | The logged dispatcher already requires FP16; `_sm70_qwen38_router_topk` then sets `SELECT_TOP16 = flag && FP16 && M in (5,10)` without another fallback. Thus that log/shape + flag + source identity proves capture-time selection. The kernel name `_sm70_qwen38_router_topk_kernel` is shared with full sort; its name alone is insufficient. A diagnostic observer at this JIT launch can record `SELECT_TOP16=True` without reading GPU data if shape attribution remains unclear. M1 logs do not prove this flag hit. |
| `QSA_MTP_TOPK` | **`Using exact SM70 MTP batch QSA selector.`** | This log proves passing `decode_batch=True` into `qsa_lexicographic_topk`, **not** its inner fast branch. Require `qsa_lexicographic_mtp_topk_kernel<...>` in target verify, not `qsa_lexicographic_topk_kernel<...>`. Native dispatch requires M5/M10 and score columns <=9216. Inside the MTP kernel only rows with <=2304 visible blocks use compact selection; larger rows use the normal body. Source log alone cannot prove compact selection. |
| `MTP_ROUTER_BATCH` | **`SM70 MTP4 batch router with ordered FP32 splits enabled.`** | `qwen38_router_batch_sm70_out` launches `router_split_quad_kernel<40>`. Require this in M5/M10 target/draft1 in place of the router projection GEMM. Packed weight allocation alone does not prove runtime admission. |
| `MTP_SHARED_BATCH` | **`SM70 MTP4 exact shared-expert batch projection enabled.`** | Look for `shared_up_batch_kernel`, `shared_up_reduce_silu_kernel` and `shared_gate_mul_kernel` at the shared-expert positions. M5/M10 gate; the latter fuses scalar-gate sigmoid/output multiply. Match the complete group, not a generic activation kernel. |
| `MTP_HC_BATCH` | **`SM70 TP4 batched HC enabled (FP32 concurrent partials=False).`** | Require `False` for the MTP FP16-partial contract. Noncooperative route uses `hc_down_partials<..., true>` plus gathers and `hc_up_batch`; a cooperative route uses the kernel below. Loader messages about HC modules being enabled/packed are weaker than this runtime log. |
| `MTP_HC_COOPERATIVE` | HC batch runtime log above, then native **`hc_cooperative<false>`** or **`hc_cooperative<true>`** | The batch log alone does not distinguish cooperative execution. Require the cooperative kernel in a measured graph and disappearance of the corresponding separate down/gather/up/gather launches. HC batch must first qualify. |
| `MTP_HC_FULL_UNROLL` | Native **`hc_cooperative<true>`** | `<false>` is partial unroll. Use the full demangled name (the short name may discard template arguments). No separate Python info-once message exists. It only affects the cooperative route; setting this with cooperative disabled has no effect. |
| `QWEN38_GDN_INPUT_BATCH` | **`SM70 Qwen3.8 checkpoint-FP16 batched GDN input enabled.`** | Native `gdn_input_batch_kernel<true,...>` is the packed input route. The log **`SM70 GDN batched projection split-copy fusion enabled.`** is a different fallback and is not evidence for packed GDN. M2..16, packed weights and FP16 contract required. Keep `QWEN38_BATCH_FASTPATH=0` to isolate this flag, because runtime admission accepts either flag. |
| `QWEN38_BATCH_FASTPATH` | **`SM70 Qwen3.8 exact small-batch dense projections enabled.`** plus `dense_batch_kernel` | On the current base, generic batch setup excludes speculative config; this is not the MTP-specific HC/router flag. On PR #796 the contract widens, so audit changed source first. A packed-GDN hit alone cannot establish generic dense projection admission. Remains an E3/hold candidate for newly admitted MTP dense paths. |
| `NVFP4_MOE_GROUPED_MTP5` (already G) | **`Experimental SM70 grouped native-NVFP4 decode selected (tokens=5, W13/W2 share route groups, MTP split4/reduce=True).`** | Require tokens=5 and `True`, with `nvfp4_grouped_w2_batch_reduce_sm70_out` routing in `nvfp4_sm70_moe.py`. A generic grouped decode message at M1/M8 is not the MTP5 route. |
| `RMSNORM_GATED_EXACT` (already G) | **`SM70 exact native gated RMSNorm fusion enabled (N128, sigmoid/SiLU, M1 and batch).`** | Native `rmsnorm_gated_exact_kernel<...>` in target/draft positions; it replaces the earlier norm/gate sequence. Do not count this gain again in later D cells. |

## Practical collection

Keep the complete service log, then search the exact positive strings above.
Do not use broad terms such as `enabled` or `batch` as the verdict. Also retain
fallback warnings (missing native op, geometry/numerical-policy rejections).
Read each TP rank's startup environment if inheritance is uncertain.

For an exported Nsight SQLite trace, the following query shape obtains full
kernel names for a **caller-selected** rank and closed decode window. Bind
`pid`, `start_ns`, and `end_ns` from that trace's own process/NVTX records;
do not transplant PID or timestamp values from another run:

```sql
SELECT s.value AS kernel, COUNT(*) AS launches,
       SUM(k.end - k.start) / 1000000.0 AS service_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON s.id = k.demangledName
WHERE k.globalPid = :pid
  AND k.start >= :start_ns AND k.start < :end_ns
GROUP BY s.value
ORDER BY service_ms DESC;
```

Join by process as well as correlation ID if relating runtime calls to kernels:
correlation IDs can collide across TP processes. Include graph-node kernels,
not just eager ones. Sum service times only for attribution; overlapping
service time is not an end-to-end saving. Reconcile a graph's capture-time
selection with the graph actually replayed by the timed request.

For D1/D4, record constexpr selection or the exact branch's log plus the
matching captured graph when kernel names do not distinguish variants.
For D5, inspect actual column capacity and visible-block counts separately:
a 262K-configured score buffer can cause fallback even with a short live
request, unless context-bucket slicing bounds the passed page table. At long
contexts the generic selector is an expected result. For HC full-unroll,
retain template arguments in the exported names.

These checks require no new per-token host synchronization. If a temporary
Python observer is used, record only shapes/dtypes/env/constexprs, and remove
it for the profiler-free throughput measurement. Do not add `.item()`, tensor
printing or extra device synchronization to certify a route.
