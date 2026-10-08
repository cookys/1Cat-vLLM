# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host-only route diagnostics; no policy, tensor reads, or tensor allocations.

FULL replay does not call the Python attention backend. Match its capture
records to the runner's live dispatch records by padded token/request counts.
No token IDs, request IDs, block IDs, tensor values, or pointers are logged.
"""

import json
import os

KNOB = "VLLM_FLASH_V100_GROUPED_VERIFY_DIAGNOSTICS"
_seen: set[str] = set()


def enabled() -> bool:
    return os.getenv(KNOB, "0") == "1"


def emit(logger, route: str, **fields) -> None:
    if not enabled():
        return
    record = json.dumps(dict(route=route, **fields), sort_keys=True)
    if record not in _seen:
        _seen.add(record)
        logger.info("SM70 grouped route diagnostic: %s", record)


def metadata_fields(meta, query) -> dict:
    table = getattr(meta, "block_table", None)
    qsl = getattr(meta, "query_start_loc", None)
    table_rows = None if table is None else table.shape[0]
    return dict(
        B=getattr(meta, "num_reqs", table_rows),
        block_table_rows=table_rows,
        query_start_loc_slots=None if qsl is None else qsl.numel() - 1,
        num_actual_tokens=meta.num_actual_tokens,
        max_query_len=meta.max_query_len,
        query_rows=query.shape[0],
        is_dflash_selector_target=bool(
            getattr(meta, "is_dflash_selector_target", False)
        ),
        is_mtp_verify_target=bool(getattr(meta, "is_mtp_verify_target", False)),
        # B1 views deliberately inherit the parent's actual-token count/qsl.
        metadata_scope="B1_view" if hasattr(meta, "_meta") else "batch",
    )


def backend(logger, route, meta, query, *, capture, kv_dtype, **fields) -> None:
    emit(
        logger,
        route,
        **metadata_fields(meta, query),
        capture=capture,
        kv_dtype=kv_dtype,
        **fields,
    )


def gate(
    logger,
    impl,
    meta,
    query,
    key,
    value,
    *,
    num_query_tokens,
    allowed,
    single_request_shape,
    batched_request_shape,
    num_reqs,
    any_batch,
    nvfp4_available,
    capture,
) -> None:
    """Explain the existing gate AFTER it decides, without changing its result.

    Checks below are diagnostics only. The authoritative admission expression
    stays in flash_attn_v100.py; an unclassified rejection is explicit.
    """
    if not enabled():
        return
    reasons = []
    if not allowed:
        if not single_request_shape and not batched_request_shape:
            if num_reqs == 1:
                reasons.append("single_request_shape")
            else:
                if not impl.use_dflash2_batched_grouped_verify:
                    reasons.append("batched_disabled")
                if impl.dflash2_grouped_verify_request_major_abi_version < 1:
                    reasons.append("request_major_abi")
                if not (
                    num_reqs in (2, 4, 8)
                    or (
                        any_batch
                        and 2
                        <= num_reqs
                        <= impl.dflash2_grouped_verify_any_batch_max_reqs
                    )
                ):
                    reasons.append("batch_capacity_or_admission_set")
                if meta.max_query_len != 8:
                    reasons.append("max_query_len_ne_8")
                if num_query_tokens != num_reqs * 8:
                    reasons.append("tokens_ne_B_times_8")
        table, seq = meta.block_table, meta.seq_lens
        checks = {
            "grouped_disabled": impl.use_dflash2_grouped_verify,
            "op_unavailable": impl.flash_attn_grouped_verify_paged is not None,
            "target_marker": (
                getattr(meta, "is_dflash_selector_target", False)
                or (getattr(meta, "is_mtp_verify_target", False) and num_reqs == 1)
            ),
            "model_len": getattr(meta, "max_model_len", 0)
            >= impl.dflash2_grouped_verify_min_model_len,
            "causal": getattr(meta, "causal", True),
            "window": impl._flash_v100_window_size(causal=True) == (-1, -1),
            "query_shape": tuple(query.shape) == (num_query_tokens, 6, 256),
            "query_dtype": str(query.dtype) == "torch.float16",
            "query_contiguous": query.is_contiguous(),
            "cache_layout": (
                key.ndim == 4
                and value.ndim == 4
                and key.shape[1]
                in (1648, 1728, 3296, 3456, *impl.dflash2_grouped_verify_extra_pages)
                and tuple(key.shape[2:])
                == (1, 144 if impl.kv_cache_dtype == "nvfp4" else 256)
                and value.shape == key.shape
                and key.device == value.device == query.device
                and str(key.dtype) == str(value.dtype) == "torch.uint8"
                and key.stride(-1) == value.stride(-1) == 1
            ),
            "kv_dtype_or_probe": impl.kv_cache_dtype == "fp8_e5m2"
            or (impl.kv_cache_dtype == "nvfp4" and nvfp4_available),
            "block_table_layout": (
                table is not None
                and table.ndim == 2
                and table.shape[0] == num_reqs
                and table.device == query.device
                and str(table.dtype) == "torch.int32"
                and table.is_contiguous()
            ),
            "seq_lens_layout": (
                seq is not None
                and seq.ndim == 1
                and seq.shape[0] == num_reqs
                and seq.device == query.device
                and str(seq.dtype) == "torch.int32"
                and seq.is_contiguous()
            ),
        }
        reasons.extend(name for name, passed in checks.items() if not passed)
        if not reasons:
            reasons.append("unclassified_gate_rejection")
    backend(
        logger,
        "gate_accept" if allowed else "gate_reject",
        meta,
        query,
        capture=capture,
        kv_dtype=impl.kv_cache_dtype,
        reasons=reasons,
        native_max_q=impl.dflash2_grouped_verify_max_query_tokens,
        gate_tokens=num_query_tokens,
        single_shape=single_request_shape,
        batch_shape=batched_request_shape,
        batched_enabled=impl.use_dflash2_batched_grouped_verify,
        any_batch=any_batch,
        capacity=getattr(impl, "dflash2_grouped_verify_any_batch_max_reqs", None),
    )


def dispatch(logger, batch_desc, input_batch, *, dummy_run, is_profile) -> None:
    """Log live host batch and final graph descriptor, including FULL replay.

    InputBatch's *_np arrays are already host-side. Never read GPU qsl/lengths.
    This describes dispatch, not a fresh execution of the backend's gate.
    """
    if not enabled():
        return
    from collections import Counter

    lengths = input_batch.num_scheduled_tokens
    histogram = sorted(Counter(int(n) for n in lengths).items())
    emit(
        logger,
        "v2_dispatch",
        B_live=input_batch.num_reqs,
        B_bucket=batch_desc.num_reqs,
        num_actual_tokens=input_batch.num_tokens,
        padded_tokens=batch_desc.num_tokens,
        max_query_len=max((int(n) for n in lengths), default=0),
        query_length_histogram=histogram,
        prefill_requests=int(input_batch.is_prefilling_np.sum()),
        uniform_token_count=batch_desc.uniform_token_count,
        runtime_mode=str(batch_desc.cg_mode),
        attention_context_bucket=batch_desc.attention_context_bucket,
        dummy_run=dummy_run,
        is_profile=is_profile,
        # This is called outside capture, including before FULL replay.
        capture=False,
    )
