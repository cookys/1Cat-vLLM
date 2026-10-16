# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped nospec policy and independent graph evidence (ATTN-18/21/24/30)."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar

_capture_routes: ContextVar[list[dict] | None] = ContextVar(
    "flash_v100_nospec_capture_routes", default=None
)


def nospec_long_scope(
    *, configured: bool, batch: int, q_len: int, capacity: int, graph_variant: int
) -> bool:
    # Context-bucket/semantic variants need a separate capture contract.
    return (
        configured
        and 2 <= batch <= 16
        and q_len == 1
        and capacity >= 32768
        and graph_variant == 0
    )


@contextmanager
def collect_attn_routes():
    """A separate list per actual graph capture, including nested captures."""
    routes: list[dict] = []
    token = _capture_routes.set(routes)
    try:
        yield routes
    finally:
        _capture_routes.reset(token)


def record_attn_route(
    *,
    plan,
    workspace_key,
    active_num_partitions,
    batch_capacity,
    metadata,
    native_route,
    log,
) -> None:
    """Native host out-arg = grid.x, template P, grid.z, stream capture flag."""
    native_b, native_p, native_z, native_capture = native_route
    fixed = getattr(metadata, "flash_v100_decode_partition_size_hint", None)
    active_p = getattr(metadata, "flash_v100_decode_active_partition_size", None)
    if (
        native_b != batch_capacity
        or native_p != plan.partition_size
        or native_p != workspace_key[5]
        or native_z <= 0
        or native_capture not in (0, 1)
        or (
            fixed is not None
            and (native_p != fixed or (active_p is not None and active_p != fixed))
        )
    ):
        raise RuntimeError("ATTN_ROUTE native partition/active/workspace mismatch")
    capacity = max(
        1,
        getattr(metadata, "flash_v100_decode_max_seq_len_hint", 0) or 0,
        getattr(metadata, "flash_v100_decode_workspace_seq_capacity_hint", 0) or 0,
    )
    route = dict(
        mode="nospec",
        B=native_b,
        P=native_p,
        planner="fixed" if fixed is not None else "hint",
        # ATTN-30: logical key uses actual capture/native inputs, not policy.
        ws_key=(
            workspace_key[0],
            native_b,
            workspace_key[3],
            workspace_key[4],
            native_p,
            capacity,
        ),
        allocator_ws_key=tuple(
            str(x) if not isinstance(x, (int, type(None))) else x for x in workspace_key
        ),
        launch_partitions=native_z,
        workspace_partitions=plan.workspace_num_partitions,
        active_P=active_p or plan.partition_size,
        active_ptr=active_num_partitions.data_ptr(),
    )
    routes = _capture_routes.get()
    if not native_capture:
        if routes is not None:
            raise RuntimeError("ATTN_ROUTE collector outside native stream capture")
        log_attn_routes([route], phase="eager", graph_id="none", log=log)
        return
    if routes is None:
        raise RuntimeError("ATTN_ROUTE native capture without graph collector")
    # Retain all distinct routes; all layers sharing one key collapse to one row.
    if route not in routes:
        routes.append(route)


def _metadata_items(metadata):
    for group in metadata if isinstance(metadata, list) else [metadata]:
        if isinstance(group, dict):
            yield from group.values()


def attn_live_batch(metadata):
    """Legacy FULL wrapper: scoped metadata is q1, so actual tokens = live rows."""
    counts = {
        item.num_actual_tokens
        for item in _metadata_items(metadata)
        if getattr(item, "flash_v100_nospec_long_scope", False)
    }
    if len(counts) != 1:
        raise RuntimeError("ATTN_ROUTE missing or inconsistent live batch")
    return counts.pop()


def validate_attn_active_units(routes, metadata) -> None:
    """Check ON's runtime active units against captured P without CUDA reads."""
    active_units = {}
    for item in _metadata_items(metadata):
        active = getattr(item, "flash_v100_decode_active_num_partitions", None)
        if active is not None:
            active_units[active.data_ptr()] = getattr(
                item, "flash_v100_decode_active_partition_size", None
            )
    for route in routes:
        if route["planner"] == "fixed" and (
            active_units.get(route["active_ptr"]) != route["P"]
        ):
            raise RuntimeError("ATTN_ROUTE replay active-buffer unit mismatch")


def bind_attn_graph(routes, graph):
    """Bind native capture evidence to the actual graph, not its cache entry."""
    return [dict(route, graph_id=id(graph)) for route in routes]


def attn_graph_batch(live_B, capture_sizes):
    """Cold nospec q1 lookup for the legacy FULL wrapper."""
    if not isinstance(live_B, int) or live_B <= 0:
        raise RuntimeError("ATTN_ROUTE missing unpadded live_B")
    for size in sorted(capture_sizes):
        if size >= live_B:
            return size
    raise RuntimeError("ATTN_ROUTE no runtime graph candidate")


def attn_runtime_keys(metadata, *, live_B, graph_B, config):
    """Independent replay lookup. No capture route or allocator key input.

    P is the cold-nospec graph policy evaluated NOW, not the long eager hint.
    All operations inspect host config/tensor attributes; no device readbacks.
    """
    if not isinstance(live_B, int) or not 0 < live_B <= graph_B:
        raise RuntimeError("ATTN_ROUTE replay requires unpadded live_B")
    model = config.model_config
    heads = model.get_num_attention_heads(config.parallel_config)
    dim = model.get_head_size()
    p = 1024 if os.getenv("VLLM_FLASH_V100_NOSPEC_LONG_P1024") == "1" else 256
    keys = set()
    for item in _metadata_items(metadata):
        if not getattr(item, "flash_v100_nospec_long_scope", False):
            continue
        table = item.block_table
        device = table.device.index if table.device.index is not None else -1
        capacity = min(
            table.shape[1] * config.cache_config.block_size, model.max_model_len
        )
        keys.add((device, graph_B, heads, dim, p, capacity))
    if not keys:
        raise RuntimeError("ATTN_ROUTE missing live lookup inputs")
    return keys


def validate_attn_replay(routes, metadata, *, graph, runtime_keys):
    """Independently check selected object and runtime lookup before launch."""
    if any(route["graph_id"] != id(graph) for route in routes):
        raise RuntimeError("ATTN_ROUTE graph-id-mismatch")
    if {route["ws_key"] for route in routes} != runtime_keys:
        raise RuntimeError("ATTN_ROUTE key-mismatch")
    validate_attn_active_units(routes, metadata)


def log_attn_routes(routes, *, phase, graph_id, log, live_B=None, runtime_keys=None):
    for route in routes:
        key = route["ws_key"]
        if phase == "replay":
            # The caller validated equality, but output comes from the fresh
            # runtime lookup. It is never copied from the capture record.
            if runtime_keys is None or len(runtime_keys) != 1:
                raise RuntimeError("ATTN_ROUTE replay missing independent key")
            key = next(iter(runtime_keys))
        graph_b, p = key[1], key[4]
        if phase == "replay" and not (
            isinstance(live_B, int) and 0 < live_B <= graph_b
        ):
            raise RuntimeError("ATTN_ROUTE replay requires unpadded live_B")
        live = f" live_B={live_B}" if phase == "replay" else ""
        allocator = (
            " allocator_ws_key="
            + json.dumps(route["allocator_ws_key"], separators=(",", ":"))
            if phase == "capture"
            else ""
        )
        active_label = "capture_active_P" if phase == "replay" else "active_P"
        source = "runtime-lookup" if phase == "replay" else "native-grid"
        log(
            "ATTN_ROUTE phase=%s mode=nospec B=%s graph_B=%s P=%s "
            "planner=%s ws_key=%s%s launch_partitions=%s actual_partitions=%s "
            "source=%s graph_id=%s%s pid=%s workspace=%s %s=%s active_ptr=%s",
            phase,
            graph_b,
            graph_b,
            p,
            ("fixed" if p == 1024 else "hint")
            if phase == "replay"
            else route["planner"],
            json.dumps(key, separators=(",", ":")),
            live,
            route["launch_partitions"],
            route["launch_partitions"],
            source,
            graph_id,
            allocator,
            os.getpid(),
            route["workspace_partitions"],
            active_label,
            route["active_P"],
            route["active_ptr"],
        )
