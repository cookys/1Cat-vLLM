# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts: real builder -> backend -> interface -> fake native ABI.

No vLLM import/platform discovery or native extension load. Run --noconftest.
Graph tests execute the real wrapper with a fake CUDA graph, not metadata-only
"replay" logging. These are not numerical CUDA parity tests.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import os
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from enum import IntEnum
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "vllm/v1/attention/backends/flash_attn_v100.py"
INTERFACE = ROOT / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
POLICY = ROOT / "vllm/v1/attention/backends/flash_attn_v100_nospec.py"
GRAPH = ROOT / "vllm/compilation/cuda_graph.py"
spec = importlib.util.spec_from_file_location("nospec_policy", POLICY)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def runtime_config():
    return NS(
        model_config=NS(
            max_model_len=262144,
            get_num_attention_heads=lambda _: 6,
            get_head_size=lambda: 256,
        ),
        parallel_config=NS(tensor_parallel_size=4),
        cache_config=NS(block_size=4096),
    )


def nodes(path):
    return ast.parse(path.read_text()).body


def execute(selected, scope, filename):
    tree = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), scope)


@pytest.fixture
def h(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("VLLM_FLASH_V100_"):
            monkeypatch.delenv(key)
    state = NS(
        capturing=False,
        native=[],
        native_kwargs=[],
        logs=[],
        replays=0,
        fail_replay=False,
    )

    def native(*args, **kwargs):
        route_info = kwargs.get("route_info")
        state.native.append(args)
        state.native_kwargs.append(kwargs)
        if route_info is not None:
            route_info.copy_(
                torch.tensor(
                    [args[0].shape[0], args[11], args[12], int(state.capturing)]
                )
            )
        args[3].zero_()
        return args[3]

    i = dict(
        __name__=__name__,
        torch=torch,
        os=os,
        dataclass=dataclasses.dataclass,
        DEFAULT_DECODE_PARTITION_SIZE=256,
        VALID_DECODE_PARTITION_SIZES=(256, 512, 1024),
        _decode_plan_cache={},
        _decode_workspace_cache={},
        _cuda_graph_capture_active=lambda: state.capturing,
        _workspace_stream_id=lambda _: 17,
        _can_cache_workspace=lambda _: True,
        flash_attn_decode_paged_xqa_available=lambda: True,
        flash_attn_v100_cuda=NS(
            decode_paged_xqa_fwd=native, decode_paged_xqa_route_version=1
        ),
    )
    wanted = {
        "_DecodePlan",
        "_DecodeWorkspace",
        "_allocate_decode_workspace",
        "_get_decode_plan",
        "_get_decode_workspace_for_plan",
        "_round_decode_partition_capacity",
        "_get_decode_partition_size",
        "_select_default_decode_partition_size",
        "_validate_decode_partition_size",
        "_decode_dynamic_partitions_enabled",
        "_xqa_staged_pv_enabled",
        "_assert_decode_launch_covers_seq_lens",
        "maybe_contiguous",
        "flash_attn_decode_paged_xqa",
    }
    execute(
        [n for n in nodes(INTERFACE) if getattr(n, "name", None) in wanted],
        i,
        INTERFACE,
    )
    env = NS(
        VLLM_FLASH_V100_DECODE_PARTITION_SIZE=None,
        VLLM_FLASH_V100_ATTN_ROUTE_TRACE=True,
    )
    b = dict(
        __name__=__name__,
        torch=torch,
        os=os,
        partial=partial,
        envs=env,
        nospec_long_scope=policy.nospec_long_scope,
        record_attn_route=policy.record_attn_route,
        logger=NS(info=lambda fmt, *args: state.logs.append(fmt % args)),
        _as_flash_v100_metadata=lambda x: x,
        _decode_dynamic_partitions_enabled=i["_decode_dynamic_partitions_enabled"],
        _decode_partition_size_for_metadata=i["_select_default_decode_partition_size"],
        _trace_decode_active_metadata=lambda **kw: None,
        _log_fp8_kv_cache_route=lambda *a: None,
        _trace_decode_active=lambda **kw: None,
        _record_route=lambda *a: None,
        _g6_aligned_page_partition_size_hint=lambda *a: None,
        _callable_accepts_keyword=lambda fn, key: key
        in inspect.signature(fn).parameters,
    )
    top = nodes(BACKEND)
    classes = {n.name: n for n in top if isinstance(n, ast.ClassDef)}
    selected = [n for n in top if getattr(n, "name", None) == "_split_paged_kv_cache"]
    methods = (
        "_attach_decode_shape_hints",
        "_ensure_decode_active_num_partitions",
        "_update_decode_active_num_partitions",
    )
    selected += [
        n
        for n in classes["FlashAttnV100MetadataBuilder"].body
        if getattr(n, "name", None) in methods
    ]
    selected += [
        n
        for n in classes["FlashAttnV100Impl"].body
        if getattr(n, "name", None) == "_flash_v100_decode"
    ]
    execute(selected, b, BACKEND)
    builder_type = type("Builder", (), {name: b[name] for name in methods})

    def case(on=True, batch=4, length=1):
        if on:
            monkeypatch.setenv("VLLM_FLASH_V100_NOSPEC_LONG_P1024", "1")
        else:
            monkeypatch.delenv("VLLM_FLASH_V100_NOSPEC_LONG_P1024", raising=False)
        builder = builder_type()
        builder._nospec_long_configured = True
        builder._nospec_long_p1024 = on
        builder._decode_active_num_partitions = None
        builder.block_size = 4096
        builder.device = torch.device("cpu")
        builder.vllm_config = runtime_config()
        metadata = NS(
            num_actual_tokens=batch,
            seq_lens=torch.full((batch,), length, dtype=torch.int32),
            seq_lens_cpu=torch.full((batch,), length, dtype=torch.int32),
            block_table=torch.zeros((batch, 64), dtype=torch.int32),
        )
        common = NS(
            max_query_len=1,
            max_seq_len=262144,
            block_table_tensor=metadata.block_table,
            cudagraph_graph_variant=0,
        )
        q = torch.zeros((batch, 6, 256), dtype=torch.float16)
        cache = torch.zeros((1, 2, 4096, 1, 144), dtype=torch.uint8)
        impl = NS(
            prefix_anchored_decode_window=None,
            kv_cache_dtype="nvfp4",
            nvfp4_kv_available=True,
            use_decode_xqa=True,
            scale=0.0625,
            flash_attn_decode_paged_xqa=i["flash_attn_decode_paged_xqa"],
            _flash_v100_window_size=lambda **kw: (-1, -1),
        )
        layer = NS(_k_scale_float=1.0, _v_scale_float=1.0)

        def prepare(length=1, capture=True):
            metadata.seq_lens.fill_(length)
            metadata.seq_lens_cpu.fill_(length)
            builder._attach_decode_shape_hints(metadata, common, static_decode=capture)
            builder._update_decode_active_num_partitions(
                metadata, stage="capture" if capture else "build"
            )

        def forward():
            return b["_flash_v100_decode"](
                impl, layer, q, q, q, cache, metadata, torch.empty_like(q)
            )

        return NS(
            builder=builder,
            metadata=metadata,
            common=common,
            prepare=prepare,
            forward=forward,
            impl=impl,
        )

    return NS(state=state, i=i, b=b, env=env, case=case)


@pytest.mark.parametrize("on,p", [(False, 256), (True, 1024)])
@pytest.mark.parametrize("batch", [4, 8, 16])
@pytest.mark.parametrize("share", ["0", "1"])
def test_capture_four_invariants_and_live_boundaries(
    h, monkeypatch, on, p, batch, share
):
    monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", share)
    c = h.case(on, batch)
    c.prepare()
    h.state.capturing = True
    with policy.collect_attn_routes() as routes:
        c.forward()
    native = h.state.native[-1]
    assert native[11:13] == (p, 262144 // p)
    assert c.metadata.flash_v100_decode_active_partition_size == p
    assert routes[0]["P"] == routes[0]["ws_key"][4] == p
    assert routes[0]["allocator_ws_key"][2] == (None if share == "1" else batch)
    assert native[6].shape[2] == 262144 // p
    assert len(h.i["_decode_workspace_cache"]) == 1
    if on:
        for length in (1, 1024, 1025, 32767, 32768, 253953, 254208, 262144):
            c.prepare(length, capture=False)
            assert (
                c.metadata.flash_v100_decode_active_num_partitions.item()
                == (length + 1023) // 1024
            )
            policy.validate_attn_active_units(routes, {"layer": c.metadata})


@pytest.mark.parametrize("batch", [1, 17])
def test_out_of_scope_batch_preserves_hint(h, batch):
    c = h.case(True, batch)
    c.prepare()
    assert c.metadata.flash_v100_decode_partition_size_hint is None


@pytest.mark.parametrize("change", ["prefill", "short", "variant", "unconfigured"])
def test_scope_exclusions(h, change):
    c = h.case()
    if change == "prefill":
        c.common.max_query_len = 8
    if change == "short":
        c.builder.vllm_config.model_config.max_model_len = 8192
    if change == "variant":
        c.common.cudagraph_graph_variant = 1
    if change == "unconfigured":
        c.builder._nospec_long_configured = False
    c.prepare()
    assert c.metadata.flash_v100_decode_partition_size_hint is None


@pytest.mark.parametrize("conflict", ["override", "dynamic", "xqa"])
def test_conflicting_knobs_rejected(h, monkeypatch, conflict):
    if conflict == "override":
        h.env.VLLM_FLASH_V100_DECODE_PARTITION_SIZE = "1024"
    if conflict == "dynamic":
        monkeypatch.setenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "0")
    if conflict == "xqa":
        monkeypatch.setenv("VLLM_FLASH_V100_DECODE_USE_XQA", "0")
    with pytest.raises(RuntimeError, match="requires dynamic XQA"):
        h.case().prepare()


def test_workspace_keys_separate_off_and_on(h):
    keys = []
    for on in (False, True):
        c = h.case(on)
        c.prepare()
        h.state.capturing = True
        with policy.collect_attn_routes() as routes:
            c.forward()
        keys.append(routes[0]["ws_key"])
    assert keys[0] != keys[1]
    assert len(h.i["_decode_workspace_cache"]) == 2


def test_zero_padding_and_changing_batch_replay_units(h):
    c = h.case(True, 8)
    c.prepare()
    h.state.capturing = True
    with policy.collect_attn_routes() as routes:
        c.forward()
    original_ptr = routes[0]["active_ptr"]
    for length in (253953, 1, 262144):
        c.prepare(length, False)
        c.metadata.seq_lens[-1] = 0  # seven live / one padding row
        policy.validate_attn_active_units(routes, {"layer": c.metadata})
        assert (
            c.metadata.flash_v100_decode_active_num_partitions.data_ptr()
            == original_ptr
        )
    c.metadata.flash_v100_decode_active_partition_size = 256
    with pytest.raises(RuntimeError, match="unit mismatch"):
        policy.validate_attn_active_units(routes, {"layer": c.metadata})


def graph_wrapper(h, c):
    class Mode(IntEnum):
        NONE = 0
        FULL = 1

    class FakeGraph:
        def replay(self):
            if h.state.fail_replay:
                raise RuntimeError("fake replay failed")
            h.state.replays += 1

    @contextmanager
    def graph(*args, **kwargs):
        h.state.capturing = True
        try:
            yield
        finally:
            h.state.capturing = False

    fc = NS(
        batch_descriptor=7,
        cudagraph_runtime_mode=Mode.FULL,
        is_dummy_run=True,
        attn_metadata={"layer": c.metadata},
    )
    scope = dict(
        __name__=__name__,
        dataclasses=dataclasses,
        ExitStack=ExitStack,
        torch=NS(cuda=NS(CUDAGraph=FakeGraph, graph=graph)),
        CUDAGraphMode=Mode,
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: fc,
        validate_cudagraph_capturing_enabled=lambda: None,
        current_stream=lambda: None,
        get_offloader=lambda: NS(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
        set_graph_pool_id=lambda _: None,
        current_platform=NS(graph_pool_handle=lambda: None),
        weak_ref_tensors=lambda x: x,
        sm70_trace_call=lambda label, fn: fn(),
        compilation_counter=NS(num_cudagraph_captured=0),
        attn_graph_batch=policy.attn_graph_batch,
        attn_runtime_keys=policy.attn_runtime_keys,
        bind_attn_graph=policy.bind_attn_graph,
        collect_attn_routes=policy.collect_attn_routes,
        attn_live_batch=policy.attn_live_batch,
        validate_attn_replay=policy.validate_attn_replay,
        log_attn_routes=policy.log_attn_routes,
        logger=NS(info=lambda fmt, *args: h.state.logs.append(fmt % args)),
    )
    top = nodes(GRAPH)
    cls = next(n for n in top if getattr(n, "name", None) == "CUDAGraphWrapper")
    selected = [n for n in top if getattr(n, "name", None) == "CUDAGraphEntry"]
    selected += [n for n in cls.body if getattr(n, "name", None) == "__call__"]
    execute(selected, scope, GRAPH)
    wrapper_type = type("Wrapper", (), {"__call__": scope["__call__"]})
    w = wrapper_type()
    w.vllm_config = c.builder.vllm_config
    w.compilation_config = NS(cudagraph_capture_sizes=[1, 2, 4, 8, 16])
    w.runnable = c.forward
    w.runtime_mode = Mode.FULL
    w._attn_route_trace = True
    w.concrete_cudagraph_entries = {}
    w.cudagraph_options = NS(
        debug_log_enable=False, gc_disable=False, weak_ref_output=False
    )
    w._collect_tensor_addresses = lambda *a: []
    w.check_input_addresses = False
    w._graph_pool_for_descriptor = lambda _: None
    return w, fc


@pytest.mark.parametrize("on,p", [(False, 256), (True, 1024)])
def test_real_wrapper_capture_and_first_non_dummy_replay(h, on, p):
    c = h.case(on)
    c.prepare()
    w, fc = graph_wrapper(h, c)
    w()
    assert len(h.state.logs) == 1
    assert f"P={p}" in h.state.logs[0] and "phase=capture" in h.state.logs[0]
    w()  # dummy replay must not consume the first-runtime evidence slot
    assert len(h.state.logs) == 1
    fc.is_dummy_run = False
    c.prepare(253953, False)
    for _ in range(3):
        w()
    assert len(h.state.logs) == 2
    assert f"P={p}" in h.state.logs[1] and "phase=replay" in h.state.logs[1]
    assert h.state.replays == 4


def test_failed_replay_does_not_emit_success(h):
    c = h.case()
    c.prepare()
    w, fc = graph_wrapper(h, c)
    w()
    c.prepare(253953, False)
    fc.is_dummy_run = False
    h.state.fail_replay = True
    with pytest.raises(RuntimeError, match="fake replay failed"):
        w()
    assert len(h.state.logs) == 1
    assert not w.concrete_cudagraph_entries[7].attn_route_replayed


def test_eager_forward_is_not_capture_evidence(h):
    c = h.case()
    c.prepare(253953, False)
    c.forward()
    assert policy._capture_routes.get() is None


def test_recapture_has_fresh_route_evidence(h):
    c = h.case()
    c.prepare()
    w, fc = graph_wrapper(h, c)
    w()
    w.concrete_cudagraph_entries.clear()
    w()
    assert len(h.state.logs) == 2
    assert all("phase=capture" in log for log in h.state.logs)


def test_batch_one_then_batch_eight_restores_active_units(h):
    c = h.case(True, 8)
    c.prepare()
    saved = c.metadata.seq_lens, c.metadata.seq_lens_cpu
    c.metadata.seq_lens = torch.ones(1, dtype=torch.int32)
    c.metadata.seq_lens_cpu = torch.ones(1, dtype=torch.int32)
    c.prepare(1, False)
    assert c.metadata.flash_v100_decode_active_partition_size == 256
    c.metadata.seq_lens, c.metadata.seq_lens_cpu = saved
    c.prepare(253953, False)
    assert c.metadata.flash_v100_decode_active_partition_size == 1024
    assert c.metadata.flash_v100_decode_active_num_partitions.item() == 249


def test_wrong_interface_partition_rejected(h):
    c = h.case()
    c.prepare()
    c.metadata.flash_v100_decode_active_partition_size = 256
    h.state.capturing = True
    with policy.collect_attn_routes(), pytest.raises(RuntimeError, match="mismatch"):
        c.forward()


def test_trace_off_has_no_observer_or_logs(h):
    h.env.VLLM_FLASH_V100_ATTN_ROUTE_TRACE = False
    c = h.case()
    c.prepare()
    h.state.capturing = True
    with policy.collect_attn_routes() as routes:
        c.forward()
    assert not routes
    assert h.state.native[-1][11] == 1024


def test_fixed_policy_cannot_silently_fall_back(h):
    c = h.case()
    c.prepare()
    c.impl.use_decode_xqa = False
    with pytest.raises(RuntimeError, match="cannot fall back"):
        c.forward()


@pytest.mark.parametrize("value", [None, "0", "", "1", "2", "01", "true", " 1"])
def test_env_presence_exact_one(monkeypatch, value):
    key = "VLLM_FLASH_V100_NOSPEC_LONG_P1024"
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    tree = ast.parse((ROOT / "vllm/envs.py").read_text())
    entry = next(
        val
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for k, val in zip(node.keys, node.values)
        if isinstance(k, ast.Constant) and k.value == key
    )
    fn = eval(compile(ast.Expression(entry), "envs.py", "eval"), {"os": os})
    assert fn() is (value == "1")


@pytest.mark.parametrize(
    "bad",
    [
        "spec",
        "dtype",
        "page",
        "heads",
        "kvheads",
        "dim",
        "swa",
        "cpu",
        "graph",
        "tp",
        "trace_only",
        "both_off",
    ],
)
def test_real_builder_configuration_gate(bad):
    cls = next(
        n
        for n in nodes(BACKEND)
        if getattr(n, "name", None) == "FlashAttnV100MetadataBuilder"
    )
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    stmt = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_nospec_long_configured"
            for t in n.targets
        )
    )
    obj = NS(
        _nospec_long_p1024=True,
        device=NS(type="cuda"),
        decode_cudagraph_enabled=True,
        block_size=4096,
        headdim=256,
        num_heads_q=6,
        num_heads_kv=1,
        decode_sliding_window=None,
        vllm_config=NS(parallel_config=NS(tensor_parallel_size=4)),
    )
    cfg = NS(cache_dtype="nvfp4")
    scope = dict(
        self=obj,
        envs=NS(VLLM_FLASH_V100_ATTN_ROUTE_TRACE=True),
        spec_config=None,
        cache_config=cfg,
        current_platform=NS(is_device_capability=lambda _: True),
    )
    if bad in ("trace_only", "both_off"):
        obj._nospec_long_p1024 = False
        scope["envs"].VLLM_FLASH_V100_ATTN_ROUTE_TRACE = bad == "trace_only"
    if bad == "spec":
        scope["spec_config"] = object()
    if bad == "dtype":
        cfg.cache_dtype = "fp8_e4m3"
    if bad == "page":
        obj.block_size = 2048
    if bad == "heads":
        obj.num_heads_q = 8
    if bad == "kvheads":
        obj.num_heads_kv = 2
    if bad == "dim":
        obj.headdim = 128
    if bad == "swa":
        obj.decode_sliding_window = 1024
    if bad == "cpu":
        obj.device.type = "cpu"
    if bad == "graph":
        obj.decode_cudagraph_enabled = False
    if bad == "tp":
        obj.vllm_config.parallel_config.tensor_parallel_size = 1
    execute([stmt], scope, BACKEND)
    assert obj._nospec_long_configured is (bad == "trace_only")


def v2_manager(h):
    """Execute actual V2 candidate/capture/replay bodies with fake CUDA only."""

    class Mode(IntEnum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2
        FULL_AND_PIECEWISE = 3

        def decode_mode(self):
            return Mode.FULL

        def mixed_mode(self):
            return Mode.PIECEWISE

        def separate_routine(self):
            return True

    @contextmanager
    def graph(*args, **kwargs):
        h.state.capturing = True
        try:
            yield
        finally:
            h.state.capturing = False

    class FakeGraph:
        def replay(self):
            if h.state.fail_replay:
                raise RuntimeError("fake replay failed")
            h.state.replays += 1

    path = ROOT / "vllm/v1/worker/gpu/cudagraph_utils.py"
    scope = dict(
        __name__=__name__,
        dataclass=dataclasses.dataclass,
        defaultdict=defaultdict,
        CUDAGraphMode=Mode,
        nullcontext=nullcontext,
        torch=NS(
            inference_mode=lambda: lambda fn: fn,
            cuda=NS(graph=graph, CUDAGraph=FakeGraph),
        ),
        graph_capture=lambda **kw: nullcontext(),
        sm70_decode_graph_compilation=lambda *a: nullcontext(),
        is_global_first_rank=lambda: False,
        envs=NS(VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=False),
        compilation_counter=NS(num_cudagraph_captured=0),
        get_offloader=lambda: NS(
            sync_prev_onload=lambda: None, join_after_forward=lambda: None
        ),
        attn_graph_batch=policy.attn_graph_batch,
        attn_runtime_keys=policy.attn_runtime_keys,
        bind_attn_graph=policy.bind_attn_graph,
        collect_attn_routes=policy.collect_attn_routes,
        log_attn_routes=policy.log_attn_routes,
        validate_attn_replay=policy.validate_attn_replay,
        logger=NS(
            debug=lambda *a: None,
            info=lambda fmt, *args: h.state.logs.append(fmt % args),
        ),
    )
    cls = next(n for n in nodes(path) if getattr(n, "name", None) == "CudaGraphManager")
    methods = ("_init_candidates", "dispatch", "capture", "run_fullgraph")
    selected = [
        n
        for n in nodes(path)
        if getattr(n, "name", None) in ("BatchExecutionDescriptor", "_is_compatible")
    ]
    selected += [n for n in cls.body if getattr(n, "name", None) in methods]
    execute(selected, scope, path)
    typ = type("V2Manager", (), {n: scope[n] for n in methods})
    m = typ()
    m.vllm_config = runtime_config()
    m.device = "cpu"
    m.pool = None
    m.graphs = {}
    m._attn_route_trace = True
    m._attn_routes = {}
    m._attn_route_replays = set()
    m._graphs_captured = False
    m._capture_descs = {}
    m._candidates = []
    m._sm70_dflash2_tail_graphs = False
    m.decode_query_lens = (1,)
    m.decode_query_len = 1
    m.max_num_reqs = 16
    m.cudagraph_mode = Mode.FULL_AND_PIECEWISE
    cfgpath = ROOT / "vllm/config/vllm.py"
    cfg = {}
    execute(
        [
            n
            for n in nodes(cfgpath)
            if getattr(n, "name", None) == "_sm70_nomtp_cudagraph_capture_sizes"
            or (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id == "_SM70_NOMTP_CUDAGRAPH_CAPTURE_SIZES"
                    for t in n.targets
                )
            )
        ],
        cfg,
        cfgpath,
    )
    m._capture_sizes = cfg["_sm70_nomtp_cudagraph_capture_sizes"](16)
    m._init_candidates()
    return m, Mode


def test_real_v2_dispatch_capture_table(h):
    m, mode = v2_manager(h)
    assert m._capture_sizes == [1, 2, 4, 8, 16]
    m._graphs_captured = True
    assert {c: m.dispatch(c, c, 1).num_reqs for c in (4, 7, 16)} == {4: 4, 7: 8, 16: 16}
    assert m.dispatch(4, 32, 8).cg_mode == mode.NONE


@pytest.mark.parametrize("on,p", [(False, 256), (True, 1024)])
def test_real_v2_capture_replay_each_live_batch(h, on, p):
    m, mode = v2_manager(h)
    desc = next(d for d in m._capture_descs[mode.FULL] if d.num_reqs == 8)
    m._capture_descs = {mode.FULL: [desc]}
    c = h.case(on, 8)
    c.prepare()

    def factory(desc):
        return lambda _: c.forward(), NS(attn_metadata={"layer": c.metadata})

    m.capture(factory)
    captures = [s for s in h.state.logs if "phase=capture" in s]
    assert len(captures) == 1 and f"P={p}" in captures[0]
    assert "graph_B=8" in captures[0] and "live_B=" not in captures[0]
    c.prepare(253953, False)
    kw = dict(attn_metadata={"layer": c.metadata})
    m.run_fullgraph(desc, live_B=8, dummy_run=True, **kw)
    for live in (7, 7, 8, 7, 8):
        m.run_fullgraph(desc, live_B=live, **kw)
    replay = [s for s in h.state.logs if "phase=replay" in s]
    assert len(replay) == 2
    assert "live_B=7" in replay[0] and "live_B=8" in replay[1]
    capture_key = captures[0].split("ws_key=")[1].split()[0]
    assert all(f"ws_key={capture_key}" in s and f"P={p}" in s for s in replay)
    assert len([s for s in h.state.logs if "phase=eager" in s]) == 1
    assert h.state.replays == 6


def test_v2_failed_replay_not_marked(h):
    m, mode = v2_manager(h)
    d = next(d for d in m._capture_descs[mode.FULL] if d.num_reqs == 4)
    m._capture_descs = {mode.FULL: [d]}
    c = h.case()
    c.prepare()
    m.capture(lambda _: (lambda _: c.forward(), None))
    c.prepare(253953, False)
    h.state.fail_replay = True
    with pytest.raises(RuntimeError, match="fake replay failed"):
        m.run_fullgraph(d, live_B=4, attn_metadata={"layer": c.metadata})
    assert not m._attn_route_replays
    assert not any("phase=replay" in s for s in h.state.logs)


def test_native_diagnostic_is_required_not_python_hint(h):
    c = h.case()
    c.prepare()
    h.i["flash_attn_v100_cuda"].decode_paged_xqa_route_version = 0
    with pytest.raises(RuntimeError, match="rebuilt native"):
        c.forward()
    h.i["flash_attn_v100_cuda"].decode_paged_xqa_route_version = 1
    h.i["flash_attn_v100_cuda"].decode_paged_xqa_fwd = lambda *a, **kw: a[3]
    with pytest.raises(RuntimeError, match="native partition"):
        c.forward()  # Native left sentinel untouched: MUST NOT fill from Python.


def test_replay_log_uses_fresh_key_not_capture_copy(h):
    c = h.case()
    c.prepare()
    h.state.capturing = True
    with policy.collect_attn_routes() as routes:
        c.forward()
    # Output plumbing must use caller's independently rebuilt key. Production
    # rejects this mismatch before replay; here exercise output in isolation.
    runtime_keys = {(-1, 8, 6, 256, 256, 262144)}
    policy.log_attn_routes(
        routes,
        phase="replay",
        graph_id=123,
        live_B=7,
        runtime_keys=runtime_keys,
        log=lambda fmt, *a: h.state.logs.append(fmt % a),
    )
    assert " P=256 " in h.state.logs[-1]
    assert " graph_B=8 " in h.state.logs[-1]
    assert "ws_key=[-1,8,6,256,256,262144]" in h.state.logs[-1]
    assert " graph_id=123 " in h.state.logs[-1]


def test_launch_partition_evidence_is_native_value(h):
    c = h.case()
    c.prepare()
    h.state.capturing = True
    original = h.i["flash_attn_v100_cuda"].decode_paged_xqa_fwd

    def distinct_native_grid(*a, route_info=None):
        result = original(*a, route_info=route_info)
        route_info[2] = 177  # deliberately different from Python launch hint
        return result

    h.i["flash_attn_v100_cuda"].decode_paged_xqa_fwd = distinct_native_grid
    with policy.collect_attn_routes() as routes:
        c.forward()
    assert routes[0]["launch_partitions"] == 177
    assert h.state.native[-1][12] == 256


def test_model_runner_passes_unpadded_request_count():
    path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(path.read_text())
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run_fullgraph"
    )
    seen = {}
    runner = NS(cudagraph_manager=NS(run_fullgraph=lambda d, **kw: seen.update(kw)))
    eval(
        compile(ast.Expression(call), str(path), "eval"),
        dict(
            self=runner,
            batch_desc="graph8",
            input_batch=NS(num_reqs=7, num_reqs_after_padding=8),
            attn_metadata="real metadata",
            dummy_run=False,
        ),
    )
    assert seen == {"live_B": 7, "attn_metadata": "real metadata", "dummy_run": False}


@pytest.mark.parametrize("batch", [1, 17])
def test_dispatch_rejects_forged_fixed_policy_outside_batch_scope(h, batch):
    c = h.case(True, batch)
    c.prepare()
    # Bypass the builder deliberately: test dispatch's own independent guard.
    c.metadata.flash_v100_nospec_long_scope = True
    c.metadata.flash_v100_decode_partition_size_hint = 1024
    c.metadata.flash_v100_decode_active_partition_size = 1024
    with pytest.raises(RuntimeError, match="dispatch shape mismatch"):
        c.forward()
    assert not h.state.native


@pytest.mark.parametrize("flag", [-1, 2])
def test_native_capture_flag_domain_rejected(h, flag):
    c = h.case()
    c.prepare()
    h.state.capturing = True
    native = h.i["flash_attn_v100_cuda"].decode_paged_xqa_fwd

    def invalid_flag(*args, **kw):
        out = native(*args, **kw)
        kw["route_info"][3] = flag
        return out

    h.i["flash_attn_v100_cuda"].decode_paged_xqa_fwd = invalid_flag
    with policy.collect_attn_routes(), pytest.raises(RuntimeError, match="mismatch"):
        c.forward()


def test_eager_rejects_unclosed_capture_collector(h):
    c = h.case()
    c.prepare()
    with (
        policy.collect_attn_routes(),
        pytest.raises(RuntimeError, match="outside native"),
    ):
        c.forward()


def test_off_trace_off_uses_legacy19_without_optional_kw(h):
    c = h.case(False)
    h.env.VLLM_FLASH_V100_ATTN_ROUTE_TRACE = False
    c.prepare()
    c.forward()
    assert len(h.state.native[-1]) == 19
    assert h.state.native_kwargs[-1] == {}
    assert h.state.native[-1][11] == 256
    assert not h.state.logs


def captured_v2_eight(h):
    m, mode = v2_manager(h)
    desc = next(d for d in m._capture_descs[mode.FULL] if d.num_reqs == 8)
    m._capture_descs = {mode.FULL: [desc]}
    c = h.case(True, 8)
    c.prepare()
    m.capture(lambda _: (lambda _: c.forward(), None))
    c.prepare(253953, False)
    return m, desc, c


def test_replay_wrong_graph_object_fails_before_launch(h):
    m, desc, c = captured_v2_eight(h)
    original = m.graphs[desc]
    m.graphs[desc] = type(original)()  # wrong object behind the correct table key
    with pytest.raises(RuntimeError, match="graph-id-mismatch"):
        m.run_fullgraph(desc, live_B=7, attn_metadata={"layer": c.metadata})
    assert h.state.replays == 0
    assert not any("phase=replay" in line for line in h.state.logs)


def test_replay_runtime_policy_change_fails_key_match(h, monkeypatch):
    m, desc, c = captured_v2_eight(h)
    # Captured route, metadata labels and active pointer all still say ON.
    # Only the independently evaluated runtime policy changes.
    monkeypatch.delenv("VLLM_FLASH_V100_NOSPEC_LONG_P1024")
    with pytest.raises(RuntimeError, match="key-mismatch"):
        m.run_fullgraph(desc, live_B=7, attn_metadata={"layer": c.metadata})
    assert h.state.replays == 0
    assert not m._attn_route_replays


def test_replay_wrong_descriptor_detected_by_fresh_dispatch(h):
    m, desc, c = captured_v2_eight(h)
    # live_B=4 must relookup graph4; passed desc8 can otherwise accommodate it.
    with pytest.raises(RuntimeError, match="key-mismatch"):
        m.run_fullgraph(desc, live_B=4, attn_metadata={"layer": c.metadata})
    assert h.state.replays == 0


def test_capture_replay_ids_are_actual_graph_object(h):
    m, desc, c = captured_v2_eight(h)
    m.run_fullgraph(desc, live_B=7, attn_metadata={"layer": c.metadata})
    logs = [s for s in h.state.logs if "phase=eager" not in s]
    assert len(logs) == 2
    assert all(f"graph_id={id(m.graphs[desc])}" in s for s in logs)
    assert "allocator_ws_key=" in logs[0]
    assert "allocator_ws_key=" not in logs[1]


def test_legacy_wrong_graph_object_fails_before_launch(h):
    c = h.case()
    c.prepare()
    w, fc = graph_wrapper(h, c)
    w()
    entry = w.concrete_cudagraph_entries[fc.batch_descriptor]
    assert f"graph_id={id(entry.cudagraph)}" in h.state.logs[0]
    entry.cudagraph = type(entry.cudagraph)()
    c.prepare(253953, False)
    fc.is_dummy_run = False
    with pytest.raises(RuntimeError, match="graph-id-mismatch"):
        w()
    assert h.state.replays == 0


def test_replay_capacity_is_from_runtime_inputs(h):
    m, desc, c = captured_v2_eight(h)
    c.metadata.block_table = c.metadata.block_table[:, :32]
    with pytest.raises(RuntimeError, match="key-mismatch"):
        m.run_fullgraph(desc, live_B=7, attn_metadata={"layer": c.metadata})
    assert h.state.replays == 0


def test_replay_checks_active_units_before_launch(h):
    m, desc, c = captured_v2_eight(h)
    # The selected graph and independently recomputed key still match capture.
    # Only the units of its live active-partitions buffer are inconsistent.
    c.metadata.flash_v100_decode_active_partition_size = 256
    with pytest.raises(RuntimeError, match="replay active-buffer unit mismatch"):
        m.run_fullgraph(desc, live_B=7, attn_metadata={"layer": c.metadata})
    assert h.state.replays == 0
    assert not m._attn_route_replays
    assert not any("phase=replay" in line for line in h.state.logs)


def test_legacy_padded_replay_uses_graph_batch_in_lookup_key(h):
    c = h.case(True, 8)
    c.prepare()
    w, fc = graph_wrapper(h, c)
    fc.batch_descriptor = 8
    w()
    entry = w.concrete_cudagraph_entries[8]
    c.prepare(253953, False)
    c.metadata.num_actual_tokens = 7
    c.metadata.seq_lens[-1] = 0
    c.metadata.seq_lens_cpu[-1] = 0
    fc.is_dummy_run = False
    w()
    assert h.state.replays == 1
    assert entry.attn_route_replayed == {7}
    replay = [line for line in h.state.logs if "phase=replay" in line]
    assert len(replay) == 1
    assert " graph_B=8 " in replay[0] and " live_B=7 " in replay[0]
    assert "ws_key=[-1,8,6,256,1024,262144]" in replay[0]
    assert f" graph_id={id(entry.cudagraph)} " in replay[0]
