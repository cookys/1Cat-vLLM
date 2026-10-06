# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for packed NVFP4 KV; native GPU readers are tested separately.

Use CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 and --noconftest.
"""

import os
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flash_attn_v100 as f
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: pytest.fail("CPU contract test attempted CUDA initialization"),
    )
    yield
    assert not torch.cuda.is_initialized()


def impl():
    obj = f.FlashAttnV100Impl.__new__(f.FlashAttnV100Impl)
    obj.kv_cache_dtype = "nvfp4"
    obj.nvfp4_kv_available = True
    obj.nvfp4_paged_kv_to_fp16 = None
    obj.num_kv_heads = 1
    obj.attn_type = AttentionType.DECODER
    obj.alibi_slopes = None
    obj.logits_soft_cap = 0
    obj.sinks = None
    obj.sliding_window = (-1, -1)
    obj.prefix_anchored_decode_window = None
    # Real __init__ always sets these; default = grouped verifier disabled.
    obj.use_dflash2_grouped_verify = False
    obj.use_dflash2_batched_grouped_verify = False
    obj.dflash2_grouped_verify_request_major_abi_version = 0
    obj.dflash2_grouped_verify_max_query_tokens = 16
    obj.dflash2_grouped_verify_min_model_len = 0
    obj.dflash2_grouped_verify_extra_pages = ()
    obj.flash_attn_grouped_verify_paged = None
    return obj


def metadata(lengths):
    qsl = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    seq = torch.tensor([32] * len(lengths), dtype=torch.int32)
    return NS(
        num_actual_tokens=sum(lengths),
        max_query_len=max(lengths),
        causal=True,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq,
        seq_lens_cpu=seq,
        block_table=torch.tensor([[0, 1]] * len(lengths), dtype=torch.int32),
    )


@pytest.mark.parametrize("block", [16, 416, 1024, 2784, 2912, 4096])
def test_packed_shape(block):
    assert f.FlashAttnV100Backend.get_kv_cache_shape(3, block, 1, 256, "nvfp4") == (
        3,
        2,
        block,
        1,
        144,
    )
    assert "nvfp4" in f.FlashAttnV100Backend.supported_kv_cache_dtypes
    assert "nvfp4" not in TritonAttentionBackend.supported_kv_cache_dtypes


@pytest.mark.parametrize("dtype", ["auto", "float16", "fp8_e5m2", "fp8_e4m3"])
def test_legacy_shape_unchanged(dtype):
    assert f.FlashAttnV100Backend.get_kv_cache_shape(
        3, 16, 2, 128, dtype
    ) == TritonAttentionBackend.get_kv_cache_shape(3, 16, 2, 128, dtype)


@pytest.mark.parametrize("head,block", [(128, 16), (256, 819)])
def test_bad_nvfp4_shape_rejected(head, block):
    with pytest.raises(ValueError):
        f.FlashAttnV100Backend.get_kv_cache_shape(3, block, 1, head, "nvfp4")


@pytest.mark.parametrize("available", [False, True, None])
def test_capability_probe_not_just_dtype_or_python_symbol(monkeypatch, available):
    package = NS(flash_attn_decode_paged_xqa=Mock())
    if available is not None:
        package.flash_attn_nvfp4_kv_available = lambda min_version=1: available
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
    if available:
        assert f._require_nvfp4_xqa_reader() is None
    else:
        with pytest.raises(RuntimeError, match="NVFP4"):
            f._require_nvfp4_xqa_reader()


def _grouped_impl(extra_pages=(4096,)):
    obj = impl()
    obj.scale = 0.0625
    obj.use_dflash2_grouped_verify = True
    obj.use_dflash2_batched_grouped_verify = False
    obj.dflash2_grouped_verify_request_major_abi_version = 0
    obj.dflash2_grouped_verify_max_query_tokens = 16
    obj.dflash2_grouped_verify_min_model_len = 0
    obj.dflash2_grouped_verify_extra_pages = tuple(extra_pages)
    obj.flash_attn_grouped_verify_paged = Mock()
    obj._call_dflash2_grouped_verify = Mock()
    obj._flash_v100_small_query_prefill_as_decode = Mock()
    return obj


def _grouped_case(obj, page=4096, q_len=5, mtp=True):
    q = torch.zeros(q_len, 6, 256, dtype=torch.float16)
    out = torch.empty_like(q)
    cache = torch.empty(2, 2, page, 1, 144, dtype=torch.uint8)
    meta = metadata([q_len])
    meta.num_reqs = 1
    meta.max_model_len = 32768
    meta.is_mtp_verify_target = mtp
    obj._call_dflash2_grouped_verify.return_value = out
    obj._flash_v100_small_query_prefill_as_decode.return_value = out
    layer = NS(_k_scale_float=0.5, _v_scale_float=2.0)
    assert obj.forward(layer, q, q, q, cache, meta, out) is out


def _fake_probe(monkeypatch, version):
    package = NS(
        flash_attn_nvfp4_kv_available=lambda min_version=1: version >= min_version
    )
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)


def test_grouped_gate_rejects_nvfp4_without_probe_v3(monkeypatch):
    _fake_probe(monkeypatch, 2)
    obj = _grouped_impl()
    q = torch.zeros(5, 6, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 4096, 1, 144, dtype=torch.uint8)
    meta = metadata([5])
    meta.num_reqs = 1
    meta.max_model_len = 32768
    meta.is_mtp_verify_target = True
    assert not obj._dflash2_grouped_verify_allowed(
        q, cache[:, 0], cache[:, 1], meta, num_query_tokens=5
    )


def test_nvfp4_verify_takes_grouped_when_gate_admits(monkeypatch):
    _fake_probe(monkeypatch, 3)
    obj = _grouped_impl()
    _grouped_case(obj)
    obj._call_dflash2_grouped_verify.assert_called_once()
    obj._flash_v100_small_query_prefill_as_decode.assert_not_called()


@pytest.mark.parametrize(
    "version,page,mtp",
    [(2, 4096, True), (3, 2048, True), (3, 4096, False)],
)
def test_nvfp4_verify_falls_back_to_per_row_when_gate_rejects(
    monkeypatch, version, page, mtp
):
    _fake_probe(monkeypatch, version)
    obj = _grouped_impl()
    _grouped_case(obj, page=page, mtp=mtp)
    obj._call_dflash2_grouped_verify.assert_not_called()
    obj._flash_v100_small_query_prefill_as_decode.assert_called_once()


def test_nvfp4_verify_grouped_disabled_uses_per_row(monkeypatch):
    _fake_probe(monkeypatch, 3)
    obj = _grouped_impl()
    obj.use_dflash2_grouped_verify = False
    _grouped_case(obj)
    obj._call_dflash2_grouped_verify.assert_not_called()
    obj._flash_v100_small_query_prefill_as_decode.assert_called_once()


@pytest.mark.parametrize("q_per_kv", [4, 6, 8])
def test_real_decode_dispatch_passes_packed_dtype_and_scales(q_per_kv):
    obj = impl()
    obj.scale = 0.0625
    obj.use_decode_xqa = True
    obj.flash_attn_decode_paged_xqa = Mock()
    obj.flash_attn_decode_paged = Mock()
    obj._flash_decode_paged_kwargs = set()
    obj._sm70_scalar_tail_attention = Mock(side_effect=AssertionError("FP8-only tail"))
    q = torch.zeros(2, q_per_kv, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    layer = NS(_k_scale_float=0.5, _v_scale_float=2.0)
    obj.forward(layer, q, q, q, cache, metadata([1, 1]), torch.empty_like(q))
    called = obj.flash_attn_decode_paged_xqa
    assert called.call_count == 1
    assert called.call_args.args[1].shape[-1] == 144
    assert called.call_args.kwargs["kv_cache_dtype"] == "nvfp4"
    assert called.call_args.kwargs["k_scale"] == 0.5
    assert called.call_args.kwargs["v_scale"] == 2.0
    obj._sm70_scalar_tail_attention.assert_not_called()
    obj.flash_attn_decode_paged.assert_not_called()
    for side, view in enumerate(called.call_args.args[1:3]):
        assert view.data_ptr() == cache[:, side].data_ptr()
        assert view.stride() == (2 * 16 * 144, 144, 144, 1)
        assert view.data_ptr() % 4 == 0


@pytest.mark.parametrize("q_per_kv", [4, 6, 8])
def test_smallq_capture_uses_static_rows_and_never_grouped(monkeypatch, q_per_kv):
    obj = impl()
    obj.scale = 0.0625
    obj.use_dflash2_grouped_verify = True
    obj.use_smallq_decode_xqa = True
    obj.flash_attn_decode_paged_xqa = Mock()
    obj._call_dflash2_grouped_verify = Mock(side_effect=AssertionError("grouped"))
    obj.flash_attn_grouped_e4m3_fp32_paged = Mock(side_effect=AssertionError("E4M3"))
    q = torch.zeros(10, q_per_kv, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    meta = metadata([5, 5])
    meta.smallq_decode_block_table = meta.block_table.repeat_interleave(5, dim=0)
    meta.smallq_decode_seq_lens = torch.tensor(
        [28, 29, 30, 31, 32] * 2, dtype=torch.int32
    )
    meta.smallq_query_start_loc = meta.query_start_loc
    meta.smallq_decode_max_seq_len_hint = 32
    meta.smallq_decode_workspace_seq_capacity_hint = 32
    monkeypatch.setattr(f, "_is_cuda_graph_capturing", lambda _: True)
    obj.forward(
        NS(_k_scale_float=1.0, _v_scale_float=1.0),
        q,
        q,
        q,
        cache,
        meta,
        torch.empty_like(q),
    )
    call = obj.flash_attn_decode_paged_xqa.call_args
    assert call.args[3].data_ptr() == meta.smallq_decode_block_table.data_ptr()
    assert call.args[4].data_ptr() == meta.smallq_decode_seq_lens.data_ptr()
    assert call.kwargs["kv_cache_dtype"] == "nvfp4"
    obj._call_dflash2_grouped_verify.assert_not_called()
    obj.flash_attn_grouped_e4m3_fp32_paged.assert_not_called()


@pytest.mark.parametrize("lengths,route", [([1, 1], "decode"), ([5, 5], "verify")])
def test_forward_routes_packed_bytes_only_to_qualified_readers(lengths, route):
    obj = impl()
    q = torch.zeros(sum(lengths), 6, 256, dtype=torch.float16)
    out = torch.empty_like(q)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    obj._flash_v100_decode = Mock(return_value=out)
    obj._flash_v100_small_query_prefill_as_decode = Mock(return_value=out)
    obj._run_fp8_prefill_bridge = Mock(side_effect=AssertionError("FP8 bridge"))
    layer = NS(_k_scale_float=0.5, _v_scale_float=2.0)
    assert obj.forward(layer, q, q, q, cache, metadata(lengths), out) is out
    assert obj._flash_v100_decode.call_count == (route == "decode")
    assert obj._flash_v100_small_query_prefill_as_decode.call_count == (
        route == "verify"
    )
    obj._run_fp8_prefill_bridge.assert_not_called()


@pytest.mark.parametrize("lengths", [None, [9], [32, 1]])
def test_p1_rejects_profile_and_large_prefill(lengths):
    obj = impl()
    q = torch.empty(33, 6, 256, dtype=torch.float16)
    meta = metadata(lengths) if lengths is not None else None
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="prefill"):
        obj.forward(None, q, q, q, cache, meta, torch.empty_like(q))


@pytest.mark.parametrize("route", ["decode", "smallq"])
def test_p1_disabled_xqa_cannot_fall_back_to_scalar(route):
    obj = impl()
    obj.scale = 0.0625
    obj.use_decode_xqa = obj.use_smallq_decode_xqa = False
    obj.flash_attn_decode_paged_xqa = Mock()
    obj.flash_attn_decode_paged = Mock()
    obj.flash_attn_grouped_e4m3_fp32_paged = Mock()
    obj.use_dflash2_grouped_verify = False
    lengths = [1] if route == "decode" else [5]
    q = torch.zeros(sum(lengths), 6, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="scalar fallback"):
        obj.forward(
            NS(_k_scale_float=1.0, _v_scale_float=1.0),
            q,
            q,
            q,
            cache,
            metadata(lengths),
            torch.empty_like(q),
        )
    obj.flash_attn_decode_paged_xqa.assert_not_called()
    obj.flash_attn_decode_paged.assert_not_called()
    obj.flash_attn_grouped_e4m3_fp32_paged.assert_not_called()


def test_p1_bridge_direct_entry_is_rejected():
    with pytest.raises(NotImplementedError, match="bridge"):
        impl()._run_fp8_prefill_bridge(
            query=None,
            key_cache=None,
            value_cache=None,
            block_table=None,
            seq_lens=None,
            seq_len=0,
            k_scale=1.0,
            v_scale=1.0,
            causal=True,
            window_size=(-1, -1),
            out=None,
        )


@pytest.mark.parametrize("bad", ["alignment", "page_padding", "hnd"])
def test_p1_native_view_contract_rejects_other_layouts(bad):
    shape = (3, 2, 16, 2, 144)
    if bad == "alignment":
        raw = torch.empty(3 * 2 * 16 * 2 * 144 + 1, dtype=torch.uint8)
        cache = raw[1:].reshape(shape)
    elif bad == "page_padding":
        side = 16 * 2 * 144
        cache = torch.empty_strided(
            shape, (2 * side + 64, side, 288, 144, 1), dtype=torch.uint8
        )
    else:
        cache = torch.empty(3, 2, 2, 16, 144, dtype=torch.uint8).permute(0, 1, 3, 2, 4)
    with pytest.raises(ValueError, match="aligned base"):
        f._validate_nvfp4_xqa_cache(cache, 2)


@pytest.mark.parametrize(
    "bad", ["reader", "noncausal", "tree", "dtype", "output_scale"]
)
def test_forward_fail_closed(bad):
    obj = impl()
    meta = metadata([2])
    q = torch.empty(2, 6, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    extra = {}
    if bad == "reader":
        obj.nvfp4_kv_available = False
    if bad == "noncausal":
        meta.causal = False
    if bad == "tree":
        meta.ddtree_parent_ids = torch.zeros(1)
    if bad == "dtype":
        cache = cache.half()
    if bad == "output_scale":
        extra["output_scale"] = torch.ones(1)
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
        obj.forward(None, q, q, q, cache, meta, torch.empty_like(q), **extra)


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)])
@pytest.mark.parametrize("padded", [False, True])
def test_actual_writer_matches_p071_bytes_and_does_not_touch_neighbors(scales, padded):
    from vllm.models.qwen4_exp.nvidia.ops.nvfp4_kv import (
        reshape_and_cache_nvfp4_reference,
    )

    obj = impl()
    rows = 8
    qkv = torch.randn(
        rows, 3, 1, 256, generator=torch.Generator().manual_seed(18)
    ).half()
    key, value = qkv[:, 0], qkv[:, 2]
    page = 2 * 16 * 144
    stride = page + (64 if padded else 0)
    raw = torch.full((3 * stride,), 0x7F, dtype=torch.uint8)
    ref_raw = raw.clone()
    shape, strides = (3, 2, 16, 1, 144), (stride, 16 * 144, 144, 144, 1)
    cache = torch.as_strided(raw, shape, strides)
    ref = torch.as_strided(ref_raw, shape, strides)
    slots = torch.tensor([0, 15, 16, 47, -1, 48], dtype=torch.int64)
    layer = NS(_k_scale=torch.tensor(scales[0]), _v_scale=torch.tensor(scales[1]))
    obj.do_kv_cache_update(layer, key, value, cache, slots)
    reshape_and_cache_nvfp4_reference(
        key, value, ref, slots, k_scale=layer._k_scale, v_scale=layer._v_scale
    )
    assert torch.equal(raw, ref_raw)


@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("symbol", ["absent", "noncallable", "callable"])
def test_versioned_probe_guards_prefill_independently(monkeypatch, version, symbol):
    bridge = Mock()
    package = NS(
        flash_attn_decode_paged_xqa=Mock(),
        flash_attn_nvfp4_kv_available=lambda min_version=1: version >= min_version,
    )
    if symbol != "absent":
        package.nvfp4_paged_kv_to_fp16 = bridge if symbol == "callable" else None
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
    if version == 0:
        with pytest.raises(RuntimeError, match="capability"):
            f._require_nvfp4_xqa_reader()
    else:
        f._require_nvfp4_xqa_reader()
    if version < 2:
        assert f._get_nvfp4_prefill_bridge_op() is None
    elif symbol == "callable":
        assert f._get_nvfp4_prefill_bridge_op() is bridge
    else:
        with pytest.raises(RuntimeError, match="version >=2"):
            f._get_nvfp4_prefill_bridge_op()


@pytest.mark.parametrize("oom", [False, True])
def test_p3_profile_reserves_full_fp16_bridge_before_kv_pool(monkeypatch, oom):
    obj = impl()
    obj.nvfp4_paged_kv_to_fp16 = Mock()
    obj._nvfp4_bridge_profile_blocks = 337
    workspace = Mock(return_value=None if oom else (1, 2, 3))
    monkeypatch.setattr(f, "_get_fp8_prefill_bridge_workspace", workspace)
    native_profile = Mock()
    monkeypatch.setattr(f, "_profile_sm70_prefill_workspace", native_profile)
    q = torch.empty(32, 6, 256, dtype=torch.float16)
    out = torch.full_like(q, 99)
    if oom:
        with pytest.raises(RuntimeError, match="reserve NVFP4"):
            obj.forward(None, q, q, q, torch.empty(0), None, out)
        assert torch.all(out == 99)
        native_profile.assert_not_called()
    else:
        assert obj.forward(None, q, q, q, torch.empty(0), None, out) is out
        assert torch.count_nonzero(out) == 0
        native_profile.assert_called_once()
    assert workspace.call_args.kwargs["head_dim"] == 256
    assert workspace.call_args.args[1] == 337
    obj.nvfp4_paged_kv_to_fp16.assert_not_called()


def test_p3_workspace_decodes_256_and_reuses_profiled_buffer():
    f._fp8_prefill_bridge_workspaces.clear()
    try:
        shape_only = torch.empty(0, 16, 1, 144, dtype=torch.uint8)
        prof = f._get_fp8_prefill_bridge_workspace(shape_only, 3, head_dim=256)
        cache = torch.empty(4, 16, 1, 144, dtype=torch.uint8)
        live = f._get_fp8_prefill_bridge_workspace(cache, 2, head_dim=256)
        assert prof[0].shape == (3, f._FP8_PREFILL_BRIDGE_PAGE_SIZE, 1, 256)
        assert live[0].shape == (2, f._FP8_PREFILL_BRIDGE_PAGE_SIZE, 1, 256)
        assert live[0].data_ptr() == prof[0].data_ptr()
        assert live[1].data_ptr() == prof[1].data_ptr()
        assert live[2].tolist() == [[0, 1]]
    finally:
        f._fp8_prefill_bridge_workspaces.clear()


@pytest.mark.parametrize("destination", [False, True])
def test_p3_mixed_prefill_uses_actual_lengths_and_ignores_padding(destination):
    obj = impl()
    obj.nvfp4_paged_kv_to_fp16 = Mock()
    q = torch.zeros(38, 6, 256, dtype=torch.float16)  # 33 live + 5 graph padding
    out = torch.full_like(q, -99)
    cache = torch.empty(4, 2, 16, 1, 144, dtype=torch.uint8)
    meta = metadata([32, 1, 0])
    meta.seq_lens = torch.tensor([48, 35, 0], dtype=torch.int32)
    meta.seq_lens_cpu = torch.tensor([63, 39, 0], dtype=torch.int32)  # upper bounds
    calls = []

    def bridge(**kw):
        calls.append(kw)
        filled = torch.full_like(kw["out"], len(calls))
        if destination:
            kw["out"].copy_(filled)
        return (kw["out"] if destination else filled), destination

    obj._run_fp8_prefill_bridge = bridge
    layer = NS(_k_scale_float=0.75, _v_scale_float=1.25)
    assert obj.forward(layer, q, q, q, cache, meta, out) is out
    assert [c["seq_len"] for c in calls] == [48, 35]
    assert [int(c["seq_lens"][0]) for c in calls] == [48, 35]
    assert [c["query"].shape[1] for c in calls] == [32, 1]
    assert all(c["key_cache"].data_ptr() == cache[:, 0].data_ptr() for c in calls)
    assert all(c["k_scale"] == 0.75 and c["v_scale"] == 1.25 for c in calls)
    assert torch.all(out[:32] == 1) and torch.all(out[32:33] == 2)
    assert torch.all(out[33:] == -99)


@pytest.mark.parametrize("failure", ["capture", "allocation"])
def test_p3_prefill_failure_does_not_fall_through(monkeypatch, failure):
    obj = impl()
    obj.nvfp4_paged_kv_to_fp16 = Mock()
    obj._run_fp8_prefill_bridge = Mock(return_value=None)
    monkeypatch.setattr(f, "_is_cuda_graph_capturing", lambda _: failure == "capture")
    q = torch.empty(32, 6, 256, dtype=torch.float16)
    cache = torch.empty(2, 2, 16, 1, 144, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="eager|unavailable"):
        obj.forward(
            NS(_k_scale_float=1.0, _v_scale_float=1.0),
            q,
            q,
            q,
            cache,
            metadata([32]),
            torch.empty_like(q),
        )
    assert obj._run_fp8_prefill_bridge.call_count == (failure != "capture")


def test_p3_actual_bridge_helper_passes_unpacked_shape_and_scales_once(monkeypatch):
    obj = impl()
    obj.scale = 0.0625
    calls = []

    def bridge(k, v, table, seq, k_out, v_out, ks, vs):
        calls.append((k, v, table, seq, ks, vs))
        assert k_out.shape[-1] == v_out.shape[-1] == 256
        assert k_out.dtype == v_out.dtype == torch.float16
        k_out.fill_(2 * ks)
        v_out.fill_(3 * vs)

    obj.nvfp4_paged_kv_to_fp16 = bridge
    obj.fp8_e4m3_paged_kv_to_fp16 = Mock(side_effect=AssertionError("FP8 bridge"))
    obj.fp8_e5m2_paged_kv_to_fp16 = Mock(side_effect=AssertionError("FP8 bridge"))
    monkeypatch.setattr(f, "_try_sm70_fa2_d256_prefill", Mock(return_value=None))
    # The production helper keys buffers by the active CUDA device. These
    # tests exercise dispatch/scales using CPU cu_seqlens, not the CUDA helper.
    monkeypatch.setattr(
        f,
        "_uniform_cu_seqlens",
        lambda query, batch_size, query_len, kv_len: (
            torch.tensor([0, query_len], dtype=torch.int32),
            torch.tensor([0, kv_len], dtype=torch.int32),
        ),
    )
    q = torch.zeros(1, 32, 6, 256, dtype=torch.float16)
    out = torch.empty_like(q)
    obj.flash_attn_prefill_paged = Mock(return_value=out)
    cache = torch.empty(4, 2, 16, 1, 144, dtype=torch.uint8)
    table = torch.tensor([[3, 0, 2, 1]], dtype=torch.int32)
    seq = torch.tensor([49], dtype=torch.int32)
    f._fp8_prefill_bridge_workspaces.clear()
    try:
        result, in_place = obj._run_fp8_prefill_bridge(
            query=q,
            key_cache=cache[:, 0],
            value_cache=cache[:, 1],
            block_table=table,
            seq_lens=seq,
            seq_len=49,
            k_scale=0.75,
            v_scale=1.25,
            causal=True,
            window_size=(-1, -1),
            out=out,
        )
        assert result is out and not in_place
        assert len(calls) == 1 and calls[0][-2:] == (0.75, 1.25)
        call = obj.flash_attn_prefill_paged.call_args
        assert call.kwargs["kv_cache_dtype"] == "auto"
        assert call.kwargs["k_scale"] == call.kwargs["v_scale"] == 1.0
        assert torch.all(call.args[1] == 1.5) and torch.all(call.args[2] == 3.75)
        assert call.args[1].shape[-1] == 256 and call.args[3].tolist() == [[0]]
    finally:
        f._fp8_prefill_bridge_workspaces.clear()


@pytest.mark.parametrize("version", [1, 2])
def test_real_constructor_sets_versioned_profile_budget(monkeypatch, version):
    import vllm.config

    bridge = Mock()
    package = NS(
        flash_attn_decode_paged_xqa=Mock(),
        flash_attn_nvfp4_kv_available=lambda min_version=1: version >= min_version,
        nvfp4_paged_kv_to_fp16=bridge,
    )
    monkeypatch.setitem(sys.modules, "flash_attn_v100", package)
    monkeypatch.setattr(f, "_get_flash_ops", lambda: (Mock(),) * 9)
    monkeypatch.setattr(f, "_get_flash_grouped_verify_op", lambda: None)
    monkeypatch.setattr(f, "_get_fp8_e5m2_paged_kv_bridge_op", lambda: None)
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: NS(
            cache_config=NS(block_size=2912), model_config=NS(max_model_len=262144)
        ),
    )
    obj = f.FlashAttnV100Impl(
        num_heads=6,
        head_size=256,
        scale=0.0625,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="nvfp4",
    )
    assert obj.nvfp4_kv_available
    if version == 2:
        assert obj.nvfp4_paged_kv_to_fp16 is bridge
        assert obj._nvfp4_bridge_profile_blocks == 338
    else:
        assert obj.nvfp4_paged_kv_to_fp16 is None
        assert not hasattr(obj, "_nvfp4_bridge_profile_blocks")


@pytest.mark.parametrize("bad", ["profile_budget", "table_capacity"])
def test_p3_bridge_rejects_unprofiled_growth_and_short_block_table(monkeypatch, bad):
    obj = impl()
    obj.nvfp4_paged_kv_to_fp16 = Mock()
    obj._nvfp4_bridge_profile_blocks = 1
    workspace = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(f, "_get_fp8_prefill_bridge_workspace", workspace)
    cache = torch.empty(2, 2, 1024, 1, 144, dtype=torch.uint8)
    seq_len = 1025 if bad == "profile_budget" else 2049
    with pytest.raises((RuntimeError, ValueError), match="profiled|table capacity"):
        obj._run_fp8_prefill_bridge(
            query=None,
            key_cache=cache[:, 0],
            value_cache=cache[:, 1],
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_len=seq_len,
            k_scale=1.0,
            v_scale=1.0,
            causal=True,
            window_size=(-1, -1),
            out=None,
        )
    workspace.assert_not_called()
    obj.nvfp4_paged_kv_to_fp16.assert_not_called()
