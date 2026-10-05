# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the expert-routing dump (VLLM_SM70_EXPERT_ROUTING_DUMP_DIR).

Nothing here needs a GPU: the staging buffers, the host ring and the writer are
plain torch / numpy, so the same code that is recorded into the CUDA graphs on
the box is exercised on CPU tensors.  What these tests cannot show (graph
capture of the staging copies, their cost, the real row counts) is listed in
the delivery note, not asserted here.
"""

import hashlib
import inspect
import json
import os
import re
import stat
import threading
import time
import types

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from vllm.model_executor.layers.fused_moe import expert_routing_dump as erd

LT, LD, KD, K, NEXP = 3, 1, 2, 10, 512


@pytest.fixture(autouse=True)
def _reset_active():
    erd.ACTIVE = None
    yield
    if erd.ACTIVE is not None:
        erd.ACTIVE.close()
    erd.ACTIVE = None


def make_dumper(
    tmp_path,
    *,
    steps=3,
    max_rows=8,
    weights=True,
    rank=0,
    draft_layers=LD,
    draft_steps=KD,
    record_padded=False,
    **kwargs,
):
    return erd.ExpertRoutingDumper(
        str(tmp_path),
        rank=rank,
        num_target_layers=LT,
        num_draft_layers=draft_layers,
        num_draft_steps=draft_steps,
        top_k=K,
        num_experts=NEXP,
        steps_per_file=steps,
        max_rows=max_rows,
        with_weights=weights,
        record_padded=record_padded,
        device="cpu",
        meta_extra={"model": "unit-test", "tp": 4, **kwargs.pop("meta_extra", {})},
        **kwargs,
    )


def stg(d, layer_name, ids, w):
    """What the bound router callback does for that layer."""
    slot = d.slot_for(layer_name)
    if slot is not None:
        d.stage(slot, ids, w)


def synth_ids(m, seed):
    """[m, K] int32 with K distinct ids per row, deterministic."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(NEXP, generator=g)[:K] for _ in range(m)]
    return torch.stack(rows).to(torch.int32)


def synth_w(m, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.rand(m, K, generator=g, dtype=torch.float32)


def fake_batch(tokens_per_req, *, req_ids=None, prefilling=None, start_pos=100, pad=3):
    tokens_per_req = list(tokens_per_req)
    n = sum(tokens_per_req)
    req_ids = req_ids or [f"r{i}" for i in range(len(tokens_per_req))]
    positions = torch.arange(start_pos, start_pos + n, dtype=torch.int64)
    positions = torch.cat([positions, torch.full((pad,), 9999, dtype=torch.int64)])
    return types.SimpleNamespace(
        num_tokens=n,
        num_tokens_after_padding=n + pad,
        req_ids=req_ids,
        num_scheduled_tokens=np.array(tokens_per_req, dtype=np.int32),
        positions=positions,
        is_prefilling_np=np.array(
            prefilling if prefilling is not None else [False] * len(tokens_per_req)
        ),
        num_draft_tokens=0,
    )


def stage_all_layers(d, batch, seed, *, skip=()):
    expected = {}
    m = batch.num_tokens_after_padding  # the layer sees the padded batch
    for layer in range(LT):
        if layer in skip:
            continue
        ids, w = synth_ids(m, seed * 100 + layer), synth_w(m, seed * 100 + layer)
        stg(d, f"model.layers.{layer}.mlp.experts", ids, w)
        expected[layer] = (ids, w)
    return expected


def run_step(d, batch, seed, *, draft=True, skip=()):
    """One real step: target layers staged, target hook, draft layers + hooks."""
    expected = stage_all_layers(d, batch, seed, skip=skip)
    d.record_target_step(batch)
    draft_expected = []
    if draft and d.num_draft_layers:
        for step in range(d.num_draft_steps):
            rows = batch.num_tokens_after_padding if step == 0 else len(batch.req_ids)
            ids = synth_ids(rows, seed * 100 + 50 + step)
            stg(d, f"mtp.layers.{LT}.mlp.experts", ids, synth_w(rows, 0))
            d.record_draft_step(step, len(batch.req_ids))
            draft_expected.append(ids)
    return expected, draft_expected


def load(path, name, rank=0):
    with np.load(os.path.join(path, f"rank{rank}", name)) as z:
        return {k: z[k] for k in z.files}


def npz_files(path, rank=0):
    return sorted(
        f for f in os.listdir(os.path.join(path, f"rank{rank}")) if f.endswith(".npz")
    )


def index_lines(path, rank=0):
    with open(os.path.join(path, f"rank{rank}", "index.jsonl")) as f:
        return [json.loads(line) for line in f.read().splitlines()]


def meta_of(path, rank=0):
    with open(os.path.join(path, f"rank{rank}", "meta.json")) as f:
        return json.load(f)


def enable_kwargs(**over):
    kwargs = dict(
        num_target_layers=LT,
        num_draft_layers=LD,
        num_draft_steps=KD,
        top_k=K,
        num_experts=NEXP,
        device="cpu",
    )
    kwargs.update(over)
    return kwargs


# --------------------------------------------------------------------- (a) off
def test_env_unset_creates_no_dumper(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", raising=False)
    assert erd.maybe_enable(**enable_kwargs()) is None and erd.ACTIVE is None


def test_nonpositive_steps_disable(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path / "x"))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "0")
    assert erd.maybe_enable(**enable_kwargs()) is None
    assert not (tmp_path / "x").exists()


def test_env_set_builds_the_dumper_and_logs_once(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "7")
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS", "9")
    caplog.set_level(logging.INFO)
    d = erd.maybe_enable(**enable_kwargs(num_target_layers=48, num_draft_steps=4))
    assert d is erd.ACTIVE and d.steps_per_file == 7 and d.max_rows == 9
    assert (tmp_path / "rank0").is_dir()
    assert f"dir={tmp_path} steps=7 max_rows=9 layers=48+4" in caplog.text


@pytest.mark.parametrize(
    "seqs,k,env,expect",
    [
        (4, 4, None, 128),  # production MTP4, max-num-seqs 4: M=20, floor 128
        (16, 4, None, 128),  # M=80 still under the floor
        (64, 4, None, 320),  # 64 x 5 beats the floor
        (16, 0, None, 128),  # nospec n16: M=16
        (4, 4, "8192", 8192),  # explicit: whole prefill chunks
    ],
)
def test_max_rows_default_covers_every_decode_step(
    monkeypatch, tmp_path, seqs, k, env, expect
):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    if env is None:
        monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS", env)
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "2")
    d = erd.maybe_enable(
        **enable_kwargs(num_draft_steps=k, num_draft_layers=1 if k else 0),
        max_num_seqs=seqs,
    )
    assert d.max_rows == expect
    assert d.max_rows >= seqs * (k + 1)


def test_rank_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_RANKS", "1,2")
    monkeypatch.setattr(erd, "_tp_rank", lambda: 0)
    assert erd.maybe_enable(**enable_kwargs()) is None and erd.ACTIVE is None
    monkeypatch.setattr(erd, "_tp_rank", lambda: 2)
    d = erd.maybe_enable(**enable_kwargs())
    assert d is not None and (tmp_path / "rank2").is_dir()


class _OpLog(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        self.ops.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


def _fake_router(ids, weights):
    """A real BaseRouter (the select_experts template) with a canned top-k."""
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    class CannedRouter(BaseRouter):
        routing_method_type = None

        def _compute_routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            return weights, ids

    return CannedRouter(top_k=K, global_num_experts=NEXP)


def test_knob_off_router_path_adds_no_op_and_is_bit_identical(tmp_path):
    """The hook is one attribute check; with no callback no op is emitted."""
    ids, weights = synth_ids(5, 1), synth_w(5, 1)
    x = torch.randn(5, 8)
    router = _fake_router(ids, weights)
    assert router.dump_fn is None

    with _OpLog() as off_log:
        w_off, i_off = router.select_experts(x, x)

    d = make_dumper(tmp_path, max_rows=8)
    router.set_dump_fn(lambda w, i: stg(d, "model.layers.1.mlp.experts", i, w))
    with _OpLog() as on_log:
        w_on, i_on = router.select_experts(x, x)

    # the same tensors come back untouched, on and off
    assert w_off is weights and w_on is weights and i_off is ids and i_on is ids
    assert "copy_" not in off_log.ops
    assert on_log.ops.count("copy_") == 2  # ids + weights, nothing else
    assert d._t_ids[:5, 1].tolist() == ids.to(torch.int16).tolist()
    router.set_dump_fn(None)
    with _OpLog() as again:
        router.select_experts(x, x)
    assert "copy_" not in again.ops


def test_callback_sees_the_final_ids_after_dtype_conversion():
    ids, weights = synth_ids(3, 4), synth_w(3, 4)
    router = _fake_router(ids, weights)
    seen = []
    router.set_dump_fn(lambda w, i: seen.append((w, i)))
    w_out, i_out = router.select_experts(torch.zeros(3, 8), torch.zeros(3, 8))
    assert len(seen) == 1 and seen[0][0] is w_out and seen[0][1] is i_out


def _fake_moe(name, router):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    moe = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(moe)
    moe.layer_name = name
    moe.router = router
    return moe


def test_bind_routers_wires_every_target_and_draft_layer(tmp_path):
    d = make_dumper(tmp_path, max_rows=8)
    target = torch.nn.ModuleList(
        [
            _fake_moe(
                f"model.layers.{i}.mlp.experts",
                _fake_router(synth_ids(4, i), synth_w(4, i)),
            )
            for i in range(LT)
        ]
    )
    draft = torch.nn.ModuleList(
        [
            _fake_moe(
                f"mtp.layers.{LT}.mlp.experts",
                _fake_router(synth_ids(4, 90), synth_w(4, 90)),
            )
        ]
    )
    assert d.bind_routers([target, draft]) == LT + LD
    for moe in list(target) + list(draft):
        assert moe.router.dump_fn is not None
        moe.router.select_experts(torch.zeros(4, 8), torch.zeros(4, 8))
    for i in range(LT):
        assert d._t_ids[:4, i].tolist() == synth_ids(4, i).to(torch.int16).tolist()
    assert d._d_ids[:4, 0].tolist() == synth_ids(4, 90).to(torch.int16).tolist()
    assert d._layer_names["draft"] == {"0": f"mtp.layers.{LT}.mlp.experts"}


def test_bind_routers_warns_when_the_layer_count_is_off(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    d = make_dumper(tmp_path)
    only = torch.nn.ModuleList(
        [
            _fake_moe(
                "model.layers.0.mlp.experts",
                _fake_router(synth_ids(2, 0), synth_w(2, 0)),
            )
        ]
    )
    assert d.bind_routers([only]) == 1
    assert "bound 1 MoE routers, expected 4" in caplog.text


def test_hooks_are_guarded_at_every_call_site_and_traced_files_are_untouched():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    router = open(
        os.path.join(root, "vllm/model_executor/layers/fused_moe/router/base_router.py")
    ).read()
    mr = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()
    sp = open(
        os.path.join(root, "vllm/v1/worker/gpu/spec_decode/eagle/speculator.py")
    ).read()
    assert (
        "if self.dump_fn is not None:\n            self.dump_fn(topk_weights, topk_ids)"
        in router
    )
    assert re.search(
        r"if expert_routing_dump\.ACTIVE is not None and not dummy_run:\n"
        r"(\s+#[^\n]*\n)*\s+expert_routing_dump\.ACTIVE\.record_target_step\(",
        mr,
    )
    assert re.search(
        r"routing = expert_routing_dump\.ACTIVE\n\s+if routing is not None and "
        r"self\._dump_this_round:",
        sp,
    )
    # moe_runner.py / layer.py are inlined by torch.compile: their bytes are part
    # of the compile-cache key, so the dump must not import or mention it there.
    for traced in ("runner/moe_runner.py", "layer.py"):
        path = os.path.join(root, "vllm/model_executor/layers/fused_moe", traced)
        assert "expert_routing_dump" not in open(path).read()


def test_dump_env_vars_are_not_compile_cache_factors(monkeypatch):
    from vllm import envs

    names = [
        n
        for n in envs.environment_variables
        if n.startswith("VLLM_SM70_EXPERT_ROUTING_DUMP_")
    ]
    assert len(names) == 9
    base = envs.compile_factors()
    for n in names:
        assert n not in base
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", "/tmp/somewhere")
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "3")
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_CONCURRENCY", "16")
    assert envs.compile_factors() == base


# ---------------------------------------------------------- (b) round trip
def test_round_trip_multiple_files_and_exact_schema(tmp_path):
    d = make_dumper(tmp_path / "run-a", steps=3, max_rows=16, concurrency=2)
    shapes = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
    truth = []
    for i, tpr in enumerate(shapes):
        batch = fake_batch(tpr)
        truth.append((batch, *run_step(d, batch, i)))
    d.close()
    run = tmp_path / "run-a"

    index = index_lines(run)
    assert [tuple(e) for e in index] == [erd.INDEX_KEYS] * 3  # exact keys, in order
    assert [(e["first_step"], e["last_step"], e["valid_steps"]) for e in index] == [
        (0, 2, 3),
        (3, 5, 3),
        (6, 6, 1),
    ]
    for e in index:
        path = run / "rank0" / e["file"]
        assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert e["dropped_steps"] == 0 and e["dropped_rows"] == 0
    assert not [p for p in os.listdir(run / "rank0") if p.endswith(".tmp")]

    step = 0
    for e in index:
        z = load(run, e["file"])
        s_count = e["valid_steps"]
        assert set(z) == set(erd.NPZ_ARRAYS)
        m_file = 10
        expect = {
            "step_idx": (np.int32, (s_count,)),
            "step_wall_ns": (np.int64, (s_count,)),
            "num_tokens": (np.int32, (s_count,)),
            "padded_num_tokens": (np.int32, (s_count,)),
            "step_phase": (np.int8, (s_count,)),
            "active_requests": (np.int32, (s_count,)),
            "request_uid": (np.int32, (s_count, m_file)),
            "position": (np.int32, (s_count, m_file)),
            "token_is_prefill": (np.bool_, (s_count, m_file)),
            "row_is_padding": (np.bool_, (s_count, m_file)),
            "target_topk": (np.int16, (s_count, m_file, LT, K)),
            "target_topk_weight": (np.float16, (s_count, m_file, LT, K)),
            "draft_topk": (np.int16, (s_count, KD, m_file, LD, K)),
            "draft_num_rows": (np.int32, (s_count, KD)),
            "shared_expert_used": (np.bool_, (s_count, m_file, LT)),
        }
        for name, (dtype, shape) in expect.items():
            assert z[name].dtype == dtype, name
            assert z[name].shape == shape, name
        for j in range(s_count):
            batch, expected, draft_expected = truth[step + j]
            n = batch.num_tokens
            assert z["step_idx"][j] == step + j and z["num_tokens"][j] == n
            assert z["padded_num_tokens"][j] == n + 3  # padding is its own field
            assert z["active_requests"][j] == 2 and z["step_phase"][j] == 0
            assert not z["row_is_padding"][j, :n].any()
            for layer in range(LT):
                ids, w = expected[layer]
                np.testing.assert_array_equal(
                    z["target_topk"][j, :n, layer], ids[:n].numpy().astype(np.int16)
                )
                np.testing.assert_array_equal(
                    z["target_topk_weight"][j, :n, layer],
                    w[:n].numpy().astype(np.float16),
                )
            assert z["shared_expert_used"][j, :n].all()
            np.testing.assert_array_equal(
                z["position"][j, :n], np.arange(100, 100 + n, dtype=np.int32)
            )
            # two requests, five rows each; uids are small stable integers
            uids = list(z["request_uid"][j, :n])
            assert uids == [0] * 5 + [1] * 5
            # draft: step 0 on all token rows, later steps one row per request
            assert list(z["draft_num_rows"][j]) == [n, 2]
            np.testing.assert_array_equal(
                z["draft_topk"][j, 0, :n, 0],
                draft_expected[0][:n].numpy().astype(np.int16),
            )
            np.testing.assert_array_equal(
                z["draft_topk"][j, 1, :2, 0],
                draft_expected[1][:2].numpy().astype(np.int16),
            )
            assert (z["draft_topk"][j, 1, 2:] == -1).all()
        step += s_count

    meta = meta_of(run)
    assert set(meta) == set(erd.META_KEYS)
    assert meta["run_id"] == "run-a" and meta["concurrency"] == 2
    assert meta["cell_label"] is None and meta["synthetic"] is False
    assert meta["tp_rank"] == 0 and meta["tp"] == 4 and meta["model"] == "unit-test"
    assert meta["target_layers"] == LT and meta["draft_layers"] == LD
    assert meta["num_experts"] == NEXP and meta["top_k"] == K
    assert meta["num_draft_steps"] == KD and meta["num_speculative_tokens"] == KD
    assert meta["padding_traffic"] == "counted" and meta["padded_rows_invalid"] == 0
    assert meta["round_definition"] == erd.ROUND_DEFINITION
    assert meta["valid_steps"] == 7 and meta["dropped_steps"] == 0
    assert meta["dropped_rows"] == 0 and meta["files_written"] == 3
    assert meta["steps_per_file"] == 3 and meta["max_rows"] == 16
    assert meta["layers"]["target"] == {
        str(i): f"model.layers.{i}.mlp.experts" for i in range(LT)
    }
    assert meta["layers"]["draft"] == {"0": f"mtp.layers.{LT}.mlp.experts"}
    assert meta["dump_dir"] == str(run)
    assert "request_map.jsonl" not in os.listdir(run / "rank0")


def test_meta_extra_and_kv_info_and_labels(tmp_path):
    d = make_dumper(
        tmp_path,
        cell_label="mtp4-c1",
        concurrency="1",
        meta_extra={"num_speculative_tokens": 4, "secret_thing": "x"},
    )
    d.set_kv_info(292, 16)
    run_step(d, fake_batch([5]), 0)
    d.close()
    meta = meta_of(tmp_path)
    assert meta["kv_pool_blocks"] == 292 and meta["block_size"] == 16
    assert meta["num_speculative_tokens"] == 4 and meta["cell_label"] == "mtp4-c1"
    assert "secret_thing" not in meta  # only whitelisted extras survive


def test_weights_knob_off_skips_the_weight_array_and_buffers(tmp_path):
    d = make_dumper(tmp_path, steps=2, weights=False)
    assert d._t_w is None and d._ring.t_w is None
    run_step(d, fake_batch([2]), 1)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert "target_topk_weight" not in z and "target_topk" in z


def test_a_draft_step_that_did_not_run_is_all_padding(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    batch = fake_batch([1, 1])
    stage_all_layers(d, batch, 0)
    d.record_target_step(batch)
    stg(d, f"mtp.layers.{LT}.mlp.experts", synth_ids(5, 9), synth_w(5, 9))
    d.record_draft_step(0, 2)  # draft step 1 never ran (e.g. a prefill-only round)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["draft_num_rows"][0]) == [2, 0]
    assert (z["draft_topk"][0, 1] == -1).all()
    assert (z["draft_topk"][0, 0, :2] >= 0).all()


def test_draft_hook_without_an_open_step_and_dummy_layers_are_ignored(tmp_path):
    d = make_dumper(tmp_path)
    d.record_draft_step(0, 2)  # before any target step
    stg(d, "model.layers.99.mlp.experts", synth_ids(2, 1), synth_w(2, 1))
    stg(d, "model.embed_tokens", synth_ids(2, 1), synth_w(2, 1))
    assert d._filled == 0
    assert (d._t_ids == -1).all()


def test_draft_layer_names_never_collide_with_target_layers(tmp_path):
    d = make_dumper(tmp_path)
    assert d.slot_for(f"mtp.layers.{LT}.mlp.experts") == ("d", 0)  # continued numbering
    assert d.slot_for("mtp.layers.0.mlp.experts") == ("d", 0)  # draft-local numbering
    assert d.slot_for("mtp.layers.1.mlp.experts") is None  # only one draft layer
    assert d.slot_for("model.layers.2.mlp.experts") == ("t", 2)
    assert d.slot_for(f"model.layers.{LT}.mlp.experts") == ("d", 0)  # unnamed draft
    assert d.slot_for(f"model.layers.{LT + 1}.mlp.experts") is None


def test_nothing_is_recorded_during_graph_capture(tmp_path, monkeypatch):
    d = make_dumper(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    d.record_target_step(fake_batch([2]))
    d.record_draft_step(0, 1)
    assert d._filled == 0 and d._run.step_counter == 0


def test_close_is_idempotent_and_stops_recording(tmp_path):
    d = make_dumper(tmp_path)
    erd.ACTIVE = d
    run_step(d, fake_batch([1]), 0)
    d.close()
    d.close()
    assert erd.ACTIVE is None
    run_step(d, fake_batch([1]), 1)  # ignored after close
    assert len(index_lines(tmp_path)) == 1


# ------------------------------------------------------------- padded rows
def test_padded_rows_are_recorded_with_a_mask_when_the_knob_is_on(tmp_path):
    d = make_dumper(tmp_path, steps=3, max_rows=32, record_padded=True)
    truth = []
    for i in range(3):
        batch = fake_batch([5, 5], pad=2)
        truth.append(run_step(d, batch, i)[0])
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape == (3, 12, LT, K)  # Mmax = padded rows
    assert list(z["num_tokens"]) == [10] * 3  # the real count is untouched
    assert list(z["padded_num_tokens"]) == [12] * 3
    assert z["row_is_padding"].dtype == np.bool_
    assert not z["row_is_padding"][:, :10].any() and z["row_is_padding"][:, 10:].all()
    for i in range(3):
        for layer in range(LT):
            np.testing.assert_array_equal(
                z["target_topk"][i, :, layer],
                truth[i][layer][0][:12].numpy().astype(np.int16),
            )
            np.testing.assert_array_equal(
                z["target_topk_weight"][i, :, layer],
                truth[i][layer][1][:12].numpy().astype(np.float16),
            )
    # padding rows carry no request, position or phase
    assert (z["request_uid"][:, 10:] == -1).all()
    assert (z["position"][:, 10:] == -1).all()
    assert not z["token_is_prefill"][:, 10:].any()
    assert z["shared_expert_used"][:, :10].all()
    meta = meta_of(tmp_path)
    assert meta["padding_traffic"] == "recorded" and meta["dump_padded"] is True
    assert meta["padded_rows_invalid"] == 0 and meta["dropped_steps"] == 0


def test_knob_off_records_only_real_rows_and_says_counted(tmp_path):
    d = make_dumper(tmp_path, steps=3, max_rows=32, record_padded=False)
    for i in range(3):
        run_step(d, fake_batch([5, 5], pad=2), i)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape == (3, 10, LT, K)  # real rows only
    assert list(z["padded_num_tokens"]) == [12] * 3  # still counted
    assert not z["row_is_padding"].any()
    meta = meta_of(tmp_path)
    assert meta["padding_traffic"] == "counted" and meta["dump_padded"] is False


def test_padded_knob_comes_from_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path / "a"))
    monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_PADDED", raising=False)
    assert erd.maybe_enable(**enable_kwargs()).record_padded is True  # default 1
    erd.ACTIVE.close()
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path / "b"))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_PADDED", "0")
    assert erd.maybe_enable(**enable_kwargs()).record_padded is False


def test_a_bad_padding_row_is_blanked_counted_and_never_drops_the_step(tmp_path):
    d = make_dumper(tmp_path, steps=2, max_rows=32, record_padded=True)
    batch = fake_batch([5, 5], pad=3)  # padding rows 10, 11, 12
    stage_all_layers(d, batch, 0)
    dup = synth_ids(13, 7)
    dup[10, 4] = dup[10, 0]  # repeated expert id in a padding row
    dup[12, 2] = NEXP + 5  # out-of-range id in another one
    stg(d, "model.layers.1.mlp.experts", dup, synth_w(13, 7))
    d.record_target_step(batch)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_idx"]) == [0]  # the step is kept
    assert (z["target_topk"][0, 10] == -1).all()  # whole row blanked, all layers
    assert (z["target_topk"][0, 12] == -1).all()
    assert (z["target_topk_weight"][0, 10] == -1.0).all()
    assert (z["target_topk"][0, 11] >= 0).all()  # the good padding row stays
    assert z["row_is_padding"][0, 10:].all()  # still flagged as padding
    assert (z["target_topk"][0, :10, 1] >= 0).all()  # real rows untouched
    meta = meta_of(tmp_path)
    assert meta["padded_rows_invalid"] == 2
    assert meta["valid_steps"] == 1 and meta["dropped_steps"] == 0


def test_padding_rows_the_layers_never_wrote_count_as_invalid(tmp_path):
    d = make_dumper(tmp_path, steps=2, max_rows=32, record_padded=True)
    batch = fake_batch([5], pad=3)  # the layers only saw the 5 real rows
    for layer in range(LT):
        stg(
            d,
            f"model.layers.{layer}.mlp.experts",
            synth_ids(5, layer),
            synth_w(5, layer),
        )
    d.record_target_step(batch)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape[1] == 8
    assert (z["target_topk"][0, 5:] == -1).all() and z["row_is_padding"][0, 5:].all()
    assert meta_of(tmp_path)["padded_rows_invalid"] == 3
    assert meta_of(tmp_path)["valid_steps"] == 1


def test_padding_beyond_max_rows_is_cut_but_the_true_count_is_kept(tmp_path):
    d = make_dumper(tmp_path, steps=2, max_rows=12, record_padded=True)
    batch = fake_batch([5, 5], pad=8)  # padded 18 > max_rows 12
    stage_all_layers(d, batch, 0)
    d.record_target_step(batch)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape[1] == 12
    assert list(z["padded_num_tokens"]) == [18] and list(z["num_tokens"]) == [10]
    assert z["row_is_padding"][0, 10:].all()
    assert meta_of(tmp_path)["valid_steps"] == 1


def test_padding_reads_only_the_padded_prefix_per_step(tmp_path):
    """Size of a step: rows * layers * k * 4 bytes (ids + weights), no more."""
    d = make_dumper(tmp_path, steps=2, max_rows=64, record_padded=True)
    batch = fake_batch([5], pad=0)  # exact capture size: nothing to pad
    run_step(d, batch, 0)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape[1] == 5 and not z["row_is_padding"].any()
    assert list(z["padded_num_tokens"]) == [5]


# --------------------------------------------------- phase and request uids
def test_phase_token_prefill_and_active_requests(tmp_path):
    d = make_dumper(tmp_path, steps=4, max_rows=16)
    run_step(d, fake_batch([1, 1]), 0)  # decode
    run_step(d, fake_batch([6], prefilling=[True]), 1)  # prefill chunk
    run_step(d, fake_batch([1, 6], prefilling=[False, True]), 2)  # mixed
    run_step(d, fake_batch([5, 5, 5]), 3)  # MTP verify, not a prefill
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_phase"]) == [0, 1, 2, 0]  # mixed is never 0
    assert list(z["active_requests"]) == [2, 1, 2, 3]
    assert not z["token_is_prefill"][0].any()
    assert z["token_is_prefill"][1, :6].all() and not z["token_is_prefill"][1, 6:].any()
    assert list(z["token_is_prefill"][2, :7]) == [False] + [True] * 6
    assert not z["token_is_prefill"][3].any()
    assert list(z["num_tokens"]) == [2, 6, 7, 15]


def test_request_uid_is_stable_anonymous_and_map_is_opt_in(tmp_path):
    d = make_dumper(tmp_path / "off", steps=3, max_rows=16)
    for i in range(3):
        run_step(d, fake_batch([1, 1], req_ids=["cmpl-secret-A", f"cmpl-new-{i}"]), i)
    d.close()
    z = load(tmp_path / "off", npz_files(tmp_path / "off")[0])
    assert z["request_uid"].dtype == np.int32
    assert list(z["request_uid"][:, 0]) == [0, 0, 0]  # same request, same uid
    assert list(z["request_uid"][:, 1]) == [1, 2, 3]  # first sight, monotonic
    assert not (tmp_path / "off" / "rank0" / "request_map.jsonl").exists()
    for f in os.listdir(tmp_path / "off" / "rank0"):
        assert b"cmpl-secret" not in (tmp_path / "off" / "rank0" / f).read_bytes()

    d = make_dumper(tmp_path / "on", steps=3, max_rows=16, dump_ids=True)
    for i in range(3):
        run_step(d, fake_batch([1, 1], req_ids=["cmpl-secret-A", f"cmpl-new-{i}"]), i)
    d.close()
    lines = (tmp_path / "on" / "rank0" / "request_map.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"uid": 0, "request_id": "cmpl-secret-A"},
        {"uid": 1, "request_id": "cmpl-new-0"},
        {"uid": 2, "request_id": "cmpl-new-1"},
        {"uid": 3, "request_id": "cmpl-new-2"},
    ]


# ------------------------------------------------------------ (c) dropped
def test_a_step_with_more_rows_than_max_rows_is_dropped_whole(tmp_path):
    d = make_dumper(tmp_path, steps=4, max_rows=6)
    run_step(d, fake_batch([1, 1]), 0)
    run_step(d, fake_batch([9], prefilling=[True]), 1)  # 9 > 6: dropped as a step
    run_step(d, fake_batch([5, 1]), 2)  # exactly 6: kept
    run_step(d, fake_batch([1, 1]), 3)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_idx"]) == [0, 2, 3]  # the hole is visible
    assert z["target_topk"].shape[1] == 6
    assert (z["target_topk"][:, :, :, :] >= -1).all()
    (entry,) = index_lines(tmp_path)
    assert entry["valid_steps"] == 3 and entry["dropped_steps"] == 1
    assert entry["dropped_rows"] == 9
    meta = meta_of(tmp_path)
    assert (meta["valid_steps"], meta["dropped_steps"], meta["dropped_rows"]) == (
        3,
        1,
        9,
    )


def test_a_step_missing_a_target_layer_is_dropped_never_written_partial(tmp_path):
    d = make_dumper(tmp_path, steps=3)
    run_step(d, fake_batch([2]), 0)
    run_step(d, fake_batch([3]), 1, skip=(1,))  # layer 1 never staged
    run_step(d, fake_batch([2]), 2)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_idx"]) == [0, 2]
    (entry,) = index_lines(tmp_path)
    assert entry["valid_steps"] == 2 and entry["dropped_steps"] == 1
    assert entry["dropped_rows"] == 3


def test_stale_staging_from_the_previous_step_never_leaks(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    run_step(d, fake_batch([2]), 0)
    # step 1: layer 2 does not run; its buffer still holds step 0's ids
    # unless the recorder cleared it after the snapshot
    run_step(d, fake_batch([2]), 1, skip=(2,))
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_idx"]) == [0]
    assert meta_of(tmp_path)["dropped_steps"] == 1


def test_warmup_residue_never_lands_in_a_file(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    # eager warmup / dummy forwards write the buffers but are never recorded
    stage_all_layers(d, fake_batch([4]), 77)
    stg(d, f"mtp.layers.{LT}.mlp.experts", synth_ids(7, 78), synth_w(7, 78))
    batch = fake_batch([2])
    expected, _ = run_step(d, batch, 1)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    for layer in range(LT):
        np.testing.assert_array_equal(
            z["target_topk"][0, :2, layer],
            expected[layer][0][:2].numpy().astype(np.int16),
        )


def test_repeated_ids_in_a_row_fail_validation(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    batch = fake_batch([2])
    stage_all_layers(d, batch, 0)
    bad = synth_ids(5, 3)
    bad[1, 4] = bad[1, 0]  # a row with a repeated expert id
    stg(d, "model.layers.1.mlp.experts", bad, synth_w(5, 3))
    d.record_target_step(batch)
    run_step(d, fake_batch([2]), 1)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert list(z["step_idx"]) == [1]
    assert meta_of(tmp_path)["dropped_steps"] == 1


def test_out_of_range_ids_fail_validation(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    batch = fake_batch([2])
    stage_all_layers(d, batch, 0)
    bad = synth_ids(5, 3)
    bad[0, 0] = NEXP + 1
    stg(d, "model.layers.0.mlp.experts", bad, synth_w(5, 3))
    d.record_target_step(batch)
    d.close()
    assert npz_files(tmp_path) == []
    assert meta_of(tmp_path)["dropped_steps"] == 1


def test_ring_overrun_drops_steps_and_never_stalls_serving(tmp_path, monkeypatch):
    d = make_dumper(tmp_path, steps=2)
    gate = threading.Event()
    real_write = d._write_file

    def slow_write(ring):
        assert gate.wait(30)
        real_write(ring)

    monkeypatch.setattr(d, "_write_file", slow_write)
    done = threading.Event()

    def producer():
        # ring A (steps 0,1) -> writer; ring B (2,3) -> writer queue; step 4
        # finds no free ring and is dropped, as are 5 and 6
        for i in range(7):
            run_step(d, fake_batch([1, 1]), i)
        done.set()

    t = threading.Thread(target=producer)
    t.start()
    assert done.wait(20), "the step loop blocked on a slow writer"
    gate.set()
    t.join(5)
    deadline = time.time() + 10
    while time.time() < deadline and d._free_rings.qsize() == 0:
        time.sleep(0.05)
    run_step(d, fake_batch([1, 1]), 7)  # the writer caught up: recording resumes
    d.close()
    steps = []
    for f in npz_files(tmp_path):
        steps += list(load(tmp_path, f)["step_idx"])
    assert steps == [0, 1, 2, 3, 7]  # 4, 5, 6 are gaps
    meta = meta_of(tmp_path)
    assert meta["valid_steps"] == 5 and meta["dropped_steps"] == 3
    assert meta["dropped_rows"] == 6
    assert sum(e["dropped_steps"] for e in index_lines(tmp_path)) == 3


def test_a_failing_write_is_counted_and_the_next_file_lands(tmp_path, monkeypatch):
    d = make_dumper(tmp_path, steps=1)
    real_write = d._write_file
    calls = []

    def flaky(ring):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        real_write(ring)

    monkeypatch.setattr(d, "_write_file", flaky)
    for i in range(3):
        run_step(d, fake_batch([1]), i)
        time.sleep(0.05)  # let the writer free its ring
    d.close()
    meta = meta_of(tmp_path)
    assert meta["write_errors"] == 1 and meta["files_written"] >= 1


def test_idle_flush_writes_the_partial_file_without_close(tmp_path, monkeypatch):
    monkeypatch.setattr(erd, "_IDLE_FLUSH_S", 0.05)
    d = make_dumper(tmp_path, steps=50)
    run_step(d, fake_batch([5]), 0)
    deadline = time.time() + 10
    while time.time() < deadline and not (tmp_path / "rank0" / "index.jsonl").exists():
        time.sleep(0.1)
    assert (tmp_path / "rank0" / "index.jsonl").exists()
    assert index_lines(tmp_path)[0]["valid_steps"] == 1
    d.close()


def test_ring_budget_shrinks_file_length_for_huge_row_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(erd, "_RING_BYTES", 1 << 20)
    d = make_dumper(tmp_path, steps=256, max_rows=64)
    assert d.steps_per_file < 256
    run_step(d, fake_batch([5]), 0)
    d.close()
    meta = meta_of(tmp_path)
    assert meta["steps_per_file"] == d.steps_per_file
    assert meta["steps_per_file_requested"] == 256


# --------------------------------------------------------- nospec (no draft)
def test_nospec_has_m1_rows_and_an_empty_draft_axis(tmp_path):
    d = make_dumper(tmp_path, steps=4, draft_layers=0, draft_steps=0, max_rows=128)
    for i in range(4):
        run_step(d, fake_batch([1] * 4, req_ids=[f"q{j}" for j in range(4)]), i)
    d.close()
    z = load(tmp_path, npz_files(tmp_path)[0])
    assert z["target_topk"].shape == (4, 4, LT, K)  # M = 1 row per request
    assert list(z["num_tokens"]) == [4] * 4 and list(z["active_requests"]) == [4] * 4
    assert z["draft_topk"].shape == (4, 0, 4, 0, K)
    assert z["draft_topk"].dtype == np.int16
    assert z["draft_num_rows"].shape == (4, 0)
    meta = meta_of(tmp_path)
    assert meta["draft_layers"] == 0 and meta["num_draft_steps"] == 0
    assert meta["num_speculative_tokens"] == 0 and meta["layers"]["draft"] == {}
    d2 = make_dumper(tmp_path / "x", draft_layers=0, draft_steps=0)
    d2.record_draft_step(0, 1)  # harmless without draft layers
    d2.close()


# -------------------------------------------------------------- run rotation
def test_rotate_json_starts_a_new_run_and_keeps_the_old_one_whole(tmp_path):
    base = tmp_path / "mtp4-c1"
    d = make_dumper(base, steps=50, max_rows=32, concurrency=1)
    for i in range(3):
        run_step(d, fake_batch([5]), i)
    (base / "rotate.json").write_text(
        json.dumps(
            {"seq": 1, "subdir": "mtp4-c4", "concurrency": 4, "cell_label": "c4"}
        )
    )
    for i in range(2):
        run_step(d, fake_batch([5, 5, 5, 5]), 10 + i)
    d.close()
    old, new = tmp_path / "mtp4-c1", tmp_path / "mtp4-c4"
    assert index_lines(old)[0]["valid_steps"] == 3
    assert meta_of(old)["concurrency"] == 1 and meta_of(old)["valid_steps"] == 3
    z = load(new, npz_files(new)[0])
    assert list(z["step_idx"]) == [0, 1]  # the new run restarts at step 0
    assert list(z["active_requests"]) == [4, 4]
    mnew = meta_of(new)
    assert mnew["run_id"] == "mtp4-c4" and mnew["concurrency"] == 4
    assert mnew["cell_label"] == "c4" and mnew["valid_steps"] == 2
    # an old or malformed request is not applied twice
    d2 = make_dumper(tmp_path / "again", steps=5)
    assert d2._run.run_id == "again"
    d2.close()


def test_rotate_json_with_a_path_in_subdir_is_refused(tmp_path):
    base = tmp_path / "a"
    d = make_dumper(base)
    (base / "rotate.json").write_text(json.dumps({"seq": 1, "subdir": "../evil"}))
    run_step(d, fake_batch([2]), 0)
    d.close()
    assert not (tmp_path / ".." / "evil").exists()
    assert d._run.run_id == "a"


# ------------------------------------------------------------------ security
SECRET_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)")


def _walk_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj


def test_meta_never_carries_env_argv_or_secret_looking_values(tmp_path, monkeypatch):
    import sys

    monkeypatch.setenv("VLLM_API_KEY", "sk-should-never-appear")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok-should-never-appear")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "--api-key", "argv-secret"])
    d = make_dumper(tmp_path, cell_label="mtp4-c1", concurrency=1, dump_ids=True)
    run_step(d, fake_batch([5]), 0)
    d.close()
    meta = meta_of(tmp_path)
    assert "env" not in meta and "argv" not in meta
    assert set(meta) == set(erd.META_KEYS)
    for value in _walk_values(meta):
        assert not SECRET_RE.search(str(value)), value
    raw = (tmp_path / "rank0" / "meta.json").read_text()
    for needle in ("should-never-appear", "argv-secret", "CUDA_VISIBLE_DEVICES"):
        assert needle not in raw
    for name in os.listdir(tmp_path / "rank0"):
        assert b"should-never-appear" not in (tmp_path / "rank0" / name).read_bytes()


def test_files_are_private(tmp_path):
    old_umask = os.umask(0o022)
    try:
        d = make_dumper(tmp_path / "x" / "run", dump_ids=True)
        run_step(d, fake_batch([5]), 0)
        d.close()
    finally:
        os.umask(old_umask)
    run = tmp_path / "x" / "run"
    for directory in (run, run / "rank0"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    for name in os.listdir(run / "rank0"):
        mode = stat.S_IMODE((run / "rank0" / name).stat().st_mode)
        assert mode == 0o600, (name, oct(mode))
    assert {"meta.json", "index.jsonl", "request_map.jsonl"} <= set(
        os.listdir(run / "rank0")
    )


# ----------------------------------------------------- (d) graph safety
def test_stage_is_graph_safe_by_construction():
    src = inspect.getsource(erd.ExpertRoutingDumper.stage)
    body = src.split('"""', 2)[2]  # drop the docstring
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        ".numpy(",
        "synchronize",
        "torch.cuda",
        "torch.empty",
        "torch.zeros",
        "torch.tensor",
        ".to(",
        ".clone(",
        ".contiguous(",
        "non_blocking",
    ):
        assert forbidden not in body, forbidden
    assert body.count(".copy_(") == 3  # target ids, target weights, draft ids
    assert "layer_name" not in body and "re." not in body  # slot is pre-resolved


def test_stage_emits_only_device_copies_into_static_storage(tmp_path):
    d = make_dumper(tmp_path)
    bufs = [d._t_ids, d._t_w, d._d_ids]
    ptrs = [b.data_ptr() for b in bufs]
    ids, w = synth_ids(4, 3), synth_w(4, 3)
    stg(d, "model.layers.0.mlp.experts", ids, w)  # registers the layer
    with _OpLog() as log:
        stg(d, "model.layers.0.mlp.experts", ids, w)
        stg(d, f"mtp.layers.{LT}.mlp.experts", ids, w)
    # views and shape reads are metadata only; copy_ is the one kernel launched
    allowed = {"copy_", "__getitem__", "dim", "__get__"}
    assert set(log.ops) <= allowed, log.ops
    assert log.ops.count("copy_") == 3
    assert ptrs == [b.data_ptr() for b in bufs]  # no reallocation


def test_hooks_only_enqueue_async_copies_and_one_event_sync_per_file():
    for fn in (
        erd.ExpertRoutingDumper.record_target_step,
        erd.ExpertRoutingDumper.record_draft_step,
    ):
        body = inspect.getsource(fn).split('"""', 2)[2]
        for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize", ".numpy("):
            assert forbidden not in body, (fn.__name__, forbidden)
        assert "non_blocking=True" in body
    hand_over = inspect.getsource(erd.ExpertRoutingDumper._hand_over)
    assert hand_over.count("synchronize()") == 1
    writer = inspect.getsource(erd.ExpertRoutingDumper._write_file)
    assert "synchronize" not in writer and "torch.cuda" not in writer


def test_stage_has_no_row_dependent_python_on_device_data(tmp_path):
    """Rows staged = min(static shape, max_rows): a replay needs no python."""
    d = make_dumper(tmp_path, max_rows=4)
    ids = synth_ids(9, 5)
    stg(d, "model.layers.2.mlp.experts", ids, synth_w(9, 5))
    assert d._t_ids[:, 2].tolist() == ids[:4].to(torch.int16).tolist()
    stg(d, "model.layers.2.mlp.experts", ids[:2], synth_w(2, 5))  # smaller M
    assert d._t_ids[:2, 2].tolist() == ids[:2].to(torch.int16).tolist()
    assert d._t_ids[2:, 2].tolist() == ids[2:4].to(torch.int16).tolist()  # stale rows


def test_a_stage_that_does_not_fit_disables_the_dump_with_one_warning(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    d = make_dumper(tmp_path)
    bad = torch.zeros(2, K + 1, dtype=torch.int32)
    stg(d, "model.layers.0.mlp.experts", bad, torch.zeros(2, K + 1))
    stg(d, "model.layers.1.mlp.experts", bad, torch.zeros(2, K + 1))
    assert caplog.text.count("dump disabled") == 1
    d.record_target_step(fake_batch([2]))
    assert d._filled == 0


# ------------------------------------------------- call sites in the workers
def test_speculator_call_site_records_only_real_rounds_with_the_dump_on():
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as spec

    calls = []
    stub = types.SimpleNamespace(_draft_hidden_dump=None, _dump_this_round=True)
    spec.EagleSpeculator._dump_draft_step(stub, 1, 4)  # dump off: nothing
    erd.ACTIVE = types.SimpleNamespace(
        record_draft_step=lambda step, rows: calls.append((step, rows))
    )
    spec.EagleSpeculator._dump_draft_step(stub, 2, 3)
    stub._dump_this_round = False  # dummy / profile round
    spec.EagleSpeculator._dump_draft_step(stub, 3, 3)
    erd.ACTIVE = None
    assert calls == [(2, 3)]


def _runner_stub(spec_config, steps):
    text = types.SimpleNamespace(
        num_hidden_layers=48,
        mtp_num_hidden_layers=1,
        num_experts_per_tok=10,
        num_experts=512,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=text, model="/m/flash"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        speculative_config=spec_config,
        num_speculative_steps=steps,
        max_num_reqs=4,
        device=torch.device("cpu"),
    )


def test_model_runner_builds_the_dumper_from_the_model_config(monkeypatch, tmp_path):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    d = GPUModelRunner._maybe_enable_expert_routing_dump(_runner_stub(object(), 4))
    assert d is erd.ACTIVE
    assert (d.num_target_layers, d.num_draft_layers, d.num_draft_steps) == (48, 1, 4)
    assert (d.top_k, d.num_experts, d.max_rows) == (10, 512, 128)
    assert d._extra["model"] == "/m/flash" and d._extra["tp"] == 4
    assert d._extra["num_speculative_tokens"] == 4
    erd.ACTIVE.close()
    # no speculative config: no draft layers are staged
    d2 = GPUModelRunner._maybe_enable_expert_routing_dump(_runner_stub(None, 0))
    assert d2.num_draft_layers == 0 and d2.num_draft_steps == 0
    assert d2._extra["num_speculative_tokens"] == 0
    d2.record_draft_step(0, 1)  # harmless without draft layers


def test_model_runner_does_nothing_without_the_env(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", raising=False)
    stub = types.SimpleNamespace()  # no attribute may be touched
    assert GPUModelRunner._maybe_enable_expert_routing_dump(stub) is None


def test_kv_info_call_site_is_guarded():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    mr = open(os.path.join(root, "vllm/v1/worker/gpu/model_runner.py")).read()
    assert re.search(
        r"if expert_routing_dump\.ACTIVE is not None:\n\s+"
        r"expert_routing_dump\.ACTIVE\.set_kv_info\(",
        mr,
    )
