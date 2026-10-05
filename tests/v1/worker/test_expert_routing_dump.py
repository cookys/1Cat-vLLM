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
import types

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from vllm.model_executor.layers.fused_moe import expert_routing_dump as erd

LT, LD, DSTEPS, K, NEXP = 3, 1, 2, 10, 512


@pytest.fixture(autouse=True)
def _reset_active():
    erd.ACTIVE = None
    yield
    if erd.ACTIVE is not None:
        erd.ACTIVE.close()
    erd.ACTIVE = None


def make_dumper(tmp_path, *, steps=3, max_rows=4, weights=True, rank=0):
    return erd.ExpertRoutingDumper(
        str(tmp_path),
        rank=rank,
        num_target_layers=LT,
        num_draft_layers=LD,
        num_draft_steps=DSTEPS,
        top_k=K,
        num_experts=NEXP,
        steps_per_file=steps,
        max_rows=max_rows,
        with_weights=weights,
        device="cpu",
        meta_extra={"model": "unit-test", "tp_size": 1},
    )


def stg(d, layer_name, ids, w):
    """What the bound router callback does for that layer."""
    slot = d.slot_for(layer_name)
    if slot is not None:
        d.stage(slot, ids, w)


def synth_ids(m, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, NEXP, (m, K), generator=g, dtype=torch.int32)


def synth_w(m, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.rand(m, K, generator=g, dtype=torch.float32)


def fake_batch(tokens_per_req, *, req_ids=None, prefilling=None, start_pos=100):
    tokens_per_req = list(tokens_per_req)
    n = sum(tokens_per_req)
    req_ids = req_ids or [f"r{i}" for i in range(len(tokens_per_req))]
    positions = torch.arange(start_pos, start_pos + n, dtype=torch.int64)
    # pad like a FULL-graph batch: the buffer is longer than the real tokens
    positions = torch.cat([positions, torch.full((3,), 9999, dtype=torch.int64)])
    return types.SimpleNamespace(
        num_tokens=n,
        req_ids=req_ids,
        num_scheduled_tokens=np.array(tokens_per_req, dtype=np.int32),
        positions=positions,
        is_prefilling_np=np.array(
            prefilling if prefilling is not None else [False] * len(tokens_per_req)
        ),
        num_draft_tokens=0,
    )


def run_step(d, batch, seed, *, draft_rows=(None, None)):
    """One real step: target layers staged, target hook, draft layers + hooks."""
    m = batch.num_tokens + 3  # the layer sees the padded batch
    expected = {}
    for layer in range(LT):
        ids, w = synth_ids(m, seed * 100 + layer), synth_w(m, seed * 100 + layer)
        stg(d, f"model.layers.{layer}.mlp.experts", ids, w)
        expected[layer] = (ids, w)
    d.record_target_step(batch)
    draft_expected = []
    for step in range(DSTEPS):
        rows = batch.num_tokens + 3 if step == 0 else len(batch.req_ids)
        ids = synth_ids(rows, seed * 100 + 50 + step)
        stg(d, f"mtp.layers.{LT}.mlp.experts", ids, synth_w(rows, 0))
        d.record_draft_step(step, len(batch.req_ids))
        draft_expected.append(ids)
    return expected, draft_expected


def load(tmp_path, name):
    with np.load(os.path.join(tmp_path, "rank0", name)) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------- (a) off
def test_env_unset_creates_no_dumper(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", raising=False)
    out = erd.maybe_enable(
        num_target_layers=LT,
        num_draft_layers=LD,
        num_draft_steps=DSTEPS,
        top_k=K,
        num_experts=NEXP,
        device="cpu",
    )
    assert out is None and erd.ACTIVE is None


@pytest.mark.parametrize(
    "var,value",
    [
        ("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "0"),
        ("VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS", "0"),
    ],
)
def test_nonpositive_knobs_disable(monkeypatch, tmp_path, var, value):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path / "x"))
    monkeypatch.setenv(var, value)
    assert (
        erd.maybe_enable(
            num_target_layers=LT,
            num_draft_layers=LD,
            num_draft_steps=DSTEPS,
            top_k=K,
            num_experts=NEXP,
            device="cpu",
        )
        is None
    )
    assert not (tmp_path / "x").exists()


def test_env_set_builds_the_dumper_and_logs_once(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "7")
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_MAX_ROWS", "9")
    import logging

    caplog.set_level(logging.INFO)
    d = erd.maybe_enable(
        num_target_layers=48,
        num_draft_layers=1,
        num_draft_steps=4,
        top_k=K,
        num_experts=NEXP,
        device="cpu",
    )
    assert d is erd.ACTIVE and d.steps_per_file == 7 and d.max_rows == 9
    assert (tmp_path / "rank0").is_dir()
    assert "SM70 expert routing dump enabled" in caplog.text
    assert f"dir={tmp_path} steps=7 max_rows=9 layers=48+4" in caplog.text


def test_rank_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_RANKS", "1,2")
    monkeypatch.setattr(erd, "_tp_rank", lambda: 0)
    kwargs = dict(
        num_target_layers=LT,
        num_draft_layers=LD,
        num_draft_steps=DSTEPS,
        top_k=K,
        num_experts=NEXP,
        device="cpu",
    )
    assert erd.maybe_enable(**kwargs) is None and erd.ACTIVE is None
    monkeypatch.setattr(erd, "_tp_rank", lambda: 2)
    d = erd.maybe_enable(**kwargs)
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
    assert d._t_ids[1, :5].tolist() == ids.to(torch.int16).tolist()
    router.set_dump_fn(None)
    with _OpLog() as again:
        router.select_experts(x, x)
    assert "copy_" not in again.ops


def test_callback_sees_the_final_ids_after_dtype_conversion(tmp_path):
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
        assert d._t_ids[i, :4].tolist() == synth_ids(4, i).to(torch.int16).tolist()
    assert d._d_ids[0, :4].tolist() == synth_ids(4, 90).to(torch.int16).tolist()
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
        text = open(path).read()
        assert "expert_routing_dump" not in text


def test_dump_env_vars_are_not_compile_cache_factors(monkeypatch):
    from vllm import envs

    names = [
        n
        for n in envs.environment_variables
        if n.startswith("VLLM_SM70_EXPERT_ROUTING_DUMP_")
    ]
    assert len(names) == 5
    base = envs.compile_factors()
    for n in names:
        assert n not in base
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", "/tmp/somewhere")
    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_STEPS", "3")
    assert envs.compile_factors() == base


# ---------------------------------------------------------- (b) round trip
def test_round_trip_multiple_files_index_and_meta(tmp_path):
    d = make_dumper(tmp_path, steps=3, max_rows=6)
    shapes = [[1, 1], [5], [2, 3], [1], [4, 1], [1, 1, 1], [2]]  # 7 steps = 3+3+1
    truth = []
    for i, tpr in enumerate(shapes):
        truth.append((fake_batch(tpr), *run_step(d, fake_batch(tpr), i)))
    d.close()

    index = [
        json.loads(line)
        for line in open(tmp_path / "rank0" / "index.jsonl").read().splitlines()
    ]
    assert [(e["first_step"], e["last_step"], e["steps"]) for e in index] == [
        (0, 2, 3),
        (3, 5, 3),
        (6, 6, 1),
    ]
    for e in index:
        path = tmp_path / "rank0" / e["file"]
        assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert e["bytes"] == path.stat().st_size
    assert [e["total_tokens"] for e in index] == [
        sum(map(sum, shapes[0:3])),
        sum(map(sum, shapes[3:6])),
        sum(shapes[6]),
    ]
    assert not [p for p in os.listdir(tmp_path / "rank0") if p.endswith(".tmp.npz")]

    step = 0
    for e in index:
        z = load(tmp_path, e["file"])
        s_count = e["steps"]
        m_file = z["target_topk"].shape[1]
        assert z["target_topk"].dtype == np.int16
        assert z["target_topk"].shape == (s_count, m_file, LT, K)
        assert z["target_topk_weight"].dtype == np.float16
        assert z["draft_topk"].shape == (s_count, m_file, DSTEPS * LD, K)
        assert z["position"].dtype == np.int32
        assert z["position"].shape == (s_count, m_file)
        assert z["step_idx"].dtype == np.int32 and z["num_tokens"].dtype == np.int32
        assert z["step_wall_ns"].dtype == np.int64 and z["is_prefill"].dtype == bool
        assert z["shared_expert_used"].shape == (s_count, m_file, LT)
        assert m_file == max(sum(shapes[step + j]) for j in range(s_count))
        for j in range(s_count):
            batch, expected, draft_expected = truth[step + j]
            n = batch.num_tokens
            assert z["step_idx"][j] == step + j and z["num_tokens"][j] == n
            for layer in range(LT):
                ids, w = expected[layer]
                np.testing.assert_array_equal(
                    z["target_topk"][j, :n, layer], ids[:n].numpy().astype(np.int16)
                )
                np.testing.assert_array_equal(
                    z["target_topk_weight"][j, :n, layer],
                    w[:n].numpy().astype(np.float16),
                )
            # padding: -1 in rows past the valid ones
            assert (z["target_topk"][j, n:] == -1).all()
            assert (z["target_topk_weight"][j, n:] == -1).all()
            assert z["shared_expert_used"][j, :n].all()
            assert not z["shared_expert_used"][j, n:].any()
            # positions come from the batch, padding -1
            np.testing.assert_array_equal(
                z["position"][j, :n], np.arange(100, 100 + n, dtype=np.int32)
            )
            assert (z["position"][j, n:] == -1).all()
            # request ids: each request owns its consecutive rows, pad ""
            expect_ids = [
                f"r{r}"
                for r, c in enumerate(batch.num_scheduled_tokens)
                for _ in range(c)
            ]
            assert list(z["request_ids"][j, :n]) == expect_ids
            assert all(v == "" for v in z["request_ids"][j, n:])
            # draft: step 0 on all token rows, later steps one row per request
            assert list(z["draft_num_rows"][j]) == [n, len(batch.req_ids)]
            np.testing.assert_array_equal(
                z["draft_topk"][j, :n, 0],
                draft_expected[0][:n].numpy().astype(np.int16),
            )
            nreq = len(batch.req_ids)
            np.testing.assert_array_equal(
                z["draft_topk"][j, :nreq, 1],
                draft_expected[1][:nreq].numpy().astype(np.int16),
            )
            assert (z["draft_topk"][j, nreq:, 1] == -1).all()
        step += s_count

    meta = json.load(open(tmp_path / "rank0" / "meta.json"))
    assert meta["model"] == "unit-test" and meta["tp_size"] == 1
    assert meta["num_experts"] == NEXP and meta["top_k"] == K
    assert meta["steps_per_file"] == 3 and meta["max_rows"] == 6
    assert meta["files_written"] == 3 and meta["steps_recorded"] == 7
    assert meta["layers"]["target"] == {
        str(i): f"model.layers.{i}.mlp.experts" for i in range(LT)
    }
    assert meta["layers"]["draft"] == {"0": f"mtp.layers.{LT}.mlp.experts"}
    assert isinstance(meta["argv"], list) and isinstance(meta["env"], dict)
    assert "git_tip" in meta and "shared_expert" in meta and "row_layout" in meta


def test_weights_knob_off_skips_the_weight_array_and_buffers(tmp_path):
    d = make_dumper(tmp_path, steps=2, weights=False)
    assert d._t_w is None and d._ring.t_w is None
    run_step(d, fake_batch([2]), 1)
    d.close()
    z = load(tmp_path, "routing_0_0.npz")
    assert "target_topk_weight" not in z and "target_topk" in z


def test_a_draft_step_that_did_not_run_is_all_padding(tmp_path):
    d = make_dumper(tmp_path, steps=2)
    batch = fake_batch([1, 1])
    for layer in range(LT):
        stg(
            d,
            f"model.layers.{layer}.mlp.experts",
            synth_ids(5, layer),
            synth_w(5, layer),
        )
    d.record_target_step(batch)
    stg(d, f"mtp.layers.{LT}.mlp.experts", synth_ids(5, 9), synth_w(5, 9))
    d.record_draft_step(0, 2)  # draft step 1 never ran (e.g. a prefill-only round)
    d.close()
    z = load(tmp_path, "routing_0_0.npz")
    assert list(z["draft_num_rows"][0]) == [2, 0]
    assert (z["draft_topk"][0, :, 1] == -1).all()
    assert (z["draft_topk"][0, :2, 0] >= 0).all()


def test_draft_layer_names_never_collide_with_target_layers(tmp_path):
    d = make_dumper(tmp_path)
    assert d.slot_for(f"mtp.layers.{LT}.mlp.experts") == ("d", 0)  # continued numbering
    assert d.slot_for("mtp.layers.0.mlp.experts") == ("d", 0)  # draft-local numbering
    assert d.slot_for("mtp.layers.1.mlp.experts") is None  # only one draft layer
    assert d.slot_for("model.layers.2.mlp.experts") == ("t", 2)
    assert d.slot_for(f"model.layers.{LT}.mlp.experts") == ("d", 0)  # unnamed draft
    assert d.slot_for(f"model.layers.{LT + 1}.mlp.experts") is None


def test_draft_hook_without_an_open_step_and_dummy_layers_are_ignored(tmp_path):
    d = make_dumper(tmp_path)
    d.record_draft_step(0, 2)  # before any target step
    stg(d, "model.layers.99.mlp.experts", synth_ids(2, 1), synth_w(2, 1))
    stg(d, "model.embed_tokens", synth_ids(2, 1), synth_w(2, 1))
    assert d._filled == 0
    assert (d._t_ids == -1).all()


def test_nothing_is_recorded_during_graph_capture(tmp_path, monkeypatch):
    d = make_dumper(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    d.record_target_step(fake_batch([2]))
    d.record_draft_step(0, 1)
    assert d._filled == 0 and d.step_counter == 0


def test_close_is_idempotent_and_stops_recording(tmp_path):
    d = make_dumper(tmp_path)
    erd.ACTIVE = d
    run_step(d, fake_batch([1]), 0)
    d.close()
    d.close()
    assert erd.ACTIVE is None
    run_step(d, fake_batch([1]), 1)  # ignored after close
    assert len(os.listdir(tmp_path / "rank0")) == 3  # npz + index + meta
    index = (tmp_path / "rank0" / "index.jsonl").read_text().splitlines()
    assert len(index) == 1


# -------------------------------------------------------------- writer thread
def test_the_step_loop_does_not_wait_for_a_slow_writer(tmp_path, monkeypatch):
    import threading

    d = make_dumper(tmp_path, steps=2)
    gate = threading.Event()
    real_write = d._write_file
    started = threading.Event()

    def slow_write(ring):
        started.set()
        assert gate.wait(30)
        real_write(ring)

    monkeypatch.setattr(d, "_write_file", slow_write)
    done = threading.Event()

    def producer():
        # file 1 (steps 0-1) is handed over at step 2; step 3 fills file 2
        for i in range(4):
            run_step(d, fake_batch([1, 1]), i)
        done.set()

    t = threading.Thread(target=producer)
    t.start()
    assert done.wait(20), "the producer blocked on the writer with a free ring"
    assert started.wait(5) and d.files_written == 0
    gate.set()
    t.join(5)
    d.close()
    assert d.files_written == 2 and d.write_errors == 0
    first = load(tmp_path, "routing_0_1.npz")
    second = load(tmp_path, "routing_2_3.npz")
    # each file kept its own steps: the ring that was reused did not leak
    assert list(first["step_idx"]) == [0, 1] and list(second["step_idx"]) == [2, 3]
    np.testing.assert_array_equal(
        second["target_topk"][0, :2, 0],
        synth_ids(2 + 3, 2 * 100).to(torch.int16).numpy()[:2],
    )


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
    d.close()
    assert d.write_errors == 1 and d.files_written == 2
    assert sorted(os.listdir(tmp_path / "rank0"))[:2] == ["index.jsonl", "meta.json"]
    assert json.load(open(tmp_path / "rank0" / "meta.json"))["write_errors"] == 1


# ------------------------------------------------------------- (c) overflow
def test_rows_beyond_max_rows_are_dropped_counted_and_warned_once(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    d = make_dumper(tmp_path, steps=2, max_rows=4)
    # step 0: a 6-row prefill chunk; step 1: 2 requests x 5 rows = 10 rows
    for i, tpr in enumerate([[6], [5, 5]]):
        run_step(d, fake_batch(tpr), i)
    d.close()
    z = load(tmp_path, "routing_0_1.npz")
    assert z["target_topk"].shape[1] == 4  # capped at max_rows
    assert list(z["num_tokens"]) == [6, 10]  # the real counts survive
    # target drops 2 and 6; the draft prefill step sees the same rows (2 and 6)
    assert list(z["rows_dropped"]) == [2 + 2, 6 + 6]
    assert list(z["draft_num_rows"][0]) == [4, 1]
    assert list(z["draft_num_rows"][1]) == [4, 2]
    assert d.rows_dropped_total == 16
    meta = json.load(open(tmp_path / "rank0" / "meta.json"))
    assert meta["rows_dropped_total"] == 16
    index = json.loads((tmp_path / "rank0" / "index.jsonl").read_text().splitlines()[0])
    assert index["rows_dropped"] == 16
    assert caplog.text.count("rows beyond max_rows=4 are dropped") == 1


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
    flush = inspect.getsource(erd.ExpertRoutingDumper._hand_over)
    assert flush.count("synchronize()") == 1
    writer = inspect.getsource(erd.ExpertRoutingDumper._write_file)
    assert "synchronize" not in writer and "torch.cuda" not in writer


def test_stage_has_no_row_dependent_python_on_device_data(tmp_path):
    """Rows staged = min(static shape, max_rows): a replay needs no python."""
    d = make_dumper(tmp_path, max_rows=4)
    ids = synth_ids(9, 5)
    stg(d, "model.layers.2.mlp.experts", ids, synth_w(9, 5))
    assert d._t_ids[2].tolist() == ids[:4].to(torch.int16).tolist()
    stg(d, "model.layers.2.mlp.experts", ids[:2], synth_w(2, 5))  # smaller M
    assert d._t_ids[2, :2].tolist() == ids[:2].to(torch.int16).tolist()
    assert d._t_ids[2, 2:].tolist() == ids[2:4].to(torch.int16).tolist()  # stale rows


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


# ------------------------------------------------------- (e) prefill flag
def test_prefill_flag_per_step_and_per_row(tmp_path):
    d = make_dumper(tmp_path, steps=3, max_rows=8)
    # decode, prefill chunk (mixed with a decode request), decode again
    run_step(d, fake_batch([1, 1]), 0)
    run_step(d, fake_batch([1, 6], prefilling=[False, True]), 1)
    run_step(d, fake_batch([5], prefilling=[False]), 2)  # MTP verify, not a prefill
    d.close()
    z = load(tmp_path, "routing_0_2.npz")
    assert list(z["is_prefill"]) == [False, True, False]
    assert not z["row_is_prefill"][0].any()
    assert list(z["row_is_prefill"][1, :7]) == [False] + [True] * 6
    assert not z["row_is_prefill"][2].any()
    assert list(z["num_reqs"]) == [2, 2, 1]


# ------------------------------------------------- call sites in the workers
def test_speculator_call_site_records_only_real_rounds_with_the_dump_on(tmp_path):
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


def test_model_runner_builds_the_dumper_from_the_model_config(monkeypatch, tmp_path):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.setenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", str(tmp_path))
    text = types.SimpleNamespace(
        num_hidden_layers=48,
        mtp_num_hidden_layers=1,
        num_experts_per_tok=10,
        num_experts=512,
    )
    stub = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=text, model="/m/flash"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        speculative_config=object(),
        num_speculative_steps=4,
        device=torch.device("cpu"),
    )
    d = GPUModelRunner._maybe_enable_expert_routing_dump(stub)
    assert d is erd.ACTIVE
    assert (d.num_target_layers, d.num_draft_layers, d.num_draft_steps) == (48, 1, 4)
    assert (d.top_k, d.num_experts) == (10, 512)
    assert d._meta_static["model"] == "/m/flash" and d._meta_static["tp_size"] == 4
    erd.ACTIVE.close()
    # no speculative config: no draft layers are staged
    stub.speculative_config = None
    d2 = GPUModelRunner._maybe_enable_expert_routing_dump(stub)
    assert d2.num_draft_layers == 0 and d2.num_draft_steps == 0
    d2.record_draft_step(0, 1)  # harmless without draft layers


def test_model_runner_does_nothing_without_the_env(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.delenv("VLLM_SM70_EXPERT_ROUTING_DUMP_DIR", raising=False)
    stub = types.SimpleNamespace()  # no attribute may be touched
    assert GPUModelRunner._maybe_enable_expert_routing_dump(stub) is None
