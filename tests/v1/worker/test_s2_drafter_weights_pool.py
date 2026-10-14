# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-1): ``load_model`` opens "weights" then "draft_weights",
never nested, each tag once; CPU-only (pool contexts are recording fakes)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _load_model_env(monkeypatch, order, *, split):
    from vllm.v1.worker.gpu import model_runner as mr

    @contextmanager
    def pool(tag):
        order.append(("enter", tag))
        try:
            yield
        finally:
            order.append(("exit", tag))

    class _Loader:
        def load_model(self, vllm_config, model_config):
            order.append(("load_target",))
            return "TARGET"

    class _Prof:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mr, "DeviceMemoryProfiler", _Prof)
    monkeypatch.setattr(mr, "get_model_loader", lambda cfg: _Loader())
    monkeypatch.setattr(mr, "prepare_communication_buffer_for_model", lambda m: None)
    monkeypatch.setattr(mr, "init_model_state", lambda *a, **k: "STATE")
    monkeypatch.setattr(mr, "get_offloader", lambda: SimpleNamespace(post_init=lambda: None))

    class _Spec:
        supports_draft_weights_pool = split
        model = "DRAFT"

        def load_model(self, target, draft_pool_ctx=None):
            order.append(("spec_load", draft_pool_ctx is not None))
            if draft_pool_ctx is not None:
                with draft_pool_ctx():
                    order.append(("load_draft",))
            else:
                order.append(("load_draft_in_current",))
            order.append(("alloc_draft_logits",))

    eplb = SimpleNamespace(
        prepare_load=lambda: None,
        maybe_register_speculator=lambda *a: False,
        maybe_register_model=lambda *a: False,
        maybe_start_async_loop=lambda x: None,
    )
    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(load_format="x"),
            model_config=SimpleNamespace(model="m"),
        ),
        load_config=SimpleNamespace(load_format="x"),
        eplb=eplb,
        lora_config=None,
        use_aux_hidden_state_outputs=False,
        speculator=_Spec(),
        speculative_config=None,
        model_config=SimpleNamespace(model="m"),
        parallel_config=SimpleNamespace(),
        encoder_cache=None,
        device="cpu",
        is_pooling_model=False,
        is_last_pp_rank=True,
        is_first_pp_rank=True,
    )
    return mr, fake, pool


def test_load_model_splits_pools_in_order(monkeypatch):
    order: list[tuple] = []
    mr, fake, pool = _load_model_env(monkeypatch, order, split=True)
    mr.GPUModelRunner.load_model(fake, memory_pool_ctx=pool)
    assert order == [
        ("enter", "weights"),
        ("load_target",),
        ("exit", "weights"),  # closed before the drafter pool opens: no nesting
        ("spec_load", True),
        ("enter", "draft_weights"),
        ("load_draft",),
        ("exit", "draft_weights"),
        ("alloc_draft_logits",),  # outside both pools
    ]


def test_load_model_unsplit_speculator_stays_in_weights(monkeypatch):
    order: list[tuple] = []
    mr, fake, pool = _load_model_env(monkeypatch, order, split=False)
    mr.GPUModelRunner.load_model(fake, memory_pool_ctx=pool)
    assert order == [
        ("enter", "weights"),
        ("load_target",),
        ("spec_load", False),
        ("load_draft_in_current",),
        ("alloc_draft_logits",),
        ("exit", "weights"),
    ]


def test_load_model_without_pool_ctx_still_loads(monkeypatch):
    order: list[tuple] = []
    mr, fake, _ = _load_model_env(monkeypatch, order, split=True)
    mr.GPUModelRunner.load_model(fake)
    assert ("load_target",) in order and ("load_draft",) in order
    assert not any(e[0] in ("enter", "exit") for e in order)


def test_each_tag_opens_once(monkeypatch):
    order: list[tuple] = []
    mr, fake, pool = _load_model_env(monkeypatch, order, split=True)
    mr.GPUModelRunner.load_model(fake, memory_pool_ctx=pool)
    enters = [e[1] for e in order if e[0] == "enter"]
    assert enters == ["weights", "draft_weights"]


def test_speculator_load_model_only_draft_weights_in_pool(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import speculator as sp

    order: list[tuple] = []

    @contextmanager
    def pool():
        order.append(("enter",))
        try:
            yield
        finally:
            order.append(("exit",))

    monkeypatch.setattr(sp, "get_layers_from_vllm_config", lambda cfg, cls: {})
    monkeypatch.setattr(
        sp.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda cfg: False,
        raising=False,
    )
    monkeypatch.setattr(sp, "supports_multimodal", lambda m: False)

    fake = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=None),
        load_draft_model=lambda t, names: order.append(("load_draft",)) or "DRAFT",
        _allocate_draft_logits=lambda: order.append(("alloc_logits",)),
        _validate_local_argmax_reduction=lambda: order.append(("validate",)),
    )
    sp.DraftModelSpeculator.load_model(fake, "TARGET", draft_pool_ctx=pool)
    assert order[:4] == [("enter",), ("load_draft",), ("exit",), ("alloc_logits",)]
    assert fake.model == "DRAFT"
