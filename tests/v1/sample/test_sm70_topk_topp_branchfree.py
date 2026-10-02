# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the host-sync-free SM70 compact top-k/top-p sampler.

``_apply_top_k_top_p_compact`` is called directly (it is plain torch ops plus a
CPU-capable tie sort), which bypasses the sm70/is_cuda dispatch gate. The three
``VLLM_SM70_TOPK_TOPP_BRANCHFREE`` modes must all equal the full-vocabulary
reference bit for bit, and modes 1 and 2 must not synchronize with the host.

Run on a GPU-less shell so nothing touches a device. The production venv has
no pytest, so borrow one with uv without modifying the venv:
    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \
        uv run --no-project --python <venv>/bin/python --with pytest -- \
        python -m pytest --noconftest \
        tests/v1/sample/test_sm70_topk_topp_branchfree.py \
        tests/v1/sample/test_topk_topp_tied_cutoffs.py -q
"""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.sample.ops import topk_topp_triton as triton_ops
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.sample import states as states_module

ENV = "VLLM_SM70_TOPK_TOPP_BRANCHFREE"
VOCAB = 248320
SMALL_VOCAB = 40000
MODES = (0, 1, 2)
# Mirrors the margin inside _apply_top_k_top_p_compact.
MARGIN = 8 * 128 * torch.finfo(torch.float32).eps


@pytest.fixture(autouse=True)
def _fresh_env_cache():
    # The server freezes env lookups after startup; tests need live reads.
    envs.disable_envs_cache()


@pytest.fixture(autouse=True)
def _stable_sort(monkeypatch: pytest.MonkeyPatch):
    """Make the CPU full-row sort stable, like CUDA's large-vocabulary sort.

    The reference splits a top-p tie by sorted position. CUDA sorts a 248k-wide
    row with a stable radix sort (equal logits keep vocabulary order), and the
    compact path reproduces that order. CPU ``Tensor.sort`` is not stable by
    default, so without this the reference would break ties differently from
    the device it models.
    """
    original = torch.Tensor.sort

    def stable_sort(tensor, *args, **kwargs):
        kwargs.setdefault("stable", True)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sort", stable_sort)


def _compact(
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    monkeypatch.setenv(ENV, str(mode))
    out = triton_ops._apply_top_k_top_p_compact(
        logits.clone(), k, p, float("-inf"), **kwargs
    )
    assert out is not None
    return out


def _reference(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor):
    return apply_top_k_top_p_pytorch(logits.clone(), k, p)


def _assert_bit_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Compare bit patterns so NaN payloads and signed zeros count as well.
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def _kp(ks, ps) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(ks, dtype=torch.int32), torch.tensor(ps, dtype=torch.float32)


def _background(rows: int, vocab: int = SMALL_VOCAB) -> torch.Tensor:
    return torch.full((rows, vocab), -20.0)


def _scatter_positions(
    x: torch.Tensor, row: int, values: torch.Tensor, seed: int
) -> None:
    """Put ``values`` at random distinct vocabulary slots so vocab order matters."""
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm(x.shape[1], generator=gen)[: values.numel()]
    x[row, slots] = values


# ---------------------------------------------------------------------------
# 1. random full-vocabulary batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_random_full_vocab_matches_reference(monkeypatch, mode):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(5, VOCAB, generator=gen) * 2
    k, p = _kp([20] * 5, [0.95] * 5)
    _assert_bit_equal(
        _compact(monkeypatch, mode, logits, k, p), _reference(logits, k, p)
    )


@pytest.mark.parametrize("mode", MODES)
def test_per_row_parameters_match_reference(monkeypatch, mode):
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(5, SMALL_VOCAB, generator=gen) * 3
    k, p = _kp([1, 5, 20, 64, 127], [0.5, 0.8, 0.95, 0.99, 0.9])
    _assert_bit_equal(
        _compact(monkeypatch, mode, logits, k, p), _reference(logits, k, p)
    )


@pytest.mark.parametrize("mode", MODES)
def test_mask_value_is_applied(monkeypatch, mode):
    monkeypatch.setenv(ENV, str(mode))
    gen = torch.Generator().manual_seed(2)
    logits = torch.randn(3, SMALL_VOCAB, generator=gen) * 2
    k, p = _kp([20] * 3, [0.9] * 3)
    out = triton_ops._apply_top_k_top_p_compact(logits.clone(), k, p, -1.0e4)
    ref = _reference(logits, k, p)
    ref.masked_fill_(torch.isneginf(ref), -1.0e4)
    assert out is not None
    _assert_bit_equal(out, ref)


# ---------------------------------------------------------------------------
# 2. ties and degenerate rows
# ---------------------------------------------------------------------------


def _tie_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = 5
    x = _background(rows)
    # row 0: 30 equal logits straddle the k-th value (ranks 15..44, k=20).
    _scatter_positions(
        x, 0, torch.cat([3.0 + torch.arange(15, 0, -1) / 8, torch.full((30,), 2.0)]), 10
    )
    # row 1: 200 equal logits at the cutoff, more than the 128-entry shortlist.
    _scatter_positions(
        x,
        1,
        torch.cat([3.0 + torch.arange(10, 0, -1) / 8, torch.full((200,), 1.0)]),
        11,
    )
    # row 2: fewer than 128 finite logits.
    x[2] = float("-inf")
    _scatter_positions(
        x, 2, torch.randn(60, generator=torch.Generator().manual_seed(12)), 12
    )
    # row 3: a NaN logit.
    _scatter_positions(
        x, 3, torch.randn(80, generator=torch.Generator().manual_seed(13)) * 2, 13
    )
    x[3, 5] = float("nan")
    # row 4: every logit equal.
    x[4] = 1.0
    k, p = _kp([20] * rows, [0.95] * rows)
    return x, k, p


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("top_p", [0.3, 0.6, 0.95, 1.0])
def test_ties_and_degenerate_rows_match_reference(monkeypatch, mode, top_p):
    x, k, _ = _tie_batch()
    p = torch.full((x.shape[0],), top_p)
    _assert_bit_equal(_compact(monkeypatch, mode, x, k, p), _reference(x, k, p))


@pytest.mark.parametrize("mode", MODES)
def test_ties_beyond_shortlist_keep_every_tied_logit(monkeypatch, mode):
    x, k, _ = _tie_batch()
    p = torch.full((x.shape[0],), 1.0)
    out = _compact(monkeypatch, mode, x, k, p)
    # 10 distinct + 200 tied logits survive; the shortlist alone holds only 128.
    assert int(torch.isfinite(out[1]).sum()) == 210
    _assert_bit_equal(out, _reference(x, k, p))


@pytest.mark.parametrize("mode", [0, 1])
def test_modes_0_and_1_leave_the_input_untouched(monkeypatch, mode):
    # The reference scatters into its input; mode 1 must hand it a copy.
    x, k, _ = _tie_batch()
    p = torch.full((x.shape[0],), 0.95)
    before = x.clone()
    monkeypatch.setenv(ENV, str(mode))
    out = triton_ops._apply_top_k_top_p_compact(x, k, p, float("-inf"))
    assert out is not None
    _assert_bit_equal(x, before)


# ---------------------------------------------------------------------------
# 3. top-p boundary rows (cumulative probability inside the ambiguity margin)
# ---------------------------------------------------------------------------


def _boundary_row(seed: int, top_k: int, offset: float, index: int):
    """A row whose ascending cumsum lies ``offset`` away from ``1 - p``."""
    gen = torch.Generator().manual_seed(seed)
    kept = 3.0 + torch.randn(top_k, generator=gen)
    probs = kept.sort().values.softmax(-1)
    cumulative = probs.cumsum(-1)
    target = float(cumulative[index]) + offset
    row = torch.full((SMALL_VOCAB,), -20.0)
    slots = torch.randperm(SMALL_VOCAB, generator=gen)[:top_k]
    row[slots] = kept
    return row, 1.0 - target


class _WhereSpy:
    """Record the conditions handed to ``torch.where`` (mode 1's row select)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.conditions: list[torch.Tensor] = []
        original = torch.where

        def spy(condition, *args, **kwargs):
            self.conditions.append(condition.clone())
            return original(condition, *args, **kwargs)

        monkeypatch.setattr(torch, "where", spy)


@pytest.mark.parametrize("mode", MODES)
def test_top_p_margin_rows_match_reference(monkeypatch, mode):
    index = 14
    specs = [
        # (seed, offset from the boundary, expected to be a reference row)
        (20, 1e-5, True),
        (21, -1e-5, True),
        (22, 0.0, True),
        (23, 5e-4, False),
        (24, -5e-4, False),
    ]
    rows, ps = [], []
    for seed, offset, _ in specs:
        row, p_value = _boundary_row(seed, 20, offset, index)
        rows.append(row)
        ps.append(p_value)
    x = torch.stack(rows)
    k, p = _kp([20] * len(specs), ps)

    # Ground truth for the fixture: ask mode 1 which rows it routes to the
    # reference, instead of trusting the construction above.
    with pytest.MonkeyPatch.context() as spy_patch:
        spy = _WhereSpy(spy_patch)
        spy_patch.setenv(ENV, "1")
        triton_ops._apply_top_k_top_p_compact(x.clone(), k, p, float("-inf"))
    assert len(spy.conditions) == 1
    assert spy.conditions[0].flatten().tolist() == [flag for *_, flag in specs]

    _assert_bit_equal(_compact(monkeypatch, mode, x, k, p), _reference(x, k, p))


def test_boundary_offsets_straddle_the_margin():
    # The boundary fixture assumes 1e-5 is inside and 5e-4 outside the margin.
    assert triton_ops.COMPACT_CAPACITY == 128
    assert 1e-5 < MARGIN < 5e-4


# ---------------------------------------------------------------------------
# 4. host synchronizations
# ---------------------------------------------------------------------------


class _SyncCounter:
    """Count host-visible tensor reads: bool(), .item() and .nonzero()."""

    def __init__(self) -> None:
        self.bool = 0
        self.item = 0
        self.nonzero = 0

    def run(self, fn: Callable[[], torch.Tensor | None]):
        original_bool = torch.Tensor.__bool__
        original_item = torch.Tensor.item
        original_nonzero = torch.Tensor.nonzero
        counter = self

        def counting_bool(t):
            counter.bool += 1
            return original_bool(t)

        def counting_item(t):
            counter.item += 1
            return original_item(t)

        def counting_nonzero(t, *args, **kwargs):
            counter.nonzero += 1
            return original_nonzero(t, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.Tensor, "__bool__", counting_bool)
            patch.setattr(torch.Tensor, "item", counting_item)
            patch.setattr(torch.Tensor, "nonzero", counting_nonzero)
            result = fn()
        return result, (self.bool, self.item, self.nonzero)


def _clean_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(30)
    x = torch.randn(4, SMALL_VOCAB, generator=gen) * 3
    k, p = _kp([20] * 4, [0.9] * 4)
    return x, k, p


def _syncs(monkeypatch, mode, x, k, p, **kwargs):
    monkeypatch.setenv(ENV, str(mode))
    return _SyncCounter().run(
        lambda: triton_ops._apply_top_k_top_p_compact(
            x.clone(), k, p, float("-inf"), **kwargs
        )
    )


def test_counter_sees_the_two_baseline_fences(monkeypatch):
    # Guards the counter itself: mode 0 keeps the k gate and the `.any()`.
    x, k, p = _clean_batch()
    out, counts = _syncs(monkeypatch, 0, x, k, p)
    assert counts == (2, 0, 0)
    _assert_bit_equal(out, _reference(x, k, p))


def test_mode0_nonzero_fence_when_reference_rows_exist(monkeypatch):
    x, k, _ = _tie_batch()
    p = torch.full((x.shape[0],), 0.95)
    _, counts = _syncs(monkeypatch, 0, x, k, p)
    assert counts == (2, 0, 1)


def test_mode1_old_signature_keeps_only_the_k_gate(monkeypatch):
    x, k, p = _clean_batch()
    out, counts = _syncs(monkeypatch, 1, x, k, p)
    assert counts == (1, 0, 0)
    _assert_bit_equal(out, _reference(x, k, p))


@pytest.mark.parametrize("flagged", [False, True])
def test_mode1_with_host_k_gate_has_no_syncs(monkeypatch, flagged):
    if flagged:
        x, k, _ = _tie_batch()
        p = torch.full((x.shape[0],), 0.95)
    else:
        x, k, p = _clean_batch()
    out, counts = _syncs(monkeypatch, 1, x, k, p, k_in_compact_range=True)
    assert counts == (0, 0, 0)
    _assert_bit_equal(out, _reference(x, k, p))


@pytest.mark.parametrize("flagged", [False, True])
@pytest.mark.parametrize("k_in_compact_range", [None, True])
def test_mode2_has_no_syncs(monkeypatch, flagged, k_in_compact_range):
    if flagged:
        x, k, _ = _tie_batch()
        p = torch.full((x.shape[0],), 0.95)
    else:
        x, k, p = _clean_batch()
    # None means "old signature": do not pass the keyword at all.
    kwargs = {} if k_in_compact_range is None else {"k_in_compact_range": True}
    out, counts = _syncs(monkeypatch, 2, x, k, p, **kwargs)
    assert counts == (0, 0, 0)
    _assert_bit_equal(out, _reference(x, k, p))


def test_host_k_gate_false_returns_none_without_launching_topk(monkeypatch):
    x, k, p = _clean_batch()
    monkeypatch.setenv(ENV, "0")
    topk_calls = []
    original_topk = torch.Tensor.topk

    def spy_topk(t, *args, **kwargs):
        topk_calls.append(args)
        return original_topk(t, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "topk", spy_topk)
    out, counts = _SyncCounter().run(
        lambda: triton_ops._apply_top_k_top_p_compact(
            x.clone(), k, p, float("-inf"), k_in_compact_range=False
        )
    )
    assert out is None
    assert counts == (0, 0, 0)
    assert topk_calls == []


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv(ENV, "3")
    with pytest.raises(ValueError, match=ENV):
        envs.environment_variables[ENV]()
    monkeypatch.delenv(ENV)
    assert envs.environment_variables[ENV]() == 0
    assert envs.VLLM_SM70_TOPK_TOPP_BRANCHFREE == 0


# ---------------------------------------------------------------------------
# 5. host-side k gate equals the device formula
# ---------------------------------------------------------------------------


def _device_formula(k: torch.Tensor) -> bool:
    return bool(((k > 0) & (k < 128)).all())


@pytest.mark.parametrize("value", [0, 1, 19, 20, 127, 128, 129, VOCAB])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_compact_k_in_range_single_values(value, dtype):
    arr = np.full(4, value, dtype=dtype)
    assert triton_ops.compact_k_in_range(arr) == _device_formula(torch.from_numpy(arr))
    assert triton_ops.compact_k_in_range(arr) == (0 < value < 128)


def test_compact_k_in_range_mixed_rows():
    pool = np.array([0, 1, 19, 20, 127, 128, 129, VOCAB], dtype=np.int32)
    rng = np.random.default_rng(0)
    for _ in range(200):
        arr = rng.choice(pool, size=int(rng.integers(1, 9)))
        assert triton_ops.compact_k_in_range(arr) == _device_formula(
            torch.from_numpy(arr)
        )
    assert triton_ops.compact_k_in_range(np.array([20, 127, 1], dtype=np.int32))
    assert not triton_ops.compact_k_in_range(np.array([20, 128, 1], dtype=np.int32))
    assert not triton_ops.compact_k_in_range(np.array([20, 0, 1], dtype=np.int32))


# ---------------------------------------------------------------------------
# 7. plumbing: sampler wrapper and the V2 SamplingStates call site
# ---------------------------------------------------------------------------


def test_sampler_wrapper_forwards_gate_only_to_triton(monkeypatch):
    if not hasattr(topk_topp_sampler, "apply_top_k_top_p_triton"):
        pytest.skip("Triton unavailable")
    seen = []

    def fake_triton(logits, k, p, mask_value=float("-inf"), **kwargs):
        seen.append(kwargs)
        return logits

    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p_triton", fake_triton)
    x = torch.zeros(2, SMALL_VOCAB)
    k, p = _kp([20, 20], [0.9, 0.9])
    topk_topp_sampler.apply_top_k_top_p(x, k, p)
    topk_topp_sampler.apply_top_k_top_p(x, k, p, k_in_compact_range=True)
    topk_topp_sampler.apply_top_k_top_p(x, k, p, k_in_compact_range=False)
    assert seen == [
        {"k_in_compact_range": None},
        {"k_in_compact_range": True},
        {"k_in_compact_range": False},
    ]


def _sampling_states(top_k, top_p):
    state = object.__new__(states_module.SamplingStates)
    state.vocab_size = VOCAB
    state.top_k = SimpleNamespace(
        np=np.array(top_k, dtype=np.int32), gpu=torch.tensor(top_k, dtype=torch.int32)
    )
    state.top_p = SimpleNamespace(
        np=np.array(top_p, dtype=np.float32),
        gpu=torch.tensor(top_p, dtype=torch.float32),
    )
    return state


@pytest.mark.parametrize(
    ("top_k", "top_p", "expected"),
    [
        ([20, 5, 127], [0.9, 0.9, 0.9], True),
        ([20, 128, 5], [0.9, 0.9, 0.9], False),
        ([20, VOCAB, 5], [0.9, 0.9, 0.9], False),
        ([20, 5, 6], [1.0, 1.0, 1.0], None),  # top-p inactive: no gate
        ([VOCAB, VOCAB, VOCAB], [0.9, 0.9, 0.9], None),  # top-k inactive: no gate
    ],
)
def test_sampling_states_decides_the_gate_on_the_host(
    monkeypatch, top_k, top_p, expected
):
    state = _sampling_states(top_k, top_p)
    seen = []

    def fake(logits, k, p, **kwargs):
        seen.append(kwargs)
        return logits

    monkeypatch.setattr(states_module, "apply_top_k_top_p", fake)
    logits = torch.zeros(4, 8)
    expanded = torch.tensor([0, 1, 1, 2])  # request 1 owns two rows
    state.apply_top_k_top_p(logits, expanded, np.array([0, 1, 2]))
    assert seen == [{"k_in_compact_range": expected}]


def test_sampling_states_only_reads_the_scheduled_requests(monkeypatch):
    # Request 1 (k=128) is not in this step, so the gate must stay True.
    state = _sampling_states([20, 128, 5], [0.9, 0.9, 0.9])
    seen = []
    monkeypatch.setattr(
        states_module,
        "apply_top_k_top_p",
        lambda logits, k, p, **kw: seen.append(kw) or logits,
    )
    state.apply_top_k_top_p(torch.zeros(2, 8), torch.tensor([0, 2]), np.array([0, 2]))
    assert seen == [{"k_in_compact_range": True}]
