"""renew-tail-eos-v2 worker side (CPU): MambaHybridModelState.add_request records the bias of a tail-hit request and
preprocess_state replays it in place through run_mamba_align_precopy with src == dst, token_bias = bias and one-token queries.
The Triton kernel itself is mocked (its in-place path is the existing single_token_tail path)."""
import types

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_test


def _state(mh):
    st = object.__new__(mh.MambaHybridModelState)
    st._align_mode = True
    st._renew_replay = {}
    st._mamba_state_idx_gpu = torch.full((8,), -2, dtype=torch.int32)
    st._mamba_src_col_gpu = torch.full((8,), -1, dtype=torch.int32)
    st._mamba_token_bias_gpu = torch.zeros(8, dtype=torch.int32)
    return st


def test_replay_runs_in_place_on_scratch_tables(monkeypatch):
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh

    st = _state(mh)
    st._renew_replay = {5: (2, 44), 3: (1, 100)}
    calls = []
    monkeypatch.setattr(mh, "run_mamba_align_precopy", lambda *a: calls.append(a))
    ib = types.SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([3, 5]),
        idx_mapping=torch.tensor([3, 5]),
        query_start_loc=torch.tensor([0, 7, 9], dtype=torch.int32),
    )
    st._replay_renew_tail(ib, "CTX", 16)
    assert st._renew_replay == {}
    ctx, n, dst, src, tb, ridx, qsl = calls[0]
    assert ctx == "CTX" and n == 2 and ridx.tolist() == [3, 5] and qsl.tolist() == [0, 1, 2]
    # column = cdiv(num_computed, block) - 1; src == dst; bias per request; other rows untouched
    assert dst[3].item() == src[3].item() == 6 and dst[5].item() == src[5].item() == 2
    assert tb[3].item() == 1 and tb[5].item() == 2 and tb[0].item() == 0
    # the live tables were not modified
    assert st._mamba_state_idx_gpu.tolist() == [-2] * 8 and st._mamba_token_bias_gpu.tolist() == [0] * 8


def test_replay_requires_the_request_in_the_batch(monkeypatch):
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh

    st = _state(mh)
    st._renew_replay = {5: (2, 44)}
    ib = types.SimpleNamespace(num_reqs=1, idx_mapping_np=np.array([3]), idx_mapping=torch.tensor([3]), query_start_loc=torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(AssertionError):
        st._replay_renew_tail(ib, "CTX", 16)
