# SPDX-License-Identifier: Apache-2.0
"""G-LOG block-ID accounting through the real scheduler + MambaManager (CPU).

Same scheduler, same request; the only difference is the MambaSpec the GDN
group is created with: classic (num_speculative_blocks=7) vs G-LOG (0, with
gdn_log_spec_tokens=7). The scheduler's lookahead stays 7.
"""

import pytest
import torch

from tests.v1.core import utils
from tests.v1.core.test_mixed_prefill_budget import factory  # noqa: F401
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
B = 4096
GDN_GROUPS = 6  # Qwen3.8-Flash-Next / 27B: six GDN cache groups per rank
K = 7


def _out(s, step, n_acc=1):
    ids = list(step.num_scheduled_tokens)
    toks = [[] if s.requests[r].is_prefill_chunk else [42] * n_acc for r in ids]
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=toks,
    )


def _steady_blocks(factory, spec, num_tokens):  # noqa: F811
    s = factory(
        target=0,
        async_scheduling=False,
        num_speculative_tokens=K,
        block_size=B,
        max_num_batched_tokens=B * 4,
        max_model_len=262144,
        num_blocks=4000,
        kv_cache_spec=spec,
    )
    s.need_mamba_block_aligned_split = True
    s.mamba_state_block_size = B
    r = utils.create_requests(
        1, num_tokens=num_tokens, req_ids=["r"], max_tokens=512, block_size=B
    )[0]
    s.add_request(r)
    while r.num_output_tokens < 8:
        st = s.schedule()
        s.update_from_output(st, _out(s, st))
    mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
    return len(mgr.req_to_blocks["r"])


@pytest.mark.parametrize("ctx_tokens", [31 * B + 100, 62 * B + 100])
def test_gdn_group_holds_seven_fewer_blocks_under_glog(factory, ctx_tokens):  # noqa: F811
    common = dict(
        block_size=B, shapes=((1,),), dtypes=(torch.float16,), mamba_cache_mode="align"
    )
    classic = _steady_blocks(
        factory, MambaSpec(**common, num_speculative_blocks=K), ctx_tokens
    )
    glog = _steady_blocks(
        factory,
        MambaSpec(**common, num_speculative_blocks=0, gdn_log_spec_tokens=K),
        ctx_tokens,
    )
    assert classic - glog == K, (classic, glog)
    # The product: six GDN groups -> 42 block IDs (378 MiB at 9 MiB/page) saved.
    assert GDN_GROUPS * (classic - glog) == 42


def test_blocks_per_user_arithmetic_matches_acceptance_rows():
    """Acceptance table (supplement e), from the measured per-user totals in
    Astra's port page: 254K steady 177 IDs, 126K 113. 42 saved each."""
    for total, floor in ((177, 0.20), (113, 0.33)):
        saved = 42
        assert saved / total >= floor
