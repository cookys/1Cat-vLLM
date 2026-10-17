# SPDX-License-Identifier: Apache-2.0
"""The align block-table gather under G-LOG: one page per request, PAD after."""

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

B = 4096


def _spec(**kw):
    return MambaSpec(
        block_size=B,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        **kw,
    )


def test_glog_gathers_current_page_and_pads():
    table = torch.arange(1, 7, dtype=torch.int32).view(1, 6)  # only 6 columns
    seq_lens = torch.tensor([2 * B + 5])  # current column 2
    out = mamba_get_block_table_tensor(
        table, seq_lens, _spec(num_speculative_blocks=0, gdn_log_spec_tokens=7), "align"
    )
    assert out.shape == (1, 8)
    assert out[0, 0] == 3 and bool((out[0, 1:] == PAD_SLOT_ID).all())


def test_glog_last_column_does_not_overrun_the_row():
    # The classic gather would index past the row here (needs 1+7 columns).
    table = torch.arange(1, 4, dtype=torch.int32).view(1, 3)
    out = mamba_get_block_table_tensor(
        table,
        torch.tensor([3 * B]),
        _spec(num_speculative_blocks=0, gdn_log_spec_tokens=7),
        "align",
    )
    assert out[0, 0] == 3 and out.shape == (1, 8)


def test_classic_gather_unchanged():
    table = torch.arange(1, 13, dtype=torch.int32).view(1, 12)
    out = mamba_get_block_table_tensor(
        table, torch.tensor([B + 1]), _spec(num_speculative_blocks=7), "align"
    )
    assert out.tolist() == [[2, 3, 4, 5, 6, 7, 8, 9]]
