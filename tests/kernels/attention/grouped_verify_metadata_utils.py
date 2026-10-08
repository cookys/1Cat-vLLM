# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the serving metadata class, without a fictitious ``num_reqs`` field."""

from dataclasses import fields

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata


def serving_metadata(source):
    qsl = source.query_start_loc
    empty = torch.empty(0, device=qsl.device)
    metadata = TritonAttentionMetadata(
        num_actual_tokens=source.num_actual_tokens,
        max_query_len=source.max_query_len,
        query_start_loc=qsl,
        max_seq_len=source.max_model_len,
        seq_lens=source.seq_lens,
        block_table=source.block_table,
        slot_mapping=torch.full(
            (source.num_actual_tokens,), -1, dtype=torch.int64, device=qsl.device
        ),
        seq_threshold_3D=1,
        num_par_softmax_segments=1,
        softmax_segm_output=empty,
        softmax_segm_max=empty,
        softmax_segm_expsum=empty,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    # FlashAttnV100MetadataBuilder attaches routing/CPU-shadow extensions;
    # num_reqs belongs to CommonAttentionMetadata, never this result class.
    base_fields = {f.name for f in fields(TritonAttentionMetadata)}
    for name, value in vars(source).items():
        if name not in base_fields and name != "num_reqs":
            setattr(metadata, name, value)
    assert not hasattr(metadata, "num_reqs")
    return metadata
