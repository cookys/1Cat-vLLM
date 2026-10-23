# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 GPU state migration for hybrid Mamba ``align`` prefix caching.

This is the narrow Model Runner V2 portion of upstream vLLM PR #42406.  It is
kept separate from :mod:`vllm.v1.worker.mamba_utils` because that module also
owns this tree's V1 DDTree state-selection compatibility path.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.gdn_log import HDR_WORD_EPOCH, GdnLogLayout
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

# Temporal states are much larger than conv states. Splitting their byte range
# keeps block-boundary copies from serializing on one CTA at small batch sizes.
_TEMPORAL_TILES = 16


@triton.jit
def _glog_replay_tile(
    src_page,
    dst_page,
    steps,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    READ_PREV_BANK: tl.constexpr,
    G_HV: tl.constexpr,
    G_K: tl.constexpr,
    G_V: tl.constexpr,
    G_BANK0: tl.constexpr,
    G_BANK_STRIDE: tl.constexpr,
    G_STEP_STRIDE: tl.constexpr,
    G_EPOCH: tl.constexpr,
):
    """Materialise ``H1 + steps`` logged updates of ``src_page`` into ``dst_page``.

    ``src_page``/``dst_page`` point at the FP32 SSM state of a G-LOG page. Each
    element evolves independently (``h = h * decay; h = h + delta * key``), so a
    tile can run alone and ``src_page == dst_page`` is safe. READ_PREV_BANK picks
    the bank of the last completed round relative to the page epoch: 0 while
    that round's epoch flip is still pending (postprocess), 1 once it ran
    (precopy of the next step).
    """
    epoch = tl.load((src_page + G_EPOCH).to(tl.pointer_type(tl.int32)))
    bank = (epoch & 1) ^ READ_PREV_BANK
    log = src_page + G_BANK0 + bank * G_BANK_STRIDE
    total: tl.constexpr = G_HV * G_V * G_K
    per_tile: tl.constexpr = (total + TEMPORAL_TILES - 1) // TEMPORAL_TILES
    tile_start = tile_idx * per_tile
    tile_end = tl.minimum(tile_start + per_tile, total)
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for chunk_start in range(tile_start, tile_end, COPY_BLOCK_SIZE):
        elem = chunk_start + offsets
        mask = elem < tile_end
        head = elem // (G_V * G_K)
        row = (elem % (G_V * G_K)) // G_K
        col = elem % G_K
        h = tl.load(src_page + elem, mask=mask, other=0)
        for step in range(1, steps + 1):
            rec = log + step * G_STEP_STRIDE + head * (1 + G_K + G_V)
            decay = tl.load(rec, mask=mask, other=0)
            key = tl.load(rec + 1 + col, mask=mask, other=0)
            delta = tl.load(rec + 1 + G_K + row, mask=mask, other=0)
            h = h * decay
            h = h + delta * key
        tl.store(dst_page + elem, h, mask=mask)


@triton.jit
def _copy_mamba_state_block(
    state_idx,
    block_table_row,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    state_glog_flags_ptr=None,
    GLOG: tl.constexpr = False,
    READ_PREV_BANK: tl.constexpr = 0,
    G_HV: tl.constexpr = 0,
    G_K: tl.constexpr = 0,
    G_V: tl.constexpr = 0,
    G_BANK0: tl.constexpr = 0,
    G_BANK_STRIDE: tl.constexpr = 0,
    G_STEP_STRIDE: tl.constexpr = 0,
    G_EPOCH: tl.constexpr = 0,
):
    """Copy one SD-layout conv or temporal state using V1 align semantics."""
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx).to(tl.int64)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx).to(tl.int64)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table += block_table_row * block_table_stride_req

    dst_block_id = tl.load(block_table + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dst_block_id * state_block_stride
    is_conv_state = conv_width > 0

    if is_conv_state:
        # Conv states are small and may be shifted left within the same block.
        # One CTA owns the copy so a block-local barrier can make each chunk
        # memmove-safe before lower addresses overwrite overlapping input.
        if tile_idx > 0:
            return
        src_block_id = tl.load(block_table + src_col).to(tl.int64)
        token_bytes = state_inner_size * state_elem_size
        src_addr = (
            state_base_addr
            + src_block_id * state_block_stride
            + token_bias.to(tl.int64) * token_bytes
        )
        copy_size = (conv_width - token_bias).to(tl.int64) * token_bytes
        tile_start = copy_size * 0
        tile_end = copy_size
        is_left_overlap = (dst_addr < src_addr) & (dst_addr + copy_size > src_addr)
    else:
        # Temporal state chooses the accepted speculative column.
        if GLOG:
            # G-LOG states have one page per request: the accepted state is
            # H1 plus token_bias logged updates, replayed while copying.
            is_log_state = tl.load(state_glog_flags_ptr + state_idx) != 0
            src_block_id = tl.load(
                block_table + src_col + tl.where(is_log_state, 0, token_bias)
            ).to(tl.int64)
            if is_log_state:
                _glog_replay_tile(
                    (state_base_addr + src_block_id * state_block_stride).to(
                        tl.pointer_type(tl.float32)
                    ),
                    dst_addr.to(tl.pointer_type(tl.float32)),
                    token_bias,
                    tile_idx,
                    COPY_BLOCK_SIZE,
                    TEMPORAL_TILES,
                    READ_PREV_BANK,
                    G_HV,
                    G_K,
                    G_V,
                    G_BANK0,
                    G_BANK_STRIDE,
                    G_STEP_STRIDE,
                    G_EPOCH,
                )
                return
        else:
            src_block_id = tl.load(block_table + src_col + token_bias).to(tl.int64)
        src_addr = state_base_addr + src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size
        bytes_per_tile = (copy_size + TEMPORAL_TILES - 1) // TEMPORAL_TILES
        tile_start = tile_idx.to(tl.int64) * bytes_per_tile
        tile_end = tl.minimum(tile_start + bytes_per_tile, copy_size)
        is_left_overlap = False

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for chunk_start in range(tile_start, tile_end, COPY_BLOCK_SIZE):
        byte_offsets = chunk_start + offsets
        mask = byte_offsets < tile_end
        src = (src_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(src, mask=mask)
        if is_left_overlap:
            tl.debug_barrier()
        tl.store(dst, data, mask=mask)


@triton.jit
def preprocess_mamba_align_fused_kernel(
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    src_col_ptr,
    token_bias_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    """Advance running state columns and emit any required pre-copy inputs."""
    rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rows < num_reqs
    req_indices = tl.load(idx_mapping_ptr + rows, mask=mask, other=0)

    state_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)
    num_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)
    # A new request is seeded here, after the resolved MambaSpec is available.
    # cache_config.block_size can be smaller than the actual Mamba block when
    # an FP16 DFlash cache shares the hybrid page pool, so add_request cannot
    # derive the state column from the global cache block size.
    seeded_state_idx = (num_computed + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    state_idx = tl.where(state_idx == -2, seeded_state_idx, state_idx)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_indices, mask=mask, other=1)
    token_bias = tl.maximum(num_accepted - 1, 0)
    tl.store(src_col_ptr + req_indices, state_idx, mask=mask)
    tl.store(token_bias_ptr + req_indices, token_bias, mask=mask)

    query_start = tl.load(query_start_loc_ptr + rows, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + rows + 1, mask=mask, other=0)
    computed_after = num_computed + query_end - query_start
    new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    tl.store(state_idx_ptr + req_indices, new_state_idx, mask=mask)
    crossed_boundary = (state_idx >= 0) & (state_idx != new_state_idx)
    # Ordinary one-token decode reads the canonical recurrent state. After a
    # speculative step, materialize the accepted slot there even if the request
    # stays in the same Mamba block; the non-spec kernels cannot select it.
    single_token_tail = (query_end - query_start == 1) & (token_bias > 0)
    tl.store(
        num_accepted_tokens_ptr + req_indices,
        1,
        mask=mask & (crossed_boundary | single_token_tail),
    )


@triton.jit
def _precopy_mamba_align_kernel(
    state_idx_ptr,
    src_col_ptr,
    token_bias_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    state_glog_flags_ptr=None,
    GLOG: tl.constexpr = False,
    G_HV: tl.constexpr = 0,
    G_K: tl.constexpr = 0,
    G_V: tl.constexpr = 0,
    G_BANK0: tl.constexpr = 0,
    G_BANK_STRIDE: tl.constexpr = 0,
    G_STEP_STRIDE: tl.constexpr = 0,
    G_EPOCH: tl.constexpr = 0,
):
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(state_idx_ptr + req_idx)
    token_bias = tl.load(token_bias_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    single_token_tail = query_end - query_start == 1 and token_bias > 0
    if src_col < 0 or (src_col == dst_col and not single_token_tail):
        return
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        TEMPORAL_TILES,
        state_glog_flags_ptr,
        GLOG,
        # The previous step's epoch flip already ran: its logs sit in the bank
        # this page would write next.
        1,
        G_HV,
        G_K,
        G_V,
        G_BANK0,
        G_BANK_STRIDE,
        G_STEP_STRIDE,
        G_EPOCH,
    )


@triton.jit
def _postprocess_mamba_align_kernel(
    num_accepted_snapshot_ptr,
    num_accepted_out_ptr,
    state_idx_ptr,
    new_num_computed_tokens_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    num_reqs,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    state_glog_flags_ptr=None,
    GLOG: tl.constexpr = False,
    G_HV: tl.constexpr = 0,
    G_K: tl.constexpr = 0,
    G_V: tl.constexpr = 0,
    G_BANK0: tl.constexpr = 0,
    G_BANK_STRIDE: tl.constexpr = 0,
    G_STEP_STRIDE: tl.constexpr = 0,
    G_EPOCH: tl.constexpr = 0,
):
    batch_idx = tl.program_id(0)
    state_meta_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    num_accepted = tl.load(num_accepted_snapshot_ptr + req_idx)
    src_col = tl.load(state_idx_ptr + req_idx)
    new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
    num_tokens_running_state = new_num_computed - num_accepted + 1
    aligned_new_computed = (new_num_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    if aligned_new_computed < num_tokens_running_state:
        return

    token_bias = aligned_new_computed - num_tokens_running_state
    dst_col = aligned_new_computed // MAMBA_BLOCK_SIZE - 1
    if src_col == dst_col and state_meta_idx == 0 and tile_idx == 0:
        tl.store(num_accepted_out_ptr + req_idx, 1)
    if src_col == dst_col and token_bias == 0:
        return

    _copy_mamba_state_block(
        state_meta_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        TEMPORAL_TILES,
        state_glog_flags_ptr,
        GLOG,
        # This step's round has not flipped its epoch yet: its logs are in the
        # bank the page epoch currently selects for writing.
        0,
        G_HV,
        G_K,
        G_V,
        G_BANK0,
        G_BANK_STRIDE,
        G_STEP_STRIDE,
        G_EPOCH,
    )


@triton.jit
def _glog_flip_epoch_kernel(
    state_idx_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_group_indices_ptr,
    state_glog_flags_ptr,
    idx_mapping_ptr,
    num_reqs,
    G_EPOCH_BYTES: tl.constexpr,
):
    """Publish a finished round: bump the epoch word of each request's page.

    Runs as its own launch after the postprocess kernel (and after every
    forward kernel of the step), so no verifier CTA can observe the new epoch
    while a sibling CTA of the same page is still reading the old bank.
    """
    batch_idx = tl.program_id(0)
    state_meta_idx = tl.program_id(1)
    if batch_idx >= num_reqs:
        return
    if tl.load(state_glog_flags_ptr + state_meta_idx) == 0:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return
    col = tl.load(state_idx_ptr + req_idx)
    if col < 0:
        return
    group_idx = tl.load(state_group_indices_ptr + state_meta_idx).to(tl.int64)
    block_table = tl.load(block_table_ptrs_ptr + group_idx).to(
        tl.pointer_type(tl.int32)
    )
    block_id = tl.load(block_table + batch_idx * block_table_stride_req + col).to(
        tl.int64
    )
    page = tl.load(state_base_addrs_ptr + state_meta_idx) + block_id * tl.load(
        state_block_strides_ptr + state_meta_idx
    )
    epoch_ptr = (page + G_EPOCH_BYTES).to(tl.pointer_type(tl.int32))
    tl.store(epoch_ptr, tl.load(epoch_ptr) + 1)


def _glog_kernel_args(ctx: MambaSpecDecodeGPUContext) -> dict:
    layout = ctx.glog_layout
    if layout is None:
        return {}
    return dict(
        state_glog_flags_ptr=ctx.state_glog_flags,
        GLOG=True,
        G_HV=layout.hv,
        G_K=layout.k,
        G_V=layout.v,
        G_BANK0=layout.bank0_words_from_ssm,
        G_BANK_STRIDE=layout.bank_stride_words,
        G_STEP_STRIDE=layout.step_words,
        G_EPOCH=layout.header_words_from_ssm + HDR_WORD_EPOCH,
    )


def run_mamba_align_precopy(
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    state_idx: torch.Tensor,
    src_col: torch.Tensor,
    token_bias: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    if num_reqs == 0 or not ctx.is_initialized:
        return
    grid = (num_reqs, ctx.num_states, _TEMPORAL_TILES)
    _precopy_mamba_align_kernel[grid](
        state_idx,
        src_col,
        token_bias,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_elem_sizes,
        ctx.state_inner_sizes,
        ctx.state_conv_widths,
        ctx.state_group_indices,
        idx_mapping,
        query_start_loc,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES,
        **_glog_kernel_args(ctx),
    )


def run_mamba_align_postprocess(
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    num_accepted_tokens: torch.Tensor,
    state_idx: torch.Tensor,
    new_num_computed_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> None:
    if num_reqs == 0 or not ctx.is_initialized:
        return
    # Snapshot the decision array because one program may reset a request's
    # output count while programs copying other state tensors still read it.
    snapshot = ctx.num_accepted_tokens_out
    snapshot.copy_(num_accepted_tokens)
    grid = (num_reqs, ctx.num_states, _TEMPORAL_TILES)
    _postprocess_mamba_align_kernel[grid](
        snapshot,
        num_accepted_tokens,
        state_idx,
        new_num_computed_tokens,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_elem_sizes,
        ctx.state_inner_sizes,
        ctx.state_conv_widths,
        ctx.state_group_indices,
        idx_mapping,
        num_reqs,
        MAMBA_BLOCK_SIZE=ctx.block_size,
        COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES,
        **_glog_kernel_args(ctx),
    )


def run_glog_epoch_flip(
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    state_idx: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> None:
    """Publish this step's GDN update logs (G-LOG only; call after postprocess)."""
    layout = ctx.glog_layout
    if layout is None or num_reqs == 0 or not ctx.is_initialized:
        return
    _glog_flip_epoch_kernel[(num_reqs, ctx.num_states)](
        state_idx,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_group_indices,
        ctx.state_glog_flags,
        idx_mapping,
        num_reqs,
        G_EPOCH_BYTES=(layout.header_offset - layout.ssm_offset)
        + 4 * HDR_WORD_EPOCH,
    )
