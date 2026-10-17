# SPDX-License-Identifier: Apache-2.0
"""Shared CPU fixtures for the plan 075 G-LOG tests.

The Triton kernels run through the Triton interpreter (TRITON_INTERPRET=1) on
CPU tensors. Shapes and strides are the production ones: per-rank GDN geometry
(H=4 key heads, HV=12 value heads, K=V=128), the 1,179,648 B page of the
C1-fix serve (--block-size 4096, nvfp4 KV), a QKV view with the wider fused
projection row stride, and precomputed FP32 gating as the packed verifier
feeds it.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as _fsg  # noqa: E402
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out,
)
from vllm.v1.gdn_log import (  # noqa: E402
    HDR_WORD_EPOCH,
    LOG_STEPS,
    GdnLogLayout,
)



def _interpreter_exp(x):
    # The module-level alias ``exp = tl.exp`` is bound before the interpreter
    # patches ``tl`` and fails under it; resolve through ``tl`` at call time.
    return _fsg.tl.exp(x)


_fsg.exp = _interpreter_exp

H, HV, K, V = 4, 12, 128, 128
Q_SIZE = H * K
QKV_WIDTH = 2 * H * K + HV * V
ROW_STRIDE = QKV_WIDTH + 2 * 96  # z/b/a share the projection row in production
CONV_BYTES = 51200
PAGE_BYTES = 1179648
PAGE_WORDS = PAGE_BYTES // 4
LAYOUT = GdnLogLayout(
    conv_bytes=CONV_BYTES, hv=HV, k=K, v=V, page_size_bytes=PAGE_BYTES
)
CONV_WIDTH, CONV_DIM = 10, CONV_BYTES // 2 // 10  # (3 + 7) x 2560 FP16
K_SPEC = 7
WIDTH = K_SPEC + 1  # verifier query length == state-index width


class Pages:
    """A strided pool of GDN pages: conv bytes, FP32 SSM, logs, header."""

    def __init__(self, num_pages: int, fill: float | None = None) -> None:
        self.num_pages = num_pages
        gen = torch.Generator().manual_seed(7)
        if fill is None:
            # Arbitrary bytes everywhere (stale pages): header and logs of a
            # fresh page must not need initialisation.
            self.raw = torch.randn(num_pages * PAGE_WORDS, generator=gen) * 0.01
        else:
            self.raw = torch.full((num_pages * PAGE_WORDS,), fill)
        self.conv = torch.as_strided(
            self.raw.view(torch.float16),
            size=(num_pages, CONV_WIDTH, CONV_DIM),
            stride=(PAGE_BYTES // 2, CONV_DIM, 1),
        )
        self.ssm = torch.as_strided(
            self.raw,
            size=(num_pages, HV, V, K),
            stride=(PAGE_WORDS, V * K, K, 1),
            storage_offset=CONV_BYTES // 4,
        )

    def words_i32(self) -> torch.Tensor:
        return self.raw.view(torch.int32)

    def epoch_index(self, page: int) -> int:
        return (
            page * PAGE_WORDS + LAYOUT.header_offset // 4 + HDR_WORD_EPOCH
        )

    def epoch(self, page: int) -> int:
        return int(self.words_i32()[self.epoch_index(page)])

    def set_epoch(self, page: int, value: int) -> None:
        self.words_i32()[self.epoch_index(page)] = value

    def record(self, page: int, bank: int, step: int, head: int):
        base = page * PAGE_WORDS + CONV_BYTES // 4 + LAYOUT.record_word_index(
            bank, step, head
        )
        rec = self.raw[base : base + LAYOUT.record_words]
        return rec[0], rec[1 : 1 + K], rec[1 + K :]

    def replay(self, page: int, bank: int, steps: int) -> torch.Tensor:
        """Reference materialisation: H1 plus ``steps`` logged updates, using
        the verifier's operation order (decay multiply, then rank-1 add)."""
        h = self.ssm[page].clone()
        for step in range(1, steps + 1):
            for head in range(HV):
                decay, key, delta = self.record(page, bank, step, head)
                h[head] = h[head] * decay
                h[head] = h[head] + delta[:, None] * key[None, :]
        return h


def make_round_inputs(num_reqs: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    tokens = num_reqs * WIDTH
    base = torch.zeros(tokens, ROW_STRIDE, dtype=torch.float16)
    base[:, :QKV_WIDTH] = (torch.randn(tokens, QKV_WIDTH, generator=gen) * 0.7).to(
        torch.float16
    )
    mixed_qkv = base[:, :QKV_WIDTH]
    assert mixed_qkv.stride() == (ROW_STRIDE, 1)
    g = -(torch.rand(tokens, HV, generator=gen) * 1.5 + 0.05)
    beta = torch.sigmoid(torch.randn(tokens, HV, generator=gen))
    cu = torch.arange(num_reqs + 1, dtype=torch.int32) * WIDTH
    return mixed_qkv, g, beta, cu


def run_verify(
    ssm: torch.Tensor,
    indices: torch.Tensor,
    selectors: torch.Tensor,
    inputs,
    *,
    gdn_log: GdnLogLayout | None,
) -> torch.Tensor:
    mixed_qkv, g, beta, cu = inputs
    out = torch.zeros(mixed_qkv.shape[0], 1, HV, V, dtype=torch.float16)
    fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
        A_log=torch.zeros(HV),
        a=g,
        b=beta,
        dt_bias=torch.zeros(HV),
        mixed_qkv=mixed_qkv,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        scale=K**-0.5,
        initial_state=ssm,
        out=out,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=selectors,
        use_qk_l2norm_in_kernel=True,
        precomputed_g=g,
        precomputed_beta=beta,
        quantize_state_each_step=False,
        match_recurrent_schedule=True,
        match_recurrent_numerics=True,
        gdn_log=gdn_log,
    )
    return out


def make_align_ctx(
    pages: Pages,
    block_table: torch.Tensor,
    *,
    glog: bool,
    block_size: int = 4096,
    max_num_reqs: int | None = None,
) -> SimpleNamespace:
    """The context the MRV2 align kernels read, built like
    MambaSpecDecodeGPUContext.initialize_from_forward_context builds it."""
    conv, ssm = pages.conv, pages.ssm
    return SimpleNamespace(
        is_initialized=True,
        num_layers=1,
        num_state_types=2,
        num_states=2,
        block_table_ptrs=torch.tensor([block_table.data_ptr()], dtype=torch.int64),
        block_table_stride_req=block_table.stride(0),
        state_base_addrs=torch.tensor(
            [conv.data_ptr(), ssm.data_ptr()], dtype=torch.int64
        ),
        state_block_strides=torch.tensor(
            [conv.stride(0) * 2, ssm.stride(0) * 4], dtype=torch.int64
        ),
        state_elem_sizes=torch.tensor([2, 4], dtype=torch.int32),
        state_inner_sizes=torch.tensor(
            [conv.stride(1), ssm[0].numel()], dtype=torch.int64
        ),
        state_conv_widths=torch.tensor([conv.size(1), 0], dtype=torch.int32),
        state_group_indices=torch.zeros(2, dtype=torch.int32),
        block_size=block_size,
        num_accepted_tokens_out=torch.zeros(
            max_num_reqs or block_table.shape[0], dtype=torch.int32
        ),
        glog_layout=LAYOUT if glog else None,
        state_glog_flags=torch.tensor([0, 1 if glog else 0], dtype=torch.int32),
    )
