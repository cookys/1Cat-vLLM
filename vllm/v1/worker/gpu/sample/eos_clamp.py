# SPDX-License-Identifier: Apache-2.0
"""renew-tail-eos-v3 option C (RULING-eos-parity-v2, DESIGN-renew-tail-eos-v1 AMEND 4): clamp the accepted run of a
speculative step to its first stop position, on device, before ``postprocess_sampled``.

Why: under async scheduling step k+1 is already enqueued when the CPU sees a stop inside step k's accepted run
(m < g). If step k+1 feeds the bonus token the post-EOS state is gone and the tail cannot be registered. Clamping
num_sampled (g -> m), last_sampled_tokens (-> the stop token) and num_computed_tokens (+m, not +g) together makes
step k+1 feed the stop token first, so its first-token state is the exact post-stop state.

The clamped copies feed ONLY ``postprocess_sampled``; AsyncOutput and the drafter keep the raw tensors, so the CPU
scheduler's accounting (g) is unchanged. Stop rule = scheduler ``check_stop``: after output token i of the run
(new_total = total_len + i + 1): stop iff new_total - prompt_len >= min_tokens and (token in {eos, stop_token_ids}
or new_total >= min(prompt_len + max_tokens, max_model_len)). ignore_eos drops the eos id.
"""
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

MAX_NUM_CLAMP_STOP_IDS = 128


def stop_ids_of(sp: SamplingParams) -> list[int]:
    """The ids check_stop() treats as a stop token (eos is None when ignore_eos)."""
    ids = set(sp.stop_token_ids or ())
    if sp.eos_token_id is not None:
        ids.add(sp.eos_token_id)
    return sorted(ids)


class EosClampState:
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.num_stop = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.stop_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_CLAMP_STOP_IDS), dtype=torch.int32, device=device
        )
        # absolute output-length thresholds, in total_len units (prompt + output)
        self.min_total = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.max_total = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(
        self, req_idx: int, prompt_len: int, sp: SamplingParams, max_model_len: int
    ) -> None:
        ids = stop_ids_of(sp)
        if len(ids) > MAX_NUM_CLAMP_STOP_IDS:
            raise ValueError(
                f"Too many stop token ids: {len(ids)}. The max size is {MAX_NUM_CLAMP_STOP_IDS}."
            )
        self.num_stop.np[req_idx] = len(ids)
        if ids:
            self.stop_ids.stage_write(req_idx, 0, ids)
        self.min_total.np[req_idx] = prompt_len + sp.min_tokens
        max_tokens = sp.max_tokens if sp.max_tokens is not None else max_model_len
        self.max_total.np[req_idx] = min(prompt_len + max_tokens, max_model_len)

    def apply_staged_writes(self) -> None:
        self.num_stop.copy_to_uva()
        self.stop_ids.apply_write()
        self.min_total.copy_to_uva()
        self.max_total.copy_to_uva()

    def clamp(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        total_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return clamp_accepted_run(
            idx_mapping, sampled_tokens, num_sampled, num_rejected, total_len,
            self.min_total.gpu, self.max_total.gpu, self.num_stop.gpu, self.stop_ids.gpu,
        )


@triton.jit
def _clamp_kernel(
    idx_mapping_ptr,
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    out_sampled_ptr,
    out_rejected_ptr,
    total_len_ptr,
    min_total_ptr,
    max_total_ptr,
    num_stop_ptr,
    stop_ids_ptr,
    stop_ids_stride,
    BLOCK_RUN: tl.constexpr,
    BLOCK_STOP: tl.constexpr,
):
    r = tl.program_id(0)
    g = tl.load(num_sampled_ptr + r)
    rej = tl.load(num_rejected_ptr + r)
    m = g
    s = tl.load(idx_mapping_ptr + r)
    if s >= 0:
        if g > 1:
            total = tl.load(total_len_ptr + s)
            min_total = tl.load(min_total_ptr + s)
            max_total = tl.load(max_total_ptr + s)
            n_stop = tl.load(num_stop_ptr + s)
            ib = tl.arange(0, BLOCK_RUN)
            sb = tl.arange(0, BLOCK_STOP)
            toks = tl.load(sampled_ptr + r * sampled_stride + ib, mask=ib < g, other=-1)
            sids = tl.load(stop_ids_ptr + s * stop_ids_stride + sb, mask=sb < n_stop, other=-1)
            match = (toks[:, None] == sids[None, :]) & (sb[None, :] < n_stop)
            is_stop = tl.sum(match.to(tl.int32), axis=1) > 0
            new_total = total + ib + 1
            hit = (ib < g) & (new_total >= min_total) & (is_stop | (new_total >= max_total))
            first = tl.min(tl.where(hit, ib, BLOCK_RUN), axis=0)
            m = tl.where(first < BLOCK_RUN, first + 1, g)
    tl.store(out_sampled_ptr + r, m)
    tl.store(out_rejected_ptr + r, rej + (g - m))


def clamp_accepted_run(
    idx_mapping, sampled_tokens, num_sampled, num_rejected, total_len, min_total, max_total, num_stop, stop_ids
):
    """Returns (num_sampled', num_rejected') as NEW tensors; inputs are left untouched."""
    num_reqs = idx_mapping.shape[0]
    out_sampled = torch.empty_like(num_sampled)
    out_rejected = torch.empty_like(num_rejected)
    _clamp_kernel[(num_reqs,)](
        idx_mapping, sampled_tokens, sampled_tokens.stride(0), num_sampled, num_rejected,
        out_sampled, out_rejected, total_len, min_total, max_total, num_stop,
        stop_ids, stop_ids.stride(0),
        BLOCK_RUN=triton.next_power_of_2(sampled_tokens.shape[1]),
        BLOCK_STOP=triton.next_power_of_2(MAX_NUM_CLAMP_STOP_IDS),
    )
    return out_sampled, out_rejected


def clamp_reference(
    idx_mapping, sampled_tokens, num_sampled, num_rejected, total_len, min_total, max_total, stops
):
    """Pure-python/torch statement of the same rule. stops: list (per request-state slot) of id sets."""
    out_s, out_r = num_sampled.clone(), num_rejected.clone()
    for r in range(idx_mapping.shape[0]):
        s = int(idx_mapping[r])
        g = int(num_sampled[r])
        if s < 0 or g <= 1:
            continue
        for i in range(g):
            nt = int(total_len[s]) + i + 1
            if nt >= int(min_total[s]) and (int(sampled_tokens[r, i]) in stops[s] or nt >= int(max_total[s])):
                out_s[r] = i + 1
                out_r[r] = num_rejected[r] + (g - i - 1)
                break
    return out_s, out_r
