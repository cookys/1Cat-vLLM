# SPDX-License-Identifier: Apache-2.0
"""renew-tail-eos-v3 option C: CPU torch reference of the accepted-run clamp (worker/gpu/sample/eos_clamp.py).

Given sampled_token_ids [reqs, spec+1], num_sampled g, eos/stop id sets, min_tokens and the length budget, the clamp gives
num_sampled' = m (first stop position), num_rejected' = num_rejected + (g - m); post_update then makes the last sampled
token the stop token and advances num_computed by m. The Triton kernel is run through the Triton CPU interpreter when
available and must equal the reference."""
import os

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import eos_clamp as ec

pytestmark = pytest.mark.cpu_test

EOS = 100


def _sp(**kw):
    sp = SamplingParams(**kw)
    sp.update_from_generation_config({}, eos_token_id=EOS)
    return sp


def test_stop_ids_follow_check_stop():
    assert ec.stop_ids_of(_sp(max_tokens=8)) == [EOS]
    assert ec.stop_ids_of(_sp(max_tokens=8, stop_token_ids=[7, 9])) == [7, 9, EOS]
    assert ec.stop_ids_of(_sp(max_tokens=8, ignore_eos=True)) == []  # ignore_eos drops eos
    assert ec.stop_ids_of(_sp(max_tokens=8, ignore_eos=True, stop_token_ids=[7])) == [7]


def _case(sps, prompt_lens, total_lens, sampled, num_sampled, num_rejected, idx):
    n = len(sps)
    min_total = torch.tensor([p + sp.min_tokens for p, sp in zip(prompt_lens, sps)], dtype=torch.int32)
    max_total = torch.tensor([min(p + sp.max_tokens, 4096) for p, sp in zip(prompt_lens, sps)], dtype=torch.int32)
    stops = [set(ec.stop_ids_of(sp)) for sp in sps]
    return dict(
        idx_mapping=torch.tensor(idx, dtype=torch.int32),
        sampled_tokens=torch.tensor(sampled, dtype=torch.int64),
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32),
        num_rejected=torch.tensor(num_rejected, dtype=torch.int32),
        total_len=torch.tensor(total_lens, dtype=torch.int32),
        min_total=min_total, max_total=max_total, stops=stops)


def _ref(c):
    return ec.clamp_reference(**c)


def test_clamp_reference_cases():
    sps = [_sp(max_tokens=64), _sp(max_tokens=64), _sp(max_tokens=64, min_tokens=50), _sp(max_tokens=3), _sp(max_tokens=64, ignore_eos=True), _sp(max_tokens=64)]
    prompt = [40] * 6
    total = [45, 45, 45, 41, 45, 45]  # outputs so far: 5,5,5,1,5,5
    sampled = [
        [1, EOS, 3, 4],    # 0: EOS at position 1 -> m=2
        [1, 2, 3, 4],      # 1: no stop -> m=g=4
        [1, EOS, 3, 4],    # 2: min_tokens not reached -> m=4
        [1, 2, 3, 4],      # 3: length cap: outputs 1 + 2 = 3 -> m=2
        [1, EOS, 3, 4],    # 4: ignore_eos -> m=4
        [EOS, 2, 3, 4],    # 5: EOS first -> m=1
    ]
    g = [4] * 6
    rej = [0] * 6
    c = _case(sps, prompt, total, sampled, g, rej, list(range(6)))
    s, r = _ref(c)
    assert s.tolist() == [2, 4, 4, 2, 4, 1]
    assert r.tolist() == [2, 0, 0, 2, 0, 3]
    # last token after post_update = sampled[m-1]; num_computed advances by m (query_len - rejected')
    last = [sampled[i][int(s[i]) - 1] for i in range(6)]
    assert last[0] == EOS and last[5] == EOS and last[3] == 2
    # inputs are untouched
    assert c["num_sampled"].tolist() == g


def test_clamp_skips_negative_idx_and_g_le_1():
    sps = [_sp(max_tokens=64)]
    c = _case(sps, [40], [45], [[EOS, 2], [EOS, 2], [EOS, 2]], [2, 1, 0], [0, 0, 0], [-1, 0, 0])
    s, r = _ref(c)
    assert s.tolist() == [2, 1, 0] and r.tolist() == [0, 0, 0]


def _triton_cpu_ok():
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _triton_cpu_ok(), reason="triton not importable")
def test_kernel_equals_reference_on_cpu_interpreter():
    # TRITON_INTERPRET must be set before triton is imported: run the body in a fresh interpreter
    import subprocess
    import sys

    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", __file__], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "KERNEL_EQ_REFERENCE_OK" in r.stdout


def _kernel_vs_reference():
    g = torch.Generator().manual_seed(0)
    n, k = 24, 5
    sps = [_sp(max_tokens=int(torch.randint(3, 40, (1,), generator=g)),
               min_tokens=0 if i % 2 else 1 + i % 2,
               ignore_eos=bool(torch.randint(0, 5, (1,), generator=g) == 0),
               stop_token_ids=[7] if i % 3 == 0 else None) for i in range(n)]
    prompt = [int(x) for x in torch.randint(5, 60, (n,), generator=g)]
    total = [p + int(x) for p, x in zip(prompt, torch.randint(0, 12, (n,), generator=g))]
    sampled = torch.randint(0, 12, (n, k), generator=g)
    sampled[torch.rand(n, k, generator=g) < 0.2] = EOS
    ns = [int(x) for x in torch.randint(0, k + 1, (n,), generator=g)]
    nrej = [k - x for x in ns]
    c = _case(sps, prompt, total, sampled.tolist(), ns, nrej, list(range(n)))
    ref_s, ref_r = _ref(c)
    stop_ids = torch.zeros(n, ec.MAX_NUM_CLAMP_STOP_IDS, dtype=torch.int32)
    num_stop = torch.zeros(n, dtype=torch.int32)
    for i, st in enumerate(c["stops"]):
        num_stop[i] = len(st)
        for j, t in enumerate(sorted(st)):
            stop_ids[i, j] = t
    out_s, out_r = ec.clamp_accepted_run(
        c["idx_mapping"], c["sampled_tokens"], c["num_sampled"], c["num_rejected"], c["total_len"],
        c["min_total"], c["max_total"], num_stop, stop_ids)
    assert out_s.tolist() == ref_s.tolist() and out_r.tolist() == ref_r.tolist()
    assert c["num_sampled"].tolist() == ns  # not modified in place
    print("KERNEL_EQ_REFERENCE_OK")


if __name__ == "__main__":
    _kernel_vs_reference()
