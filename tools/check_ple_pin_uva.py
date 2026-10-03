# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-run GPU smoke for the exact PLE pin path and full-range UVA reads.

Launch with torchrun on an owner-reserved GPU set. Default footprint is only
16 MiB/rank; this checks byte/address contracts, not large-startup pressure.
"""

import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.mib <= 0 or args.iterations <= 0:
        parser.error("mib and iterations must be positive")

    import torch

    from vllm.models.qwen4_exp.nvidia.ple_layer import _ple_pinned_host_empty
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    size = args.mib * 1024**2
    started = time.monotonic()
    storage = _ple_pinned_host_empty((size // 256, 256), torch.float8_e4m3fn)
    pin_s = time.monotonic() - started
    assert storage.is_pinned()
    raw = storage.view(torch.uint8)
    pointer = storage.data_ptr()
    view = get_accelerator_view_from_cpu_tensor(storage).view(torch.uint8)
    assert view.shape == storage.shape and view.stride() == storage.stride()
    assert view.is_cuda and view.is_contiguous()
    # Probe both ends and multiple bytes of every OS page, not just its first byte.
    pages = torch.arange(0, size, os.sysconf("SC_PAGE_SIZE"), dtype=torch.int64)
    indices = torch.unique(
        torch.cat((pages, pages + 37, pages + 255, torch.tensor([0, size - 1])))
    )
    indices = indices[indices < size]
    gpu_indices = indices.to(device=view.device)
    pattern = torch.arange(256, dtype=torch.int64)
    for iteration in range(args.iterations):
        raw.copy_(((pattern + rank + iteration) % 256).to(torch.uint8))
        expected = raw.reshape(-1)[indices]
        actual = view.reshape(-1).index_select(0, gpu_indices).cpu()
        assert torch.equal(expected, actual), f"UVA byte mismatch on rank {rank}"
        assert storage.data_ptr() == pointer
    print(
        json.dumps(
            {
                "rank": rank,
                "pid": os.getpid(),
                "bytes": size,
                "pin_s": pin_s,
                "sampled_bytes_per_iteration": indices.numel(),
                "iterations": args.iterations,
                "uva_byte_parity": True,
                "host_pointer": pointer,
                "device_pointer": view.data_ptr(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
