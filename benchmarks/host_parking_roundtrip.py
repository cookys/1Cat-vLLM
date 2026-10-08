# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lead-run raw-byte gate. Default mode prints the plan without CUDA calls.

Use --execute-gpu only in the assigned isolated window. Run one process per TP
rank to test simultaneous transfers; output filenames must differ per rank.
This synthetic gate checks physical copies, not serving/state correctness.
"""

import argparse
import json
import os
from pathlib import Path

PAGE = 1179648
# Six 8-layer committed GDN groups, two 8-layer NVFP4 full-attention groups,
# one 5-layer FP16 DFlash sliding-window group. KV physical tensors shared.
GROUPS = [(8, 837632)] * 6 + [(8, PAGE)] * 2 + [(5, 1048576)]


def plan():
    return {
        "physical_page_bytes": PAGE,
        "groups": [
            {"group": g, "layers": n, "valid_bytes_per_layer": valid}
            for g, (n, valid) in enumerate(GROUPS)
        ],
        "gpu_bytes_per_rank": 8 * 36 * PAGE,
        "host_bytes_per_rank": sum(n for n, _ in GROUPS) * 4 * PAGE,
        "cycles": "all groups x distinct/shuffled destinations x slot reuse",
        "scope": "synthetic raw transfer; no serving arithmetic/parity assertion",
    }


def run(rank):
    import torch

    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
    )
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
    from vllm.v1.kv_offload.cpu.parking import bind_memory_node
    from vllm.v1.kv_offload.cpu.parking_pool import ParkingPool
    from vllm.v1.kv_offload.cpu.parking_transfer import RecoveringHandler

    torch.cuda.set_device(rank)
    backing = [
        torch.full((36, PAGE), -71, dtype=torch.int8, device=f"cuda:{rank}")
        for _ in range(8)
    ]
    refs = [[CanonicalKVCacheRef(i, valid) for i in range(n)] for n, valid in GROUPS]
    canonical = CanonicalKVCaches(
        [CanonicalKVCacheTensor(t, PAGE) for t in backing], refs
    )
    with bind_memory_node(0):
        pool = ParkingPool(plan()["host_bytes_per_rank"])
    handlers = CpuGpuOffloadingHandlers(
        canonical,
        1,
        4,
        group_page_sizes={g: n * PAGE for g, (n, _) in enumerate(GROUPS)},
        group_num_blocks={g: 4 for g in range(9)},
        cpu_tensor_factory=pool.allocate,
    )
    down = RecoveringHandler(handlers.gpu_to_cpu_handler, pool)
    up = RecoveringHandler(handlers.cpu_to_gpu_handler, pool)
    rows = []
    pattern = (
        torch.arange(PAGE, dtype=torch.int32, device=f"cuda:{rank}")
        .remainder_(251)
        .to(torch.int8)
    )
    try:
        for cycle in range(3):
            for g, (n, valid) in enumerate(GROUPS):
                src, dst = 4 * g + cycle % 2, 4 * g + 2 + cycle % 2
                value = (g * 17 + cycle * 7 + rank) % 127
                sizes = [0] * 9
                sizes[g] = 1
                source = GPULoadStoreSpec([src], sizes, [0] * 9)
                dest = GPULoadStoreSpec([dst], sizes, [0] * 9)
                cpu = CPULoadStoreSpec([cycle % 4])
                # Enqueue writes without a host synchronize; handler must wait
                # for the compute stream before reading or restoring a page.
                for layer, t in enumerate(backing[:n]):
                    t[src, :valid].copy_(pattern[:valid].bitwise_xor(value + layer))
                    t[dst].fill_(-71)
                jid = cycle * 100 + 2 * g
                assert down.transfer_async(jid, (source, cpu))
                down.wait({jid})
                dr = down.get_finished()
                assert len(dr) == 1 and dr[0].success
                assert up.transfer_async(jid + 1, (cpu, dest))
                up.wait({jid + 1})
                ur = up.get_finished()
                assert len(ur) == 1 and ur[0].success
                equal = all(
                    torch.equal(t[src, :valid], t[dst, :valid]) for t in backing[:n]
                )
                padding_ok = all(
                    bool((t[dst, valid:] == -71).all()) for t in backing[:n]
                )
                rows.append(
                    {
                        "cycle": cycle,
                        "group": g,
                        "byte_equal": equal,
                        "padding_untouched": padding_ok,
                        "payload_bytes": n * valid,
                        "d2h_s": dr[0].transfer_time,
                        "h2d_s": ur[0].transfer_time,
                    }
                )
                if not (equal and padding_ok):
                    raise AssertionError(f"raw roundtrip mismatch group={g}")
    finally:
        # Both directions quiesce before host registration is released.
        down.shutdown()
        up.shutdown()
    return {"rank": rank, "status": "PASS", "plan": plan(), "transfers": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-gpu", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = p.parse_args()
    result = (
        run(args.rank) if args.execute_gpu else {"status": "SKIP_GPU", "plan": plan()}
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        path = Path(str(args.output).replace("{rank}", str(args.rank)))
        path.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
