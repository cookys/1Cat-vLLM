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
        "gpu_bytes_per_rank": 8 * 72 * PAGE,
        "host_bytes_per_rank": sum(n for n, _ in GROUPS) * 4 * PAGE,
        "cycles": "single + three shuffled blocks per group; three slot-reuse cycles",
        "limitations": (
            "reset generation and abort are scheduler tests, not raw-copy "
            "tests; short token tails are not committed snapshots"
        ),
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
        torch.full((72, PAGE), -71, dtype=torch.int8, device=f"cuda:{rank}")
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
                for count in (1, 3):
                    src_ids = [8 * g + x for x in (2, 0, 1)[:count]]
                    dst_ids = [8 * g + x for x in (5, 7, 6)[:count]]
                    sizes = [0] * 9
                    sizes[g] = count
                    source = GPULoadStoreSpec(src_ids, sizes, [0] * 9)
                    dest = GPULoadStoreSpec(dst_ids, sizes, [0] * 9)
                    cpu = CPULoadStoreSpec([3, 1, 2][:count])
                    # Source writes must precede native copy-stream reads.
                    for j, (src, dst) in enumerate(zip(src_ids, dst_ids)):
                        value = (g * 17 + cycle * 7 + rank + j) % 119
                        for layer, t in enumerate(backing[:n]):
                            t[src, :valid].copy_(
                                pattern[:valid].bitwise_xor(value + layer)
                            )
                            t[dst].fill_(-71)
                    jid = cycle * 1000 + 10 * g + count * 2
                    assert down.transfer_async(jid, (source, cpu))
                    down.wait({jid})
                    dr = down.get_finished()
                    assert len(dr) == 1 and dr[0].success
                    assert up.transfer_async(jid + 1, (cpu, dest))
                    up.wait({jid + 1})
                    ur = up.get_finished()
                    assert len(ur) == 1 and ur[0].success
                    equal = all(
                        torch.equal(t[src, :valid], t[dst, :valid])
                        for t in backing[:n]
                        for src, dst in zip(src_ids, dst_ids)
                    )
                    padding_ok = all(
                        bool((t[dst, valid:] == -71).all())
                        for t in backing[:n]
                        for dst in dst_ids
                    )
                    rows.append(
                        {
                            "cycle": cycle,
                            "group": g,
                            "blocks": count,
                            "source_ids": src_ids,
                            "destination_ids": dst_ids,
                            "byte_equal": equal,
                            "padding_untouched": padding_ok,
                            "payload_bytes": count * n * valid,
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
