# SPDX-License-Identifier: Apache-2.0
"""CPU-only 27B TP4 capacity accounting using the real allocator.

Model geometry: 48 GDN + 16 attention layers, DFlash2: 5 FP16 SW2048
layers. Budgets are inputs, not predictions of the GPU memory profiler.
Native version >=2 has XQA plus a prefill bridge, still GPU-unvalidated.
--reserve-bridge deducts its profiled FP16 workspace from an OLD baseline
budget. Do not deduct it again from a measured new Available KV budget.
The CLI block is included in Platform._align_hybrid_block_size alignment;
do not confuse --block-size 2048 with the minimum backend multiple (16).
"""

import argparse
import json
from types import SimpleNamespace as NS

import torch

from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MambaAttentionBackendEnum,
    MambaSpec,
    SlidingWindowSpec,
)


def geometry(mode, kv, cli_block):
    steps = {"nospec": 0, "mtp4": 4, "dflash7": 7}[mode]
    state = MambaSpec(
        block_size=-1,
        shapes=((2560, 3 + steps), (12, 128, 128)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=steps,
    )
    per_token = 288 if kv == "nvfp4" else 512
    alignment = max(16, cli_block)
    block = alignment * -(-state.page_size_bytes // (alignment * per_token))
    page = block * per_token
    attn = FullAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4 if kv == "nvfp4" else KVQuantMode.NONE,
    )
    from dataclasses import replace

    state = replace(state, block_size=block, page_size_padded=page)
    specs = {f"model.layers.{i}": attn if i % 4 == 3 else state for i in range(64)}
    if mode == "mtp4":
        specs["mtp.layers.0"] = attn
    if mode == "dflash7":
        sw = SlidingWindowSpec(
            block_size=block,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.float16,
            sliding_window=2048,
        )
        specs.update({f"draft.layers.{i}": sw for i in range(5)})
    config = NS(
        parallel_config=NS(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=NS(
            max_model_len=262144,
            get_num_kv_heads=lambda _: 1,
            get_total_num_hidden_layers=lambda: 64,
        ),
        cache_config=NS(
            user_specified_mamba_block_size=True,
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            block_size=block,
            enable_prefix_caching=True,
            hash_block_size=None,
        ),
        scheduler_config=NS(
            disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=4096
        ),
        max_in_flight_tokens=8192,
        kv_transfer_config=None,
    )
    return specs, config


def capacity(mode, kv, cli_block, budget_gib, reserve_bridge=False):
    specs, config = geometry(mode, kv, cli_block)
    block = config.cache_config.block_size
    # Version >=2 bridge shared across attention layers on a stream.
    # Account for it when reusing a baseline budget predating NVFP4 profiling.
    # K + V, FP16, head256, one local KV head, rounded to 784-token pages.
    capacity_tokens = -(-262144 // block) * block
    bridge_bytes = -(-capacity_tokens // 784) * 784 * 256 * 2 * 2 + 4 * -(
        -capacity_tokens // 784
    )
    deducted = bridge_bytes if kv == "nvfp4" and reserve_bridge else 0
    groups = U.get_kv_cache_groups(config, specs)
    cache = U.get_kv_cache_config_from_groups(
        config,
        groups,
        int(budget_gib * 2**30) - deducted,
    )
    rows = []
    for group in cache.kv_cache_groups:
        spec = group.kv_cache_spec
        needed = -(-spec.max_memory_usage_bytes(config) // spec.page_size_bytes)
        rows.append(
            dict(
                kind=type(spec).__name__,
                layers=len(group.layer_names),
                block=spec.block_size,
                page_bytes=spec.page_size_bytes,
                ids_per_request=needed,
            )
        )
    per_request = sum(row["ids_per_request"] for row in rows)
    scheduler, hash_block = U.resolve_kv_cache_block_sizes(cache, config)
    return dict(
        scope="allocator_only_not_serving_validation",
        reader_status=(
            "xqa_and_bridge_v2_gpu_unvalidated" if kv == "nvfp4" else "legacy"
        ),
        bridge_reserve_applied=bool(deducted),
        mode=mode,
        kv=kv,
        cli_block=cli_block,
        budget_gib=budget_gib,
        bridge_reserve_bytes=deducted,
        attention_block=block,
        scheduler_block=scheduler,
        hash_block=hash_block,
        pool_ids=cache.num_blocks,
        ids_per_256k_request=per_request,
        users_at_256k=cache.num_blocks // per_request,
        equivalent_tokens=int(cache.num_blocks / per_request * 262144),
        groups=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["nospec", "mtp4", "dflash7"], required=True)
    parser.add_argument("--kv", choices=["fp8", "nvfp4"], default="nvfp4")
    parser.add_argument("--cli-block", type=int, default=2048)
    parser.add_argument("--budget-gib", type=float, required=True)
    parser.add_argument(
        "--reserve-bridge",
        action="store_true",
        help="Deduct the profiled FP16 bridge from an OLD baseline KV budget",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            capacity(
                args.mode, args.kv, args.cli_block, args.budget_gib, args.reserve_bridge
            ),
            indent=2,
        )
    )
