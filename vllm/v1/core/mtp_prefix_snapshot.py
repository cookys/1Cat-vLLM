# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host certificates for completed, immutable MTP prefill checkpoints.

No device allocation/copy or sampling operation belongs here. The align
allocator already preserves a boundary state when the next chunk moves to a
new block. We retain one additional boundary and certify the existing blocks.
Certificates are deliberately unavailable until the producer finishes normally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.request import Request


@dataclass(eq=False, slots=True)
class MTPPrefixCertificate:
    boundary: int
    next_token: int
    # One object shared by the terminal block of EVERY cache group. Identity
    # couples the groups and is a generation token, including same-hash reuse.
    committed: bool = False


def eligible_request(request: Request) -> bool:
    """First version: ordinary text requests, never resumed/preempted/embedded."""
    return bool(
        request.prompt_token_ids is not None
        and request.prompt_embeds is None
        and not request.mm_features
        and request.lora_request is None
        and not request.resumable
        and request.num_preemptions == 0
    )


class MTPPrefixSnapshots:
    def __init__(
        self,
        alignment: int,
        *,
        telemetry: bool = False,
        cacheable_group_ids: tuple[int, ...] | None = None,
    ):
        self.alignment = alignment
        self.telemetry = telemetry
        self.cacheable_group_ids = cacheable_group_ids
        self.pending: dict[str, list[MTPPrefixCertificate]] = {}
        self.lookup_hits = 0
        self.lookup_saved_tokens = 0

    def pool_stats(self, pool) -> tuple[int, int, int]:
        """Diagnostic only: ordinary eviction owns these blocks, no private pin."""
        return (
            sum(block.mtp_prefix_only for block in pool.blocks),
            pool.mtp_prefix_evictions,
            pool.get_num_free_blocks(),
        )

    def extra_boundaries(self, request: Request) -> tuple[int, ...]:
        if not eligible_request(request):
            return ()
        # Strictly inside the original prompt: token[B] must be known, and a
        # later target prefill must execute before the producer's final output.
        boundary = (request.num_prompt_tokens - 1) // self.alignment * self.alignment
        return (boundary,) if boundary > 0 else ()

    def record_prefill(
        self,
        request: Request,
        end: int,
        managers: Sequence[SingleTypeKVCacheManager],
    ) -> bool:
        if not (
            eligible_request(request)
            and request.num_output_tokens == 0
            and request.num_output_placeholders == 0
            and request.num_computed_tokens < end < request.num_prompt_tokens
            and end % self.alignment == 0
        ):
            return False
        terminal_blocks = []
        for manager in managers:
            if not manager.kv_cache_spec.prefix_cacheable:
                # Only gated QSA raw rings may enter here. At an aligned
                # compression-group boundary they need no historical rows.
                continue
            index = end // manager.block_size - 1
            blocks = manager.req_to_blocks[request.request_id]
            if not 0 <= index < len(blocks):
                return False
            block = blocks[index]
            # Do not certify a shared prefix or an unmaterialized/null state.
            if block.is_null or block.block_hash is None or block.ref_cnt != 1:
                return False
            if block.mtp_prefix_certificate is not None:
                return False
            terminal_blocks.append(block)
        if not terminal_blocks:
            return False
        certificate = MTPPrefixCertificate(end, request.prompt_token_ids[end])
        for block in terminal_blocks:
            block.mtp_prefix_certificate = certificate
        self.pending.setdefault(request.request_id, []).append(certificate)
        return True

    def finish(self, request_id: str, *, publish: bool) -> int:
        certificates = self.pending.pop(request_id, ())
        if publish:
            for certificate in certificates:
                certificate.committed = True
        # Unpublished certificates can remain on cached blocks, but can never
        # be used by this policy. reset_hash removes them on pool eviction.
        return len(certificates) if publish else 0

    def accepts(
        self,
        request: Request,
        blocks_by_group: Sequence[Sequence[KVCacheBlock]],
        boundary: int,
    ) -> bool:
        if (
            not eligible_request(request)
            or not 0 < boundary < request.num_prompt_tokens
        ):
            return False
        if self.cacheable_group_ids is not None:
            if any(i >= len(blocks_by_group) for i in self.cacheable_group_ids):
                return False
            blocks_by_group = [blocks_by_group[i] for i in self.cacheable_group_ids]
        if not blocks_by_group or any(not blocks for blocks in blocks_by_group):
            return False
        certificate = blocks_by_group[0][-1].mtp_prefix_certificate
        if not (
            certificate is not None
            and certificate.committed
            and certificate.boundary == boundary
            and certificate.next_token == request.prompt_token_ids[boundary]
        ):
            return False
        # Ordinary block lookup has already validated every prefix hash. The
        # extra token guards the MTP shift: KV[B-1] depends on target token[B].
        return all(
            not blocks[-1].is_null
            and blocks[-1].block_hash is not None
            and blocks[-1].mtp_prefix_certificate is certificate
            for blocks in blocks_by_group
        )


class MTPPrefixBlockPoolView:
    """Lookup-only view; experimental states still need all-group validation."""

    def __init__(self, block_pool):
        self.block_pool = block_pool
        self.null_block = block_pool.null_block

    def get_cached_block(self, block_hash, group_ids):
        return self.block_pool.get_cached_block(
            block_hash, group_ids, allow_mtp_prefix=True
        )


def configure_mtp_prefix_snapshots(config, coordinator, *, has_connector: bool):
    """Fail closed when the explicitly enabled experiment leaves its scope."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
    )

    text = config.model_config.hf_text_config
    spec = config.speculative_config
    parallel = config.parallel_config
    managers = coordinator.single_type_managers
    mamba = [m for m in managers if isinstance(m, MambaManager)]
    groups = coordinator.kv_cache_config.kv_cache_groups
    layers = {
        name: manager
        for group, manager in zip(groups, managers)
        for name in group.layer_names
    }
    # The standalone drafter registers its real attention modules in the shared
    # static context. Verify their presence in the final scheduler cache config;
    # merely recognizing the model's architecture name is insufficient.
    draft = f"mtp.layers.{getattr(text, 'num_hidden_layers', -1)}.self_attn"
    draft_full = (draft + ".attn", draft + ".indexer.compressed_key_cache")
    ratio = getattr(text, "indexer_compress_ratio", 0)
    rings_safe = (
        isinstance(ratio, int)
        and ratio > 0
        and all(
            type(manager) is CircularBufferManager
            and group.layer_names
            and all(
                name.endswith(".self_attn.indexer.raw_key_cache")
                for name in group.layer_names
            )
            and manager.block_size % ratio == 0
            and coordinator.lcm_block_size % manager.block_size == 0
            for group, manager in zip(groups, managers)
            if not manager.kv_cache_spec.prefix_cacheable
        )
    )
    supported = (
        config.use_v2_model_runner
        and not config.scheduler_config.async_scheduling
        and config.cache_config.enable_prefix_caching
        and config.cache_config.mamba_cache_mode == "align"
        and getattr(text, "model_type", None) == "qwen4_exp_text"
        and getattr(text, "mtp_num_hidden_layers", 0) == 1
        and spec is not None
        and spec.method == "mtp"
        and spec.num_speculative_tokens == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.data_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not has_connector
        and isinstance(coordinator, HybridKVCacheCoordinator)
        and all(
            type(m) in (FullAttentionManager, MambaManager, CircularBufferManager)
            for m in managers
        )
        and all(
            type(layers.get(name)) is FullAttentionManager
            and layers[name].kv_cache_spec.prefix_cacheable
            for name in draft_full
        )
        and type(layers.get(draft + ".indexer.raw_key_cache")) is CircularBufferManager
        and rings_safe
        and {m.kv_cache_spec.mamba_type for m in mamba}
        == {MambaAttentionBackendEnum.GDN_ATTN, MambaAttentionBackendEnum.SHORT_CONV}
        and all(
            m.mamba_cache_mode == "align" and m.block_size == coordinator.lcm_block_size
            for m in mamba
        )
        and all(
            m.kv_cache_spec.prefix_cacheable
            for m in managers
            if type(m) is not CircularBufferManager
        )
    )
    if not supported:
        raise ValueError(
            "VLLM_SM70_MTP_COMMITTED_PREFIX_CACHE requires Qwen4Exp V2 MTP4, "
            "one draft layer, align prefix caching, GDN + PLE short-conv, "
            "registered draft main/compressed KV, aligned QSA raw rings, "
            "synchronous scheduling, PP/DP/DCP/PCP=1 "
            "and no KV/EC connector"
        )
    return MTPPrefixSnapshots(
        coordinator.lcm_block_size,
        cacheable_group_ids=tuple(
            i for i, m in enumerate(managers) if m.kv_cache_spec.prefix_cacheable
        ),
    )
