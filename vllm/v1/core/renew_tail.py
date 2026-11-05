# SPDX-License-Identifier: Apache-2.0
"""renew-tail-eos-v1: length-keyed "tail hash" for a finished request.

A finished conversation's next turn normally hits the prefix cache only up to
the last lcm-aligned block (the partial tail block has no hash and the GDN
state only exists at aligned boundaries). This module registers the partial
tail block of every group under a hash keyed by (aligned parent hash, tail
tokens, T) when the request finishes, and lets a later request that starts
with exactly those T tokens resume at the non-aligned position T.

Everything here is behind ``C4140_RENEW_TAIL_EOS=1`` (read once by
KVCacheManager); with the flag off this module is never constructed.
See DESIGN-renew-tail-eos-v1-fable-20261105.md.

Which tokens / which state page (derivation, live config = Model Runner V2 +
G-LOG, async scheduling on)
-----------------------------------------------------------------------
Let N = request.num_tokens when the request stops (the last token is the
sampled EOS, which has not been fed to the model yet).

* The mamba "state at T" is NOT stored as a ready page after a speculative
  step. With G-LOG the page of the running column holds H1, the state after
  the *first* input token of the last step, and the accepted state is
  "H1 plus (accepted-1) logged updates" replayed lazily by the next step's
  pre-copy (worker/gpu/mamba_align.py:136-160, token_bias at :210-211,
  pre-copy launch condition at :270-275). Without G-LOG the accepted state is
  page ``src_col + token_bias`` (:164). Only ``token_bias == 0`` (one accepted
  token) therefore leaves a directly usable page: the running column itself.
  The running column of a step is ``cdiv(num_computed + num_scheduled, bs) - 1``
  (mamba_align.py:217-219).

* sync (``num_in_flight_tokens == 0`` at stop): T = N - 1. With G-LOG the page
  of the running column holds H1 plus (g-1) logged updates (g = tokens the
  step accepted); the state at T is H1 plus (m-1) of them, m = tokens kept
  after the stop truncated the accepted run (m <= g; EOS inside the run or
  max_tokens inside it). The entry stores bias = m-1 and the worker replays it
  in place when a request hits the tail (mamba_hybrid.py). Accounting check:
  num_computed == N - 1 + (g - m), else ``tail_skip_acct``. Without G-LOG the
  accepted state is slot r+(m-1). The column is the one the manager
  allocated: ``len(blocks) - 1 - nspec`` (the same expression the V1 path
  uses, mamba_utils.py:983-995).

* async (``num_in_flight_tokens = s' > 0`` at stop): step k+1 was scheduled
  before the stop was known and feeds [EOS, drafts...] (s' tokens), so after
  it ran there is a page holding H1 = the state after EOS = the state at
  T = N, with ``token_bias = 0`` by construction. Its column is the running
  column of step k+1: ``cdiv((N-1) + s', bs) - 1`` (mamba_align.py:217-219 with
  the true num_computed N-1 after step k). The scheduler's own table gives
  ``len(blocks) - 1 - nspec`` computed from the optimistic num_computed (the
  worker corrects it on the GPU, gpu_model_runner.py:2523-2531 / :2690-2706);
  the two differ only when a block boundary lies inside the rejected-draft
  window, in which case we skip (``tail_skip_r_ambiguous``). If the stop
  truncated the accepted run (m < g), v3 option C (worker/gpu/sample/eos_clamp.py)
  has clamped num_sampled / last_sampled / num_computed on device to m, so step
  k+1 still feeds the stop token first and the same page is registered; the CPU
  count then carries g - m extra. (``tail_skip_async_trunc`` is no longer
  produced; the counter key stays.) Attention KV of
  the EOS token at position N-1 is written by the same step, so the tail
  block holds positions aligned..N-1.

* Hazards: free() runs while step k+1 may still be on the GPU. The earliest
  reader is a new request scheduled at >= k+2, ordered after k+1 on the same
  stream. The tail blocks are hashed and so sit at the tail of the free queue;
  they can only be taken by eviction, which unhashes them (callback drops the
  index entry), and any new owner writes after k+1 in stream order. No extra
  guard is needed.

Draft (sliding-window) group: a tail hit needs the pages covering
[T-W+1, T). The tail page and a partially covered first page are registered
under run keys (consumed on hit); fully covered interior pages that carry no
hash yet get their ordinary full-block hash (the KV is written) and are
looked up through the new request's own hash chain.
"""

from __future__ import annotations

import hashlib
import os
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

_LOG_EVERY = 100
_ROOT_PARENT = b""

STAT_KEYS = (
    "tail_hit",
    "tail_miss_absent",
    "tail_miss_hash",
    "tail_miss_evicted",
    "tail_miss_len",
    "tail_miss_draft",
    "tail_reg",
    "tail_reg_sync",
    "tail_reg_async",
    "tail_skip_acct",
    "tail_skip_async_trunc",
    "tail_reg_bias",
    "tail_skip_aligned",
    "tail_skip_layout",
    "tail_skip_ineligible",
    "tail_skip_r_ambiguous",
    "tail_error",
)


@dataclass
class _DraftRun:
    first_idx: int
    last_idx: int
    run_keys: dict[int, bytes]  # page index -> BlockHashWithGroupId


@dataclass
class _Entry:
    length: int
    tail_hash: BlockHash
    group_ids: tuple[int, ...]  # full + mamba groups holding the tail block
    bias: int = 0  # G-LOG updates the worker replays in place at hit time (m - 1)
    owned_keys: list[bytes] = field(default_factory=list)
    draft: dict[int, _DraftRun] | None = None


@dataclass
class TailHit:
    """A usable tail hit: per-group block lists and the new hit length."""

    length: int
    num_aligned_blocks: int
    parent: bytes
    owned_keys: list[bytes]
    tail_group_ids: tuple[int, ...]
    draft_group_ids: tuple[int, ...]
    blocks: tuple[list[KVCacheBlock], ...]
    bias: int = 0


def compute_tail_hash(
    parent: bytes, tail_tokens: Sequence[int], length: int
) -> BlockHash:
    h = hashlib.sha256()
    h.update(b"renew-tail-v1")
    h.update(parent)
    h.update(length.to_bytes(8, "little"))
    h.update(array("q", tail_tokens).tobytes())
    return BlockHash(h.digest())


def _run_key(tail_hash: bytes, gid: int, j: int) -> bytes:
    d = hashlib.sha256(b"rt-draft" + tail_hash + gid.to_bytes(4, "big") + j.to_bytes(8, "big")).digest()
    return make_block_hash_with_group_id(BlockHash(d), gid)


class RenewTail:
    def __init__(self, coordinator, use_eagle: bool) -> None:
        self.coordinator = coordinator
        self.pool = coordinator.block_pool
        self.stats: dict[str, int] = dict.fromkeys(STAT_KEYS, 0)
        self.enabled = False
        self.disabled_reason: str | None = None
        self.index: OrderedDict[bytes, _Entry] = OrderedDict()
        self._by_hash: dict[bytes, bytes] = {}  # owned key -> parent
        self._max_entries = max(1, self.pool.num_gpu_blocks)
        self._seen: set[str] = set()
        self.pending: dict[str, TailHit] = {}
        self._full_gids: list[int] = []
        self._mamba_gids: list[int] = []
        self._draft_gids: list[int] = []
        # G-LOG (default on for the dflash2 preset) keeps H1 + logs in the page
        # of the running column, so only a 1-token accept leaves a ready state.
        self.verbose = os.environ.get("C4140_RENEW_TAIL_VERBOSE", "0") == "1"
        self.glog = os.environ.get("VLLM_DFLASH2_GDN_LOG", "0") == "1"
        self._setup(use_eagle)
        if self.enabled:
            self.pool.evict_callback = self._on_evict

    # ------------------------------------------------------------------ setup
    def _disable(self, reason: str) -> None:
        self.enabled = False
        self.disabled_reason = reason
        logger.warning("RENEW_TAIL disabled: %s", reason)

    def _setup(self, use_eagle: bool) -> None:
        c = self.coordinator
        if not hasattr(c, "attention_groups") or not hasattr(c, "lcm_block_size"):
            return self._disable("coordinator is not hybrid")
        if use_eagle:
            return self._disable("use_eagle (final-block recompute) not supported")
        if self.pool.enable_kv_cache_events:
            return self._disable("kv cache events enabled")
        lcm = c.lcm_block_size
        for gid, mgr in enumerate(c.single_type_managers):
            spec = c.kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            if not spec.prefix_cacheable:
                continue
            if gid in c.draft_group_ids:
                if isinstance(mgr, SlidingWindowManager) and lcm % mgr.block_size == 0:
                    self._draft_gids.append(gid)
                    continue
                return self._disable(f"draft group {gid} unsupported")
            if type(mgr) is FullAttentionManager and mgr.block_size == lcm:
                self._full_gids.append(gid)
            elif (
                isinstance(mgr, MambaManager)
                and mgr.mamba_cache_mode == "align"
                and mgr.block_size == lcm
            ):
                self._mamba_gids.append(gid)
            else:
                return self._disable(
                    f"group {gid} ({type(mgr).__name__}, bs={mgr.block_size}) "
                    f"unsupported, lcm={lcm}"
                )
        if not self._full_gids and not self._mamba_gids:
            return self._disable("no supported group")
        self.lcm = lcm
        self.enabled = True

    # ------------------------------------------------------------- bookkeeping
    def _bump(
        self,
        key: str,
        request: Request | None = None,
        length: int = -1,
        groups="",
        extra: str = "",
    ) -> None:
        self.stats[key] += 1
        if self.verbose:
            # fixed format, one line per event (parity cell: few renewals);
            # `extra` (accounting fields) is appended after groups=
            logger.info(
                "RENEW_TAIL event=%s req=%s L=%d groups=%s%s",
                key,
                request.request_id if request is not None else "-",
                length,
                groups if groups != "" else "-",
                (" " + extra) if extra else "",
            )

    def _maybe_log(self) -> None:
        s = self.stats
        n = sum(
            s[k]
            for k in (
                "tail_hit",
                "tail_miss_hash",
                "tail_miss_evicted",
                "tail_miss_len",
                "tail_miss_draft",
            )
        )
        if n and n % _LOG_EVERY == 0:
            logger.info("RENEW_TAIL stats %s", dict(s))

    def _on_evict(self, key: bytes) -> None:
        parent = self._by_hash.get(key)
        if parent is not None:
            self._forget(parent, evict_pool=True)

    def _forget(self, parent: bytes, *, evict_pool: bool) -> None:
        ent = self.index.pop(parent, None)
        if ent is None:
            return
        for key in ent.owned_keys:
            self._by_hash.pop(key, None)
        if evict_pool:
            for key in ent.owned_keys:
                blk = self.pool.cached_block_hash_to_block.get_one_block(key)
                if blk is not None:
                    self.pool._maybe_evict_cached_block(blk)

    def _parent_of(self, request: Request, aligned: int) -> bytes | None:
        hs = self.coordinator.hash_block_size
        if aligned == 0:
            return _ROOT_PARENT
        idx = aligned // hs - 1
        if idx >= len(request.block_hashes):
            return None
        return bytes(request.block_hashes[idx])

    def _page_hashes(self, request: Request, bsd: int):
        hs = self.coordinator.hash_block_size
        if bsd == hs:
            return request.block_hashes
        return BlockHashListWithBlockSize(request.block_hashes, hs, bsd)

    @staticmethod
    def _eligible(request: Request) -> bool:
        return (
            not request.lora_request
            and not request.mm_features
            and request.cache_salt is None
            and not getattr(request, "skip_reading_prefix_cache", False)
        )

    # --------------------------------------------------------------- register
    def on_free(self, request: Request) -> None:
        hint = getattr(request, "renew_tail_hint", None)
        if hint is None or not self.enabled:
            return
        request.renew_tail_hint = None
        g, in_flight, m = hint  # raw accepted, tokens of a step still in flight, kept after stop truncation
        if request.status not in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            return
        n = request.num_tokens
        nc = request.num_computed_tokens
        is_async = in_flight > 0
        L0 = n if is_async else n - 1
        if not self._eligible(request):
            return self._bump("tail_skip_ineligible", request, L0)
        acct = f"n={n} nc={nc} in_flight={in_flight} g={g} m={m}"

        # The page the last step left is H1 (state after its first input token)
        # plus g-1 logged updates; the state at the registered length is H1 plus
        # bias = m-1 of them, replayed by the worker when a request hits it.
        if is_async:
            # step k+1 (in flight) re-wrote the same page with H1 of its first
            # input token. Option C (renew-tail-eos-v3) clamps the accepted run on
            # device at the stop position, so that token is the last kept one for
            # every m <= g. The CPU count still carries the raw g: (g - m) extra.
            length = n  # state after EOS, written by the in-flight step k+1
            bias = 0
            ok = nc == n - 1 + in_flight + (g - m)
        else:
            length = n - 1
            bias = m - 1 if self.glog else 0
            # sync: the scheduler rolled num_computed back to c0+g; kept tokens end at c0+m
            ok = nc == n - 1 + (g - m)
        if not ok or length <= 0:
            return self._bump("tail_skip_acct", request, L0, extra=acct)
        lcm = self.lcm
        aligned = length // lcm * lcm
        if aligned == length:
            return self._bump("tail_skip_aligned", request, L0)
        parent = self._parent_of(request, aligned)
        if parent is None:
            return self._bump("tail_skip_layout", request, L0)
        rid = request.request_id
        mgrs = self.coordinator.single_type_managers
        tail_idx = length // lcm
        picks: list[tuple[int, KVCacheBlock]] = []
        for gid in self._full_gids:
            blocks = mgrs[gid].req_to_blocks.get(rid)
            if not blocks or len(blocks) <= tail_idx:
                return self._bump("tail_skip_layout", request, L0)
            picks.append((gid, blocks[tail_idx]))
        for gid in self._mamba_gids:
            mgr = mgrs[gid]
            blocks = mgr.req_to_blocks.get(rid)
            if not blocks:
                return self._bump("tail_skip_layout", request, L0)
            col_table = len(blocks) - 1 - mgr._nspec(rid)
            if is_async:
                col = cdiv(n - 1 + in_flight, lcm) - 1
                if col != col_table:
                    return self._bump("tail_skip_r_ambiguous", request, L0)
            else:
                # non-G-LOG: the accepted state is slot r + (g - 1)
                col = col_table + (m - 1 if not self.glog else 0)
                if m - 1 > mgr._nspec(rid):
                    return self._bump("tail_skip_layout", request, L0)
            if col < tail_idx or col >= len(blocks):
                return self._bump("tail_skip_layout", request, L0)
            picks.append((gid, blocks[col]))
        for _, blk in picks:
            if blk.is_null or blk.block_hash is not None or blk.ref_cnt != 1:
                return self._bump("tail_skip_layout", request, L0)

        tail_hash = compute_tail_hash(
            parent, request.all_token_ids[aligned:length], length
        )
        draft_plan = self._plan_draft(request, length, tail_hash)
        if draft_plan is False:
            return self._bump("tail_skip_layout", request, L0)

        old = self.index.get(parent)
        if old is not None:
            self._forget(parent, evict_pool=True)
        entry = _Entry(length, tail_hash, tuple(gid for gid, _ in picks), bias=bias)
        for gid, blk in picks:
            key = make_block_hash_with_group_id(tail_hash, gid)
            self._own(blk, key, parent, entry)
        if draft_plan:
            entry.draft = {}
            for gid, (first_idx, last_idx, actions) in draft_plan.items():
                run_keys: dict[int, bytes] = {}
                for kind, j, blk, key in actions:
                    if kind == "run":
                        self._own(blk, key, parent, entry)
                        run_keys[j] = key
                    else:  # "real": ordinary full-block hash, KV is written
                        blk.block_hash = key
                        self.pool.cached_block_hash_to_block.insert(key, blk)
                entry.draft[gid] = _DraftRun(first_idx, last_idx, run_keys)
        self.index[parent] = entry
        while len(self.index) > self._max_entries:
            self._forget(next(iter(self.index)), evict_pool=True)
        gs = ",".join(str(x) for x in list(entry.group_ids) + list(entry.draft or ()))
        self.stats["tail_reg"] += 1
        if bias:
            self.stats["tail_reg_bias"] += 1
        self._bump(
            "tail_reg_async" if is_async else "tail_reg_sync",
            request,
            length,
            gs,
            extra=f"bias={bias} " + acct,
        )

    def _own(self, blk: KVCacheBlock, key: bytes, parent: bytes, entry: _Entry) -> None:
        blk.block_hash = key
        self.pool.cached_block_hash_to_block.insert(key, blk)
        self._by_hash[key] = parent
        entry.owned_keys.append(key)

    def _plan_draft(self, request: Request, length: int, tail_hash: BlockHash):
        """Pages of each draft group covering [T-W+1, T).

        Returns {} (no draft data, e.g. K=0 request), False (unusable layout)
        or {gid: (first_idx, last_idx, [(kind, j, block, key)])}.
        """
        plan: dict[int, tuple[int, int, list]] = {}
        rid = request.request_id
        for gid in self._draft_gids:
            mgr = self.coordinator.single_type_managers[gid]
            blocks = mgr.req_to_blocks.get(rid)
            if not blocks:
                return {}  # K=0 request: tail usable only where draft is skipped
            bsd = mgr.block_size
            start = max(0, length - mgr.sliding_window + 1)
            first_idx = start // bsd
            last_idx = (length - 1) // bsd
            if len(blocks) <= last_idx:
                return False
            hashes = self._page_hashes(request, bsd)
            actions = []
            for j in range(first_idx, last_idx + 1):
                blk = blocks[j]
                if blk.is_null or blk.ref_cnt != 1:
                    return False
                if blk.block_hash is not None:
                    continue  # ordinary hash: resolved through the hash chain
                interior = j != last_idx and j * bsd >= start and j < len(hashes)
                if interior:
                    actions.append(
                        ("real", j, blk, make_block_hash_with_group_id(hashes[j], gid))
                    )
                else:
                    actions.append(("run", j, blk, _run_key(tail_hash, gid, j)))
            plan[gid] = (first_idx, last_idx, actions)
        return plan

    # ----------------------------------------------------------------- lookup
    def lookup(
        self,
        request: Request,
        max_length: int,
        with_draft: bool,
        count: bool = True,
    ) -> TailHit | None:
        """Best tail hit with length <= max_length, or None. Read-only."""
        if not self.enabled or not self._eligible(request):
            return None
        first = count and request.request_id not in self._seen
        if first:
            self._seen.add(request.request_id)
        hit, outcome = self._lookup(request, max_length, with_draft)
        if first:
            gs = ""
            if hit is not None:
                gs = ",".join(str(g) for g in hit.tail_group_ids + hit.draft_group_ids)
            self._bump(outcome, request, hit.length if hit is not None else max_length, gs)
            self._maybe_log()
        return hit

    def _lookup(
        self, request: Request, max_length: int, with_draft: bool
    ) -> tuple[TailHit | None, str]:
        c = self.coordinator
        lcm = self.lcm
        hs = c.hash_block_size
        n_hashed_aligned = len(request.block_hashes) * hs // lcm
        kmax = min(max_length // lcm, n_hashed_aligned)
        outcome = "tail_miss_absent"
        for k in range(kmax, -1, -1):
            parent = self._parent_of(request, k * lcm)
            if parent is None:
                continue
            ent = self.index.get(parent)
            if ent is None:
                continue
            if ent.length > max_length:
                outcome = "tail_miss_len"
                continue
            aligned = k * lcm
            want = compute_tail_hash(
                parent, request.all_token_ids[aligned : ent.length], ent.length
            )
            if want != ent.tail_hash:
                outcome = "tail_miss_hash"
                continue
            if with_draft and self._draft_gids and ent.draft is None:
                outcome = "tail_miss_draft"
                continue
            hit = self._build_hit(request, k, parent, ent, with_draft)
            if hit is None:
                outcome = (
                    "tail_miss_draft"
                    if with_draft and self._draft_gids
                    and self._last_fail == "draft"
                    else "tail_miss_evicted"
                )
                continue
            return hit, "tail_hit"
        return None, outcome

    _last_fail = ""

    def _build_hit(
        self, request: Request, k: int, parent: bytes, ent: _Entry, with_draft: bool
    ) -> TailHit | None:
        c = self.coordinator
        pool = self.pool
        lcm = self.lcm
        num_groups = len(c.single_type_managers)
        out: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        tail_blocks: dict[int, KVCacheBlock] = {}
        for gid in ent.group_ids:
            key = make_block_hash_with_group_id(ent.tail_hash, gid)
            blk = pool.cached_block_hash_to_block.get_one_block(key)
            if blk is None:
                self._last_fail = "tail"
                return None
            tail_blocks[gid] = blk
        if set(ent.group_ids) != set(self._full_gids) | set(self._mamba_gids):
            self._last_fail = "tail"
            return None
        hs = c.hash_block_size
        if k > 0:
            for spec_group in c.attention_groups:
                gids = [g for g in spec_group.group_ids if g in self._full_gids]
                if not gids or spec_group.manager_cls is not FullAttentionManager:
                    continue
                spec = spec_group.spec
                bh = (
                    request.block_hashes
                    if spec.block_size == hs
                    else BlockHashListWithBlockSize(
                        request.block_hashes, hs, spec.block_size
                    )
                )
                got = FullAttentionManager.find_longest_cache_hit(
                    block_hashes=bh,
                    max_length=k * lcm,
                    kv_cache_group_ids=gids,
                    block_pool=pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=False,
                    alignment_tokens=lcm,
                )
                if any(len(b) != k for b in got):
                    self._last_fail = "tail"
                    return None
                for gid, blocks in zip(gids, got):
                    out[gid] = list(blocks)
        for gid in self._full_gids:
            out[gid].append(tail_blocks[gid])
        for gid in self._mamba_gids:
            out[gid] = [pool.null_block] * k + [tail_blocks[gid]]
        owned = [make_block_hash_with_group_id(ent.tail_hash, g) for g in ent.group_ids]
        draft_gids: tuple[int, ...] = ()
        if with_draft and self._draft_gids and ent.draft is not None:
            for gid, run in ent.draft.items():
                bsd = c.single_type_managers[gid].block_size
                hashes = self._page_hashes(request, bsd)
                pages: list[KVCacheBlock] = []
                for j in range(run.first_idx, run.last_idx + 1):
                    key = run.run_keys.get(j)
                    if key is not None:
                        blk = pool.cached_block_hash_to_block.get_one_block(key)
                        if blk is not None:
                            owned.append(key)
                    else:
                        got = (
                            pool.get_cached_block(hashes[j], [gid])
                            if j < len(hashes)
                            else None
                        )
                        blk = got[0] if got else None
                    if blk is None:
                        self._last_fail = "draft"
                        return None
                    pages.append(blk)
                out[gid] = [pool.null_block] * run.first_idx + pages
            draft_gids = tuple(ent.draft)
        self._last_fail = ""
        return TailHit(
            length=ent.length,
            num_aligned_blocks=k,
            parent=parent,
            owned_keys=owned,
            tail_group_ids=ent.group_ids,
            draft_group_ids=draft_gids,
            blocks=tuple(out),
            bias=ent.bias,
        )

    # --------------------------------------------------------------- consume
    def consume(self, request: Request) -> None:
        """The request now owns the tail blocks: unhash them (called after the
        blocks were touched by allocate_new_computed_blocks)."""
        hit = self.pending.pop(request.request_id, None)
        if hit is None:
            return
        rid = request.request_id
        self._forget(hit.parent, evict_pool=False)
        for key in hit.owned_keys:
            blk = self.pool.cached_block_hash_to_block.get_one_block(key)
            if blk is not None:
                self.pool._maybe_evict_cached_block(blk)
        mgrs = self.coordinator.single_type_managers
        for gid in hit.tail_group_ids + hit.draft_group_ids:
            mgr = mgrs[gid]
            tp = (hit.length - 1) // mgr.block_size  # page holding token T-1
            blk = mgr.req_to_blocks[rid][tp]
            # The tail page becomes a normal full page once it fills up: let
            # cache_blocks() hash it then (unless it kept an ordinary hash).
            mgr.num_cached_block[rid] = tp if blk.block_hash is None else tp + 1
        request.renew_tail_bias = hit.bias

    def on_request_free(self, request_id: str) -> None:
        self._seen.discard(request_id)
        self.pending.pop(request_id, None)
