# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host reference for the Q2.16 option C ("NoPlan") expert grouping.

``csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu`` ``w13_kernel<S, I, true>``
derives each expert group from the route ids with warp ballots instead of reading
the output of ``plan_kernel``. This module replays that exact lane-level procedure
(32-lane ballot / match_any / shfl semantics, the same bit operations) in numpy so
the grouping rules can be checked on a CPU, and provides the routing generators
shared by the CPU and GPU tests.

Semantics being mirrored (see the kernel's header comment):
  * ids <0 or >=512 share the bucket 512 (zero accumulators, zero rows written);
  * route slot ``t`` leads a group iff it is the 8k-th occurrence of its expert
    in slot order; the group's rows are the next <=8 slots of that expert;
  * group numbers are leader ranks in ascending slot order; ``total`` is written
    by slot 0 only.
``plan_kernel`` assigns group numbers and in-group row order through atomicAdd
races, so those are unspecified there; nothing downstream depends on them, so the
comparison against it is by group content, not by number.
"""

from __future__ import annotations

import numpy as np

EXPERTS = 512
PACK = 8
FULL = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


def sanitize(expert: int) -> int:
    return EXPERTS if (expert < 0 or expert >= EXPERTS) else int(expert)


def popc(x: int) -> int:
    return bin(x).count("1")


def nth_set_bit(mask: int, n: int) -> int:
    """Mirror of the device helper: clear the n lowest set bits, then ffsll - 1."""
    for _ in range(n):
        mask &= mask - 1
    assert mask != 0, "mask must have more than n set bits"
    return (mask & -mask).bit_length() - 1


def ballot(pred: np.ndarray) -> int:
    return sum(1 << lane for lane in range(32) if pred[lane])


def match_any(vals: np.ndarray) -> np.ndarray:
    """__match_any_sync(full, v): per-lane mask of lanes holding an equal value."""
    eq = vals[:, None] == vals[None, :]
    return np.array([ballot(eq[lane]) for lane in range(32)], dtype=np.int64)


def lane_ids(ids: list[int], routes: int) -> tuple[np.ndarray, np.ndarray]:
    id_a = np.full(32, -1, dtype=np.int64)
    id_b = np.full(32, -1, dtype=np.int64)
    for lane in range(32):
        if lane < routes:
            id_a[lane] = sanitize(ids[lane])
        if lane + 32 < routes:
            id_b[lane] = sanitize(ids[lane + 32])
    return id_a, id_b


def cta(t: int, ids: list[int], routes: int):
    """One w13 CTA (blockIdx.y == t, blockIdx.x == 0). None if it exits as non-leader.

    Returns the CTA's group (expert, count, per-lane route table) and the global
    writes its warp 0 performs: rows[index] = value, experts[g], sizes[g], total.
    """
    id_a, id_b = lane_ids(ids, routes)
    # --- prologue, every warp ---
    expert = int((id_a if t < 32 else id_b)[t & 31])  # __shfl_sync broadcast
    same = ballot(id_a == expert) | (ballot(id_b == expert) << 32)
    below = (1 << t) - 1
    ordinal = popc(same & below)
    if ordinal % PACK != 0:
        return None
    count = min(PACK, popc(same) - ordinal)
    rest = same & ~below & MASK64
    # --- per-lane row (mma_row); lanes with quad == 0 publish slot_rows ---
    route = np.zeros(32, dtype=np.int64)
    mma_row = np.array([(lane & 3) + (4 if lane & 16 else 0) for lane in range(32)])
    quad = np.array([(lane >> 2) & 3 for lane in range(32)])
    for lane in range(32):
        if mma_row[lane] < count:
            route[lane] = nth_set_bit(rest, int(mma_row[lane]))
    slot_rows = {int(mma_row[lane]): int(route[lane]) for lane in range(32) if quad[lane] == 0 and mma_row[lane] < count}
    assert sorted(slot_rows) == list(range(count))
    # --- epilogue emission (warp 0, blockIdx.x == 0) ---
    lane_ix = np.arange(32)
    low = ((1 << lane_ix) - 1) & FULL
    same_a, same_b = match_any(id_a), match_any(id_b)
    earlier_a = np.zeros(32, dtype=np.int64)
    for j in range(32):  # earlier_a += __shfl_sync(full, idA, j) == idB
        earlier_a += id_a[j] == id_b
    lead_a = np.array([id_a[i] >= 0 and popc(int(same_a[i]) & int(low[i])) % PACK == 0 for i in range(32)])
    lead_b = np.array(
        [id_b[i] >= 0 and (int(earlier_a[i]) + popc(int(same_b[i]) & int(low[i]))) % PACK == 0 for i in range(32)]
    )
    lead_lo, lead_hi = ballot(lead_a), ballot(lead_b)
    if t < 32:
        g = popc(lead_lo & ((1 << t) - 1))
    else:
        g = popc(lead_lo) + popc(lead_hi & ((1 << (t - 32)) - 1))
    rows_writes = {g * PACK + int(mma_row[lane]): int(route[lane]) for lane in range(32) if quad[lane] == 0 and mma_row[lane] < count}
    total = popc(lead_lo) + popc(lead_hi) if t == 0 else None
    return {
        "expert": expert,
        "count": count,
        "g": g,
        "slot_rows": slot_rows,
        "rows": rows_writes,
        "total": total,
    }


def run_all(ids: list[int], routes: int) -> dict:
    """All 5*routes CTAs of one call (only tile 0 emits; the others compute rows)."""
    rows: dict[int, int] = {}
    experts: dict[int, int] = {}
    sizes: dict[int, int] = {}
    totals: list[int] = []
    leaders = []
    for t in range(routes):
        r = cta(t, ids, routes)
        if r is None:
            continue
        leaders.append(t)
        g = r["g"]
        assert g not in experts, f"two leaders claim group {g}"
        experts[g], sizes[g] = r["expert"], r["count"]
        for k, v in r["rows"].items():
            assert k not in rows, f"rows[{k}] written twice"
            rows[k] = v
        if r["total"] is not None:
            totals.append(r["total"])
    assert len(totals) == 1, "total must be written by exactly one CTA (slot 0)"
    return {"rows": rows, "experts": experts, "sizes": sizes, "total": totals[0], "leaders": leaders}


def plan_reference(ids: list[int], routes: int) -> list[tuple[int, tuple[int, ...]]]:
    """What plan_kernel produces up to its unspecified order: (expert, rows) groups.

    Rows of one expert are chunked by 8 in ascending slot order here; plan_kernel's
    chunking/order is an atomicAdd race, identical as a set whenever an expert has
    <=8 rows (always so for distinct per-token top-k at M=5, at most 5 rows).
    """
    by_expert: dict[int, list[int]] = {}
    for t in range(routes):
        by_expert.setdefault(sanitize(ids[t]), []).append(t)
    groups = []
    for e, slots in by_expert.items():
        for i in range(0, len(slots), PACK):
            groups.append((e, tuple(slots[i : i + PACK])))
    return sorted(groups)


def plan_kernel_mirror(ids: list[int], routes: int, rng: np.random.Generator) -> dict:
    """Old plan_kernel with its atomicAdd races resolved in a random arrival order.

    Thread t's ordinal is the arrival rank among threads of the same expert; a
    group number is handed out in arrival order to each thread with ordinal % 8 == 0.
    Any permutation is a legal schedule, so the result is one sample of what the
    kernel may produce; ``rows`` is indexed like the kernel's output.
    """
    arrival = rng.permutation(routes)  # thread ids in the order their atomics land
    counts: dict[int, int] = {}
    ordinal = [0] * routes
    for t in arrival:
        e = sanitize(ids[t])
        ordinal[t] = counts.get(e, 0)
        counts[e] = ordinal[t] + 1
    leaders = [t for t in rng.permutation(routes) if ordinal[t] % PACK == 0]
    group_of, experts, sizes = {}, {}, {}
    for g, t in enumerate(leaders):
        e = sanitize(ids[t])
        group_of[(e, ordinal[t] // PACK)] = g
        experts[g] = e
        sizes[g] = min(PACK, counts[e] - ordinal[t])
    rows = {}
    for t in range(routes):
        e = sanitize(ids[t])
        rows[group_of[(e, ordinal[t] // PACK)] * PACK + ordinal[t] % PACK] = t
    return {"rows": rows, "experts": experts, "sizes": sizes, "total": len(leaders)}


def equivalent_to_plan_kernel(new: dict, old: dict) -> None:
    """New leader-rank plan vs one old-kernel schedule: same groups as sets.

    Experts with <=8 rows (every distinct-top-k M=5 call): identical row sets.
    More rows: which rows share a pack is schedule-dependent in the old kernel, so
    compare per expert the multiset of pack sizes and the union of rows.
    """
    def per_expert(out):
        d: dict[int, list[tuple[int, ...]]] = {}
        for g in range(out["total"]):
            rs = tuple(sorted(out["rows"][g * PACK + j] for j in range(out["sizes"][g])))
            assert 1 <= len(rs) <= PACK
            d.setdefault(out["experts"][g], []).append(rs)
        return d

    a, b = per_expert(new), per_expert(old)
    assert a.keys() == b.keys()
    for e in a:
        assert sorted(map(len, a[e])) == sorted(map(len, b[e])), e
        assert sorted(r for rs in a[e] for r in rs) == sorted(r for rs in b[e] for r in rs), e
        if sum(map(len, a[e])) <= PACK:
            assert sorted(a[e]) == sorted(b[e]), e


def groups_from_arrays(out: dict) -> list[tuple[int, tuple[int, ...]]]:
    groups = []
    for g in range(out["total"]):
        c = out["sizes"][g]
        groups.append((out["experts"][g], tuple(out["rows"][g * PACK + j] for j in range(c))))
    return sorted(groups)


def check_plan(ids: list[int], routes: int) -> dict:
    out = run_all(ids, routes)
    total = out["total"]
    assert sorted(out["experts"]) == list(range(total)), "group numbers must be dense 0..total-1"
    assert len(out["leaders"]) == total
    got = groups_from_arrays(out)
    # no row missing, none duplicated
    flat = sorted(r for _, rs in got for r in rs)
    assert flat == list(range(routes)), "every route slot must appear exactly once"
    for e, rs in got:
        assert 1 <= len(rs) <= PACK
        assert all(sanitize(ids[r]) == e for r in rs)
        assert list(rs) == sorted(rs)
    assert got == plan_reference(ids, routes)
    # group number == rank of the leader (first row) in ascending slot order
    first_rows = [out["rows"][g * PACK] for g in range(total)]
    assert first_rows == sorted(first_rows) == sorted(out["leaders"])
    return out


# --------------------------------------------------------------- routing makers


def routing_with_multiplicity(
    u: int, seed: int, mode: str = "uniform", rows: int = 5, top_k: int = 10, experts: int = EXPERTS
) -> np.ndarray:
    """[rows, top_k] distinct-per-row ids whose union is exactly ``u`` experts.

    How many tokens route to an expert (its row count 1..rows) follows ``mode``:
    ``uniform`` spreads the surplus over random experts, ``skewed`` piles it onto
    as few experts as possible (as many 5-row experts as fit), ``staircase`` gives
    expert i a cap of 1 + i % rows. Then the token/expert incidence is built by
    degree-constrained greedy filling (largest multiplicity first onto the tokens
    with the most free slots), so every token holds ``top_k`` distinct experts.
    """
    total = rows * top_k
    if not (top_k <= u <= min(total, experts)):
        raise ValueError(f"U={u} infeasible for rows={rows}")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        mult = np.ones(u, dtype=int)
        extra = total - u
        if mode == "uniform":
            while extra > 0:
                i = int(rng.integers(u))
                if mult[i] < rows:
                    mult[i] += 1
                    extra -= 1
        elif mode == "skewed":
            for i in rng.permutation(u):
                add = min(rows - 1, extra)
                mult[i] += add
                extra -= add
                if extra == 0:
                    break
        elif mode == "staircase":
            cap = 1 + (rng.permutation(u) % rows)
            while extra > 0:
                moved = False
                for i in range(u):
                    if extra > 0 and mult[i] < cap[i]:
                        mult[i] += 1
                        extra -= 1
                        moved = True
                if not moved:
                    cap = np.full(u, rows)
        else:
            raise ValueError(mode)
        assert mult.sum() == total and mult.max() <= rows
        free = np.full(rows, top_k)
        picks: list[list[int]] = [[] for _ in range(rows)]
        ok = True
        pool = rng.permutation(experts)[:u]
        for e_idx in np.argsort(-mult, kind="stable"):
            m = int(mult[e_idx])
            order = sorted(range(rows), key=lambda r: (-free[r], rng.random()))
            chosen = order[:m]
            if any(free[r] == 0 for r in chosen):
                ok = False
                break
            for r in chosen:
                picks[r].append(int(pool[e_idx]))
                free[r] -= 1
        if ok and not free.any():
            return np.array([rng.permutation(p) for p in picks], dtype=np.int32)
    raise RuntimeError("could not build routing")


def multiplicity_histogram(ids: np.ndarray) -> dict[int, int]:
    _, counts = np.unique(ids, return_counts=True)
    return {int(m): int((counts == m).sum()) for m in sorted(set(counts.tolist()))}
