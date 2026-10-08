# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in grouped slot ratios. Pure integer CPU arithmetic, before any pinning."""


def proportional_group_slots(page_sizes, ratios, budget_per_rank):
    """Spend the same byte budget in proportion to requested *slot* ratios.

    First floor a common fractional multiplier; then distribute at most one
    extra slot per group in descending fractional-remainder order, with stable
    group-ID ties. A large page that does not fit is skipped. No floating point,
    overcommit, worker-local observation, or runtime resizing is involved.
    """
    expected = {str(g) for g in page_sizes}
    if not isinstance(ratios, dict) or set(ratios) != expected:
        raise ValueError("parking_group_slot_ratios must name every group exactly")
    if any(type(ratios[str(g)]) is not int or ratios[str(g)] <= 0 for g in page_sizes):
        raise ValueError("parking_group_slot_ratios values must be positive integers")
    if type(budget_per_rank) is not int or budget_per_rank <= 0:
        raise ValueError("positive per-rank byte budget required")
    if not page_sizes or any(type(p) is not int or p <= 0 for p in page_sizes.values()):
        raise ValueError("positive physical page sizes required")
    weight = {g: ratios[str(g)] for g in page_sizes}
    unit_bytes = sum(page_sizes[g] * weight[g] for g in page_sizes)
    slots = {g: budget_per_rank * weight[g] // unit_bytes for g in page_sizes}
    if min(slots.values()) < 1:
        raise ValueError("ratio budget must fit at least one slot in every group")
    remaining = budget_per_rank - sum(page_sizes[g] * slots[g] for g in page_sizes)
    for g in sorted(
        page_sizes, key=lambda g: (-(budget_per_rank * weight[g] % unit_bytes), g)
    ):
        if page_sizes[g] <= remaining:
            slots[g] += 1
            remaining -= page_sizes[g]
    assert sum(page_sizes[g] * slots[g] for g in slots) <= budget_per_rank
    return slots
