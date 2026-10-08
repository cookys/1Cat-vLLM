# m37b: deadline, partial evidence, and per-group capacity audit

Date: 2026-10-09. CPU-only follow-up to
[the parking preregistration](c4140_idle_kv_host_parking.md).
Base `7b72034ec`; m37b actually ran `addf06bb4`. No GPU/service actions, pool
resizing, cache policy changes, or numerical changes are part of this delta.

## What stopped m37b

At `addf06bb4:benchmarks/host_parking_window.py:562–579`, startup has its own
900-second limit; after health/NUMA inspection, every arm gets
`signal.alarm(2700)`. That is **45 minutes after ready**, separate from the
shell's 10,800-second outer timeout. `on1` includes 100 serial 262000-token
returns and 10 bursts of ten 32768-token returns (`:399–425`); ordinary arms
run only one of each. Cases, returns and copy counters were saved **after**
all returns (`:714–738`). Missing files therefore do not imply missing work.

Evidence: `/data/bench/m37b/window-error.json` says interruption 14;
`cgroup-events.json` has OOM delta 0. Offline audit and input SHA256s:
`/data/bench/astra-m37b-timeout/{analyze.py,analysis.json}`.

| Observation | Evidence | Meaning |
|---|---|---|
| on1 ready/NUMA file 03:52:13; expected alarm 04:37:13 | `analysis.json`, file-mtime proxies | Consistent with the 2700-second arm alarm, not the outer cap |
| off1/on1 traffic each 84 completed, zero errors | `m37b/{off1,on1}.traffic.json` | Both traffic stages completed; one pair cannot close the three-pair interference gate |
| 100 successful 258048-token loads; scheduler-arrival p50 0.4490 s, p99 0.4545 s | `on1.serve.log:5222–7321`, `analysis.json` | Genuine c1 transfer events; client-arrival records are missing, so these cannot replace gate 3 |
| c1 arrival interval median 11.9754 s | same events | Transfer time alone understates repeat cost: settle, tail recompute and driver work remain |
| c10 phase: one 28672-token success after generation 18 | `on1.serve.log:7340,7499` | No per-group miss counters existed; it does not identify which group caused other misses |

The projected full on1 duration is **58.1 minutes including startup**, assuming
c10 misses cost the observed OFF cold burst. Components: suite/reset 240 s,
traffic/settle 724 s, c1 cold producer/reset 145 s, 100 c1 cycles 1198 s,
c10 producer/reset 91 s, ten cold-return bursts 847 s, startup 241 s.
These are scenario estimates from `analysis.json:on1_budget_projection`, not
a completed ON measurement. An on1 arm is expected to take longer than OFF
because of unequal repetition counts; this alone is not a parking slowdown.

## Driver and diagnostic changes

- `host_parking_window.py:38–58,670–678`: arm caps are on1 **5400 s**, other
  normal arms **2700 s**, fault arms **1500 s**, with CLI overrides. The shell
  asks `--budget-only` for an outer cap covering preflight, raw probe, every
  selected arm, startup and cleanup. `WindowInterrupted(BaseException)` cannot
  be swallowed by the ordinary request `except Exception` handler.
- `--arms` selects normal/fault arms without renaming them. Normal arms must
  precede fault arms; duplicate names are rejected. The default full sequence
  and original sample counts remain unchanged.
- Atomic cases checkpoints after each parity/chain response; returns after
  each burst; recovery cases after each response/abort; copy counters directly
  after traffic. `phases.jsonl` records begin/complete/interrupted with wall and
  monotonic timestamps. `<arm>.deadline.json` records the armed deadline.
- `--reuse-from` copies only nonselected-arm results/logs and, with
  `--skip-raw`, four prior raw results. `reused-evidence.json` records source
  paths and SHA256. Old evidence is not relabelled as independent replicates.
  Fault judging now finds cases by label and refuses partial checkpoints.
- `--parking-diagnostics` adds opt-in connector extra config
  `parking_diagnostics=true` (default absent/false). `ParkingManager.lookup`
  records absent/pending/invalid lookups by group, capacity, and used slots at
  first lookup. `build_connector_meta` samples `ParkingManager.stats()` once
  per scheduler step after store reservations. `HOST_PARKING step_stats` emits
  current used/capacity slots, bytes, in-flight refs, miss counts and reported
  evictions. It never prints hashes or request IDs.
- Reported evictions are a **lower bound** if a grouped reservation fails:
  the native manager may already have evicted another group before returning
  `None` (`cpu/manager.py:317–332`). A separate failed-reservations count and
  actual used-slot sample expose this case; diagnostics do not repeat lookup
  or touch/reorder LRU. Disabled diagnostics perform no pool scan.
- Diagnostic scans/logging can affect timing. Gate 4 is mechanically
  **INCONCLUSIVE** when enabled, retaining the measured subreport. Quantiles
  p50<=2 s/p99<=4 s, 100 returns per regime, and three OFF/ON pairs are unchanged.

Runtime overlay relative to `7b72034ec`: `vllm/v1/kv_offload/cpu/parking_manager.py`,
`parking_spec.py`, and
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`.
Driver, judge and capacity calculator are standalone `benchmarks/` files.
The prior preflight fixes remain inherited. Do not copy the whole worktree
onto the serving PYTHONPATH: the driver creates a staging directory containing
only its `vllm` symlink, preserving installed native libraries.

## 10 × 32K: total budget fits, current group quotas do not

`benchmarks/host_parking_capacity.py:analyze` uses the actual
`SchedulerOffloadConfig`, native retention mask and `HostParkingSpec` geometry.
Input fixture is `tests/v1/kv_offload/cpu/test_host_parking.py:real_layout`,
with `mamba_state_slots_reference_tokens=32768`, matching
`m37b/on1.serve.log:7,940–944,3171`. Output:
`/data/bench/astra-m37b-timeout/capacity.json`. No pinned/device allocation.

| Groups | Physical slot / rank | Current slots per group | 10 independent completed 32K prompts need | Deficit per group |
|---|---:|---:|---:|---:|
| GDN 0–5, eight layers each | 9 MiB | 36 | 20 | 0 |
| Full attention 6–7, eight layers each | 9 MiB | 91 | 80 | 0 |
| Draft SW 8, five layers | 5.625 MiB | 91 | 160 | **69** |

Important correction to the initial GDN hypothesis: `retention_interval=0`
does **not** send all eight intermediate GDN states. The core hashes only the
reachable terminal boundaries; the handoff skips blocks without a hash.
`kv_cache_coordinator.py:104–137`,
`single_type_kv_cache_manager.py:1499–1532`, `block_pool.py:271` implement this.
The native MambaManager regression steps through all eight chunks and observes
only **28672 and 32768** handoffs. The fixture uses DFlash2, not EAGLE drop.

In contrast, draft SW uses 1024-token blocks, a two-block tail and a four-block
alignment segment (`offloading/scheduler.py:131–225,892–913`). Each of eight
segments can store its trailing two blocks: **16/request**, not eight and
not just two. The fixed allocator currently gives FA and SW the same *number*
of slots (`cpu/spec.py:142–192`), despite this 2:1 demand for completed 32K
producers. This predicts a draft-SW bottleneck even when GDN fits.

Physical storage across TP4 (padding included):

- Current allocation: **15.99170 GiB** (17,170,956,288 bytes).
- All sparse-GDN + FA + historical-SW data above: **13.359375 GiB**.
- Only one jointly-restorable boundary at 28672 for all ten prompts:
  **7.470703125 GiB**. This is a pruning-policy lower bound, not current behavior.
- Candidate quota vector `[30]*6 + [80]*2 + [160]`:
  **15.46875 GiB**, allowing three GDN states/request (two terminals plus a
  junction), but no spare FA/SW history beyond this workload. It is a sizing
  example, **not a universal safe configuration or implemented change**.

Thus **16 GiB is enough in aggregate for this independent-32K scenario**;
increasing total RAM is not the first fix. Extra old sessions, active copies,
shared-prefix junctions, prompt-length heterogeneity and event-pinned entries
change occupancy, so the table is not a general ten-session admission guarantee.

Native LRU counterexample (`test_native_sw_lru_cyclic_miss_and_capacity_counterfactual`):
ten producers, 16 SW keys each, then returns in the same sequential order;
request B=28672 needs draft pages 26/27. At 91 slots, return misses and re-stores
continually evict later requests: **0/10** hit. At 160 slots: **10/10** hit.
This demonstrates the mechanism; the real concurrent GPU completion order was
not replayed. It does not prove the exact m37b missed group or a 28% missing-key
fraction. The new per-step diagnostic is the decisive runtime evidence.

Proposed fixes, ordered:

1. Size independent group pools by native token grid and *host* retention
   demand. FA:SW demand is 1:2 here, with sparse GDN separately budgeted.
   Keep the aggregate 16 GiB ceiling; use optional workload quotas and validate
   identical scheduler/worker layouts before allocating. No such policy change
   is in this patch; obtain group-miss confirmation first.
2. Retain a complete checkpoint bundle across groups, pruning intermediate
   SW suffixes with no retained GDN state. Per-entry events/refcounts and shared
   hashes must be respected; delete only after all TP jobs/readers release.
   This needs ownership/indexing and new eviction/generation tests.
3. Request-aware admission or protective LRU can reduce cyclic churn but cannot
   manufacture slots. Arbitrary LRU reorder alone offers no ten-resident guarantee.

## Lead-only minimal rerun

Do not edit the old `/data/bench/ab/m37-parking.sh` or reuse an existing output
directory. The external wrapper owns production stop/restore. This command
is a recipe only; Astra has not executed it:

```sh
M37_GO=1 M37_WORKTREE=/data/src/1cat-wt-astra-parking-rerun \
  M37_OUT=/data/bench/m37c \
  bash /data/src/1cat-wt-astra-parking-rerun/benchmarks/host_parking_window.sh \
  --execute --arms on1 fault-pre_submit fault-completed_copy \
  --reuse-from /data/bench/m37b --skip-raw --parking-diagnostics
```

This selects on1 + both faults, reuses G1/OFF1 evidence with hashes, and keeps
the original 16 GiB / 1800 GPU blocks / P8 OFF recipe. Estimated **90–110 min**:
on1 ~58 min from the observed-stage model; faults provisionally 15–25 min each
(unmeasured, capped). Hard outer timeout is **12,360 s / 206 min**, deliberately
larger than those estimates. Adding `on2` after `on1` adds roughly **28 min**
(OFF1 startup+cell proxy), and raises the hard cap by 3840 s. Actual next-run
phase timestamps will replace these estimates.

For shortest group-miss reconnaissance, `--arms on2` uses one burst/regime
instead of 100 returns and is about the observed **28-min** OFF cell; no p99
claim can follow. Current quota may again fail c10; this is an intentional
diagnostic, not a promised capacity PASS. A subsequent policy A/B must disable
diagnostics and run all three normal pairs. A subset never closes the absent
OFF2/OFF3/ON3 parity, utility, memory or interference gates.

## CPU validation

Exact serving-venv invocation and outputs are recorded in
`/data/bench/astra-m37b-timeout/tests-full-final.log`. The command is:

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 PYTHONPATH=/data/src/1cat-wt-astra-parking-rerun:/data/bench/astra-host-parking/test-deps \
 /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
 -p tests.v1.kv_connector.unit.offloading_connector.conftest \
 tests/v1/kv_offload/cpu/test_host_parking.py \
 tests/v1/kv_offload/cpu/test_host_parking_window.py \
 tests/v1/kv_offload/cpu/test_host_parking_rerun.py \
 tests/v1/kv_offload/cpu/test_manager.py \
 tests/v1/kv_offload/cpu/test_grouped_manager.py \
 tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
 tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py -q
```

225 passed. Coverage includes real geometry/GDN handoffs/native LRU, diagnostic
lookup equivalence (including pending and partially rolled-back reservations),
opt-in gating, deadline propagation, partial checkpoints, evidence provenance,
arm selection and refusal to pass missing/instrumented evidence. CPU tests
cannot establish GPU stream correctness, real c10 hit rates or timing.

Real CPU import preflight: `astra-m37b-timeout/live-preflight/` contains
`cpu-smoke.imports.json` PASS (`cuda_initialized=false`, vllm-only staging,
all other packages/native binaries resolve into the serving venv), plus the
expected `old-path-rejected.json` negative control. `rerun-plan.json` records
the nonexecuting arm/argv/deadline plan. Ruff, shell syntax and diff checks pass.
