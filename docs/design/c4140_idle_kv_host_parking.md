# 27B NVFP4 + DFlash2 bounded host parking PoC

Status (2026-10-09): CPU PoC; lead reports m37 raw-byte PASS, serving gates
INCONCLUSIVE after startup import failure. m37b rerun awaits review/window.
Base: de01359b8 (P7 + prefix retention fix + P8 code, P8 default OFF).
Owner authorization: fleet 01M4E13FYEVE19811XNEKNSETW;
completed-boundary/recovery clarification: 01M4E1BXKF98QD6ERVWXP13AQG.

## Scope and invariants

V1 uses **completed-boundary immediate offload**, not a 30/120-second idle
timer. The latter can miss a prefix already evicted before the timer fires.
Native offload saves committed prompt KV/state as it becomes available;
the GPU cache can subsequently reclaim it. It does not increase the active
GPU working-set limit. No disk tier, persistent snapshots, or requantization.

Opt-in is `OffloadingConnector` + custom `HostParkingSpec` +
`host_parking=true`; absent this configuration the serving path is unchanged.
Budget is **16 GiB TP4 total by default**, hard maximum 32 GiB. Above 25 GiB
requires prior lead/las notice and `parking_large_pool_ack=true`. No large
pool is allocated by CPU tests. CPU tests use ordinary byte tensors/mocks.

The native scheduler's `_build_boundary_state_store_jobs` consumes exact
Mamba handoffs; `_build_store_jobs` excludes positional Mamba tables. Native
worker D2H is deferred until the next step and waits on the compute stream.
Host keys become readable only after all TP worker completions. H2D waits
on destination initialization; the request stays WAITING_FOR_REMOTE_KVS
until its load events have completed. See native `offloading/scheduler.py`,
`offloading/worker.py`, `cpu/gpu_worker.py` at this branch.

Parking adds a namespace (model/revision, quant, group geometry, physical
pages, fixed unit layer-scale contract) to the prefix hash and group key.
In-page NVFP4 block scales are part of the bytes. Worker startup checks actual
K/V layer scales equal 1 and rejects dynamic scale calculation. Cache reset
increments generation; old completion IDs cannot publish into a new cache.
Cross-engine sharing is forbidden. GPU block reuse stays fenced by native
pending-job tracking; completed requests additionally retain page ownership
until all TP store events finish. An aborted load retains destination ownership
until all ranks finish, then frees it without serving any data.

Copy error recovery is **snapshot discard + cold recompute**, not engine
termination. A partial-copy failure must first drain the device, then report
failure across TP and invalidate the host key. Hybrid load failure invalidates
the whole external hit including GDN; no partially restored state is published.
If device synchronization itself fails, safe ownership cannot be established:
log `HOST_PARKING fatal_copy_context total=...` and fail the engine. A lost TP
process likewise requires engine teardown; no worker-local continue is safe.
Neither a failed copy nor an abort may reuse an in-flight buffer.

## GPU preregistration (freeze before the first window)

All tests are run by lead. Production baseline is current **P7 batched**;
P8 `mixed_prefill_step_latency_ms=0`, no mixed-prefill CLI flags. Use identical
model/draft revisions, quant, block sizes, graphs, seed, prompt order, pool IDs,
and AOT-loader status on both arms. Record all argv and source/extension hashes.
S is the primary steady/interfered-window recorder; R is an annotation only.
Do not mix recorder denominators or claim a lower concurrency arm won.

1. **Raw bytes: zero mismatches.** All TP ranks and all cache groups (target
   NVFP4 data+scales, committed GDN, draft SW) round-trip through native
   handlers to distinct destination IDs. Compare valid physical group payload,
   including state and scales; report allocator padding separately. GPU raw-copy scope: one block and three shuffled blocks per group, three
   slot-reuse cycles. It does NOT prove scheduler reset/abort. Short token
   tails are not committed full-block snapshots and must not be restored;
   serving boundary±1, reset generation and abort are separate gate-7 cases. Any valid-byte mismatch is FAIL, not an E2 candidate.
2. **Serving parity:** OFF/OFF floor and OFF/ON instances with the same production compile policy
   (do not enable a disabled AOT cache just to obtain loaders), 12 fixed-seed
   cases plus 3 chains × 4 turns per instance, 128 output tokens/call. All token
   IDs must match. For common-prefix raw API logprobs, require ON/OFF maximum
   absolute difference ≤ max(1e-6, OFF/OFF maximum); mean ≤
   max(1e-7, OFF/OFF mean). Any token mismatch rejects E1 regardless of floor. Preserve
   raw differences and do not change the rule after seeing data. Byte equality
   alone does not prove identical serving arithmetic/chunk boundaries.
3. **Restore latency:** at least 100 returns at c1 and 100 returns in c10
   bursts; both sets require arrival→all-rank event-observation **p50 ≤ 2.0 s
   and p99 ≤ 4.0 s** (engineering acceptance targets, not measured promises).
   Also report TTFT,
   actual payload, H2D and D2H separately. A 16 GiB pool cannot hold ten full
   262K images simultaneously: c10 transport uses quota-fitting images (e.g.
   32K), with full-262K c1 separate. Ten full images is a quota-rejection test,
   not a latency sample. No unapproved expansion above the pool limit.
4. **Decode interference:** c10 traffic with continuous request completions,
   so immediate-mode D2H is actually exercised. At least three cells per arm,
   fixed OFF/ON/ON/OFF/OFF/ON order with matched plans. Measure each cell's
   S steady and interfered tok/s, round latency, whole-turn p10, time interfered,
   D2H GB/s, summed copy time and union copy-stream busy fraction. Report
   H2D bursts independently. Before unblinding ON, define noise floor as the
   maximum relative pairwise spread among the three OFF cells for each S
   metric. GO requires median ON degradation no larger than this floor for
   each S decode metric; missing activity/metric = INCONCLUSIVE. R cannot
   rescue an S failure. Immediate mode can transfer much more than the idle
   simulation's offered rate; record actual bytes, never substitute that model.
5. **Useful work:** count host hits and avoided-prefill tokens/seconds against
   identical pressure/return plans, including cold first turns and quota misses.
   A capacity GO requires avoided-prefill seconds > transfer+recovery delay
   and positive median net return-TTFT gain in all three paired cells, while
   passing gates 1–4. No claim about concurrent-active capacity expansion.
6. **RAM/las:** default 16 GiB total; verify allocated/pinned bytes <= quota,
   NUMA placement, MemAvailable, memcg events, and same-PID las VmSwap before,
   peak, after. Stop if las total swap reaches the existing 2 GiB ceiling or
   grows >128 MiB from the pre-cell value, or cgroup OOM occurs. Record all
   samples; do not turn cache reclaim into an unreported service tradeoff.
7. **Failure gates:** inject pre-submit refusal and completed failed copy on
   one TP rank; all ranks must settle, affected request must recompute and
   complete, later unrelated request must work, host entry must miss, quotas
   must return to baseline. Injected fatal rank loss tests engine teardown only
   in an isolated test process. No stale memory may be served after recovery.

All speed results remain pending until these gates run. Unapproved changes to
thresholds require a new preregistration version. The same GPU window will be
coordinated by lead with the kernel line; this document starts no services.

## Implementation, budget, and evidence

- `cpu/parking.py:22` is the opt-in/config guard. No new global env defaults.
  `cpu/parking_spec.py:30` rejects non-TP4/DP, non-NVFP4, missing prefix cache,
  non-DFlash, dynamic scales, non-align state, and P8 ON before allocation.
- `cpu/parking_pool.py:10` allocates one private anonymous mmap per rank,
  page aligned, with exact-size `cudaHostRegister`. This bypasses the pinned
  caching allocator's size classes. `parking_spec.py:122` serializes TP startup
  by engine-private flock and applies per-thread MPOL_BIND(node0) during
  MADV_POPULATE_WRITE. Restore previous policy afterwards. No persistent mmap
  file, cross-process host pointers, per-request pin, or disk tier.
- Both transfer directions hold the region owner; both native shutdown paths
  must finish before unregister/unmap. Native copy dependencies are at
  `cpu/gpu_worker.py:332`. Recoverable failure drains before reporting failed
  jobs (`cpu/parking_transfer.py:27`); all-TP completion controls publication
  and release (`offloading/scheduler.py:1194`). Hybrid failed loads are discarded
  at `core/sched/scheduler.py:2784`. Reset is rejected while a parking transfer
  is pending, then increments namespace generation. This is a retryable reset,
  not an engine failure.
- `tests/v1/kv_offload/cpu/test_host_parking.py` exercises the real allocator
  and real worker canonicalization, with CPU tensors replacing device storage.
  CUDA registration/events are mocked. Tests cover TP staggered completion,
  abort, healthy-context copy failure, fatal context, exact source state,
  quota, group LRU/refcounts, rollback tombstone pruning, generation and NUMA
  policy restoration. They cannot establish hardware stream ordering.

At 16 GiB aggregate, retention_interval=0 and a 262144-token reference, the
actual allocator/spec yields (CPU calculation; no large allocation):

| Group | Count | Token block | Physical bytes/layer/page | Valid copied bytes/layer/page | Host slots/group |
|---|---:|---:|---:|---:|---:|
| committed GDN | 6 × 8 layers | 4096 | 1,179,648 | 837,632 | 9 |
| main NVFP4 full attention | 2 × 8 layers | 4096 | 1,179,648 | 1,179,648 | 152 |
| DFlash FP16 sliding window | 1 × 5 layers | 1024 | 1,179,648 | 1,048,576 | 152 |

Pinned payload = **17,100,177,408 B TP4 total**, within 17,179,869,184 B.
Each pool remains a fixed per-group quota. Full-attention data for 152 pages
is not a guarantee of 152 restorable full snapshots: every hit needs the
matching GDN state and draft SW suffix. Immediate mode retains intermediate SW
boundary suffixes too. Independent group LRU can evict one required component;
lookup then walks back or misses safely. Do not reuse the simulation's ideal
image-count table as a claim about this PoC. Measure jointly-restorable hit
rate under the real request sequence before proposing pool rebalancing or
selective checkpoint retention. Small-group capacity/eviction is a utility
risk, not permission to serve a partial snapshot.

CPU evidence (local scratch, not committed):
`/data/bench/astra-host-parking/tests-final.log`: **155 passed** for parking,
native managers, grouped pools, native offload scheduler, and worker metadata.
Import and dry-plan evidence: `import-smoke.log`, `roundtrip-plan.json` in the
same directory. No GPU tests have run.

## Reproduction and GPU window handoff

CPU command (the serving venv has no pytest; `test-deps` contains symlinks only
to existing pytest modules, not a different torch installation):

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/data/src/1cat-wt-astra-host-parking:/data/bench/astra-host-parking/test-deps \
  /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
  -p tests.v1.kv_connector.unit.offloading_connector.conftest \
  tests/v1/kv_offload/cpu/test_host_parking.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/test_grouped_manager.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py
```

Use an isolated venv/overlay derived from the same base in both arms. Python
only; no extension rebuild. Exact overlay list is obtained with
`git diff --name-only de01359b8 HEAD -- vllm`. Keep P8 flags absent, fixed GPU
pool IDs, production P7/BATCHED flags, and the same cache/draft revisions.
OFF omits the transfer config. ON adds the following single JSON CLI argument:

```text
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"spec_name":"HostParkingSpec","spec_module_path":"vllm.v1.kv_offload.cpu.parking_spec","host_parking":true,"cpu_bytes_to_use":17179869184,"parking_numa_node":0,"parking_mode":"completed_boundary","offload_prompt_only":true,"store_threshold":0,"eviction_policy":"lru"}}'
```

**Lead only, isolated GPU window**: raw copy gate below uses ~324 MiB backing
GPU memory and 310.5 MiB exact pinned memory per rank (plus small temporaries).
It does not launch a server. It tests all nine canonical groups, valid bytes
including quant scales, padding canaries, different source/destination IDs,
three slot-reuse cycles, and async compute-to-copy dependencies. Estimated
1–3 minutes including four Python startups, unmeasured. Default invocation
without `--execute-gpu` is CPU-only and emits `SKIP_GPU`.

```sh
PYTHONPATH=/data/src/1cat-wt-astra-host-parking \
  CUDA_VISIBLE_DEVICES=0,1,2,3 TMPDIR=/data/tmp \
  /data/venvs/1cat-main-integp7pr-p8/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  /data/src/1cat-wt-astra-host-parking/benchmarks/host_parking_roundtrip.py \
  --execute-gpu --output '/data/bench/astra-host-parking/roundtrip-rank{rank}.json'
```

This synthetic gate is prerequisite, not a substitute for completed GDN state
and serving parity. Follow the seven preregistered gates above: pressure-evict
A from GPU while keeping host entries, then return A; observe nonzero
`CPU_to_GPU` payload and external hits, and compare cold recomputation output.
Abort/copy refusal and TP loss require isolated fault-injection harnesses;
CPU mocks alone do not close those GPU gates. Do not promote this branch until
those cases are run. Normal all-TP rank completion is counted by native
`KVOutputAggregator`; a missing rank does not allow partial publication.

Native metrics (`offloading/metrics.py:106,112`):
`vllm:kv_offload_total_bytes{transfer_type="GPU_to_CPU"}` and
`...{transfer_type="CPU_to_GPU"}` distinguish D2H/H2D; divide byte deltas by
wall time for offered GB/s. `vllm:kv_offload_total_time` sums event intervals;
it is **not** a union busy fraction. For the latter, take union of copy
intervals per rank/direction from the short nsys window and divide by the
same wall window. Never sum four rank-time counters as wall time. Keep the
unprofiled S cells as the speed denominator. Failure counters are the explicit
`HOST_PARKING copy_failure`, `snapshot_discard`, `fatal_copy_context` log totals.

## Review revision (2026-10-09; GPU has not run)

- Namespace v2 includes configured and resolved draft model/revision, dtype,
  quantization, HF config, method, speculative width and KV format.
- A non-WAITING hybrid reader is quarantined, invalid hashes evicted, pages
  retained until all-rank receive events, then recomputed. It does not kill
  the engine. `reader_quarantined` and `snapshot_discard` count the fallbacks.
- Validation only: `parking_validation=true`,
  `parking_test_fail_direction=CPU_to_GPU|GPU_to_CPU`,
  `parking_test_fail_mode=pre_submit|completed_copy`. Only TP rank 0 wraps
  `RecoveringHandler.inner`. The latter reports failure after the real event;
  it does not poison CUDA or pretend to test device loss. Never enable in prod.
- `HOST_PARKING load_timing` records request arrival, submit and all-rank
  event-observation epoch nanoseconds, actual external tokens, generation and
  success. Observation includes scheduler polling delay, so latency is an
  upper bound. Actual event durations remain native transfer metrics.
- Avoided prefill seconds are measured with identical OFF cold-return inputs:
  report OFF TTFT minus ON arrival→event-observation as an **upper-bound proxy**
  (OFF TTFT also includes first decode/sampling); acceptance additionally needs
  positive paired end-to-end TTFT savings. Do not call this isolated GPU prefill
  time. Save raw per-request records so the proxy can be audited.
- Tombstone sweep CPU ns/keys/count are accumulated only when nonempty, logged
  at shutdown. A normal empty sweep does not read a clock or scan the pool.
- Raw probe now covers shuffled multi-block transfers; generation and abort
  CPU tests are separate evidence. Gate 7 still needs real serving reset/abort
  cases in the window; CPU success alone is not a GPU PASS.

## Third-review handoff: m37 contract (2026-10-09)

Test runner: `benchmarks/host_parking_window.py` and CPU judge
`benchmarks/host_parking_judge.py`. Entrypoint (scratch, lead executes only):
`/data/bench/ab/m37-parking.sh`. A bare invocation emits a CPU-only JSON plan;
`M37_GO=1 bash /data/bench/ab/m37-parking.sh --execute` uses its own localhost
18037 test server. Production down/restore belongs to lead's wrapper. The
script uses a 48 GiB / 16-CPU fence (reduced by system reserve), 3-hour outer
cap, 900-second startup cap and 2700-second per-cell cap. Estimated 90–150
minutes is unmeasured. It has never been run on GPU by Astra.

The test matrix is OFF/ON/ON/OFF/OFF/ON, fixed 1800 GPU block IDs in both arms,
P7 grouped+BATCHED, P8 absent/zero. Each arm runs 12 synthetic fixed-seed parity
cases plus 3×4 append/return chains, the frozen m19 c10fit plan, one 262000-token
return and a ten-request 32768-token return. ON1 additionally attempts 100 c1
returns and 10 c10 bursts (100 returns). An attempt counts for latency only if
external-token telemetry proves restoration of at least prompt length−4096.
Tiny surviving prefixes, unsuccessful copies and cold recomputations cannot
inflate the sample count. Missing evidence yields INCONCLUSIVE; a numerical
threshold violation yields FAIL. No result automatically promotes production.

For these quota-fitting tests only, set the existing
`mamba_state_slots_reference_tokens=32768`: real CPU allocator yields 36 slots
per GDN group and 91 each for main attention / draft SW, aggregate **17,170,956,288
bytes**, below 16 GiB. The default 262K reference still yields 9/152/152. This is
an explicit experimental pool mix, not an increase in host budget. Independent
LRUs and intermediate SW suffixes can still prevent full c10 restoration; that
is a utility FAIL, not permission to silently measure only the hits.

Gate-4 copy occupancy is the **mean TP-rank union of same-direction job-event
intervals**: native `gpu_worker.py` serializes each direction using the previous
end event, so sum(D2H durations)/(4×cell wall time) is this mean. It is not the
cross-rank union, a copy-engine-utilization counter, or joint bidirectional
occupancy. Both metric endpoints are taken after six seconds with unchanged
transfer counters and zero running/waiting requests. Native stats aggregate
all ranks. Save raw `/metrics` endpoints and directional byte deltas.

Fault injection now requires **both** explicit config and
`VLLM_HOST_PARKING_FAULT_INJECT=1`. `parking_test_fail_rank` is an integer 0–3;
`parking_test_fail_nth` selects the Nth submission in that direction (not the
Nth completion). m37 injects the **second H2D job on rank 1**, once per process,
for each of `pre_submit` and `completed_copy`. It first obtains a successful
return, then forces the affected return, retries the same key, sends an
unrelated request, resets both caches and tests aborts. The judge requires
failed-rank count 1, failed H2D telemetry, matching recomputed output IDs,
unrelated success, `invalid_lookup_miss`, zero logical quota after reset,
four pool-release logs at teardown, and actual abort-with-pending-load/store
logs. Timing-sensitive aborts that never overlap a copy are INCONCLUSIVE.
Actual device-loss/fatal-rank injection remains a separate isolated test;
these wrappers deliberately do not poison CUDA or kill a TP worker.

Two types of counters must not be confused:

- `ParkingPool.stats()` reports physical view reservations (`in_use_bytes`,
  `in_use_slots`), fixed pinned capacity and direction owners. They go to zero
  only after both directions drain and unregister; raw allocation stays fixed
  while a server is alive. `registered` logs include exact mapping base/size
  so m37 can inspect that mapping's `/proc/PID/numa_maps` entry.
- `ParkingManager.stats()` reports logical resident slots/bytes per rank,
  live read/write refs and failed-key tombstones. A failed read can retain its
  tombstoned LRU slot until eviction (no unsafe reuse while another reader
  holds it). After quiescent external reset, all four values must be zero.

Avoided-prefill evidence in this first window uses a deliberate **GPU-prefix
reset, host-cache retained** surrogate with identical OFF/ON inputs. This
isolates transport/restore utility. It does not establish natural-LRU or
20-user capacity benefit; that needs a later pressure/idle trace. The JSON/MD
labels this limitation and always records `production_go=false`.

CPU reproduction log now includes shell-expanded complete command and venv:
`/data/bench/astra-host-parking/tests-review-final.log`, generated by
`/data/bench/astra-host-parking/test-review.sh`. Latest suite includes parking,
window judge, native CPU/grouped managers, native connector scheduler and
worker metadata. Do not substitute a different venv and compare test counts.

## m37 startup recovery / m37b (2026-10-09)

Lead reports G1 raw-byte PASS for m37; serving gates remain INCONCLUSIVE.
`/data/bench/m37/off1.serve.log:335–353` shows the source-tree FlashQLA loader
trying a JIT extension instead of the venv's bundled SM70 binary. The harness
put the entire engine worktree in `PYTHONPATH`, exposing sibling packages as
well as the intended `vllm` overlay. This is an import-path failure, not evidence
about parking parity, throughput, or recovery.

The new `benchmarks/host_parking_window.py` makes `<out>/python-staging/`
containing **only** `vllm -> <worktree>/vllm`. Server, raw TP4 probe and import
preflight use this staging path as both `PYTHONPATH` and working directory.
`PYTHONSAFEPATH=1` and `PYTHONNOUSERSITE=1` exclude incidental cwd/script and
user-package paths; inherited `PYTHONPATH`/`PYTHONHOME` are replaced/removed.
The serving venv and engine computation/route code are unchanged.

Before the raw probe and **each server start**, the serving Python runs
`benchmarks/host_parking_import_check.py` with no visible GPUs. It checks all
six top-level spec origins before importing anything: `vllm` must resolve to
the patched package; `torch`, `numpy`, `triton`, `flash_qla`, and `flashinfer`
must resolve inside the intended venv. Actual module `__file__` values are
then printed/checked. The bundled FlashQLA SM70 native module is imported
directly, without calling its JIT-capable `_load_ext`. Missing/wrong origins,
wrong interpreter prefix, import errors or unexpected CUDA initialization
refuse the window. Artifacts: `<label>.imports.{json,log}` and argv JSON.

CPU evidence in `/data/bench/astra-host-parking/import-recovery/`:
`smoke.imports.json` PASS with all six modules and the installed SM70 `.so`,
`cuda_initialized=false`; `old-path-rejected.json` FAIL with the original
whole-worktree path **before any package import**. This preflight validates
import provenance and native loading, not a GPU execution or full startup.
Full serving-venv CPU suite: **189 passed**, exact command and venv in
`/data/bench/astra-host-parking/tests-import-recovery-final.log`; Ruff,
`bash -n` and ShellCheck pass. Added cases reject shadowing, missing/wrong
prebuilt modules, symlink escapes, server/raw launch after failed preflight,
and overwriting existing outputs through either shell or direct entrypoint.

The shell wrapper is now versioned as `benchmarks/host_parking_window.sh` and
copied to `/data/bench/ab/m37-parking.sh`. `M37_OUT` defaults to
`/data/bench/m37b`; explicit `--out PATH` or `--out=PATH` takes precedence.
The shell and direct Python entrypoint refuse an existing output directory.
A dry plan does not create it. Lead's rerun command, **after review/window**:

```bash
M37_GO=1 M37_OUT=/data/bench/m37b bash /data/bench/ab/m37-parking.sh --execute
```

No preregistered numerical thresholds, GPU blocks, test cases, copy behavior,
or production flags change. The prior 90–150-minute estimate remains unmeasured;
CPU preflight adds up to nine short imports, each capped at 120 seconds.

## Fourth-review preflight requirements (2026-10-09)

Changes are isolated in `/data/src/1cat-wt-astra-host-parking-review`, branch
`p072-astra-host-parking-preflight-review`, based on `addf06bb4`. The running
m37b worktree and `/data/bench/ab/m37-parking.sh` remain unchanged. Do not
cherry-pick into that worktree or replace its scripts until the window ends.

- `host_parking_import_check.py:checked_staging` checks the actual directory
  entries equal **exactly `{vllm}`** before any package import, including hidden
  entries. The only entry must be a symlink to this worktree's `vllm` directory.
- The top-level package list now includes `flash_attn_v100`. Its `__init__.py`
  must reside in the venv; separately, the loaded
  `flash_attn_v100.flash_attn_v100_cuda` must be a native extension from that
  same venv. Both spec origin and actual `__file__` are checked, as for the
  existing FlashQLA native extension. A Python wrapper alone cannot pass the
  native-binary gate.
- `host_parking_window.py:import_preflight` runs a real negative-control
  subprocess **once per window, before the raw GPU probe**. It restores the
  old whole-worktree `PYTHONPATH` and requires exit 1, `status=FAIL`, and a
  `RuntimeError` for FlashQLA's out-of-venv origin before any package import.
  Success, unrelated errors, or imports before rejection fail the gate.
  `<out>/old-path-rejected.{json,log}` and argv JSON retain the evidence.

CPU evidence: `/data/bench/astra-host-parking/review4/tests-final.log` records
the exact serving-venv command and **204 passed**. The tests include a genuine
old-path subprocess regression, not just mocked origins. Live CPU preflight
artifacts are under `review4/live-preflight/`: `cpu-smoke.imports.json` PASS
with seven top-level packages, both venv native binaries and
`cuda_initialized=false`; `old-path-rejected.json` has the expected
`RuntimeError`. Ruff and `git diff --check` pass. No GPU execution was performed
for these changes; no parking runtime or numerical code changed.
