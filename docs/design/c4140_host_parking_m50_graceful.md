# m50 parking: graceful teardown and clean full-matrix rerun

Status: CPU implementation; GPU unrun. Lead/Fable decision received 2026-10-09
(`01M4FHF929NB02C243H372HSWR`) supersedes the initial partial/reuse recipe.
Only the lead runs the GPU window, after m47, before production restoration.
This change does not modify any engine module or the seven-gate judge.

## Why parent-only TERM also needs a nonzero shutdown timeout

The old `host_parking_window.py::server` sent SIGTERM to the entire process group.
In m48p `fault-pre_submit.serve.log:916–919`, all four workers report termination;
there is no `HOST_PARKING pool_released` line. The successful HTTP recovery cases
therefore do not establish explicit pool release. G7 remains INCONCLUSIVE.

The native CLI defaults `--shutdown-timeout` to **0 = abort**
(`vllm/engine/arg_utils.py:1581–1586`). API shutdown forwards that value to the
engine (`vllm/entrypoints/launcher.py:103–119`,
`vllm/entrypoints/openai/api_server.py:151–154`).
`vllm/v1/utils.py:593–621` sends TERM, joins only within that budget, then calls
`kill_process_tree` on survivors. A zero budget skips the join. Consequently a
parent-only signal with unchanged argv would still race worker cleanup.

Both OFF and ON now explicitly use **`--shutdown-timeout 90`**. This argv change
affects teardown; compute/quantization/placement/parking knobs are unchanged.
The driver sends TERM only to its API child, created with `start_new_session`.
It then waits up to **120 s** for **all live processes in that session**, including
workers that outlive the API parent. Only after that deadline does it SIGKILL
still-owned process groups, with **30 s** additional exit allowance. No process
group signal is sent on the normal path. Foreign sessions and exited zombies
are excluded; PID start ticks are rechecked before forced escalation.

Native cleanup chain: EngineCore shutdown
(`vllm/v1/engine/core.py:768–773`) → executor closes worker death pipes
(`vllm/v1/executor/multiproc_executor.py:442–466`) → worker finally/shutdown
(`:908–918`, `:769–779`) → GPU worker connector shutdown
(`vllm/v1/worker/gpu_worker.py:1397–1400`) → offloading handlers drain
(`vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:328–334`,
`vllm/v1/kv_offload/cpu/parking_transfer.py:86–95`) → pool unregister/close
(`vllm/v1/kv_offload/cpu/parking_pool.py:77–88`). The executor retains its native
4 s + 4 s escalation policy; this patch does not weaken G7 if that path cannot
finish unregistering in time. GPU evidence still must show all four releases.

Each arm writes `<arm>.shutdown.json`: parent PID, observed PID/start/SID/PGID,
signals, graceful/forced status, elapsed time, exit code, distinct release PIDs.
Exiting processes without release logs does **not** produce fake release evidence.
The original G7 judge is unchanged; errors/forced cleanup stay visible.

## Frozen window recipe

Use the tracked wrapper at the reviewed branch tip. Final commit and SHA256
manifest are handed to Fable separately, so the document need not self-hash.
The output directory must not already exist; Astra has not created it.

```sh
M37_GO=1 M37_WORKTREE=/data/src/1cat-wt-astra-parking-rerun \
 M37_PYTHON=/data/venvs/1cat-main-integp7pr-p8/bin/python \
 M37_OUT=/data/bench/m50p \
 bash /data/src/1cat-wt-astra-parking-rerun/benchmarks/host_parking_window.sh \
 --execute --arms off1 on1 on2 off2 off3 on3 \
 fault-pre_submit fault-completed_copy \
 --parking-quota balanced-32k --async-admission \
 --on1-timeout-s 5400 --cell-timeout-s 2700 --fault-timeout-s 2400
```

* **No `--reuse-from`, no `--skip-raw`, no `--parking-diagnostics`.** G1 reruns
  four-rank raw byte checks. All six cells are newly collected in this window;
  old m37b/m48p results are contextual evidence only, never new samples.
* Order is the judge's OFF/ON/ON/OFF/OFF/ON, followed by the two fault arms. Same
  c10fit plan `/data/bench/m19/Ap_c10fit.json`, fixed 1800 GPU IDs, TP4, 16 GiB
  aggregate balanced host pool, test port 18037. No P8/cadence flag.
* Wrapper uses fence 48 GiB / 16 CPUs. It owns only the isolated test processes;
  the external lead wrapper owns :8001 stop/restore. Restore on every exit code.
* Hard outer cap = **34200 s**: 240 initial + 600 raw + 8×(120 import + 900
  startup + 150 cleanup) + 5400 on1 + 5×2700 other cells + 2×2400 faults + 300
  final margin. On timeout, TERM then kill-after 180 s. Lead's wrapper allowance
  **34800 s** (inner cap + 600) accommodates escalation; production restoration
  still belongs to its finalizer. These are hard bounds, not expected runtime.
* Engineering estimate **3.7–4.5 hours**, not a measurement guarantee. m48p on1
  body measured 2588.44 s (parity 232.22, traffic 736.92, returns 1619.30);
  fault recovery bodies measured 92.68/91.75 s, excluding startup. Sources:
  `/data/bench/m48p/on1.cell.json`, `phases.jsonl`. Budget roughly 50–60 min for
  extensive on1, 25–35 min each for the other five cells, 5–10 min per fault,
  plus raw/setup. The ranges allow ~3.2–4.8 h; 3.7–4.5 h is the planning range.

## Evidence needed per gate (unchanged)

| Gate | New evidence |
|---|---|
| G1 byte equality | Four raw-rank reports, 9 groups, shuffled/multiblock/tail/reset cycles |
| G2 parity | All six `cases.json`; OFF/OFF numerical floor and matched ON/OFF, 12 fixed + 12 chain turns per cell |
| G3 restore latency | New on1: 100 c1 + 100 c10 attempts, genuine external restore, p50≤2 s/p99≤4 s. m48p is comparison only |
| G4 interference | All six clean `traffic.json` + copy deltas; S steady/interfered degradation within three-OFF spread, R annotation; actual c10 and D2H activity |
| G5 saved work | All three OFF/ON pairs, 11 matching returns each, real host restoration and positive net TTFT saving |
| G6 RAM/las | This window's memory samples/cgroup deltas and all three ON NUMA/quota records; same original thresholds |
| G7 recovery | Both new fault arms, after-aborts followup, generation/reset/quota/failure evidence and ≥4 release logs each; also inspect distinct worker PIDs in shutdown.json |

The balanced quota's additional 95/100 c10 full-hit gate also remains. Missing
evidence is INCONCLUSIVE; a forced process exit is not proof of cudaHostUnregister.
The judge's `production_go=False` requires lead's adoption decision even if all
mechanical gates pass.

## CPU validation

The shutdown tests mock time, child state, process tables and all signals. They
exercise API-before-worker exit, already-exited API, timeout-only group KILL,
PID-reuse refusal, foreign-session/zombie exclusion and nonzero CLI grace.
They do not start or signal GPU/server processes.

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 env PYTHONPATH=/data/src/1cat-wt-astra-parking-rerun:/data/bench/astra-host-parking/test-deps \
 /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
 tests/v1/core/test_host_parking_admission.py \
 tests/v1/core/test_mixed_prefill_budget.py \
 tests/v1/core/test_mixed_prefill_off_reference.py \
 tests/v1/kv_offload/cpu/test_host_parking.py \
 tests/v1/kv_offload/cpu/test_host_parking_window.py \
 tests/v1/kv_offload/cpu/test_host_parking_rerun.py \
 tests/v1/kv_offload/cpu/test_host_parking_shutdown.py -q
```

CPU result: **199 passed** (14 upstream Torch deprecation warnings), 11.28 s,
using the exact serving venv/command above. Ruff 0.14.0 format/check and wrapper
`bash -n` passed; CPU-only `--budget-only` returned **34200**, with the command
plan preserved in `plan.json`. Full results and frozen manifest:
`/data/bench/astra-m50-review/` (CPU only).
