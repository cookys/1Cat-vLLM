# C4140 PLE registration pacing and host-pressure validation

2026-10-03; Python-only follow-up to exact-size pin commit `dbbc5f13e`.
Worktree `/data/src/1cat-wt-astra`, branch `p069-astra`. Service/GPU validation
belongs to the serving owner; Astra ran CPU tests and a read-only monitor smoke.

## Implemented behavior

`_ple_pinned_host_empty` retains the original contiguous CPU allocation, one
`cudaHostRegister(..., Portable | Mapped)` for the entire tensor, the process
lifetime reference, and the existing UVA-view creation. Placement, dtype,
shape, shard-copy arithmetic and the inference path are unchanged.

Opt-in startup controls add:

1. A host-local `flock` around allocation/registration, so cooperating TP ranks
   register one at a time. Order is whichever rank acquires the lock; no rank
   collective or rank-order assumption is required. A crash releases the lock.
   A timeout fails startup instead of proceeding concurrently. The shared
   lock file is never unlinked, which would split waiters onto different inodes.
2. Before allocation, `POSIX_FADV_DONTNEED` for the exact PLE tensor ranges
   identified by the local safetensors index and file headers. Only complete
   pages inside those ranges are advised. No checkpoint tensor data is read
   for this discovery, no whole-system cache drop occurs, and other weights
   sharing a checkpoint file are excluded.
3. After the original synchronous shard copy, an additional advice for the
   file-backed source shard. The mapping's inode/device identity is checked;
   anonymous, deleted, noncontiguous and unsupported source mappings are skipped.
   Other ranks may reread the unchanged file if they still need those pages.
4. Optional pause while holding the lock after registration. Logs separate
   lock wait, cache-advice time, allocation/registration time and pause time.
5. In paced mode, exact-registration failure does not silently fall back to
   a power-of-two pinned allocation. It raises instead of changing the intended
   memory footprint. A successful registration that fails the pinned check
   is unregistered before any legacy fallback; failed cleanup retains the
   allocation and raises so the driver never references freed storage.

All new controls default off. The existing exact-pin default stays on.

| Environment | Default | Effect |
| --- | --- | --- |
| `VLLM_QWEN4EXP_PLE_PIN_SERIALIZE` | `0` | `1`: one cooperating local process can allocate/register at a time |
| `VLLM_QWEN4EXP_PLE_PIN_DROP_CACHE` | `0` | `1`: pre-pin PLE checkpoint advice plus copied-shard advice |
| `VLLM_QWEN4EXP_PLE_PIN_LOCK_PATH` | `/tmp/vllm-ple-pin-UID.lock` under the process temp directory | All local TP ranks must resolve to the same local filesystem path; never use a per-rank path |
| `VLLM_QWEN4EXP_PLE_PIN_TIMEOUT_S` | `600` | Finite positive lock wait timeout; failure releases the fd |
| `VLLM_QWEN4EXP_PLE_PIN_PAUSE_MS` | `0` | Finite nonnegative post-registration pause; recommend trying `250` only with serialization |
| Existing `VLLM_QWEN4EXP_PLE_EXACT_PIN` | `1` | Keep `1` for this experiment; `0` explicitly selects the old caching allocator |

The lock path must be a user-owned regular file without group/world write
permissions or extra hardlinks; symlinks are rejected. Its containing directory
must be stable and visible to every local rank. All same-user servers sharing
the default path cooperate. A nonlocal checkpoint/model ID without a readable
index emits a cache-advice warning and keeps the model-loading behavior.

## E1 argument and limits

`flock` changes only the time at which a rank allocates/registers its table.
File advice does not write checkpoint bytes, replace pointers, alter the TP
row split or discard anonymous/COW tensor contents. The existing blocking
copy completes before post-copy advice. The UVA view still spans one fully
registered contiguous allocation, retained for the process lifetime.

No `MADV_DONTNEED` is applied to a private writable safetensors mapping:
discarding private modified pages could corrupt a tensor. CPU tests deliberately
modify a private COW mapping, call the real advice path and check that both
the private bytes and the unchanged backing file survive. Linux can keep
mapped/dirty pages resident despite `fadvise`; logged `advised_bytes` measures
the hint's coverage, **not bytes actually freed**. See the
[Linux file-advice contract](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
and [mapping advice semantics](https://man7.org/linux/man-pages/man2/madvise.2.html).

Segmented registration is deliberately not implemented in this version.
The native view helper obtains one device pointer at the allocation base;
splitting registrations would require proving that every chunk's device alias
is contiguous across the whole view and that no page belongs to two registered
ranges. The [CUDA registration contract](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
does not make this a safe assumption for arbitrary devices. Keeping one
registration avoids changing that E1 contract. Chunking is optional follow-up
if serialization/cache advice do not sufficiently reduce measured pressure.

This does not reduce the final **47.684 GiB** of TP4 host table storage. It
removes concurrent registration and gives the kernel explicit clean-file
reclaim candidates. Persistent capacity pressure can still swap CADO pages.
Other ranks may copy source shards while the next rank registers; the lock
does not serialize the entire several-minute model loader.

Automatic placement still evaluates available host memory before allocation.
Record each rank's placement and compare with the baseline. For this specific
G all-host PLE case, the owner can fix `VLLM_QWEN4EXP_PLE_HOST_GIB=12` in **both**
control and candidate to prevent timing-dependent budget capping from changing
placement; the table is only 11.92 GiB/rank, so that budget keeps the same rows.
Do not apply that value blindly to other models. Keep all four rank placement
logs and require identical row counts/bytes in the comparison.

## Validation already run

```bash
cd /data/src/1cat-wt-astra
OMP_NUM_THREADS=1 .venv/bin/python tests/models/qwen4_exp/test_ple_host_startup_cpu.py
```

Fourteen tests passed, including three-process nonoverlap, timeout, exception
and terminated-owner cleanup, symlink rejection, exact PLE header ranges,
page-aligned advice, malformed-header fallback, real COW-byte preservation,
anonymous-memory exclusion, advice/allocation/registration order under the
lock, registration failure handling, materialization without an ambient model
config, and 12 shard-copy
parity cases (two ranks, three host/device splits, advice on/off).

Read-only parsing of the real local E4M3 overlay found 128 PLE tensors in
10 files, exactly 51,200,245,760 bytes (47.6839447 GiB); no real checkpoint
cache eviction or weight-content read was performed. The monitor completed
a one-second read-only host smoke, with stable process identities and no
new las swap during that interval. That smoke is not a startup performance
result. CUDA smoke and full-model parity remain for the owner.

## Owner-run GPU and startup checks

Use the candidate worktree with the already-established native-extension
overlay. Both `ple_layer.py` and the new `common/ple_host_startup.py` are needed.
The commands below are examples for an owner-reserved window; Astra did not
run them or change the installed venv.

First, a small TP4 process smoke checks the existing UVA view across every
OS page with three changing FP8-byte patterns (16 MiB/rank):

```bash
env PYTHONPATH=/data/src/1cat-wt-astra \
  VLLM_QWEN4EXP_PLE_EXACT_PIN=1 \
  VLLM_QWEN4EXP_PLE_PIN_SERIALIZE=1 \
  VLLM_QWEN4EXP_PLE_PIN_PAUSE_MS=250 \
  /data/venvs/1cat-m589/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  /data/src/1cat-wt-astra/tools/check_ple_pin_uva.py --mib 16 --iterations 3
```

Require four `uva_byte_parity=true` records and nonoverlapping `PLE_PIN begin`
through `end` intervals. This intentionally does not establish 12-GiB pressure
behavior. Then monitor the actual startup (after the owner stops the intended
test service), preserving the same CADO processes and server settings:

```bash
env PYTHONPATH=/data/src/1cat-wt-astra \
  VLLM_QWEN4EXP_PLE_EXACT_PIN=1 \
  VLLM_QWEN4EXP_PLE_HOST_GIB=12 \
  VLLM_QWEN4EXP_PLE_PIN_SERIALIZE=1 \
  VLLM_QWEN4EXP_PLE_PIN_DROP_CACHE=1 \
  VLLM_QWEN4EXP_PLE_PIN_PAUSE_MS=250 \
  /data/src/1cat-wt-astra/.venv/bin/python \
  /data/src/1cat-wt-astra/tools/monitor_ple_startup.py \
  --out /data/bench/ab/ple-paced-startup.jsonl \
  --interval 0.2 --duration 1800 --settle-seconds 60 \
  --ready-url http://127.0.0.1:8001/v1/models -- \
  /home/cookys/projects/llm-playground/scripts/c4140-1cat-flashnext-serve.sh up
```

The monitor launches only the supplied command, once. It never stops a server,
changes sysctls, drops caches or imports CUDA. It refuses to reuse an output
file and, when launching with a readiness URL, refuses an already-ready
endpoint. Timeout/Ctrl-C leaves service lifecycle control with the owner.
Without a command or URL it can be used as a standalone read-only monitor.

Compare baseline (all new controls 0), serialization only, cache advice only,
then combined if startup budget permits. Keep `HOST_GIB=12` consistent in all
arms when using it. Collect same-PID/start-time las `VmSwap` changes, total
swap peak, min `MemAvailable`/`MemFree`, cache extrema, `pswpout`, `allocstall*`,
`pgscan_direct*`, `pgsteal_direct*`, time-to-ready, and the PLE timing logs.
Process churn and independent CADO activity invalidate a simple aggregate
before/after comparison; the JSON explicitly records missing/new identities.
Extrema are sampled at 200 ms, not guaranteed continuous peaks. NVIDIA page
registration need not appear fully in Linux `Mlocked`, so do not use that
field alone as the pin-byte counter. Finish with the owner's fixed parity
suite and unchanged-placement check before considering the candidate E1-qualified.

## Expected cost and measurable success

Cell H logs put the simultaneous four-rank registration window at about
11 seconds (07:19:18--07:19:29), within an approximately 15-minute startup.
If an isolated registration also took 11 seconds, serial execution would
take 44 seconds instead of 11: **+33 seconds**, plus 4 x 250 ms = **+1 second**
of explicit pauses. This illustrative +34 seconds is about 3.8% of 15 minutes;
it is not a measured prediction because the original calls shared contention.
Measure `sum(pin_s)`, each rank's `wait_s`, and actual time-to-ready.

Cache advice adds syscalls and may cause source rereads. Its extra time is
roughly reread bytes divided by effective storage/copy bandwidth, plus reclaim
work; the logs alone do not bound it. Eager/warm loader differences matter.
The full 47.684-GiB hint is not a claim that 47.684 GiB will be reread or freed.
Report the actual startup delta, rather than promising a fixed sub-minute cost.

Success means lower same-process las swap growth and reclaim counters at
identical table placement and token parity, with an acceptable measured startup
cost. Expected pressure reduction is plausible; the number of MiB of avoided
swap cannot be derived from the registration byte count alone.
