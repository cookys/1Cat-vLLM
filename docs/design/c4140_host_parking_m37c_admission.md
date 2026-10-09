# m37c G3：尾段計算阻塞 H2D admission（2026-10-09）

狀態：CPU修法；GPU未驗。branch `p072-astra-parking-rerun`，本次parent
`7982f2cfc`（先修driver的http模組遮蔽，Fable複審PASS）。不改m37c產物、
執行中wrapper、服務或venv。本文件的引擎路徑均相對本worktree。

## 1. 證據與機制

原始資料 `/data/bench/m37c/on1.returns.json`、`on1.serve.log`、
`on1.deadline.json`、`phases.jsonl`。離線重算工具
`/data/bench/astra-m37c-review/analyze_g3.py`，輸出`g3-timing.json`
含原始檔SHA256，只輸出合成case標籤與時間，不印token/prompt。

第7批10個client arrival跨 **66.588 ms**。其中後4個
server-arrival→submit為 **5.813–5.829 s**，前6個為12–30ms；
該批最慢client-arrival→all-rank-event-observed **8.301 s**。
`submit`是scheduler把job加入metadata，**不是DMA起點**；
`observed`是host收到全部rank完成通知，包含polling時間，兩者差不能當純H2D。
欄位來源：`offloading/scheduler.py` 的`update_state_after_alloc`及
`update_connector_output`；正式G3仍用client-arrival起算，不偷換分母。

driver排除：`benchmarks/host_parking_window.py`的`complete`在建好HTTP
Request/JSON後、`urlopen`前寫client arrival；`returns`的ThreadPoolExecutor
同批並發，`pool.map`全回完才checkpoint；reset/settle在批次之前。
`server`只設alarm、不sleep。該臂deadline5400s，returns phase正常complete，
該慢批不是alarm中斷。server `Request.arrival_time`源於input processing
（`vllm/v1/engine/input_processor.py:278–289`），**也不是engine add_request時間**，
所以新診斷另記engine收件時間，不把兩者混稱。

Fable獨立server鑑識與本次讀碼一致：修法前
`vllm/v1/core/sched/scheduler.py:932`的waiting loop要求`token_budget>0`；
`_select_waiting_queue_for_scheduling`在FCFS下優先skipped_waiting。
已完成H2D的請求被promote後，其4096-token tail能吃滿4096 budget，
後面新waiting請求連connector lookup/submit都進不了。不是只發生在running
迴圈，**同一waiting loop中前面的promote也能耗光budget**。
Fable發現的5個tail約1.1s/個與5.8s等待同量級；精確逐步因果仍由新timestamp
驗證，舊log不含engine收件時間。來源：m37c server log第30066–30116行
（以LF實體行計數；startup的CR會使`splitlines()`行號不同）。

## 2. 修法契約與開關

native `--kv-transfer-config`的`kv_connector_extra_config`新增：

```json
{"host_parking_async_admission": true, "parking_diagnostics": true}
```

`host_parking_async_admission`預設False，嚴格JSON boolean，非HostParkingSpec
connector啟用會拒絕。driver對應`--async-admission`預設不帶，OFF命令不變。
diagnostics獨立；算效能正式格兩臂同設，G4無插樁格仍關diagnostics。

budget>0維持原流程；budget=0時，waiting loop僅允許**WAITING且computed=0**
的新請求進lookup。已完成/未完成receive、grammar/streaming等待皆不promote。
lookup為async hit才呼叫原allocate_slots，`num_new_tokens=0`、
`delay_cache_blocks=True`，原lookahead限制保留；配置的是**H2D目的KV頁**，
不能宣稱「完全不配置slot」。不新增model-runner計算列、不消耗token budget。
然後走原connector submit→WAITING_FOR_REMOTE_KVS流程，完成事件/失敗重算
生命週期不改。非async miss或未就緒請求暫存至本pass skipped queue，
最後原順序放回，每步每請求至多lookup一次，不在同一步重試風暴。

既有max-num-running-reqs、KV配置失敗、preemption與pause gate保留。
GPU目的頁不夠時不submit，下步重新判；不以強制逐出running頁突破容量。
native host manager的`prepare_load`是已命中entry的ref-count pin，沒有另一次
可回傳None的host配額預留；host未ready用lookup的None延期，GPU容量不足用
allocate_slots的None延期。沒有加新的host pool/quota或變更量化。

OFF六步async-receive場景對固定parent `7982f2cfc`的schedule output、
request state、兩個waiting隊列、block refcounts/free數產生同SHA256，fixture在
`tests/v1/core/host_parking_admission_off.json`，另保留P7原始OFF回歸。
ON提前H2D可能影響混批/allocator時序及copy競爭，**不能由CPU測試預先宣稱E1**；
G2 parity與G4 decode干擾仍必驗，不因G3改善而豁免。

## 3. 診斷、測試與overlay

`parking_diagnostics=true`新增：

- scheduler waiting-loop前：`waiting_scan time_ns/token_budget/running/waiting/skipped/async_admission`。
- `add_request`：`engine_received_ns`；load_submitted與load_timing亦攜帶同時間戳。
- ParkingManager的`step_stats` JSON加`time_ns`。

runtime overlay只有三檔（相對7982f2cfc）：
`vllm/v1/core/sched/scheduler.py`、
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`、
`vllm/v1/kv_offload/cpu/parking_manager.py`。
window driver亦更新CLI與prereg記錄；沒有.so/CUDA/allocator layout變更。
CPU入口`tests/v1/core/test_host_parking_admission.py`：sync/async scheduler、
running tail與completed-receive tail、前方cold miss、pending receive、
KV容量失敗/connector pending次步重試、seq上限、pause、config拒絕及OFF fixture。

```sh
cd /data/src/1cat-wt-astra-parking-rerun
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 env PYTHONPATH=/data/src/1cat-wt-astra-parking-rerun:/data/bench/astra-host-parking/test-deps \
 /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
 tests/v1/core/test_host_parking_admission.py \
 tests/v1/core/test_mixed_prefill_budget.py \
 tests/v1/core/test_mixed_prefill_off_reference.py \
 tests/v1/kv_offload/cpu/test_host_parking.py \
 tests/v1/kv_offload/cpu/test_host_parking_window.py \
 tests/v1/kv_offload/cpu/test_host_parking_rerun.py -q
```

結果 **191 passed**；測試log `/data/bench/astra-m37c-review/full-tests.log`（含完整命令、Python/pytest版本）。fixture生成以
`git show 7982f2cfc:vllm/v1/core/sched/scheduler.py`作class來源，非拿新實作
自產期望值；來源hash及輸出hash均存fixture。

## 4. 下窗預登記（門檻不改）

lead wrapper選新輸出目錄，在既有balanced-32k/on1+fault臂配方加入
`--async-admission --parking-diagnostics`；先新venv副本overlay/import smoke，
staging只含vllm symlink；不重用舊PYTHONPATH遮蔽其他binary package。
driver命令可先用同旗標的PLAN_ONLY列完整argv，執行仍須lead的M37_GO/GPU窗。

G3照舊每regime100個genuine restores，p50≤2s、p99≤4s；c10完整host hit
≥95/100；G1–G7各自原閘不變。額外查每批最慢四段：client→server arrival、
server→engine_received、engine_received→submit、submit→observed；後兩段
不是純DMA。預測：engine已收件且KV/seq允許的async hit，不再因token_budget=0
連等數個tail；若仍晚，依新log定位收件、allocator或worker等待，不能移門檻。
算力批次仍最多4096 tokens；先前的c10 G3 FAIL不改寫為PASS，需新窗實測。
