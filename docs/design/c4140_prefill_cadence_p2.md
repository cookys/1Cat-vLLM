# P2 · Prefill cadence 原型與 GPU 事前登記（2026-10-09）

狀態：CPU 原型；GPU 未跑、未採用。engine branch `p072-astra-prefill-cadence`，基底 `de01359b8`（P7 + prefix retention + P8 預設關閉 + V2 dispatch stats）。工作樹 `/data/src/1cat-wt-astra-cadence`。本稿的 `vllm/`、`tests/`、`benchmarks/` 皆指這個 worktree；正式知識副本在 llm-playground `doc/plan/072-prefill-cadence-p2-prereg-20261009.md`。

## 1. 動機與範圍

P0 m44 以每秒 iteration 差分得 M/D ≥9.0，只是下界；超過一秒的混合步被量化成1000ms，不能用來驗證 cadence。來源：llm-playground `doc/plan/072-concurrency-merged-ranking-20261009.md:70–82`，原始 `/data/bench/m44/{summary.json,metrics-*.jsonl,client/*.jsonl}`。本原型不估已得收益，保留現行4096 align/chunk路線，在chunk之間安排k個純decode批次。

它不能縮短一次長prefill kernel，也不減少prefill總工作量；可能延長cold TTFT。改變的是decode進度的分配，以及純decode是否能回到FULL graph。不得把P0理想公式或排程比例直接列為實測收益。

## 2. 實作契約

- `--prefill-cadence-decode-steps 0` 預設OFF；實驗ON為1、3（合法0…64）。`--prefill-cadence-step-log` 獨立預設OFF。OFF+logOFF不掃描cadence metadata、不建CUDA event。設定與CLI：`vllm/config/scheduler.py:102–113,259`、`vllm/engine/arg_utils.py:562,1447,2315`。
- cadence>0且`--mixed-prefill-step-latency-ms >0`在config階段拒絕，包含P8 min=max的靜態模式。原型拒PP/DP>1、encoder-decoder、非chunked、disable_chunked_mm_input、tree verifier；逐步timer要求V2。不能無聲退成OFF。TP4可以，所有rank接收同一SchedulerOutput。
- 任何有prefill列的**成功非空排程批次**把剩餘額度設k；接下來k個非空純decode批次各扣1。hold時running partial-prefill留原位置，waiting/skipped queues完全不取出，確保短prefill/tail也不混入「純decode」步；額度完了走原有完整budget。這也會作用於短prefill的最後一步，不是只判 `rows==4096`，避免尾段與warm append鑽過gate。來源：`core/sched/scheduler.py:597–615,657,970,1394–1419`、`prefill_cadence.py:15–32`。
- **async語義**：額度在KV配置成功、preemption處理後、`_update_after_schedule`前保留，算的是FIFO將要執行的批次；不是scheduler呼叫次數，也不等待GPU完成才再計數。engine `core.py:606–687`依序submit、appendleft/pop消費；`async_scheduler.py:35–59`管理placeholders。這是對早期提案「成功完成」文字的精確化：沒有新增GPU fence；回報不重複扣額度。若execute致命失敗，原引擎錯誤政策不變，不再繼續宣稱完成了這些步。
- 沒有eligible decoder時立即放行prefill並清舊額度；eligible decoder在配置期間被preempt/失敗導致空批次，也清gate供下次重試。空批次不被算作成功decode。沒有sleep、按秒節流或新KV保留策略。
- **prefill進度保底**：在已有可配置的prefill、原排程器budget/KV/seq槽可前進的前提下，cadence至多延後k個非空decode批次，隨後必有一次不受cadence限制的prefill機會；固定c/無資源失敗測試中每k+1批次跨一個4096邊界。不能保證過量admission、持續KV不足或原priority策略下的無條件wall-time SLA。這些不能當作cadence成功，也不新增繞過FCFS/priority的餓死風險。
- `_mamba_block_aligned_split`、align4096的config斷言、KV manager、attention/CUDA kernel均不改。來源：scheduler原函式`449–513`無diff；4096 MambaSpec測試與原始OFF schedule/refcount/queue hash回歸。

### 數值界線

OFF的排程序列/KV refcounts/queue/request state逐值等於固定P7+retention基底fixture（新的`cadence_step=None`欄位不計入fixture）；不把CPU排程等價說成GPU已驗E1。ON不改token/quant/算子，但批次形狀、接受數與state更新分段可能改變浮點結果，**不得預先宣稱E1或嚴格分布等價**。GPU要求OFF對生產基線token/logprob parity；ON記錄相同seed的token、logprob差異及self-replay底線。若數值變動，保留為需owner核准的候選，不直接採用。

## 3. 主量測：逐步server時間，不用一秒iteration倒數

`--prefill-cadence-step-log`兩臂都開。每個TP rank，在V2非dummy `execute_model`前record start，在sampling、postprocess、draft與connector post-forward排入main stream後record end。稍後`event.query()`就緒才讀`elapsed_time`，**不呼叫synchronize**；最多128 pending，超額/unfinished明印`PREFILL_CADENCE_TIMING_GAP`，不是靜默丟樣本。來源：`worker/cadence_step_timer.py:16–70`、`worker/gpu/model_runner.py:1497,1544,2001`。

每步兩種紀錄：

1. `PREFILL_CADENCE_STEP` JSON：step_id/rank、prefill/decode列、人數、pending_prefill_reqs、hold/remaining、FULL/NONE路由、padding、**stream_ms**、stream_gap_ms、host_enqueue_ms、host_begin_ns。stream_ms含兩event間的GPU串流等待/host提交空隙，並非kernel-only busy time；gap為前一個end至本step start。copy stream在main-stream event以外的尾端不包含。host_enqueue_ms是CPU enqueue跨度，不能當GPU完成時間。
2. scheduler `PREFILL_CADENCE_COMPLETE`：同step_id的**實際commit給EngineCoreOutput**的decode token數；已做stop/abort/rejection處理，排除被丟棄的raw sampled token。來源：scheduler`2067–2087`。不記request字串、token IDs或prompt。

`benchmarks/prefill_cadence_readout.py`以step_id對齊四rank與completion，時間取同step四rank的max。主rate為 `Σdecode_committed_tokens / Σ(decode_reqs × (max_rank_stream_ms + max_rank_gap_ms)/1000)`；這是server decoder人秒rate，不等同client SSE p50。無gap版並列供辨識host idle。使用max gap+max duration是保守帳，不宣稱它是同步全機wall精確值。

**干擾口徑預先固定**：step開始時仍有running prefill或waiting/skipped新請求即`pending_prefill_reqs>0`；有decoder的步歸interfered，包含cadence插入的pure-decode步。這是engine待處理prefill區間代理，不等同client first-token前區間；blocked waiting也計入，需保留計數。pending=0的decoder步歸steady。另按真正mixed/decode/prefill三類列p50/p90/p99，不能把插入的decode誤分類成steady來抬數字。

空批不納入。測試覆蓋4099.065ms仍完整保留、延後event與缺rank/重複step會判INCONCLUSIVE。每格只含一個engine lifetime；記測試前後step_id，結尾發短drain請求直到最後正式step四rank皆寫出，再用`--first-step/--last-step`排除warmup/drain。出現未補齊step、timing gap、rank metadata不一致或任一phase不足20步 → INCONCLUSIVE，補量不能挑好樣本。timer會改变allocator/event footprint，故兩臂同開，另用相同workload各一格logOFF檢查方向，不能把logON快當timer收益。

離線命令（CPU）：

```sh
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 /data/venvs/1cat-main-integp7pr-p8/bin/python \
 /data/src/1cat-wt-astra-cadence/benchmarks/prefill_cadence_readout.py \
 /data/bench/<window>/<cell>.serve.log --ranks 4 \
 --first-step <first> --last-step <last> --out /data/bench/<window>/<cell>.steps.json
```

## 4. GPU前凍結的P2門檻

決策對象為27B NVFP4 KV、TP4、DFlash2 K7、P7既有batched生產路線，P8=OFF；不夾帶P9、kernel prototype或parking變更。沿用同model/.so/venv、block/mamba4096、同max-num-seqs/batch budget、同pool IDs、同arrival/plan/seed與cache暖機；兩臂都先完成AOT後重啟成loader。先驗OFF token/logprob ==同基底生產底線；pool若不同先釘相同ID數。每格記完整argv、commit、.so hash、cache命中、preempt、KV peak、FULL比例、RAM/las swap。

**矩陣**：OFF(k0)對k1、OFF(k0)對k3分開判。每組採`A B B A / B A A B`，每臂4格（超過≥3），每個c10與c10fit plan分層；不得把k1/k3挑最佳單格、或兩個plan混成一個數。每格用同一份team_traffic_workload plan、同到達序；另跑P0固定arrival cold128K/200K各≥10個有效TTFT樣本/臂，混合與steady各≥20server steps。若窗不夠，只能先交INCONCLUSIVE的smoke，不能改門檻。

每一plan × k的**所有合併門檻**：

| 門檻 | 事前定義 |
|---|---|
| interfered | 各格server人秒rate中位數 ON/OFF **≥1.25**（§3主rate）；client windows p50/p10同時附列，不混換分母 |
| steady | 同口徑server rate中位数 ON/OFF **≥0.95** |
| whole-turn | 現行workload `summary.aggregate.per_stream_tok_s.p50`的跨格中位数 ON/OFF **≥1.15**；且`ratio−1 > max(spread_OFF, spread_ON)`，spread明定`(max_cell−min_cell)/min_cell`；這是既有whole-turn decode rate，TTFT另量 |
| cold TTFT | 固定arrival冷請求（cached_tokens≤1 block）的TTFT p90：各格p90跨格中位數 ON/OFF **≤1.50**；所有請求保留，timeout/error不得剔除。樣本不足或cache欄空 → INCONCLUSIVE |
| 進度/存活 | 所有已admit prefill最終完成；无engine error/OOM、无新增preemption；以event/log驗證k間隔，資源不足另外標記而非當成功純decode步 |
| 數值 | OFF serving parity及self-replay通過；ON對OFF IDs/logprob差分＋ON self-replay；非bitwise需owner看數值/任務品質後另決定，速度PASS不自動授權部署 |

whole-turn欄位來源：llm-playground `scripts/c4140-ab/team_traffic_workload.py:804–815,860–863`；不可改成aggregate總tok/s替代。同時列whole p10、TTFT p50/p99、client interfered時間比例、p99 token gap。任一性能/TTFT門檻失敗 → 該k/plan NO-GO；缺證據 → INCONCLUSIVE；兩組都過才可宣稱整體通過。此處沒有「先量再改」的容忍帶。

CLI差異（由lead的GPU wrapper執行，本人未啟動服務）：

```text
兩臂共用：--mixed-prefill-step-latency-ms 0 --prefill-cadence-step-log
A：--prefill-cadence-decode-steps 0
B1：--prefill-cadence-decode-steps 1
B3：--prefill-cadence-decode-steps 3
```

## 5. Overlay、CPU驗證與交付

相對base需覆蓋7個runtime Python檔：

```text
vllm/config/scheduler.py
vllm/engine/arg_utils.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/prefill_cadence.py (new)
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/cadence_step_timer.py (new)
```

無.so/CUDA/allocator改動；只在lead新venv副本overlay，勿將整個worktree當PYTHONPATH遮掉flash_qla。混入其他新補丁時按diff合併，不用舊整檔覆寫更新版。

CPU命令（驗證venv、完整命令應與log一起保存）：

```sh
cd /data/src/1cat-wt-astra-cadence
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 nice -n 19 taskset -c 0-26:2 \
 env PYTHONPATH=/data/src/1cat-wt-astra-cadence:/data/bench/astra-host-parking/test-deps \
 /data/venvs/1cat-main-integp7pr-p8/bin/python -m pytest --noconftest \
 tests/v1/core/test_prefill_cadence.py \
 tests/v1/core/test_mixed_prefill_budget.py \
 tests/v1/core/test_mixed_prefill_off_reference.py \
 tests/v1/worker/test_gpu_model_runner_v2_cudagraph_metrics.py \
 tests/v1/worker/test_gpu_model_runner_v2_prefill_dispatch.py \
 tests/v1/worker/test_cadence_step_timer_dispatch.py -q
```

結果：**133 passed**，`/data/bench/astra-cadence/final-tests.log`；含實際CLI/import且`torch.cuda.is_initialized()==False`、k1/3 × sync/async × spec0/7、4096真MambaSpec、pending async、abort/alloc失敗、running/waiting queue、原始OFF fixture、V2真execute/sample entrypoints的CPU mock。修補一個既有V2 dispatch fixture缺少的P8 timer=None欄位，未放寬production判斷。GPU數值、吞吐與每step timer開銷皆未驗。
