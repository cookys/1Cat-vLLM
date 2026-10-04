# C4140 plan 071：Qwen4Exp QSA main K/V 的 NVFP4 KV cache 設計

研究快照：2026-10-04。性質：**唯讀設計稿**，只整理事實與選項，沒有任何程式碼、
venv、服務或 GPU 變更。決策欄位標示「決策：Fable」，由 lead 定奪。

- Worktree：`/data/src/1cat-wt-fable-nvfp4kv`，分支 `p071-fable-nvfp4kv`，
  基底 `c4140-p070` = `1b5731f90`。
- Upstream `origin/main` = `98470f2b6`（2026-10-04 fetch）。p070 落後 39 個
  commit、領先 30 個。**這 39 個 commit 沒有動到** `vllm/models/qwen4_exp`、
  `kv_cache_utils.py`、`kv_cache_interface.py`、`platforms/interface.py`、
  `worker/gpu/attn_utils.py`、`flash-attention-v100`（`git log c4140-p070..origin/main -- <這些路徑>` 為空），
  內容是 GGUF 與文件。
- 證據標籤：**[code]** 讀 p070 原始碼；**[dry-run]** 在 CPU 上呼叫真正的
  allocator（`vllm.v1.core.kv_cache_utils`，用 `/data/venvs/1cat-p070` 內安裝的
  vllm `1.5.2.dev479+gc63e1e5cd`，該 venv 的 `kv_cache_utils.py`、
  `kv_cache_interface.py`、`platforms/interface.py`、`nvidia/qsa.py` 與本 worktree
  逐位元相同）並以合成 spec 驅動；**[calc]** 純算術；**[unverified]** 未驗證。
- 精確度標籤沿用 plan 069：target main K/V 量化一律 **E3**
  （輸出分佈會變，rejection sampling 保證的是量化後的 target 分佈，修不回 E4M3）。

## 1. 目標與非目標

**目標。** 把 12 個 target QSA 層的 main K/V 以 NVFP4 存進 paged KV cache：
packed E2M1 nibble，每 16 個值搭 1 byte E4M3 block scale，沿用 checkpoint 既有的
逐層純量 K/V scale 當 global scale。每 token 每層每 rank（TP4、1 個 KV head、
head_dim 256）K+V 由 E4M3 的 512 B 降到 288 B（−43.75%；FP16 為 1024 B）。
動機是 20 人團隊、每人 256K context 的常駐容量（見 §7）。

**非目標。**

- compressed index key 維持 FP16（64 B/token/層，每 4 個位置一列 128 維 FP16 key），
  QSA top-k 選取算術不動。
- raw ring（FP16 key + int64 MRoPE 位置）、GDN state（fp16 conv + fp32 SSM）、
  PLE conv state 不量化。縮 GDN SSM 精度是另一條能縮 state page 的路，
  但 `mamba_utils.py:116-121` 註解明說 fp32 是為了避免 MTP 驗證漂移，不在本案。
- V100 沒有 FP4/FP8 算術：以整數／float32 軟體解碼，unpack 成 FP16 後走 FP16
  tensor core，與現行 E4M3 路徑（軟體解碼 + FP16 dot）同一策略。
- 不移植 SM120 的 XQA／FlashInfer／TMA kernel（見 §3.5），不做 TurboQuant，
  不分析 NVFP4 × DCP2（DCP2 的 packed page 機制只被拿來當 §4 選項 C 的先例）。
- 本文件不含程式碼變更。

## 2. p070 上的錨點

行號皆為 `c4140-p070`（`1b5731f90`）。引文 ≤3 行。

### 2.1 QSA attention 與 KV 路徑

| # | 項目 | 錨點 | 內容 | 對 NVFP4 的影響 |
|---|---|---|---|---|
| a1 | backend 接受的 dtype | `models/qwen4_exp/nvidia/qsa.py:184-190` | `supported_kv_cache_dtypes = ["auto","float16","bfloat16","fp8","fp8_e4m3"]` | 加入 `"nvfp4"` |
| a2 | impl 層 admission | `qsa.py:231-240` | `NotImplementedError("Qwen4Exp QSA requires FP16/BF16 or E4M3 main KV cache")` | 同上 |
| a3 | owner 層 admission | `qsa.py:457-466`；`:468-469` | 同一則訊息；另 `kv_cache_scheme is not None` → `"Qwen4Exp QSA does not support KV quantization"` | 放寬 dtype；`kv_cache_scheme` 拒絕保留（NVFP4 走 `--kv-cache-dtype nvfp4`，不走 checkpoint scheme） |
| a4 | E4M3 容量閘 | `qsa.py:413-424`、`ops/qsa.py:2262-2270` | `qsa_e4m3_capability_reason`：需 FP16 activation 或原生 BF16 | 新增同型的 NVFP4 capability reason |
| a5 | dtype 與 scale 狀態 | `qsa.py:567-596` | `kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(...)`；非 fp8 時 `!= model dtype` → `NotImplementedError`；E4M3 禁 `calculate_kv_scales`；載入前 `k_scale/v_scale = -1.0` 哨兵 | `"nvfp4"` 解析成 `torch.uint8`（`torch_utils.py:98`），要放寬 `:578-583` 並沿用哨兵流程 |
| a6 | spec 建立 | `qsa.py:715-727` | `FullAttentionSpec(block_size, num_kv_heads, head_size=head_dim, head_size_v=head_dim, dtype=kv_cache_torch_dtype, kv_quant_mode=get_kv_quant_mode(...), dcp_sharded=...)` | `get_kv_quant_mode("nvfp4")` 已回 `KVQuantMode.NVFP4`（`kv_cache_interface.py:65-66`），**spec 的頁大小數學已就緒**（見 §2.2） |
| b1 | **寫入**（prefill、decode、draft 同一入口） | `qsa.py:804-810`（`_run_qsa` 呼叫 `impl.do_kv_cache_update`） | 每次 forward 都先寫再讀（store→attend from cache），**首個 prefill chunk 也從 cache 讀量化後的 K/V** | 見 §8 的品質差異說明 |
| b2 | 寫入實作 | `v1/attention/backends/flash_attn.py:857-890` → `_custom_ops.py:2691-2710` → `csrc/libtorch_stable/cache_kernels.cu:840-857` | `reshape_and_cache_flash(key,value,key_cache,value_cache,slot_mapping,kv_cache_dtype,layer._k_scale,layer._v_scale)`；`"nvfp4"` 分支只在 `ENABLE_NVFP4_SM100/SM120` 編譯，否則 `"NVFP4 KV cache requires SM100+ (Blackwell)..."` | **V100 不能用**；需自寫 store（§6.1） |
| c1 | **稀疏 decode/prefill 載入** | `ops/qsa.py:617-828`（`_qsa_sparse_paged_gqa_splitk_kernel`） | top-k 收集：`:695-699` 讀 indices；`:701-715` logical→physical page；`:718-735` K `[D,N]`、V `[N,D]` 載入，`mask=valid, other=0.0`；`:736-738` E4M3 軟解 | 主要改動點；`valid` mask 已是 zero-fill 的天然位置 |
| c2 | 入口與路由 | `ops/qsa.py:2273-2519` | XQA page4（CUDA）`:2362-2390`；resolved-index 快路徑 `:2392-2412`（僅 FP16）；Triton split-K `:2414-2519` | NVFP4 首版只走 Triton；其餘路由要**明確拒絕** nvfp4 |
| c3 | XQA page4 admission | `ops/qsa.py:1678-1760` | `k_cache.dtype in (float16, uint8)`、`k_cache.shape[2:] == (1,256)`、`stride(1)==256`、`stride(0) in (shape[1]*256, 2*shape[1]*256)`；`rows >= 64` 或（uint8 且 `rows > 16`） | NVFP4 末維 144 目前會被形狀檢查擋掉，但 dtype 同為 uint8，應加 `kv_cache_dtype` 明確守衛 |
| c4 | grouped page4（prefill）CUDA | `ops/qsa.py:1946-2053`；E4M3 需 ABI≥2（`:136-147`） | `flash_attn_v100_cuda.grouped_sparse_page4_fwd(..., kv_cache_dtype, k_scale, v_scale)`；`:2020-2034` W12：把 padding 列改指向第一個真實 microblock，避免 page 0（GDN state）解成 E4M3 NaN | NVFP4 首版 prefill 退到 Triton（效能風險，§6.4） |
| c5 | top-k 消費端 | `qsa.py:787-791`、`:275`；indexer 選取 `ops/qsa.py:1529-1660`（`torch.ops._C.qsa_lexicographic_topk`，`:1273`、`:1644`） | indexer 寫入 `topk_indices_buffer`（int32，寬度 `output_width`=2051），attention 以 `logical_indices` 讀取；非法值 `< 0` 或超出 page table → `valid=False` | 不改；非法索引的 zero-fill 要保證（§6.5） |
| d1 | E4M3 反量化 | `ops/qsa.py:683-688`、`:775-778`、`:2499-2518` | **整層一個純量**：K scale 乘進 `qk_scale_log2`（分數），V scale 在歸一化後乘到輸出（split-K merge 再乘一次）；E4M3 位元用 `fp8_e4m3fn_bits_to_fp32`（`deepseek_v4/common/ops/fp8_software.py:22-34`）軟解 | NVFP4 的 block scale 沿 head_dim 變動，**必須在 dot 之前套用**；純量 scale 仍可照舊折進分數／輸出（§6.2） |
| d2 | scale 來源 | `qsa.py:663-710`；`nvidia/model.py:126-141,198-250`；`common/kv_policy.py` | checkpoint 內 `...layers.{i}.self_attn.{k,v}_scale` 各一個 fp32 純量；`-e4m3kv` checkpoint 的 `kvscales-provenance.json`：target 24 個 tensor（12 層×2，`target_scale_source: published`）、MTP 2 個 tensor（`mtp_scale_source: prototype:from-layer-47`，即**複製 layer 47 的原型值，非真校準**） | NVFP4 沿用同一批純量當 global scale（§3.2）；draft 的 scale 是原型值 |
| e1 | compressed index key（必須維持 FP16） | `common/qsa_cache.py:978-990`；backend dtype 白名單 `:813-817` 只有 `auto/float16/bfloat16` | `MLAAttentionSpec(block_size=side_block_size, num_kv_heads=1, head_size=128, dtype=fp16, compress_ratio=4, dcp_sharded=False)` | 不動；新 B 只影響其頁大小（`B/4×256 B`） |
| e2 | raw ring | `qsa_cache.py:923-975` | `CircularBufferSpec`，容量 `4×ceil((4+K)/4)`（K=4→8、K=0→4），`assert block_size % capacity == 0` | 新 B 必須仍被 8（MTP4）或 4 整除 |
| f1 | 假設 K 與 V 同版面 | `ops/qsa.py:2297`（`v_cache.shape != k_cache.shape`）、`:2307`（`q.shape[2] != k_cache.shape[3]`）、`:2315-2316`（E4M3 需 uint8）、`:2331`（stride）；`qsa.py:279-281`（`kv_cache.unbind(1)`）；`flash_attn.py:140-149`（`get_kv_cache_shape` 回 `(num_blocks, 2, block_size, num_kv_heads, head_size)`） | 這些是 NVFP4 要逐一處理的假設：末維 144≠head_dim 256 | QSA backend 需覆寫 `get_kv_cache_shape`（仿 FlashInfer `flashinfer.py:357-368`，末維 `nvfp4_kv_cache_full_dim(256)=144`） |
| f2 | page 內 K/V 並列 | `worker/gpu/attn_utils.py:229-257` | `kv_tensor = kv_raw_tensor.view(dtype)`；形狀由 backend 決定；`cache_dtype` 是**全域單一字串**（`:186`、`:240`） | 混合 dtype（§5.2 (b)）時需改成逐 group／逐 spec |

### 2.2 KV cache spec、allocator 與頁佈局

**E4M3 現況的每層頁大小（公式出自程式碼）。**
`FullAttentionSpec.real_page_size_bytes = block_size × num_kv_heads × (head_size + head_size_v) × dtype_size`
（`kv_cache_interface.py:299-319`）。TP4 時 `num_kv_heads = max(1, 2//4) = 1`
（全域 KV head=2，每 rank 仍 1 個，複製）。E4M3 的 dtype 是 uint8（1 B），所以
**每 token 每層 512 B**，B=1616 時一頁 827,392 B。NVFP4 分支已存在
（`:299-313`、`AttentionSpec.real_page_size_bytes :186-204`）：
`block_size × num_kv_heads × (nvfp4_kv_cache_full_dim(256) + nvfp4_kv_cache_full_dim(256))`，
`nvfp4_kv_cache_full_dim(h) = h//2 + h//16`（`utils/torch_utils.py:464-466`）= 144，
**每 token 每層 288 B**，B=1616 時 465,408 B [dry-run 以真 spec 驗證]。

**GDN／PLE state 頁（與 block size 無關）。** `MambaSpec.real_page_size_bytes = Σ prod(shape)×dtype_size`
（`kv_cache_interface.py:697-702`）。形狀來自 `mamba_utils.py:222-243` 與
`nvidia/model.py:985-1037`，TP4、fp16 conv、fp32 SSM [calc，與 Astra 一致]：

| | GDN conv | GDN SSM | GDN 合計 | PLE conv（TP 複製） |
|---|---:|---:|---:|---:|
| MTP off（K=0） | 2560×3×2 = 15,360 | 12×128×128×4 = 786,432 | **801,792** | 10240×9×2 = 184,320 |
| MTP4（K=4） | 2560×7×2 = 35,840 | 786,432 | **822,272** | 10240×13×2 = 266,240 |

**Block size 規則。** `Platform._align_hybrid_block_size`（`platforms/interface.py:556-757`）：
`attn_block_size = align × cdiv(state_page, align × attn_page_size_1_token)`，
`align = max(kernel block 16, cache_config.block_size)`（`:672-678`、`:705-708`），
`attn_page_size_1_token` 取**全域單一 cache dtype**（`:583-643`，E4M3=512、NVFP4=288）。
接著把 mamba 頁 pad 到 attention 頁（`:737-757`）。結果：

| | 每 token 每層 | K=0 的 B | K=4 的 B |
|---|---:|---:|---:|
| E4M3 | 512 | 16×cdiv(801,792, 8192) = **1568** | 16×cdiv(822,272, 8192) = **1616** |
| NVFP4 | 288 | 16×cdiv(801,792, 4608) = **2784**（288×2784 = 801,792，**零 padding**） | 16×cdiv(822,272, 4608) = **2864**（824,832，slack 2,560 B） |

QSA ring 要求 B 被 8（K=4）或 4（K=0）整除：1616、1568、2784、2864 皆滿足。
⚠ 若改用 `--block-size 1616` 想把 NVFP4 的 B 釘回 1616，`align` 會變成 1616，
公式給出 `1616×cdiv(822,272, 1616×288) = 3232`，**反而更大**（`:672-678`）；
要取得小 B 必須改這條規則（見 §4 選項 C）。

**CSA+linear allocator 的組成（`kv_cache_utils.py`）。**

- 分類只認 `type(spec) is FullAttentionSpec`（`:2129`、`:2380`、`:2390`）。
  所以 NVFP4 要用**既有的 `FullAttentionSpec` + `kv_quant_mode=NVFP4`**，
  不能另開子類（`TQFullAttentionSpec` 會被拒）。
- 群組（`:2224-2357`）：compressed-sparse 群（12 主 KV + 12 壓縮 key，MTP 時各 13）、
  ring 群、GDN 群（36 層以 round-robin 分成 `cdiv(36, owner 數)` 群，`:2342-2344`）、PLE 群。
- 每個 main-KV **owner** 一個實體 tensor（`:1735-1768`）：
  `size = owner_page_size × num_blocks`，與同 index 的各 GDN 層**共用同一塊記憶體**
  （`shared_by`），所以 block ID 在 main KV 與 state 之間重用；另有 compressed key
  tensor 與 ring 共用。`bytes_per_block = Σ main_kv_pages + Σ compressed_pages`（`:2065-2067`）。
- state 必須裝得下：`unpadded_page > min(owner_pages)` → `ValueError`（`:2330-2336`）；
  state 頁會被 pad 成 owner 頁（`:2346-2349`）。
- **p070 新增（Astra 基底 25361ff5f 沒有）**：`_pack_csa_linear_main_kv`（`:2083-2110`）
  依「頁大小」把多個 main-KV owner 打包進同一實體頁（`cdiv(state_page, page)` 個一組，
  成員逐 kernel block 交錯），`KVCacheTensor.packed_members`（`:1758`），worker 端
  `_packed_attention_members`／`_reshape_kv_cache` 以 `as_strided` 建每個成員的 view
  （`worker/gpu/attn_utils.py:170-179`、`:258-280`）。來源是 DCP2
  （`docs/design/qwen4exp_qsa_dcp2.md`，PR #823）。pack 後 owner 數縮減，GDN 群數
  變 `cdiv(36, owners)`（DCP2 文件：六個 GDN 群）。
- 單一 ID 空間：`num_blocks = available_memory // bytes_per_block`（`:1744`）；
  `GPU KV cache size = int(max_concurrency × max_model_len)`（`:2682-2706`），
  `max_concurrency = num_blocks / Σ_groups cdiv(max_memory_usage_bytes, page_size_bytes)`
  （`:916-930`）。MTP4 E4M3：163（262144/1616 向上取整）+ ring 1 + 4 個 mamba 群
  ×(2+K=6) = 188，`292/188×262144 = 407,159`，**dry-run 以真 code 重現 407,159**。
- 實體 owner 數：MTP4 為 13（12 target + 1 draft），MTP off 為 12。
- worker 端配置：`_allocate_kv_cache`（`worker/gpu/attn_utils.py:151-167`）對每個 `KVCacheTensor`
  配一塊 `torch.zeros(size, dtype=int8)`，`shared_by` 的各層（main KV、GDN、PLE）都指向同一塊；
  之後 `_reshape_kv_cache`（`:182-346`）依各層 spec 與 backend 的 `get_kv_cache_shape` 切 view。

**MambaManager／CircularBufferManager。** `single_type_kv_cache_manager.py:1246`（`MambaManager`，
align 模式每群每 request 需 `2+num_speculative_blocks` 個 ID，
`kv_cache_interface.py:711-722`）、`:686`（`CircularBufferManager`，ring 每 request 固定 1 個 ID，
`kv_cache_interface.py:586-588`）。它們只看頁數與 block size，**與 KV dtype 無關**，
但決定了「固定 state 稅 F」（§7）。

**Align 模式的 prefill chunk 對齊。** `core/sched/scheduler.py:447-520`
（`_mamba_block_aligned_split`）把 chunk 尾端下取整到 mamba grid（預設 = block size）。
`max_num_batched_tokens=8192` 時，整數個 block 的 chunk 長度：
B=1616 → 8080；B=2864 → 5728；B=2784 → 5568；B=1568 → 7840 [calc]。
大 B 會讓每步 prefill token 數縮水約 30%，除非同步調高 `max_num_batched_tokens`
（會放大 prefill scratch，F14 的 PLE OOM 教訓）[unverified，需實測]。

### 2.3 MTP draft

- draft QSA 層的 spec：`qsa.py:715-727` 同一支 `get_kv_cache_spec`，
  `qsa_dcp_block_geometry`（`common/qsa_cache.py:855-880`）以前綴含 `mtp` 判定為 draft：
  block 為全長（不隨 DCP 切分）、`dcp_sharded=False`。
- **dtype 來源**：`mtp.py:226-230` 的 `_make_draft_vllm_config` 只換 `model_config`
  （`dataclasses.replace`），`cache_config` 原封不動，所以 draft **一定繼承 target 的
  `cache_dtype`**（`mtp.py:310-312` 還把它記成 `self._kv_cache_dtype`）。
  `SpeculativeConfig.kv_cache_dtype`（`config/speculative.py:175`）只被
  `llm_base_proposer._create_draft_vllm_config`（`:2126-2133`）讀取；V2 runner 的
  `load_eagle_model`（`worker/gpu/spec_decode/eagle/utils.py:34-60`）直接共用 `vllm_config`，
  所以這個欄位對 Flash-Next 的 MTP **目前無效** [code，未實測]。
- draft scale：`mtp.py:684-692` 以 `require_calibrated_speculative_draft=True` 呼叫
  `_finalize_qsa_e4m3_scale_load`（`model.py:198-250`），缺 scale 直接拒絕啟動
  （訊息：「...Provide calibrated draft K/V scales or keep the draft cache in FP16」）。
- **相等頁假設**：p070 的 CSA 路徑**沒有**「target 與 draft 同頁大小」的假設——
  `main_kv_pages` 是逐 owner 的清單（`kv_cache_utils.py:2273`），打包以頁大小分桶
  （`:2096-2110`）。dry-run 證實 NVFP4 target + E4M3 draft（B=1616）被 allocator 接受
  （§4）。仍存在的「單一全域 dtype」假設在 worker／platform 層：
  `interface.py:583-643`（B 規則）、`attn_utils.py:186,240`（`cache_dtype` 字串）、
  `worker/utils.py:177`（`KVBlockZeroer.init_meta(cache_dtype=...)`）。

### 2.4 可重用的 fp8 scale 機制與既有 NVFP4 元件

- `quantization/kv_cache.py:18-110`（`BaseKVCacheMethod`：`k_scale/v_scale` 載入、`-1.0` 哨兵）、
  `attention/attention.py:96-120`（`set_default_quant_scales`，device 純量 `_k_scale/_v_scale`
  加 host 浮點 `_k_scale_float/_v_scale_float`）。QSA 自己的 finalize 在 `qsa.py:663-710`。
- 既有 NVFP4 元件（**沒有任何一個是 KV 專用的 SM70 kernel**）：

| 元件 | 位置 | 說明 / 打包慣例 |
|---|---|---|
| Triton e2m1 解碼／捨入、E4M3 軟解 | `model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`：`_e2m1_inline :78`、`_apply_float32_sign_bit :103`、`_e4m3fn_byte_to_float :111`、`_round_to_fp4 :234`、`_dequantize_nvfp4_kernel :128`、SM70 閘 `:40-58` | 低 nibble＝偶數元素（`tl.interleave(low, high)`），block=16，E4M3 scale × global scale；**SM70 可執行**（scale 用整數／fp32 軟解，不用 `float8e4nv`） |
| 純 torch 參考實作 | 同檔 `ref_nvfp4_quant :533`、`ref_nvfp4_quant_dequant :550`、`break_fp4_bytes :418` | CPU 可跑，可當 unit test 的真值 |
| 軟體 E4M3 編解碼 | `models/deepseek_v4/common/ops/fp8_software.py`：`fp8_e4m3fn_bits_to_fp32 :22`、`fp32_to_fp8_e4m3fn_bits :53`（飽和、RNE） | block scale 的編碼可直接用 |
| NVFP4 KV store（Blackwell） | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`（整檔）、`quantization/fp4/nvfp4_utils.cuh:70-155`（`cvt.rn.satfinite.e2m1x2.f32`）、`:221-285`（scale 數學） | **SM100/120 only**（CMake `:1123`、`:1157` 以 `FP4_ARCHS` 限定）；格式與數學是本設計的規格來源 |
| NVFP4 KV 檢視輔助 | `utils/torch_utils.py:469-550`（`nvfp4_kv_cache_split_views`） | 假設每側（K 或 V）「先 data 後 scale」的兩區佈局 |
| 權重 QPN2 | `csrc/sm70_turbomind/ops/nvfp4_qpn2_layout.cuh:7` | nibble 順序 `[0,2,4,6,1,3,5,7]`（tensor-core 私有佈局），**與 KV 的順序不同，不可混用** |
| DeepSeek-V4 MXFP4 indexer 快取 store | `deepseek_v4/common/ops/fused_compress_quant_cache.py:648-840` | 32 值一組、UE8M0 scale、用 native e2m1 PTX；只作 Triton 切 even/odd 半邊（免 interleave）的寫法參考 |

### 2.5 Config 數字

`/data/models/Qwen3.8-Flash-Next-NVFP4-e4m3kv/config.json`（symlink 到
`../Qwen3.8-Flash-Next-NVFP4/config.json`）`text_config`：

| 欄位 | 值 |
|---|---|
| `num_hidden_layers` / `layer_types` | 48：`full_attention` 12（`full_attention_interval=4`，index 3,7,…,47）、`linear_attention` 36 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256（TP4 每 rank 6 個 query head、1 個 KV head） |
| `indexer_budget` / `indexer_compress_ratio` / `indexer_head_dim` / `indexer_kv_heads` / `indexer_n_heads` | 2048 / 4 / 128 / 1 / 4（top-k 寬度 2051） |
| `mtp_num_hidden_layers` / `mtp.layer_types` | 1 / `["full_attention"]`（1 層 draft QSA） |
| `linear_*`（GDN） | key head 16×128、value head 48×128、conv kernel 4 |
| `ple_layer_ids` / `ple_conv_kernel_size` / `ngram_size` / `hc_count` | `[2]` / 4 / 3 / 4 |
| `max_position_embeddings` | 262144 |

### 2.6 錨點漂移表（Astra 基底 `25361ff5f` → p070）

方法：以 `git diff -U0 25361ff5f c4140-p070 -- <檔案>` 的 hunk 把舊行號映射到新行號
（`25361ff5f` **不是** p070 的祖先，兩者相差 480 個 commit、522 個檔案）。

| Astra 引用（章節） | 25361ff5f 行 | p070 行 | 狀態 |
|---|---|---|---|
| `nvidia/qsa.py` backend dtype 白名單（G3/G8） | 78–88 | **184–194** | 位移 +106，內容相同 |
| `qsa.py` impl admission（G8） | 127–137 | **231–241** | 位移 +104，相同 |
| `qsa.py` owner admission／拒絕 kv scheme（G3/G8） | 262–278、264–278 | **455–469** | 位移 +193；dtype 與 `kv_cache_scheme` 檢查本身未變（`_verify_e4m3_kv_requirements` 呼叫在舊碼就有），範圍尾端的 PCP／DCP 檢查改為允許 DCP2（`:470-476`） |
| `qsa.py` `get_kv_cache_spec`（G8） | 515–524 | **715–728** | 位移 +200，3 個 hunk（加入 DCP 幾何） |
| `qsa.py` writer／`_run_qsa`（G8） | 544–625 | **748–829** | 位移 +204，無改動 |
| `ops/qsa.py` splitk kernel（G8） | 609–810 | **617–830** | 2 個 hunk，尾端 +20 |
| `ops/qsa.py` raw ring／compressor（G8） | 914–1024 | **942–1052** | +28，無改動 |
| `ops/qsa.py` page4 路徑（G8） | 1543–2060 | **1678–2195** | +135，無改動 |
| `ops/qsa.py` 入口／mqa／topk（G8） | 2127+、1104、1233+、143 | **2273、1132、1261、151** | +146／+28／+28／+8 |
| `common/qsa_cache.py` compressed key／state backend（G8） | 813–869 | **806–852（backend）、855–880（`qsa_dcp_block_geometry`，新）、923–990（cache 類別）** | 1 個 hunk；新增 DCP 幾何函式 |
| `qsa_cache.py` raw ring（G8） | 845–857 | **956–968** | +111，無改動 |
| `kv_cache_utils.py` CSA classifier（G8） | 2008–2032 | **2120–2144** | +112，無改動 |
| `kv_cache_utils.py` CSA allocator（G3/G7/G8） | 2075–2270 | **2187–2424** | **17 個 hunk，實質改寫**：新增 `_pack_csa_linear_main_kv`／`main_kv_owners`／逐 owner 頁清單／packed tensor。25361ff5f 該檔 `packed_members` 出現 0 次，p070 為 4 次 |
| `platforms/interface.py` hybrid B（G8） | 567–760 | **567–769** | 1 個 hunk（+9：`get_kv_block_size_multiple` 的 lcm，`:680-689`），B 公式不變 |
| `eagle/utils.py` `load_eagle_model`（G8） | 39–49 | **39–49** | 檔案未變 |
| `nvfp4_kv_cache_kernels.cu` 184–230、`nvfp4_utils.cuh` 70–105、`turboquant_attn.py` 182–213／389–410／1265–1375、`turboquant/config.py` 21–180（G3/G7/G8） | — | 同行號 | 檔案未變 |
| `docs/design/sm70_v100_migration_control.md` 3235–3282、25988–26013（G8） | — | 同行號 | 已讀：3235–3282 是 TQ k8v4 品質探針未過（mismatch .125、JSD .0778）與 4bit_nc 通過的紀錄；25988–26013 是 item 317 的 TQ 穩定品質基線（兩個 Flash-V100 閘設 0） |
| `c13-M94.serve.log:449–451,558–561`、`sampling_params.py:480–484`、`prompt_logprob.py:218–247` | — | 未重解（log／無關檔） | [unverified] |

**漂移的實質結論。** Astra G3／G7／G8 寫「MTP4 的 draft 保留 E4M3 而 target 改 NVFP4，
會打破 CSA allocator 目前主 KV 同 page-size 假設，需要 mixed-layout 支援」。
**這在 p070 已不成立**：逐 owner 頁清單與頁大小分桶打包（DCP2 PR #823）已在，
dry-run 顯示該混合配置被 allocator 接受（§4、§7）。剩下的阻礙在 worker／platform 層
（§5.2 (b)）。反過來，Astra 的容量表（G7）不受影響：我用 p070 真 allocator 重現了
G7 的 292／269／257 IDs（§7）。

### 2.7 上游 PR 重疊（`1CatAI/1Cat-vLLM`，2026-10-04 以 `gh` 查詢）

查詢：`nvfp4 OR fp4 OR turboquant OR 'kv cache' quant`（21 筆）、`QSA`（29）、`4-bit KV`、
`kv_cache_dtype`、`e2m1 cache`、`page size`、`hybrid kv cache`、`PLE`，另查 issue。
**沒有任何針對「SM70 上 NVFP4／4-bit KV cache」的 PR 或 issue**；相鄰的如下。

| # | 標題 | 狀態 | 與本案的關係 |
|---|---|---|---|
| 826 | Select automatic QSA E4M3 only from complete calibration | MERGED（在 p070） | `common/kv_policy.py`：只在 `cache_dtype=="auto"` 時生效，顯式 `nvfp4` 不受影響；E4M3 scale 完整性檢查的範式 |
| 824 | Admit calibrated E4M3 MTP through QSA kernel capabilities | MERGED（在 p070） | draft 的 E4M3 admission 與 `qsa_e4m3_capability_reason` 範式 |
| 664 | Calibrated E4M3 KV cache for Qwen4Exp QSA with MTP | CLOSED | 被 824／826 取代 |
| 607 | Serve an uncalibrated E4M3 QSA cache instead of refusing | MERGED | unit-scale fallback 與 `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES`；NVFP4 缺 scale 時的語意要決定 |
| 613 | Honor a checkpoint's KV-cache quantization directive only on Ampere and newer | MERGED | 確保 checkpoint 的 `kv_cache_quant_algo: nvfp4` 不會在 V100 自動啟用 |
| 823 | Support target-only QSA DCP with per-owner cache geometry | MERGED（機制在 p070） | packed pages、逐 owner 頁大小：§4 選項 C 與混合 draft 的基礎 |
| 645 | Decouple the Mamba state grid from the KV block size | MERGED（在 p070） | `--mamba-block-size` 只改 state checkpoint grid，頁仍被 pad 到 owner 頁（`kv_cache_utils.py:1139-1149`），不縮小 B |
| 829 | Preserve hybrid cache capacity with compact draft pages | MERGED（在 p070） | `_compact_sm70_hybrid_sliding_window_pages`（`kv_cache_utils.py:1059-1098`）：為避免 pad state 而縮頁的先例，但只適用 `SlidingWindowSpec` |
| 690 | Bound SM70 prefill scratch and compact hybrid cache pages | OPEN | 829 的其餘部分與 prefill scratch；同一塊 allocator 區域 |
| 828 | Use the grouped page4 route for QSA DCP prefill attention | OPEN | 改 `ops/qsa.py`、`nvidia/qsa.py` 的 page4 admission 與 packed stride；**與本案同檔，合併衝突面** |
| 647 | Stop grouped-page4 null-block padding from leaking E4M3 NaN | MERGED | NaN／zero-fill 的失敗模式（§6.5） |
| 705 | Allocate the QSA XQA page4 workspace for E4M3 KV | CLOSED | XQA workspace 與 KV dtype 的耦合 |
| 523 | Native FlashInfer SM70 batch GDN and QSA probes | OPEN（WIP，預設關） | 另一條稀疏 QSA decode kernel 路線；若落地，NVFP4 reader 要不要接是開放問題 |
| 513 | Port actual FlashInfer CUDA decode to sparse QSA (prototype) | CLOSED | 同上 |
| 821 | Reduce default Flash-Next MTP4 complete-round latency | OPEN | MTP4 round 延遲，與 KV dtype 無直接衝突 |
| 228 | [SM70][NVFP4] Admit ModelOpt on exact SM70 via proven backends | MERGED | 權重 NVFP4 的 SM70 admission（無 KV） |

此外 `c4140-p070..origin/main` 的 39 個 commit 沒有動到本案相關檔案（見檔頭）。

## 3. 格式

### 3.1 16 值一組的位元佈局

head_dim 256 沿最後一維切成 16 組，每組 16 個連續元素（一個 K 列或 V 列各 16 組）。

```text
一組 16 個值 → 8 byte 資料 + 1 byte scale

data byte j (j = 0..7) = (nib[2j+1] << 4) | nib[2j]     # 低 nibble 是偶數索引元素
nib = sign(bit 3) | magnitude(bits 2..0)
magnitude 0..7 → [0, 0.5, 1, 1.5, 2, 3, 4, 6]            # E2M1，最大 6
scale byte    = E4M3FN，非負（bias 7、最大 448、0x7F/0xFF 為 NaN）
```

一列（256 元素）= 128 B 資料 + 16 B scale = **144 B**；K、V 各一列 → **288 B/token/層/rank**。

### 3.2 scale 編碼與 global scale

規格來源是 `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:221-285`
（`cvt_warp_fp16_to_fp4`）與 `nvfp4_kv_cache_kernels.cu:95`（`global_scale = 1/k_scale`）。
令 `k_scale`／`v_scale` 為 checkpoint 的逐層 fp32 純量（與 E4M3 目前用的同一批）：

```text
sf   = (1/k_scale) × amax16 / 6          # amax16 = 該組 16 個值的絕對值最大
sf_q = RNE_saturate_e4m3fn(sf)            # 存成 1 byte
out  = sf_q != 0 ? 1 / (sf_q × k_scale) : 0
nib  = RNE_e2m1( clamp(x × out, -6, 6) )
x̂    = e2m1(nib) × sf_q × k_scale        # 反量化
```

- 若 `k_scale = amax_layer/448`（E4M3 校準的慣例，**未驗證**這批 checkpoint scale 是否如此），
  `sf_q` 最大約 448/6 = 74.7，離 E4M3 上限 448 還很遠；絕對值小於
  `6·k_scale·2^-9`（約為層 amax 的 2.6e-5 倍）的組會因 E4M3 subnormal 下限而歸零，
  實務上無害。另一種定義是 `global = k_scale/6`，把 block scale 推到滿量程（mantissa 精度相同，
  只改變下限）。**這是 §9 的開放問題**。
- 單位化：`e2m1 × sf_q`（不乘 `k_scale`）的值域 ≤ 6×448 = 2688（`k_scale = amax/448` 時 ≤ 448），
  遠小於 FP16 上限 65504，放得進 FP16。因此 kernel 可
  沿用 E4M3 路徑的做法，把純量 `k_scale` 折進分數、`v_scale` 折進歸一化後的輸出
  （`ops/qsa.py:683-688`、`:775-778`），**block scale 則必須在 dot 之前乘進 K、V tile**
  （它沿 head_dim 變動，無法像 E4M3 的純量那樣提到 dot 之外）。
- 全 0 的組：`sf=0`、資料 0、反量化為 0，不會產生 NaN。
- draft：`k_scale/v_scale` 是 `prototype:from-layer-47` 的複製值。E4M3 下缺校準曾造成
  「invalid proposals」（`model.py:222-232` 的訊息）；NVFP4 的 block scale 是逐組自適應，
  對 global scale 的準確度較不敏感（假說，未驗證）。

### 3.3 每頁版面、位元組數與對齊

沿用 upstream 的「每側兩區」版面（`utils/torch_utils.py:469-550`、
`nvfp4_kv_cache_kernels.cu:4-8,184-246`）：

```text
一頁 = B 個 token、1 個 KV head、D=256
[ K_data  : B×128 | K_scale : B×16 | V_data : B×128 | V_scale : B×16 ]   = 288·B bytes
backend 回報的邏輯形狀 = (num_blocks, 2, B, 1, 144) uint8；末維 144 是「資料 128 + scale 16」
的尺寸，不是實際 stride。`nvfp4_kv_cache_split_views` 以 8/9、1/9 的比例重算 stride。
```

| 項目 | 值 |
|---|---|
| 每 token 每層 K+V | 288 B（E4M3 512 B，FP16 1024 B）；每頁 `288·B` |
| 對齊 | `B % 16 == 0` 時，K_data 區 `B×128`（2048 的倍數）、K_scale 區 `B×16`（256 的倍數）、V 側起點 `B×144`（2304 的倍數）；頁 `288·B`（4608 的倍數）。`KVBlockZeroer` 要求頁位元組數能被 4 整除（`worker/utils.py` 的 `assert cur_bytes % 4 == 0`），皆滿足 |
| 每列 scale 讀取 | 16 B，可用單一 128 位元載入取回一個 token 的全部 16 個 scale |
| 與 upstream 的差異 | upstream 的 V scale 為 SM100 trtllm-gen **swizzle**（`nvfp4_kv_cache_kernels.cu:25-39,161-171`）。V100 設計 K、V 都用**線性**版面，仍相容 `nvfp4_kv_cache_split_views`，但不與 upstream 的 SM100 reader 位元相容（本來也不能用） |

候選替代：token 交錯的 144 B 記錄 `[128 資料 | 16 scale]`。稀疏 gather 時兩種版面每個
token 都碰 2 條 cache line 左右，**沒有資料顯示誰較快**，所以建議沿用 upstream 版面
（可重用 `nvfp4_kv_cache_split_views` 與其測試），效能留待 §8 L1/L4 實測。

### 3.4 與 repo 現有 NVFP4 權重打包的關係

| | 權重（ModelOpt／HF checkpoint、emulation 路徑） | KV（本設計） |
|---|---|---|
| 元素格式 | E2M1，兩個一 byte，**低 nibble 為偶數索引** | 相同 |
| block 與 scale | 16 元素一組、E4M3 block scale | 相同 |
| global scale | 每張量 fp32 `weight_scale_2`（乘法） | 每層 fp32 `k_scale`/`v_scale`（乘法） |
| scale 產生 | 離線 | **每個 token 線上產生**（store kernel） |
| 解碼 helper | `nvfp4_emulation_utils.py`：`_e2m1_inline :78`、`_e4m3fn_byte_to_float :111`、`_dequantize_nvfp4_kernel :128`（可在 SM70 執行） | 直接重用 |
| 不可混用 | QPN2 tensor-core 佈局的 nibble 順序 `[0,2,4,6,1,3,5,7]`（`nvfp4_qpn2_layout.cuh:7`） | KV 一律用順序 nibble |

### 3.5 SGLang QSA NVFP4 patch 的取捨（`engines/patches/sglang-qsa-nvfp4-kv/README.md`）

**可以搬的**：格式本身（packed nibble + 16 值 1 byte E4M3 scale，144 B/列 → 288 B/token/層）；
**非法 top-k 索引一律 zero-fill**（資料與 scale 兩個區都要；頁尾 padding 區的舊 scale
byte 可能剛好是 E4M3 NaN，zero-fill 後 pad 列反量化為 exact 0；Lesilva 的 kernel 只做
`mask=valid`，交錯出現的非法索引會讓真實 FlashInfer decode 的 16/16 列輸出 NaN，
README 的結論是「任何 port 必須保留 zero-fill」）；gather 的位移要用 **int64**
（`slot×heads×dim` 在約 4.2M token 池上會溢位 int32；`ops/qsa.py:717` 的 `safe_page` 已是 int64）；
global scale 從 device tensor 讀（CUDA graph 安全）；`layer_global_scale` 絕不回傳空切片；
測試設計（fused vs 兩段式 vs 純 torch 參考；以 0xFF 預填 scratch 的 poison + control；
CUDA graph replay；fp8 溢位飽和到 ±448）；品質探針方法（§8 L3）。

**不能搬的（SM120 或 SGLang 專屬）**：(1) 實際的 attention kernel 是 FlashInfer／XQA
trtllm-gen（SM120），SGLang 只是把 gather 出的 top-k 反量化到 bf16/fp8 **scratch** 再交給原
decode kernel；V100 沒有對應 kernel，且 scratch 會吃掉省下的 KV 空間（Astra G3 的警告；
SGLang README 的實測是每個 backend instance 66–132 MiB 常駐，另佔 CUDA graph private pool），所以 V100 版應該把
unpack + scale 直接融進 QSA 的稀疏載入；(2) 寫入用 `cvt.rn.satfinite.e2m1x2.f32`
（SM100+ 的 PTX），V100 必須軟體 pack；(3) V scale 的 SM100 swizzle；(4) SGLang 的
「前 4096 token 的 chunk 不讀 pooled KV」——1Cat 的 QSA 是先寫再讀（§2.1 b1），
所以量化誤差在首個 chunk 就生效（§8）。

SM120 上的外部結果（**不是 V100 證據**）：KV bytes −44%；TP2 C16 aggregate tok/s 兩段式
−4～−6%、fused + fp8 scratch 與 fp8 KV 持平（+0.7%）；teacher-forced ΔNLL 對 bf16 KV
+0.0042 [0.0024, 0.0062]（nvfp4 + fp8 scratch）、fp8 KV 為 +0.0047 [−0.0012, 0.0120]；
SWE-34 nvfp4 27–29/34 對 fp8 27–28/34（McNemar p=1.00，「未偵測到差異」）。

## 4. 頁佈局／allocator 選項

### 4.1 為什麼會有問題

GDN state 頁（801,792／822,272 B）必須放進 main-KV 的 owner 頁（§2.2 的 `:2330-2336` 守衛）。
E4M3 的 main 頁剛好比 state 大一點（B=1616：827,392 ≥ 822,272）；NVFP4 同樣 B 只剩
465,408 B，**裝不下 state**。三條路：把 B 放大讓頁追上 state（A）、讓 state 離開 attention
頁（B）、把兩個 NVFP4 頁併成一個實體頁（C）。

若 QSA 頁縮小而不處理，三處會直接擋下：`_get_kv_cache_groups_csa_linear` 的
`ValueError`（`kv_cache_utils.py:2330-2336`，「needs N bytes, but the smallest physical main-KV page has M bytes」）、
platform 的 `assert attn_page_size >= mamba_page_size`（`interface.py:739`），以及
ring 容量整除斷言（`qsa_cache.py:964-967`）。

### 4.2 [dry-run] 真 allocator 的結果

做法：用 `FullAttentionSpec`／`MLAAttentionSpec`／`CircularBufferSpec`／`MambaSpec` 的合成 spec
（TP4、1 KV head、D256、compress 4、ring 容量 4×ceil((4+K)/4)、36 個 GDN + 1 個 PLE、
12 或 13 個 QSA owner）呼叫 `get_kv_cache_groups`、`get_kv_cache_config_from_groups`、
`get_max_concurrency_for_kv_cache_config`。預算＝ M94 實測的 `292×12,100,608 B = 3.2907 GiB/rank`。
`max_model_len=262144`。「ID」指一個 block-ID（一整欄 tensor 頁）。

#### MTP4（K=4，13 個 owner）

| 配置 | B | owner 頁（B×slot） | Q（B/ID） | 線性 B/token | 固定 ID/請求 F | IDs | `GPU KV cache size` | 262K 等價 | 對 E4M3 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3（現況） | 1616 | 13×827,392 | 12,100,608 | 7,488 | 25 | 292 | **407,159** | 1.553 | 1.000× |
| **A** 全 NVFP4、放大 B | 2864 | 13×824,832 | 13,105,664 | 4,576 | 25 | 269 | 602,707 | 2.299 | 1.480× |
| **C1** target NVFP4 兩兩打包 + draft E4M3 | 1616 | 6×930,816（2×465,408）+ 1×827,392 | 7,756,800 | 4,800 | 43 | 455 | 579,007 | 2.209 | 1.422× |
| 混合、放大 B（無打包） | 2864 | 12×824,832 + 1×1,466,368 | — | — | 25 | 257 | 575,820 | 2.197 | 1.414× |
| **C2** 全 13 層 NVFP4 打包 | 1616／1440 | — | — | — | — | — | **`ValueError`**：`main-KV owners do not fill whole physical pages: {465408: 1}`（B=1616）、`{414720: 1}`（B=1440）；13 為奇數，剩一個落單 | | |

#### MTP off（K=0，12 個 owner）

| 配置 | B | owner 頁 | Q | 線性 B/token | F | IDs | `GPU KV cache size` | 262K 等價 | 對 E4M3 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3 | 1568 | 12×802,816 | 10,838,016 | 6,912 | 9 | 326 | 482,818 | 1.842 | 1.000× |
| **A** NVFP4、放大 B | 2784 | 12×801,792（零 padding） | 11,759,616 | 4,224 | 9 | 300 | 756,184 | 2.885 | 1.566× |
| **C** NVFP4 兩兩打包 | 1392 | 6×801,792（2×400,896，零 padding） | 5,879,808 | 4,224 | 15 | 600 | 771,011 | 2.941 | 1.597× |
| C′ 打包但沿用 E4M3 的 B | 1568 | 6×903,168 | 6,623,232 | 4,224 | 15 | 533 | 763,512 | 2.913 | 1.581× |

另加「MTP off 回收 draft 權重 1.247 GiB/rank」的預算（Astra G2 估計，非實測，4.5378 GiB）：
E4M3 449 IDs／664,986 tokens；A 414／1,043,534；C 828／1,063,996；C′ 735／1,052,873。

F 的來源（`get_max_concurrency_for_kv_cache_config`）：compressed-sparse 群 `ceil(262144/B)` +
ring 1 + 各 mamba 群 `2+K`。A 與 E4M3 都是 4 個 mamba 群（3 GDN + 1 PLE）；打包後 owner 數減半，
GDN 群數變 `cdiv(36, owners)` = 6，所以 7 個 mamba 群（C：7×2+1 = 15；C1：7×6+1 = 43）。
固定 state 稅 `F×Q`：E4M3 MTP4 288.5 MiB／A 312.5／C1 318.1；MTP off：E4M3 93.0／A 100.9／C 84.1。

### 4.3 三個選項

#### (A) 放大 block 使 NVFP4 頁追上 state（補齊 padding）——決策：Fable

- 作法：什麼都不改，`--kv-cache-dtype nvfp4` 後 `_align_hybrid_block_size` 自己算出
  B=2784（off）／2864（MTP4）；`AttentionSpec` 的 NVFP4 頁數學與 `FullAttentionSpec`
  分類（`type(spec) is FullAttentionSpec`）都已就緒。**不需要改 allocator、platform、
  worker**，只需要 QSA owner／kernel（§6）。
- 保留的省量：MTP4 保住理想 1.636× 的 **90.5%**（實得 1.480×）；MTP off 保住 **95.7%**（1.566×）。
- 代價：B 變成 E4M3 的 1.77×，prefix cache 重用粒度變粗（Astra G7：MTP4 退一頁的平均額外
  重算約 `1.5×(B−1616)` ≈ 1872 token/合格 call，以 4–5K prefill tok/s 約 .37–.58 s）；
  align 模式的 prefill chunk 在 `max_num_batched_tokens=8192` 下由 8080 縮到 5728（MTP4）
  或 5568（off）；固定 state 稅略升（+24 MiB／+8 MiB）。[unverified 實測影響]

#### (B) state 獨立頁池——決策：Fable（建議暫不採，高並發才有價值）

- 讓 GDN／PLE state 離開 attention 頁，B 可以維持 1568／1616 或更小，NVFP4 頁 451,584／465,408 B 不 pad。
- **現有 code 沒有支援**：`KVCacheConfig.num_blocks` 是單一純量，所有群共用一個 ID 空間與一個
  `BlockPool`（`kv_cache_utils.py:1744`、`:2656-2680`）；獨立頁池要改 coordinator、
  scheduler 的 block 預算與 prefix 邏輯 [unverified，未逐檔評估]。
- 容量上限估計 [calc，連續模型，忽略 ID 粒度與池切分碎片]：現行 `F×Q` 有近一半是浪費——PLE 群只有 1 層，
  每個 request 卻佔 `2+K` 個整欄 ID；ring 群每個 request 佔 1 個整欄 ID（真資料不到 30 KiB）；GDN 群的 ID 欄
  還附帶用不到的 compressed 頁。獨立頁池可把每個 request 的 state 稅降到真實 state 位元組：
  MTP off 55.4 MiB（A 為 100.9、C 為 84.1）、MTP4 170.9 MiB（E4M3 為 288.5、A 為 312.5）。
  收益集中在高並發（MTP off、3.2907 GiB、每人 ctx）：N=16 為 38,523（A 25,056、C 30,624），
  N=8 為 90,805（A 77,952、C 82,128）；N≤4 幾乎沒差。MTP4 時 GDN 的 fp32 state 本身就佔掉大半預算，
  N=16 在 A／C1 為 0，B 的上限也只有約 9K（§7.2）。
- 結論：N≥8 才是它的戰場；為此要重寫 allocator／coordinator／scheduler，**先用 C 的 pack-n 拿一部分**（見下）。

#### (C) 把兩個 NVFP4 頁打包進一個實體頁——決策：Fable

- 重用 DCP2 已上線的 `_pack_csa_linear_main_kv` + `packed_members` +
  `_reshape_kv_cache` 的 `as_strided` 成員 view（§2.2）；allocator 側 **dry-run 已通過**
  （上表 C、C1）。
- pack-n 一般化：`n = cdiv(state, B×288)`。n=2 之外，n=4／6／12（B=704／464／240）**也被真 allocator 接受**
  （dry-run：IDs 1188／1802／3485，blocks/請求 400／604／1168），`F×Q` 由 84.1 MiB 降到 76.6／72.9／72.5，
  MTP off、N=16 每人 ctx 由 30,624 升到 33,088／33,872／34,080（B-ideal 上限 38,523）。代價是 GDN 群數升到
  12／18／36，每個 decode step 要為每群建 GDN metadata（DCP2 文件：6 群在 #699 之前 +8.6%、之後 +3.0%），
  所以 n>2 的 decode 開銷 [unverified]，邊際收益也遞減；n=2 是合理的第一站。
- 保留的省量：MTP off **97.6%**（1.597×）；MTP4 + draft E4M3（C1）1.422×，
  相對其自己的理想 1.56× 為 91.2%，相對 1.636× 為 86.9%。**全 13 層 NVFP4 不可行（C2）**，
  除非擴充 allocator 讓最後一個落單 owner 單獨 pad 成 state 頁。
- B 可以維持 E4M3 的 1616（C1，draft 決定）或降到 1392（off，剛好 `2×1392×288 = 801,792`
  零 padding），prefix 粒度比 A 細，chunk 長度 6960／8080 也比較不縮。
- 要做的事：(1) 改 `_align_hybrid_block_size` 的 B 規則（目前 `--block-size` 反而放大，§2.2）；
  (2) 打包頁讓 `stride(0)` 變成 `n×` 頁，XQA page4／grouped page4 的 stride 檢查
  （`ops/qsa.py:1712`）會拒收——新 kernel 要自己的 admission，PR #828（OPEN）才剛把
  packed stride 加進 DCP2 的 grouped route；(3) 已知限制：packed 頁拒絕 pipeline parallel
  與 v1 runner；(4) `KVBlockZeroer` 對 packed 頁的行為在 DCP2 E4M3 上通過了閘，但 **NVFP4 + DCP1
  未驗證**；(5) F 變大（15／43），單人固定稅較 A 低或持平（MTP off 84 MiB 對 101 MiB；
  MTP4 C1 318 MiB 對 313 MiB）。

**我的建議（供 Fable 決定）。** 第一階段走 **A**：零 allocator 改動，風險最低，
容量已拿到理想值的 90–96%；B 暫不採。C／C1 當作第二階段：低並發（N≤4）時 C 對容量的增益只有
+2%（off）／−4%（MTP4，因 draft 留 E4M3），價值在 B 粒度與 chunk 長度；但**高並發時 C 有實質增益**
（MTP off、3.2907 GiB、N=16：每人 ctx 25,056 → 30,624，+22%；N=8：+5%），因為它把 PLE／ring 的整欄
ID 浪費減半。所以若 20 人團隊的目標是 N≥8，C 應提前到第二個里程碑，而不是等 TTFT 退步才做。作為對照，現成的 **E4M3 + DCP2**（K/V 切兩個 rank）在
`qwen4exp_qsa_dcp2.md` 的實測是 256K 容量 775,096 → 1,178,337 token（+52%，TP4、MTP3）；
NVFP4 × DCP2 未分析，兩者是否可疊加是開放問題。

## 5. MTP 選項

背景：draft 是第 13 個 QSA owner（`mtp.layers.*.self_attn.attn`，block 全長、不隨 DCP 切分）。
目前 draft **必然繼承 target 的 `cache_dtype`**（§2.3）。p070 的 allocator **不要求** target 與 draft
同頁大小（§2.2、§4.2 dry-run），要求「單一全域 dtype」的是 platform 與 worker 層。

### 5.1 三個選項——決策：Fable

| | (a) draft 也 NVFP4 | (b) 混合：target NVFP4 + draft E4M3 | (c) 首里程碑不開 MTP |
|---|---|---|---|
| 版面 | A：B=2864，13×824,832 B | A 的混合版（B=2864，257 IDs）或 C1（B=1616 打包，455 IDs） | A：B=2784，12×801,792 B（零 padding） |
| 容量（3.2907 GiB，262K 等價） | 2.299（MTP4） | 2.197（A 混合）／2.209（C1） | 2.885（無 MTP 回收權重）／3.981（回收 1.247 GiB 後，非實測） |
| allocator 改動 | 無 | 無（dry-run 已接受） | 無 |
| platform／worker 改動 | 無（全域 dtype 一致） | **有**，見下 | 無 |
| 打包可行性 | 不可行（13 為奇數） | C1 可行 | C 可行（12 個，剛好兩兩成對） |
| 主要風險 | draft proposal 受量化擾動（接受率下降）；draft scale 是原型值 | 工程量、混合 dtype 的 graph／zeroer／offload 回歸 | 與正式服務（MTP4）不同格；回收 draft 權重只是估計 |
| 精確度 | E3（target）；draft 本身是 E2，但接受率會變 | E3（target）；draft 不變 | E3 |

### 5.2 各選項在程式碼層要做的事

**(a) draft 也 NVFP4**——QSA owner 與 kernel 與 target 相同，無新機制：

- draft 走與 target 相同的 `Qwen4ExpQSAAttention`（`qsa.py:427`），admission 放寬後自動帶上 NVFP4。
- scale 載入：`_finalize_qsa_e4m3_scale_load`（`model.py:198-250`）目前只處理 `fp8/fp8_e4m3`，
  要推廣到 `nvfp4`，包含 `require_calibrated_speculative_draft=True` 的語意（現值是 layer 47 的原型複製）。
- 等頁假設：全部同 dtype，天然滿足。打包不可行（§4.2 C2），只能走 A。

**(b) 混合**——allocator 已 OK，缺的是「把 draft 的 dtype 單獨傳到 spec 與 worker」：

1. `mtp.py:226-230` `_make_draft_vllm_config`：以 `speculative_config.kv_cache_dtype`
   （欄位已存在，`config/speculative.py:175`）覆寫 draft 的 `cache_config.cache_dtype`；
   該欄位目前對 V2 runner 的 MTP **無效**（§2.3），需補驗證與測試。
2. draft 的 `Qwen4ExpQSAAttention` 本來就讀自己的 `cache_config.cache_dtype`（`qsa.py:567`），
   spec 會自動帶 `kv_quant_mode=FP8_PER_TENSOR`；draft scale 走既有 E4M3 路徑
   （`mtp.py:684-692`，`self.model._kv_cache_dtype` 會是 draft config 的值），不需改。
3. worker 把全域 `cache_dtype` 字串傳給 `_reshape_kv_cache`／`_update_hybrid_attention_layout`
   （`attn_utils.py:186,240,367`）與 `KVBlockZeroer.init_meta`（`worker/utils.py:177`）。要改成
   每個 attention group 以 `spec.kv_quant_mode` 或各自的 dtype 呼叫 `backend.get_kv_cache_shape`。
4. `Platform._align_hybrid_block_size`（`interface.py:583-643`）以單一 dtype 算
   `attn_page_size_1_token`：B 要改成「target 與 draft 兩種頁各自能放下 state 的最大值」
   （A 混合版 B=2864；C1 的 B=1616 是由 draft E4M3 決定）。
5. 打包（C1）才需要的：B 規則變更（§4.3）。

工作量估計見 §9。

**(c) 首里程碑不開 MTP**——最少的零件：

- 只有 target 12 層；`C4140_SPEC=off`（serve script 已支援）。GDN state 為 801,792 B，
  `B=2784` 剛好零 padding。
- 沒有 draft 擾動，E3 的來源只剩 target，品質訊號最乾淨；順便得到 no-MTP 容量
  （含可能回收的 draft 權重）。
- 缺點：正式服務是 MTP4，no-MTP 的 decode 速度與接受率行為不同，不能直接當服務候選。

### 5.3 最早可在 GPU 上端到端測的順序（我的建議，供 Fable 決定）

1. **CPU**：把 §4.2 的 dry-run 做成 unit test（spec 頁數學、allocator 組態、B 規則），
   加上純 torch 參考量化器對 `ref_nvfp4_quant` 的測試。不需要 GPU。
2. **單張 V100（分鐘級，lead 排程）**：Triton store + reader 對合成 cache 的層級測試（§8 L1）。
3. **TP4 no-MTP、版面 A（B=2784）**，262144 ctx：第一個端到端，用 kv-logprob-probe 對 E4M3 比較。
   只用 Triton（不需 nvcc）。
4. **TP4 MTP4、全 NVFP4、版面 A（B=2864）**：加上 draft，量接受率與 round_ms。
5. **視結果**：C／C1（打包、小 B）、混合 draft、CUDA kernel 路由。

先 (c) 再 (a)、最後才 (b) 的理由：(c)、(a) 都不碰 allocator／worker／platform，變數只有 QSA kernel
與 owner admission；(b) 多出 3–5 工程日去換約 −4% 容量（相對 (a)），只有當 (a) 量到接受率明顯下降或
B 粒度造成 TTFT 退步才值得。

## 6. Kernel 工作

原則：**先 Triton，後 CUDA**。Triton 只需 JIT，不需 nvcc／ninja，與「CPU 設計階段不建置」一致；
CUDA 路由只在 Triton 版的量測顯示 c≥4 decode 或 prefill 退步時才做。

### 6.1 Store（寫入時量化）

新增 Triton kernel，取代 `reshape_and_cache_flash` 的 NVFP4 分支（SM100+ only，§2.1 b2），
由 `Qwen4ExpQSAFlashAttentionImpl` 覆寫 `do_kv_cache_update`。

- 輸入：`key/value` fp16 `[T,1,256]`（RoPE、q/k norm 之後，`qsa.py:735-746`）、
  `slot_mapping` int64、device 純量 `layer._k_scale/_v_scale`（graph 安全）。
- 每 (token, K|V) 一個 program：16 組各取 `amax` → `sf` → `fp32_to_fp8_e4m3fn_bits`
  （`fp8_software.py:53`，飽和 RNE）→ 用解碼後的 `sf_q` 算 `out` → `_round_to_fp4`
  （`nvfp4_emulation_utils.py:234`）→ 以低 nibble 偶數打包 → 寫 128 B 資料區與 16 B scale 區。
- 必須處理：`slot < 0`（PAD）略過；同 slot 重複時後寫者勝（被拒絕的 draft 列會被覆寫，與
  E4M3 相同語意）；DCP 下的 dummy batch 以 `is_dummy_batch` 標記（`qsa.py:171-173`）。
- 數值規格＝ upstream 的 PTX 路徑。**捨入平手規則是否逐位元相同未驗證**，也無法在 V100 上用
  PTX 對拍；以 CPU 純 torch 參考（`ref_nvfp4_quant :533`）當真值，容許平手點差一個 code。

### 6.2 Reader（稀疏 decode 與 prefill 的共用核心）

改 `_qsa_sparse_paged_gqa_splitk_kernel`（`ops/qsa.py:617-828`）加一個 `KV_NVFP4` constexpr。

- 載入：K tile 的資料 `[128, N]` bytes 與 scale `[16, N]` bytes，V 同理；兩者共用既有的
  `valid` mask 與 `other=0`（`:724-735`）。物理位址 = page 的 K_data／K_scale／V_data／V_scale 四個區偏移
  （`B×128`、`B×16`）+ `token×128`／`token×16`。
- 反量化：nibble → `_e2m1_inline`＋`_apply_float32_sign_bit`；scale byte → `fp8_e4m3fn_bits_to_fp32`；
  `value = e2m1 × sf_q`（fp32）→ 轉 `query.dtype` 後 `tl.dot`。純量 `k_scale` 照舊乘進
  `qk_scale_log2`（`:683-688`），`v_scale` 照舊在歸一化後乘（`:775-778`，及 merge kernel
  `:831+` 與 `:2499-2518`），把 `KV_E4M3` 擴成「有純量的量化」旗標即可。
- 避免 interleave 的寫法（**效能假設，未驗證**）：K 的 nibble 以「偶／奇兩半」各算一次 dot，
  即 `scores = dot(q_even, K_lo) + dot(q_odd, K_hi)`，把 Q 預先依偶奇重排；V 的輸出同樣分偶奇兩半，
  最後寫回時交錯。DeepSeek-V4 的 MXFP4 store 也用「偶／奇半邊直接處理」避開 interleave
  （`fused_compress_quant_cache.py:15-18`）。
- 每個 query 列每層讀取量：2051 個 token × 288 B = 590,688 B，相對 E4M3 的 1,050,112 B 少 43.75%；
  ALU 解碼成本較 E4M3 高，**淨效果必須實測**（SGLang 的 SM120 結果不可外推）。
- 路由：`qsa_sparse_paged_attention`（`:2273`）加 `"nvfp4"` 分支，驗證 uint8、末維 144、
  純量 scale 為有限正值；XQA page4（`:2362-2390`）、resolved-index（`:2392-2412`）、
  grouped page4 一律以 `kv_cache_dtype` 明確拒絕 nvfp4（不要只靠形狀檢查，`:1693` 對 uint8 放行）。
  註（依程式碼推得，未實測路由）：現行 E4M3 在 `rows ≤ 16` 本來就走 Triton（MTP4 一個 request 的 verify 為 5 列，所以 c=1..3 走
  Triton；c≥4 才走 CUDA XQA，`ops/qsa.py:1747-1749`）。所以 Triton-only 的 NVFP4 在低並發 decode
  是與 E4M3 同路由的比較，在 c≥4 與 prefill 則可能退步。

### 6.3 Backend 與 owner 的介面

- `Qwen4ExpQSAFlashAttentionBackend` 覆寫 `get_kv_cache_shape`：`nvfp4` 時末維 `nvfp4_kv_cache_full_dim(head_size)=144`
  （仿 `flashinfer.py:357-368`）；dtype 白名單加 `"nvfp4"`（`qsa.py:184-190`）。
- `forward_qsa`（`qsa.py:243-328`）的 dtype 分支（`:282-288`）加 NVFP4：快取為 uint8、末維 144。
- `kv_cache_dtype_str_to_dtype("nvfp4")` 回 `uint8`（`torch_utils.py:98`），`get_kv_quant_mode`
  回 `NVFP4`；owner 的 `:578-583` 守衛要放行。
- scale：沿用 `validate_loaded_kv_scales`（`qsa.py:678-710`）與 `_finalize_qsa_e4m3_scale_load`，
  把 `("fp8","fp8_e4m3")` 條件推廣為「需要逐層純量的量化 dtype」。`KernelConfig.qsa_auto_e4m3`
  只在 `cache_dtype == "auto"` 時生效（`kv_policy.py`），不會干擾顯式 `nvfp4`。
- PR #613 的規則（auto dtype 在 Ampere 以下忽略 checkpoint 的 KV 量化指示）要保持：
  `MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP` 內有 `"nvfp4"`（`torch_utils.py:111-114`），
  checkpoint 內的 `kv_cache_quant_algo` 不應在 V100 上自動啟用 NVFP4。

### 6.4 Prefill

QSA 的 prefill 也是同一個稀疏 kernel（每個 query 列取 top-2051），只是列數多；
現行 E4M3 在大 batch 走 CUDA grouped page4（八列共用 K/V microblock）。NVFP4 首版退到 Triton
split-K：PR #828 的描述指出該 kernel 「每個 query 列各載入一次 K/V」，**prefill 吞吐是主要風險**
（量級未知，沒有任何 NVFP4 數據）。緩解選項：(i) 先量；
(ii) 有界的「selected-row 反量化到 FP16 scratch tile 再走既有 FP16 page4」，scratch 必須逐 tile
有界，不可把 8192 列×2051 條都攤開（Astra G8：單 K/V 即約 16 GiB/rank）；(iii) CUDA 路由（§6.6）。

### 6.5 非法 top-k 索引的 zero-fill 與 stale 位元組

- Triton 路徑：資料與 scale 兩個區的載入都用 `mask=valid, other=0`，所以非法／padding 列的資料為 0、
  scale 為 0 → 反量化為 0，沒有 NaN；`valid` 已涵蓋 `logical_token<0`、超出 page table、
  physical page 超出範圍（`:703-715`）。
- **為什麼必須有**：ID 在 main KV 與 GDN／PLE state 之間重用（§2.2），E4M3 scale 的 `0x7F`／`0xFF`
  會解成 NaN，而 stale 位元組可能剛好是 state 資料。#647（merged）與 W12（`ops/qsa.py:2020-2034`）都是這個失敗模式在
  E4M3 上的修補。`KVBlockZeroer`（`worker/utils.py:158-300`）會歸零新配置的 main KV block，但對
  **packed 頁在 NVFP4 上**是否正確 [unverified]。
- CUDA 路由若之後加入，必須重用 W12 的「把 padding 列指向第一個真實 microblock」。

### 6.6 CUDA kernel（第二階段，視量測而定）

- 既有 SM70 CUDA 可仿：`flash-attention-v100/kernel/fp8_kv_utils.cuh`、`paged_kv_utils.cuh`、
  `fused_mha_forward_paged.cu` 的 KV dtype 分派（`:43-46`、`:3627`、`:4057`）、
  `flash_decode_turboquant.cu`（TurboQuant packed 4-bit 的 paged decode，head_dim 分派含 256，`:542-563`）。
  grouped page4 的 ABI 需 v3 才能傳 NVFP4（目前 `ops/qsa.py:1946-1985` 只傳 `kv_cache_dtype,k_scale,v_scale`）。
- 這一段需要 nvcc 建置，**不在本設計階段**。

### 6.7 Triton vs CUDA 一覽

| 工作 | 建議 | 理由 |
|---|---|---|
| store | Triton | 軟體 e2m1／e4m3 helper 已在；PTX 版不能用 |
| reader（decode，rows≤16） | Triton | 與現行 E4M3 同路由 |
| reader（prefill、c≥4 decode） | 先 Triton，量過再決定 CUDA | E4M3 這兩處用 CUDA，會退步 |
| allocator／platform／worker | Python | §4、§5 |

## 7. 容量估計

### 7.1 錨點與公式

- **錨點**：`c10-P3-b.serve.log`（Astra G1）：gmu .94、TP4、MTP4、UVA、E4M3、`max-model-len=262144`；
  weights 22.9 GiB/rank、KV budget **3.29 GiB/rank**、292 個 ID、
  `GPU KV cache size: 407,159 tokens`。本文以 p070 的真 allocator 重現 292 IDs 與 407,159（§4.2）。
  3.29 GiB 是四捨五入值；本文用 `292×12,100,608 B = 3,533,377,536 B = 3.290714 GiB`（Astra G7 的保守錨點）。
- **每 token 每層每 rank 的位元組（程式碼導出）**：main K+V：E4M3 512、NVFP4 288（§2.2）；
  compressed index key 64（`B/4×256 B ÷ B`）。MTP4 共 13 個 QSA 層：E4M3 `13×576 = 7,488`、
  全 NVFP4 `13×352 = 4,576`、target NVFP4 + draft E4M3 `12×352 + 576 = 4,800`；MTP off 12 層：
  E4M3 6,912、NVFP4 4,224 B/token。
- **公式**（皆有程式碼對應，§2.2）：
    - `Q = Σ owner 頁 + Σ compressed 頁`（B/ID，`bytes_per_block`，`:2065-2067`）
    - `IDs = floor(budget / Q)`（`:1744`）
    - `F = n_mamba群 × (2+K) + 1`；`blocks/請求 = ceil(262144/B) + F`（`:916-930`）
    - `GPU KV cache size = floor(IDs / (ceil(262144/B)+F) × 262144)`
    - 完整 262K 條數 `= floor((IDs−1)/(ceil(262144/B)+F))`（扣 null ID）
    - 每人 ctx（N 個同時 active）`= min(262144, B × max(floor((IDs−1)/N) − F, 0))`
    - 四張 GPU 是 TP 複製同一份 context（每 rank 1 個 KV head），**容量不可乘 4**。

### 7.2 結果（budget 3.290714 GiB/rank，除非另註）

| 配置 | B | F | F×Q (MiB) | IDs | `GPU KV cache size` | 262K 等價 | 完整 262K 條數 | 每人 ctx N=2 / 4 / 8 / 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **MTP4** E4M3（現況） | 1616 | 25 | 288.5 | 292 | 407,159 | 1.553 | 1 | 193,920 / 75,952 / 17,776 / 0 |
| MTP4 **A** 全 NVFP4 | 2864 | 25 | 312.5 | 269 | 602,707 | 2.299 | 2 | 262,144 / 120,288 / 22,912 / 0 |
| MTP4 **C1** target NVFP4 打包 + draft E4M3 | 1616 | 43 | 318.1 | 455 | 579,007 | 2.209 | 2 | 262,144 / 113,120 / 21,008 / 0 |
| MTP4 **B** 上限（連續模型，全 NVFP4） | — | — | 170.9（真 state） | — | — | — | — | 262,144 / 153,869 / 57,350 / 9,090 |
| **MTP off** E4M3 | 1568 | 9 | 93.0 | 326 | 482,818 | 1.842 | 1 | 239,904 / 112,896 / 48,608 / 17,248 |
| MTP off **A** | 2784 | 9 | 100.9 | 300 | 756,184 | 2.885 | 2 | 262,144 / 180,960 / 77,952 / 25,056 |
| MTP off **C**（pack-2） | 1392 | 15 | 84.1 | 600 | 771,011 | 2.941 | 2 | 262,144 / 186,528 / 82,128 / 30,624 |
| MTP off C4（pack-4） | 704 | 27 | 76.6 | 1188 | 778,567 | 2.970 | 2 | 262,144 / 189,376 / 85,184 / 33,088 |
| MTP off **B** 上限（連續） | — | — | 55.4（真 state） | — | — | — | — | 262,144 / 195,367 / 90,805 / 38,523 |
| **MTP off + 回收 draft 權重 1.247 GiB**（budget 4.5378 GiB）E4M3 | 1568 | 9 | 93.0 | 449 | 664,986 | 2.537 | 2 | 262,144 / 161,504 / 73,696 / 29,792 |
| 同上 **A** | 2784 | 9 | 100.9 | 414 | 1,043,534 | 3.981 | 3 | 262,144 / 261,696 / 116,928 / 44,544 |
| 同上 **C** | 1392 | 15 | 84.1 | 828 | 1,063,996 | 4.059 | 4 | 262,144 / 262,144 / 122,496 / 50,112 |
| 同上 **B** 上限（連續） | — | — | 55.4 | — | — | — | — | 262,144 / 262,144 / 130,429 / 58,336 |

來源：IDs／`GPU KV cache size`／完整條數／F 皆 [dry-run]（真 allocator）；每人 ctx 是 [calc]
（上列公式；B 上限用 `(budget − N×真 state)/N/每 token 位元組` 的連續模型，忽略 ID 粒度與池切分碎片，
真 state = 36×(2+K)×GDN 頁 + (2+K)×PLE 頁 + ring）。MTP4／off 的 E4M3、A 兩列與 Astra G7
的 292／269／449／414 一致。其他 budget（graph reserve 576 MiB、gmu .96）同樣套用上述公式，
Astra G7 已列出。

### 7.3 讀法與限制

- **262K 等價容量的提升**：MTP4 為 A 1.480×、C1 1.422×；MTP off 為 A 1.566×、C 1.597×
  （理想線性比 1.636×，main-only 512/288 = 1.778×）。
- **N≥8 時 state 稅主導**：MTP4 的 GDN fp32 state 每個 request 約 171 MiB，A 與 C1 在 N=16 都是 0，
  4-bit KV 解決不了 16 人同時 active（與 Astra G7 結論一致）；即使是 B 的連續上限也只有約 9K。
  MTP off 時 N=16 才有意義（A 25K、C 31K、B 上限 39K）。
- **預算假設**：NVFP4 kernel 不新增常駐 workspace（Triton split-K 的 partial buffer 與 E4M3 同；
  CUDA 路由若加入 workspace 要另扣）[unverified]；MTP off 的 1.247 GiB 是 Astra G2 由 checkpoint header
  推算的估值，非實測；新 kernel 必須進 real-route profile，不能再重演 PLE 漏 profile（F14）。
- **「每人 ctx」是 allocator 模型預算，不是已驗證的服務 SLA**：假設互異 prefix、不計額外保留頁、
  每個 request 採保守 handoff state 預留。
- 以上都**不含**品質：E3，見 §8。

### 7.4 p070 的 KV 預算（第二組欄位）

§7.2 的 3.2907 GiB/rank 是 p069 的預算。p070 的權重為 24.16 GiB/GPU（較 p069 多 1.26 GiB），CUDA graph pool 也
計入 gmu，實測 E4M3 加 MTP4 的 pool 為 **263,538 tokens**。由這個錨點反推預算：
`floor(IDs / (163+25) × 262144) = 263,538` 只有 `IDs = 189` 成立，所以預算在
`189×12,100,608 = 2,287,014,912 B = 2.1299 GiB`（含）到 2.1412 GiB（不含）之間；下表用下界，
上界的結果只差 1 到 3 個 ID。以下皆為真 allocator 在 CPU 上的結果（`GPU KV cache size`），
每人 ctx 為 N 人同時 active 的 allocator 模型預算。不開 MTP 的列沒有計入回收 draft 權重。

| 配置 | B | p069 tokens | p069 262K 等價 | p069 N=16 | p070 tokens | p070 262K 等價 | p070 完整 262K 條數 | p070 N=8 | p070 N=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MTP4 E4M3 | 1616 | 407,159 | 1.553 | 0 | 263,538 | 1.005 | 1 | 0 | 0 |
| MTP4 A | 2864 | 602,707 | 2.299 | 0 | 389,855 | 1.487 | 1 | 0 | 0 |
| MTP4 C1 | 1616 | 579,007 | 2.209 | 0 | 374,127 | 1.427 | 1 | 0 | 0 |
| 不開 MTP，E4M3 | 1568 | 482,818 | 1.842 | 17,248 | 312,499 | 1.192 | 1 | 26,656 | 6,272 |
| 不開 MTP，A | 2784 | 756,184 | 2.885 | 25,056 | 488,999 | 1.865 | 1 | 41,760 | 8,352 |
| 不開 MTP，C | 1392 | 771,011 | 2.941 | 30,624 | 498,587 | 1.902 | 1 | 45,936 | 12,528 |

p070 的 N=2／N=4 每人 ctx：MTP4 E4M3 111,504／35,552、A 174,704／51,552、C1 166,448／48,480；
不開 MTP E4M3 150,528／67,424、A 242,208／108,576、C 247,776／112,752。
在 p070 的預算下，MTP4 於 N≥8 時 GDN 的固定 state 稅已吃光預算，任何 KV 格式都救不了；
不開 MTP、N=16 時 A 比 E4M3 多 33%、C 多 100%。

## 8. 驗證計畫骨架

**規則**：所有 GPU 執行由 **lead 排程**；hands 不用 GPU、不起停服務；`:8001` 是正式服務，不動。
本節只列「要跑什麼」，不執行任何一項。分層沿用 `serving-ab` skill 的 E1/E2/E3：target KV 量化是 **E3**，
因此「與 E4M3 逐 token 一致」的 parity **不適用**，改以統計式品質探針與 paired SWE 驗收。

| 層 | 內容 | 工具（repo 路徑） | 通過條件／產出 |
|---|---|---|---|
| **L0 CPU** | spec 頁數學、allocator 組態、B 規則（§4.2 的 dry-run 轉成 unit test）；純 torch 量化參考對 `ref_nvfp4_quant`（`nvfp4_emulation_utils.py:533`）；rounding 邊界（全部 fp16 輸入、平手點）| pytest（`tests/v1/core/`、`tests/models/qwen4_exp/`，與 PR #826 的 CPU 測試同類） | 逐 code 比對；allocator 數字與 §4.2 一致；`ValueError` 情境（C2）覆蓋 |
| **L1 單 GPU（分鐘級）** | store→read 往返；reader 對 FP16 與對 E4M3 參考的層級數值；**poison 測試**（快取預填 0xFF／NaN，含交錯非法 index、padding 列）、重複／PAD slot、CUDA graph replay、int64 位移（大 `num_blocks`）、V100 上的 `rows` 1／5／20／64／8192 | 仿 PR #826 的「四個成對 native QSA attention case」（12 head、D256、32K ctx、top-k 512／4096；E4M3 對未量化 FP32 的 relative L2 為 2.45%–2.76%）與 SGLang 的 `test_qsa_nvfp4_fused.py` 設計 | 閘門以**同一 harness 的 E4M3 臂為基準**（NVFP4 的 relative L2 與 E4M3 並列報告），不設跨 harness 的絕對值；全部有限、無 NaN。CPU 合成實驗（§8.1）顯示 NVFP4 的 attention-output 誤差約為 E4M3 的 1.4–6 倍，**不能拿它當門檻**，只作預期量級 |
| **L2 TP4 端到端（lead）** | no-MTP、版面 A、262144 ctx 啟動；route-hit 證據（log 要有明確命名 NVFP4 路由的 `info_once`，並確認是**量測請求**走到而非只有 warmup）；0 NaN／0 OOM；`GPU KV cache size` 與 §7.2 對表 | `serving-ab` skill 的重啟慣用法（fence、co-tenant 守衛）、`scripts/eval/serving-parity.py`（`--label`／`--compare`） | 先做 **determinism floor**：同一 NVFP4 server 錄兩次（label a／b）再 `--compare`，不一致就先不比較其他臂。與 E4M3 的 `--compare` 只作資訊（預期逐 token 發散） |
| **L3 品質探針** | teacher-forced logprob：窗口 2K／8K／32K／96K（再補 192K 與接近 262144 的窗口）；**每個比較必須含 bf16（或 FP16 KV）臂與 E4M3 臂各一個 rerun 地板**；另加與 NVFP4 同 B 的 **E4M3 控制臂**（分離「量化」與「B 變大造成的重算／切塊」） | `scripts/eval/kv-logprob-probe.py`；方法與門檻見 `notes/methodology/kv-cache-quality-probe.md` | 現況：該腳本的 `capture` 只呼叫 SGLang 的 `/generate`、`/server_info`、`/flush_cache`，**不能把 `--base-url` 換成 :8001 直接跑**，需先做 vLLM adapter（`/v1/completions` numeric prompt、`prompt_logprobs`，約 1–2 工程日，Astra G8 步驟 1）。只看 window-mean ΔNLL 對 rerun 地板的 95% CI（對窗口 bootstrap），per-position 指標被非確定性淹沒（SGLang 的 bf16 rerun top-1 一致僅 91.6%）。**門檻須由 owner 在看結果前定義**；Astra 建議的嚴格候選線是 CI 上界 ΔNLL ≤ .005 nats/token |
| **L4 serving A/B** | decode／prefill／chunk 與容量 | `scripts/bench-agentic-shapes.py --label <name> --out /data/bench/<name>.json`（gen／copy／chain × 1K–64K，`/metrics` delta，server 必須閒置）| 以 `round_ms` 判讀（噪音 <0.2%），不看 tok/s；預期差距 <10% 的槓桿必須 **ABBA 交錯、≥2 輪、≥4×並發數的 prompt**，並回報各臂輪間離散度當噪音地板。另外量 B 變大造成的 chunk 縮水（B=2864 時 8192 預算實得 5728 token） |
| **L5 SWE-34 與並發** | 凍結的 SWE-34 exam，與 c4140 的 E4M3 + MTP4 各格（`notes/machines/dell-c4140.md`：同一 stack 的各格為 26–29/34，彼此統計上持平）paired；N=2／4／8／16 的 admission／preemption 壓力；20-session 輪轉；實量 KV IDs、graph／workspace 峰值、OOM、cache hit 與重算 | `scripts/run-swe-personal.py`（canonical exam）、`scripts/eval/swe-contamination.py`、`footing-budget.py`、`footing-agency.py` | SWE-34 的單題 = 2.94 點、per-task run-to-run 波動 24%，只能排除大效應；「未偵測到差異」不是等價。**不得**用 decode-only 的勝利宣稱 SWE 增益（`serving-ab` skill 的規則）；保留 E4M3 回退配置 |

### 8.1 預期的量級（CPU 合成實驗，非模型資料）

2051 個 key、D=256、12 個 query head；K、V 取高斯／t(3)／含離群通道三種分佈；E4M3 用逐張量
`amax/448`，NVFP4 用逐 16 組 E4M3 scale + 同一個全域 scale；logit 標準差 1 與 4：

| 分佈 | 逐元素 relL2 E4M3／NVFP4 | attention 輸出 relL2（logit std 1）E4M3／NVFP4 | （logit std 4） |
|---|---|---|---|
| 高斯 | 0.0265／0.0954 | 0.0378／0.1321（3.5×） | 0.1009／0.3033（3.0×） |
| t(3) 重尾 | 0.0251／0.0870 | 0.0589／0.1033（1.75×） | 0.0330／0.1955（5.9×） |
| 離群通道 | 0.0266／0.0898 | 0.0727／0.1095（1.5×） | 0.1293／0.1742（1.35×） |

**怎麼讀**：NVFP4 逐元素誤差約為 E4M3 的 3.5 倍，是合成資料上的**悲觀量級**；真實模型的 SGLang SM120 量測
（§3.5）顯示 teacher-forced ΔNLL 僅 +0.004～+0.010 nats，遠比這個合成估計溫和。
⚠ **差異**：SGLang 的前 4096 token chunk **不讀 pooled KV**（2K 窗口對所有 KV dtype 逐位元相同、不帶 KV 訊號），
而 1Cat 的 QSA 是先寫再讀（§2.1 b1），**量化誤差在首個 chunk 就生效**。因此 SGLang 的 ΔNLL 在 1Cat
上**不能當上界**；L3 的 2K 窗口在這裡帶有訊號，應納入。這也是為什麼要先跑 L1 的 route-hit 證據。

## 9. 工作量估計與開放問題

### 9.1 各階段工作量（工程日，**我的估計，不是承諾**；Astra G8 估 NVFP4 sparse QSA 完整版 1–3 工程週 + 約 1 週驗收）

| 階段 | 內容 | 估計 | GPU 需求 |
|---|---|---:|---|
| S0 | CPU：owner／spec／dtype admission（§2.1 a1–a6、f1）、純 torch 參考與測試、allocator dry-run 轉 unit test | 2–3 | 無 |
| S1 | Triton store kernel + L1 往返測試 | 2–3 | 單 GPU，分鐘級 |
| S2 | Triton reader（`KV_NVFP4`、merge 旗標、路由守衛）+ L1 數值／poison／graph replay | 3–5 | 單 GPU |
| S3 | 版面：A = 0–0.5（驗證 B 規則與 spec）；C（pack-2）= 2–4（B 規則、packed stride admission、zeroer／offload 回歸）；(b) 混合 draft = 再 3–5；B（獨立頁池）= 未估（>2 週） | 0–5 | 無 |
| S4 | TP4 no-MTP 啟動、route-hit 證據、vLLM 版 logprob adapter（1–2）、第一輪 L2／L3 | 3–4 | TP4，lead 排程 |
| S5 | MTP4 全 NVFP4（版面 A）：draft scale 推廣、接受率與 round_ms | 1–2 | TP4 |
| S6（選配） | CUDA reader（decode c≥4、prefill grouped page4）+ ABI v3 + 建置 | 5–10 | 單 GPU + TP4 |
| S7 | L3 全套、L4 ABBA、L5 SWE-34 與並發 | 約 1 週牆鐘 | TP4，lead 排程 |

到「版面 A、no-MTP、第一個端到端 + 品質探針」約 **10–15 工程日**（S0–S4），與 Astra 的 1–3 週一致。
S6 只在 L4 顯示 c≥4 或 prefill 退步時啟動。

### 9.2 開放問題（需 Fable 或 owner 決定／需實測）

1. **global scale 的定義**：沿用 `k_scale`（E4M3 慣例，block scale 最大 ≈74.7）或改 `k_scale/6`（用滿 E4M3 範圍）？
   這批 checkpoint 的 `k_scale` 是否為 `amax/448` 未驗證。
2. **捨入平手**：Triton 軟體 e2m1 與 upstream PTX `cvt.rn.satfinite.e2m1x2` 是否逐 code 相同？V100 上無法用 PTX 對拍。
3. **Triton reader 的實際速度**：ALU 解碼 vs 少讀 43.75% 位元組的淨效果；偶／奇半邊免 interleave 的寫法是否有效；
   c≥4 decode 與 prefill 是否必須上 CUDA。
4. **品質閘門**：由 owner 在看結果前定義（§8 L3）；首 chunk 也讀量化 KV 使 SGLang 證據不能當上界。
5. **draft 量化**：NVFP4 draft 的接受率變化；scale 是 `prototype:from-layer-47` 的複製值。
6. **版面選擇**（決策：Fable）：目標並發 N 決定 A／C／pack-n／B 的取捨（N≤4 選 A；N≥8 考慮 C 乃至 B）；B 變大對 TTFT 與
   `max_num_batched_tokens` 的影響（chunk 縮水約 30%）需實測。
7. **packed 頁在 DCP1 的 NVFP4**：`KVBlockZeroer`、offload worker、v1 runner／PP 的拒絕行為；QSA main 群的 kernel block 大小與
   「逐 kernel block 交錯」的粒度（`attn_utils.py:266-280`）。
8. **`_align_hybrid_block_size` 的 B 規則怎麼改**才能支援 C（目前 `--block-size` 反而放大）：新增旋鈕或讓規則感知打包。
9. **NVFP4 × DCP2**：未分析；E4M3 + DCP2（+52% 容量、decode −2.7%）是否可與 NVFP4 疊加，或兩者擇一。
10. **KV offload／connector**：頁位元組與 dtype 無關，但 `nvfp4` 頁未測過（`supports_kv_connector=False` 是 QSA backend 的設定，
    grouped offload 另走 #598／#617）。
11. **CLI 與 platform 閘**：`--kv-cache-dtype nvfp4` 在 `CacheDType` 內（`config/cache.py`）；`vllm.py:3524` 只擋 MLA；
    `vllm/platforms/*.py` 內沒有 nvfp4 專屬拒絕（grep 無命中），但要實測確認沒有別處（例如 backend 選擇器的 dtype 白名單）攔下。

### 9.3 未驗證項目彙整

- 所有 §7 的每人 ctx、B 上限、與 pack-n 的 decode 開銷。
- Triton reader 的實際效能、prefill 的 NVFP4 退步幅度、CUDA 路由的必要性。
- 捨入平手與全域 scale 定義、`k_scale` 是否為 `amax/448`。
- packed 頁 + NVFP4 + `KVBlockZeroer`／offload／v1 runner；`kernel_block_size` 對 QSA main 群的實際值。
- V2 runner 下 `SpeculativeConfig.kv_cache_dtype` 對 MTP 無效（由讀碼推得，未實測）。
- c≥4 與 prefill 的 E4M3 路由（由讀碼推得，未以 route-hit 實測）。
- 1.247 GiB draft 權重回收（Astra G2 的估計）。
- `c13-M94.serve.log`、`sampling_params.py`、`prompt_logprob.py` 的 Astra 錨點（非本案必要，未重解）。

### 9.4 本文件的重現方式

- allocator dry-run：以 `/data/venvs/1cat-p070/bin/python`（唯讀使用，`CUDA_VISIBLE_DEVICES=` 空）匯入 `vllm.v1.core.kv_cache_utils`，
  建 12 或 13 個 QSA owner（`FullAttentionSpec`＋`MLAAttentionSpec`＋`CircularBufferSpec`）與 36 個 GDN＋1 個 PLE 的 `MambaSpec`，
  呼叫 `get_kv_cache_groups` → `get_kv_cache_config_from_groups` → `get_max_concurrency_for_kv_cache_config`。
  `NS` stub 的欄位：`parallel_config.{decode_context_parallel_size,prefill_context_parallel_size,pipeline_parallel_size}`、
  `model_config.{max_model_len,get_num_kv_heads}`、`cache_config.{mamba_cache_mode,num_gpu_blocks_override}`、
  `scheduler_config.disable_hybrid_kv_cache_manager`。
- 錨點漂移：`git diff -U0 25361ff5f c4140-p070 -- <檔案>` 的 hunk 映射。
- 合成量化實驗：純 torch CPU，`torch.manual_seed(0)`。
