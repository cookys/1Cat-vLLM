# C4140 plan 071：QSA main K/V 的 NVFP4 KV cache（階段 1，僅 CPU）

基底 `c4140-p070`（`1b5731f90`）。階段 1 做參考實作、dtype 准入、cache spec 與 allocator 推導；
之後依序接上 global scale（步驟 A）、Triton store 與呼叫點契約（步驟 B），並記錄編譯路徑與 reader 計畫（步驟 C）；
步驟 D 把 reader 接進稀疏注意力（先 gather 再走 FP16），步驟 E 是單 GPU 的內核數值與計時腳本。
`--kv-cache-dtype nvfp4` 現在在沒有 DCP2、沒有 MTP draft 的設定下可以執行，但**沒有在任何 GPU 上跑過**：
證據只有直譯器測試與 sm_70 編譯閘門。

## 已定決策（Fable）

- **版面採選項 A**：放大 block，讓 NVFP4 頁剛好放得下 GDN state 頁。不改 allocator、platform、worker。
  選項 C 留作團隊並發的後續：把兩個 NVFP4 頁打包進一個實體頁，不開 MTP 時 block 為 1392，重用
  DCP2 已有的 packed 頁機制，但要改 platform 的 block 規則。選項 B（state 獨立頁池）不做。
- **MTP 順序**：不開 MTP 的 A；接著 draft 也用 NVFP4 的 A；最後才做 target NVFP4 加 draft E4M3 的混合版面。
  混合版面延後，不是被排除：「混合版面會打破 allocator」這個前提來自 Astra 的基底 `25361ff5f`，
  在 p070 不成立。p070 的 allocator 有逐 owner 頁清單與 packed 頁（PR #823），dry-run 顯示該配置被接受；
  剩下的阻礙只是 worker 與 platform 層的單一全域 cache dtype。
- **dtype 名稱**：沿用既有的 `nvfp4`，不另取新名。

## dtype 名稱與 SM100 約定（SM70 參考必須與之一致）

名稱 `nvfp4`：`config/cache.py:34`（`CacheDType`）、`kv_cache_interface.py:65-66`（`get_kv_quant_mode`）、
頁大小 `:186-204`、`:299-319`、儲存 dtype `utils/torch_utils.py:98`（uint8）、列位元組
`:464-466`（`head//2 + head//16`）、SM100 分派 `csrc/libtorch_stable/cache_kernels.cu:840-857`。

SM100/SM120 的 store（`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`、`quantization/fp4/nvfp4_utils.cuh`）：

| 項目 | 約定 | 位置 |
|---|---|---|
| 頁版面 | `[K_data \| K_scale \| V_data \| V_scale]`，每側先資料後 scale | `nvfp4_kv_cache_kernels.cu:8`、`:218-245` |
| 打包 | 16 值打成 8 byte；偶數索引在低 nibble | `:139-145`；`nvfp4_utils.cuh:118-153`（`cvt.rn.satfinite.e2m1x2 b0, %3, %2`，第一個來源進高 nibble） |
| scale 位元組 | `sf = (1/k_scale) × (amax × rcp(6))`，E4M3 飽和 RNE；`out = rcp(sf_q × rcp(1/k_scale))` | `nvfp4_kv_cache_kernels.cu:95`；`nvfp4_utils.cuh:240-271` |
| K 與 V 的 scale 排列 | K 線性，V 為 SM100 的 4×4 swizzle。V100 兩者都用線性 | `nvfp4_kv_cache_kernels.cu:161-171`、`:33-39` |
| 位元級對拍 | 參考實作用同樣的運算順序，除法取代 `rcp.approx.ftz`，差異只在 fp32 中間值貼著捨入邊界時 | 見測試 |

## global scale 的決定

**決定**：每層 K、V 各一個 fp32 純量，語意是反量化乘數，`dequant = e2m1 × block × global`。
checkpoint 有就照用，沒有就是 1.0。這與 fleet 的 SGLang 實作和 SM100 store 一致，沒有語意衝突，
所以不採用「由校準 amax 推導」的替代方案。上游測試的 `amax/448` 只是測試取值，不是約定。

| 面向 | SGLang（`cookys/sglang-qsa-nvfp4-kv` @1118e72） | 1Cat 本分支 |
|---|---|---|
| 反量化 | 乘 global：`patches/qsa_nvfp4_kv.py:173-174`、`:257-262` | `nvfp4_kv.py:353-375`，`block = e4m3_decode(scale) × layer_scale` |
| 取得 | `layer_global_scale`（`:120-135`）以全域 layer id 索引，越界回 1.0；取用在 `:188-189` | `layer._k_scale`／`_v_scale`，由 `qsa.py:801-840` 從 checkpoint 槽位複製 |
| 缺值 | 寫入端存 1.0（`docs/mechanism.md:62-71`），驗收就在 1.0 下跑 | 保留預設 1.0；`model.py:168`、`:211` 對 nvfp4 只記 info，不拒絕 |
| store 端 | `1/global` 當 SM100 的 `SFScaleVal` | 相同：`scale_inverse = 1.0 / layer_scale` |

E4M3 沒有校準就拒絕啟動，NVFP4 不拒絕：block scale 已逐 16 值吸收動態範圍，
測試顯示 `amax/2688`、`amax/448`、`amax/200` 與 1.0 的 relative L2 都在 8.5% 到 10.5%。
E4M3 overlay 的 24 個純量為 K 0.0186 到 0.0367、V 0.0171 到 0.0841，它們的倍數是 448 還是 200 不在 repo 與 provenance 內，[未驗證]。

守衛：載入值必須有限、為正，且不超過 `NVFP4_KV_LAYER_SCALE_MAX = 10`（`nvfp4_kv.py:78-84`、`:253-269`，
錯誤訊息帶層名）。group 的 block scale 是 `amax16 / (6 × layer_scale)`，低於 2^-6 就成為 E4M3 次正規數、失去尾數位；
amax 至少 1 的層在 layer_scale 超過 10.67 時會發生。overlay 的 K／V amax 都大於 3，所以 10 留有餘裕，
又能擋掉為別種格式準備的 scale。`nvfp4_layer_scale_from_amax`（`amax/2688`）與 `nvfp4_block_scale_range`
留作需要推導時使用，目前沒有呼叫端。測試在 `test_nvfp4_kv_scale.py`，含從 SGLang 測試移植的案例。

**SM100 編譯守衛**：store 內唯一 SM100 以上才有的內建指令是 `cvt.rn.satfinite.e2m1x2.f32`
（`nvfp4_utils.cuh:70-153`，無條件的 inline asm，沒有 `__CUDA_ARCH__` 守衛，測試已釘住）。其餘是一般運算：
half2 的 abs 與 max、`__nv_fp8_e4m3(float)`、`rcp.approx.ftz.f32`、`__shfl_xor_sync`、整數存取。
`CMakeLists.txt:1123-1127`、`:1157-1161` 只對 10.0a／10.1a／10.3a／12.0a／12.1a 加入此檔，
`cache_kernels.cu:841-856` 其餘架構報錯。若要編 sm_70：把 `fp32_vec8_to_e2m1`／`fp32_vec16_to_e2m1` 換成軟體
捨入（`e2m1_encode` 即規格）、排除 bf16 實例化（bf16 的 half2 運算需 sm_80 以上，[未驗證]）、加 CMake 架構項與分派巨集。
這對階段 1 不需要：Triton store 用軟體 e2m1 捨入，不需要 nvcc；CUDA store 只有在 reader 存在之後才有用。沒有建置任何東西。

## 寫入路徑（store）：契約與 SM100 分支對照

呼叫點在 `Qwen4ExpQSAAttention._run_qsa`：`impl.do_kv_cache_update(self, key, value, self.kv_cache, main_metadata.slot_mapping)`，
prefill、decode、draft 都走這裡。`Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update`（`qsa.py:290-329`）對 nvfp4 轉給
`store_nvfp4_kv_triton`，其他 dtype 仍是 `reshape_and_cache_flash`。`_run_qsa` 更早就對 nvfp4 拋 `NotImplementedError`，
所以真實路徑仍是關的，store 目前只被測試呼叫。

| 參數 | 契約 |
|---|---|
| `key`、`value` | `[rows, num_kv_heads, head_dim]`，fp16（fp32、bf16 也可）。只要求最後一維連續；`value` 是 QKV 輸出的切片，token 與 head stride 照傳。`rows` 可以大於 slot 數（CUDA graph padding），多出的列不寫 |
| `slot_mapping` | int64 `[num_actual_tokens]`，`block × block_size + offset`。負值跳過（padding，或 DCP 下別的 rank 擁有的 token）；大於等於 `num_blocks × block_size` 也跳過 |
| `kv_cache` | 5 維 uint8 `(num_blocks, 2, block_size, heads, 144)`，worker 建立的 view。packed member view 的 block stride 較大，照 stride 寫，不碰鄰居 |
| `k_scale`、`v_scale` | `layer._k_scale`／`_v_scale`，0 維 fp32 device 純量，內核內讀取，所以 CUDA graph 可擷取 |
| 重複 slot | 同一次呼叫內重複的 slot，內核結果未定義，與 SM100 相同，scheduler 不會產生；參考 writer 規定後寫者勝 |

SM100 的 NVFP4 分支（`cache_kernels.cu:840-857` 分派，`nvfp4_kv_cache_kernels.cu` 內核與檢查）做什麼，以及 SM70 store 是否對齊：

| SM100 行為 | 位置 | SM70 Triton store |
|---|---|---|
| token 數取 `slot_mapping.size(0)`，key 多餘的列不碰 | `cache_kernels.cu:825-833`；`nvfp4_kv_cache_kernels.cu:191` | 相同 |
| `slot < 0` 跳過，沒有上界檢查 | `:77` | 相同，另加上界；對合法輸入沒有差別 |
| 讀 `*k_scale_ptr`、`*v_scale_ptr` 為 device 純量，用 `1/scale` | `:95` | 相同 |
| 一個 CTA 一個 token，K 與 V 在同一個內核 | `:256` | 一個 program 一個 token 與 head，K、V 各啟動一次；每層每次 forward 多一次小啟動，成本沒量 |
| 只用 `key.stride(0)`，假設 head stride 等於 head_size | `:103-104`、`:272` | 取實際 head stride，較寬鬆 |
| 只收 fp16、bf16 | `:263` | fp16、bf16、fp32 |
| `block_size % 4 == 0`，因為 V 的 scale 有 4×4 swizzle | `:210` | V100 為線性，沒有這個要求；2784、2864、1392 本來也都是 4 的倍數 |
| V 的 scale 做 swizzle，K 線性 | `:161-171` | 兩者都線性 |
| 以 stride 判斷 HND 或 NHD，scale 區緊接在資料區後 | `:216`、`:230` | 由 `nvfp4_kv_split_views` 給 stride，版面相同 |
| e2m1 轉換用 `cvt.rn.satfinite.e2m1x2` | `nvfp4_utils.cuh:118-153` | 軟體比較；tie 規則對 torch 規格窮舉驗證，與 PTX 的 tie 行為[未驗證] |

## 推導公式與結果

`Platform._align_hybrid_block_size`（`vllm/platforms/interface.py:672-708`）：

```text
attn_block_size = align × cdiv(state_page, align × attn_page_size_1_token)
align           = max(kernel block 16, cache_config.block_size)
attn_page_size_1_token = 2 × heads × (head//2 + head//16) × 1 B   # NVFP4：288；E4M3：512
```

state 頁是 `MambaSpec.real_page_size_bytes`，TP4、fp16 conv、fp32 SSM：GDN 801,792 B（K=0）、
822,272 B（MTP4）；PLE conv 184,320／266,240 B，較小。

| | 每 token 每層 | K=0 的 block | K=4 的 block |
|---|---:|---:|---:|
| E4M3（現況） | 512 | 1568 | 1616 |
| **NVFP4** | **288** | **2784** | **2864** |

- 288 B/token 推得**整數**。K=0：`2784×288 = 801,792`，頁與 state 相等，**零 padding**。
  K=4：`2864×288 = 824,832`，每頁 pad 2,560 B（0.31%）。不需退回補 padding 的做法。
- 推導與 allocator 沒有寫死 1616 之類的假設；用真正的 `_align_hybrid_block_size`（QSA backend）與真正的
  `get_kv_cache_groups` 在 CPU 上驗證。
- 代價 [未實測]：align 模式的 prefill chunk 在 `max_num_batched_tokens=8192` 下由 7840／8080 縮到 5568／5728；
  prefix 重用粒度變粗。明確傳 `--block-size` 只會放大。

## 容量：兩個 KV 預算

真 allocator 在 CPU 上算的 `GPU KV cache size`。p069 預算是 292 個 ID（3.2907 GiB/rank）。
p070 預算由實測錨點反推：E4M3 加 MTP4、B=1616 的 263,538 tokens 剛好是 189 個 ID，
所以預算在 2.1299 到 2.1412 GiB 之間；下表用下界，上界的結果只差 1 到 3 個 ID。
「262K 等價」是 `IDs / (ceil(262144/B)+F)`；每人 ctx 是 N 人同時 active 時的 allocator 模型預算，不是 SLA。

| 配置 | B | p069：tokens／262K 等價／N=16 每人 ctx | p070：tokens／262K 等價／N=16 每人 ctx |
|---|---:|---|---|
| MTP4 E4M3 | 1616 | 407,159／1.553／0 | 263,538／1.005／0 |
| MTP4 A | 2864 | 602,707／2.299／0 | 389,855／1.487／0 |
| MTP4 C1（NVFP4 打包加 E4M3 draft） | 1616 | 579,007／2.209／0 | 374,127／1.427／0 |
| 不開 MTP，E4M3 | 1568 | 482,818／1.842／17,248 | 312,499／1.192／6,272 |
| 不開 MTP，A | 2784 | 756,184／2.885／25,056 | 488,999／1.865／8,352 |
| 不開 MTP，C | 1392 | 771,011／2.941／30,624 | 498,587／1.902／12,528 |

p070 的 N=8 每人 ctx：MTP4 三種皆為 0；不開 MTP 為 26,656（E4M3）、41,760（A）、45,936（C）。
不開 MTP 時沒有計入回收 draft 權重的預算；Astra 的 1.247 GiB 是 p069 的估計，p070 的權重不同，未重估。
p070 上 MTP4 在 N≥8 時 GDN state 稅就吃光預算，4-bit KV 解決不了。

p070 欄反映目前 p070 的 pool，錨點是 263,538 tokens。這個 pool 比 p069 少，已識別的原因是上游 `b75d3cd0f`
把五個 batch pack 開關的預設由 0 改為 1，可逐項核算的額外副本約 1.23 GiB 每 rank
（llm-playground repo 的 `doc/survey/2026-10-03-c4140-p069-astra.md` §I，`:3003-3012`、`:3068-3072`）。
Fable 在 GPU 上驗證五個開關全關後，預期 pool 回到約 411K 到 414K tokens；這是預期值，不是量測。
本欄會在該實測 pool 出來後重算，這裡不自行估計。

## 已交付（皆 CPU）

| 項目 | 位置 |
|---|---|
| 純 torch 參考：量化、反量化、頁版面、writer、含非法索引 zero-fill 的 sparse gather，運算順序同 SM100 store | `vllm/models/qwen4_exp/nvidia/ops/nvfp4_kv.py` |
| dtype 准入（backend、impl、owner）、`get_kv_cache_shape`（末維 144）、執行 guard | `vllm/models/qwen4_exp/nvidia/qsa.py` |
| global scale 接線：載入、次正規數守衛、`nvfp4` 不要求校準 | `qsa.py`、`model.py`、`nvfp4_kv.py` |
| Triton fused unpack+scale gather（直譯器逐位元等於參考） | `nvidia/ops/nvfp4_kv_triton.py` |
| Triton store（量化、打包、寫 scale）與 `do_kv_cache_update` 契約（直譯器逐位元組等於參考 writer） | `nvfp4_kv_triton.py`、`qsa.py` |
| sm_70 編譯閘門：不需 GPU，檢查 PTX 與暫存器，含步驟 D 的 split-K 與 merge 內核 | `tests/models/qwen4_exp/test_nvfp4_kv_sm70_compile.py` |
| reader：gather 後走 FP16 split-K，layer scale 以 `FOLD_SCALES` 折疊，放行三道門，保留 DCP2 與 MTP draft 的拒絕 | `ops/qsa_nvfp4.py`、`ops/qsa.py`、`qsa.py` |
| 單 GPU 的內核數值與計時腳本（未執行）與它的 CPU 測試 | `benchmarks/sm70_nvfp4_kv_kernel_check.py`、`tests/models/qwen4_exp/test_sm70_nvfp4_kv_kernel_check_cpu.py` |

放在 `nvidia/ops/` 與 `qsa.py`、`qsa_kv_calibration.py` 並列，後續 kernel 從這裡取格式常數；
`nvfp4_kv_cache_full_dim` 與 `nvfp4_kv_cache_split_views` 留在 upstream 的 `utils/torch_utils.py`，只匯入不修改。
寫入端不產生負零 nibble（0x8）。非法 top-k 索引（負值、超出 block table、未映射或越界 block、無效 request）
在兩個區都 zero-fill，且不讀取 cache。

逐元素誤差（8192×256，fp16 輸入，global scale 1.0）：

| 輸入 | NVFP4 max abs／mean abs／relative L2 | E4M3 逐張量 relative L2 |
|---|---|---:|
| 高斯 | 0.619／0.0715／9.5% | 2.65% |
| Student-t(3) 重尾（max 161） | 6.14／0.112／9.1% | 2.64% |

## 測試

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD uv run --no-project --python /data/venvs/1cat-p070/bin/python \
  --with pytest -- python -m pytest --noconftest \
  tests/models/qwen4_exp/test_nvfp4_kv_{reference,admission,scale,triton,store,decode,sm70_compile}.py \
  tests/models/qwen4_exp/test_sm70_nvfp4_kv_kernel_check_cpu.py -q
343 passed, 83 skipped   # 參考 129、准入與 spec 與 allocator 65、scale 66、編譯閘門 27、步驟 E 腳本 56；Triton gather 19、store 31、decode 32 與腳本的 1 個端到端預設略過
TRITON_INTERPRET=1 同上   # 399 passed, 27 skipped；編譯閘門需要真的編譯器，直譯器下略過
```

- 參考測試含「SM100 約定釘選」：讀 `csrc` 原始碼字串，斷言頁版面、`1/k_scale`、scale 運算順序、
  e2m1 運算元順序與「store 內只有 e2m1 一種 cvt、沒有 `__CUDA_ARCH__`」；上游改了約定就會失敗。
  另有「運算順序決定位元組」的邊界案例（layer scale 7 時 30 個 fp16 amax 會分岔）。
  上游自己的 SM100 測試容差很鬆（`test_cache.py:383-387`，atol 1.5、rtol 0.5），且對 K 也套用 V 的 swizzle 解碼
  （`:358-374`），而 kernel 寫 K 為線性（`nvfp4_kv_cache_kernels.cu:161-164`）；這些測試釘不住位元，[未驗證其影響]。
- store 測試（32 個）逐位元組比對參考 writer：隨機、邊界、飽和、次正規、9 個數量級的動態範圍、padding 與越界 slot、
  多餘列、strided value、fp32 輸入、三種 head size、packed member view、`do_kv_cache_update` 契約、store 接 gather、
  輸入驗證，以及兩個內核內編碼器對 torch 在每個 tie 與每個 fp16 值上的窮舉。
  突變檢查：換 nibble 順序 17 個失敗、拿掉 tie 規則 13 個失敗、拿掉資料寫入的遮罩使行程崩潰、
  改成 `amax/6` 的運算順序 2 個失敗、輸出 scale 用錯 3 個失敗。
- 編譯閘門（27 個）：Triton 3.6.0 內附的 ptxas 對 sm_70 編 store、gather，以及步驟 D 的 split-K 與 merge 內核，
  斷言目標 sm_70、PTX 無 bf16 或 fp8 轉換、無 e2m1、wgmma、`cp.async`、新架構 mma，store 無 atomic；
  store 與 gather 暫存器不超過 64、stack 不超過 16 B，gathered 路徑的 split-K 在 2 與 4 個 warp 都沒有 stack。
  把 store 的 block scale 改成原生 fp8 轉換會讓 13 個失敗；讓 split-K 內核不折疊 scale 會讓折疊檢查失敗。
  它只證明能編譯、資源小，不證明能跑或跑多快。先前這一條寫「不溢出到 local memory」不精確：store 在 1 與 2 個 warp 有
  8 B 的 stack，閘門現在同時檢查 stack 與 local。
- decode 測試（32 個，直譯器）：NVFP4 路徑對「同一個 FP16 內核跑在解碼後的 FP16 cache」逐位元相等（TP4 形狀、
  單 tile 不經 merge、兩個 KV head）；非法 top-k 項（負值、超出 table、未映射頁、越界 block、不存在的 request）、cache 中
  被 NaN scale 毒化卻只被遮掉的 block、每頁首尾 token、共用頁與重複 token 的混合批次，都與 FP16 路徑相等；
  2 的冪次 layer scale 對「預先縮放的 FP16 K、V」逐位元相等（釘住 k_scale 乘在分數、v_scale 乘在輸出），
  單 split 與多 split 兩個分支各一；一般 layer scale 對 FP32 注意力用 rtol 3e-2 比對（FP16 機率與輸出捨入的量級）；
  gate 與 lse 輸出、列分組、page4 路由不會被呼叫、入口驗證、`forward_qsa` 的 nvfp4 路徑與它的拒絕、MTP draft 拒絕、
  halves 版 gather 等於 5 維版。Triton 直譯器跑不動現行 merge 內核（對純量與區塊做 `&`，編譯後沒有問題），
  所以 CPU 上 split 數大於 1 的測試用一個算術相同的 torch 函式取代它，兩邊比對都用同一個；真的 merge 內核搭
  `FOLD_SCALES=True` 只由編譯閘門與步驟 E 的 GPU 腳本涵蓋。
  突變檢查（在 production 程式碼上）：不遮非法項 3 個失敗、K 與 V scale 對調 4、不折疊 scale 4、K 的折疊用了 V 的 scale 4、
  validity 少了物理 block 上界 1、分組時用了全部 request 2、拿掉 forward_qsa 的 DCP 拒絕 1；
  內核裡 V scale 不折疊一開始沒有被抓到（31 個全過），原因是單 split 分支沒有非 1 純量的測試，補上後 2 個失敗。
- 既有內核沒有被改動的證據：把 split-K（E4M3 與 FP16，三種 tile 與 warp 組合）與 merge（E4M3 與 FP16）共 8 個既有組態
  編到 sm_70，去掉原始碼行號資訊後的 PTX 在修改前後逐位元相同。
- 步驟 E 腳本的 CPU 測試（57 個）：參數、從真實 config 讀出的幾何、case 與 cache 規劃、種子化的 K/V 與含非法項的 top-k
  （非法位置的預期與 `nvfp4_entry_validity` 逐項一致）、位元組比對、誤差與計時統計、ABBA 順序、verdict 與結束碼、
  沒有 CUDA 與 GPU 已被佔用時的快速失敗，以及自檢的邏輯（突變函式、tie 資料、誤差帶、各種失敗與跳過階段）；
  直譯器下另有一個小 case 從 store 到計時與自檢完整跑一遍，要求 `SELF_CHECK: PASS`。
- worktree 內 `vllm/*.so` 與 `_version.py` 是指向 `/data/venvs/1cat-p070` 的符號連結（已被 git 忽略，沒有建置），
  讓 worktree 的原始碼能載入 `vllm._C`；沒有修改任何 venv。
- 沒有可見 GPU 時，匯入 `qsa.py` 會因 `flash_attn.py` 向 GPU 詢問 FlashAttention 版本而失敗；
  相關測試檔在 `torch.cuda.is_available()` 為假時用 FA2 的答案取代（Volta 本來就是 FA2）。
- 鄰近測試（QSA 快取、E4M3、DCP、page4、hybrid block、auto E4M3 policy、kv cache utils、既有的 sparse 內核測試）：
  216 passed、30 skipped；失敗的 26 個是沒有 CUDA 時才失敗的 GPU 測試（`test_qsa_reference.py` 等，它們也涵蓋被改動的 sparse 內核，要在 GPU 上跑），
  2 個錯誤是 `--noconftest` 下找不到 fixture；`test_config.py` 的 4 個失敗在 `c4140-p070` 本來就失敗，與本案無關。
- 提交使用 `--no-verify`：共用的 `.git/hooks` 被裝上 pre-commit，首次執行要下載全部工具環境而卡住。
  改為直接跑 ruff 0.14.0 的 `check` 與 `format`、typos、markdownlint-cli2 0.21.0，以及 `tools/pre_commit` 內的
  spdx、forbidden-imports、torch-cuda、boolean-context-manager、lazy-imports、env-registration 腳本。mypy 未跑。

## 編譯路徑與 reader（步驟 C，只有計畫，沒有 reader 程式碼 — 2026-10-05 起有程式，見下）

### sm_70 編譯實測

不需要 GPU：Triton 3.6.0 內附的 ptxas 把內核編到 sm_70，再用 `cuobjdump --dump-resource-usage` 讀資源。
E4M3 那三列的 constexpr 是照內核簽名手動填的（TOPK 2051、PAGE_SIZE 1616、BLOCK_M 8、GROUP_SIZE 6、HEAD_DIM 256），不是從實際啟動擷取。

| 內核 | warps | 暫存器 | stack | shared（Triton） |
|---|---|---:|---:|---:|
| NVFP4 store，head 256，fp16 輸入 | 1／2／4 | 40／28／24 | 8／8／0 B | 128／128／256 B |
| NVFP4 store，head 256，bf16 輸入 | 1 | 38 | 8 B | 128 B |
| NVFP4 gather，head 256 | 1／2／4 | 64／32／30 | 0 | 0 |
| 現行 E4M3 split-K，BLOCK_N 16，64 splits | 4 | 243 | 0 | 24,576 B |
| 現行 E4M3 split-K，BLOCK_N 16，64 splits | 2 | 255 | 304 B | 24,576 B |
| 現行 E4M3 split-K，BLOCK_N 64，4 splits | 2 | 32 | 9,232 B | 73,728 B |
| 步驟 D 的 FP16 split-K（gather 之後），BLOCK_N 16，64 splits | 4 | 166 | 0 | 24,576 B |
| 步驟 D 的 FP16 split-K（gather 之後），BLOCK_N 16，64 splits | 2 | 255 | 0 | 24,576 B |
| merge 內核，有無折疊 scale 相同 | 2 | 255 | 72 B | 256 B |

store 與 gather 都不需要 shared memory 的調校，一個 warp 就夠；它們不是瓶頸候選。
reader 才是：E4M3 split-K 在 4 個 warp 已用 243 個暫存器，2 個 warp 用滿 255 個並溢出 304 B。
步驟 D 的路徑在內核裡讀的是 FP16，沒有解碼，4 個 warp 只要 166 個暫存器，2 個 warp 用滿 255 個但沒有 stack；
V100 上 TP4 的 decode 用 2 個 warp（`_use_sm70_qsa_two_warp_partial`），所以這是比 E4M3 好的一面。
前面三列 E4M3 的 split 數與先前版本不同（先前是 8，這裡是 decode 實際用的 64），所以數字有些許差異。

### decode 的融合 reader 需要什麼

型態直接仿 `ops/qsa.py:617-828` 的 `_qsa_sparse_paged_gqa_splitk_kernel`：
索引 → block table → masked load 原始位元組 → 軟體解碼 → `tl.dot` → online softmax，再用 `_qsa_merge_splitk_kernel` 合併。
加一個 `KV_NVFP4` constexpr 分支，不是新內核。

- **tile 與 warps**：沿用 `_qsa_sparse_launch_profile`（`ops/qsa.py:2522-2556`）的 pre-Ampere 設定，BLOCK_N 16、4 warps。
  該處註解的理由：D=256 的 64 欄 tile 超過 Turing 的 64 KiB shared，2 個 warp 在 V100 上讓 tensor core 工作序列化。
  上表也顯示 BLOCK_N 64 的 stack 是 9,232 B。NVFP4 每個元素的解碼工作比 E4M3 多，所以不要從 2 個 warp 起步。
- **sm_70 沒有的東西**：沒有原生 fp8 或 fp4 轉換，所以用整數位元解碼，E4M3 的 scale byte 重用
  `deepseek_v4/common/ops/fp8_software.py` 的 `fp8_e4m3fn_bits_to_fp32`，e2m1 用位元組裝；
  沒有 bf16 tensor core，所以只做 fp16 query（`qsa_e4m3_capability_reason`，`ops/qsa.py:2262-2270`，V100 本來就跑 fp16）；
  沒有 `cp.async`、wgmma 與 m16n8 mma，這些由編譯閘門守住。
- **解碼在 fp16 是精確的**：e2m1（0、0.5、1、1.5、2、3、4、6）乘 E4M3 block scale，乘積最多 5 位有效數字，
  最大 2688，最小非零 2^-10（fp16 的正規數），對全部 8×127 個組合在 CPU 上驗證為精確。
  所以 reader 可以先解碼成 fp16 而不乘 global scale，K 的 global scale 乘在分數上，V 的 global scale 乘在正規化後的輸出上，
  和 E4M3 路徑同位置（`ops/qsa.py:686-688`、`:776-778`），符合 Astra §E 對 scale 位置的限制。
- **不做 interleave**：K tile 讀成低 nibble 與高 nibble 兩塊 `[128, BLOCK_N]`，`scores = dot(q_even, k_even) + dot(q_odd, k_odd)`，
  query 用 stride 2 的指標載入；V 產生偶數與奇數兩半累加器，最後用 stride 2 的指標寫回。
  block scale 索引對兩塊都是 `j // 8`。累加器總大小不變，但 QK 變成兩段 fp32 部分和，
  捨入順序與 K=256 的單一 dot 不同，所以與 gather 加 FP16 路徑比對的目標是 fp16 捨入內，不是逐位元。
- **遮罩**：與 E4M3 相同，無效的 request、token、page 載入 0 並貢獻 0 機率；NVFP4 的無效 lane 要同時把 nibble 與 scale byte 載成 0。
  page 偏移用 int64，與 E4M3 相同。
- **定址**：資料區與 scale 區在頁內不同位置，用 `nvfp4_kv_split_views` 給的 stride，與 gather 內核一致。

### prefill 計畫

QSA 沒有稠密 prefill。prefill 與 decode 都是對 `[rows, 2051]` 個選中 token 做稀疏注意力，差別只有列數。
分派在 `ops/qsa.py:1733-1762`：列數不少於 `_SM70_QSA_XQA_PAGE4_MIN_ROWS`，或 uint8 cache 且列數超過 16，走 XQA page4 CUDA；
再來是 grouped page4；其餘走 Triton split-K。目前 decode 的 target M5 與 draft M1 都在 Triton split-K
（Astra §E 已確認該內核是 gather、解碼、注意力融合在一起）。

NVFP4 現在被三層擋住，reader 上線時要逐層放行：`_xqa_page4_shape_supported` 要求 cache 形狀為 `(…, 1, 256)`，
NVFP4 的末維是 144 所以被擋；`qsa_sparse_paged_attention` 對 dtype 字串 `nvfp4` 拋 `ValueError`（`ops/qsa.py:2321-2323`）；
`forward_qsa` 要求非 E4M3 的 cache dtype 等於 query dtype（`qsa.py:370-376`），uint8 會拋 `RuntimeError`。再加上 `_run_qsa` 的 guard。

| 選項 | 做法 | 成本與風險 |
|---|---|---|
| a | 同一個 Triton NVFP4 split-K 服務所有列數 | 改動最小，先得到端到端正確性。與 E4M3 的 CUDA page4 prefill 誰快沒有資料，`#441` 的 1.16 到 2.6 倍是 Triton 對舊 Triton profile，不是對 CUDA |
| b | 先解碼成 fp16 scratch，再用現成的 FP16 page4 CUDA 路由 | gather 內核已逐位元正確。scratch 每個選中 token 1,024 B（K 與 V 各 fp16 512 B），而儲存是 288 B。依列逐一展開不可行：16 列 33.6 MB、64 列 134 MB、2048 列 4.3 GB。要依 page 去重：不重複 token 最多 `min(rows×2051, context+rows)`，64K context 約 64 MiB，262K 約 256 MiB，對 p070 每 rank 約 2.1 GiB 的 KV 預算不是小數。XQA page4 已有壓縮 page 索引（`_qsa_xqa_page4_block_table`），可借其表重映射 |
| c | 在 `flash-attention-v100/kernel/flash_decode_paged.cu` 加 NVFP4 的 CUDA reader | 仿 E4M3 分支（`KV_DTYPE == KV_CACHE_DTYPE_FP8_E4M3`，`:809-909`；`fp8_kv_utils.cuh`；grouped page4 ABI v2 已帶 `kv_cache_dtype` 與 scale）。需要建置 flash_attn_v100 擴充，不能在純 CPU 階段做 |

**步驟 D 落地的是選項 b 的 Triton 變體**：所有列數走同一條路，gather 到 FP16 後用現成的 Triton FP16 split-K，不經任何
CUDA page4 路由；沒有跨列的 page 去重，改成把列分組，每組的 gather 暫存不超過 64 MiB（head 256、top-k 2051 時 31 列）。
decode 的 M5 與 M1 都在一組內；prefill 的列數大時逐組處理，預期明顯慢於 E4M3 的 CUDA page4，沒有量。

建議：先 a（decode 與 prefill 一起），量 prefill 對 E4M3 CUDA page4 的差距；只有 a 大幅落後且列數大於 16 的流量佔多數時才做 b；
團隊並發（約 20 人、256K context）若 prefill 成為瓶頸才投資 c。

### 工作量估計

以下是估計，不是量測。

| 工作 | 估計 | 需要 GPU |
|---|---|---|
| split-K 與 merge 內核加 NVFP4 分支，直譯器對 `gather_dequant_nvfp4_kv` 加 FP16 參考注意力做 parity | 2 到 3 天 | 否 |
| 放行三層守衛與 `_run_qsa`、`forward_qsa` 的 dtype 分支、page4 排除 | 0.5 天 | 否 |
| V100 上編譯與資源調校（暫存器、溢出、BLOCK_N、warps、偶奇兩段 dot 的精度） | 1 到 2 天 | 是 |
| store 到 reader 的往返、`kv-logprob-probe`（含 bf16 重跑組）、ABBA 序列化 A/B；只有勝出者才跑 SWE-34 | 2 到 3 天 | 是 |
| 里程碑 2：draft 也用 NVFP4 | 約 1 天 | 是 |

合計單人約 1.5 到 2 週，其中約 1 週要占 GPU。

### 2026-10-05：融合 reader 已建（CPU）

上面「只有計畫」的描述到 2026-10-05 為止；tip `e56be74f1`（`[Kernel][SM70] Fused NVFP4 reader in the QSA split-K kernel (KV_NVFP4, plan 071 step C)`）起有程式。
**下面各條（內容、旋鈕、編譯閘門、直譯器 parity）的證據是 CPU（直譯器與 sm_70 編譯閘門）；GPU 的 kernel 層級結果在本節最後的「GPU 結果（Q1.2）」（2026-10-05），serving 層級仍沒有【沒數據】。**

- **內容**：`_qsa_sparse_paged_gqa_splitk_kernel`（`vllm/models/qwen4_exp/nvidia/ops/qsa.py`）加 `KV_NVFP4` constexpr 分支，不是新內核。
  K tile 解成偶／奇兩半 `[HEAD_DIM/2, BLOCK_N]`，QK = 兩次 K=128 的 `tl.dot`；V 兩個半累加器，最後 stride 2 寫回；
  layer scale 照 gathered 路徑折疊（k_scale 乘在分數、v_scale 在正規化後）；無效 lane 載入 byte 0 加 scale 0，所以精確為 0；merge 內核沒改。
  解碼 helper 是 `nvfp4_kv_triton.py` 的 `_nvfp4_tile_halves`；新增的內核參數是 K／V block-scale 指標加 6 個 stride（來自 `nvfp4_side_views`）。
- **旋鈕**：環境變數 `VLLM_SM70_QSA_NVFP4_FUSED_READER`（**預設 0 = 走 gather 路徑**；登記在 `vllm/envs.py` 與 `docs/configuration/env_var_reference.md`），
  或 `qsa_sparse_paged_attention(..., nvfp4_fused_reader=True)`；判斷函式 `qsa_nvfp4.nvfp4_fused_reader_enabled()`。
  fused 一律用 pre-Ampere profile（BLOCK_N 16、4 warps），並略過 two-warp partial 與 page4 兩條路由；原本的 admission 檢查都保留。
- **sm_70 編譯閘門**（Triton ptxas，無 GPU；TOPK 2051、PAGE 2784、HEAD 256、GROUP 6、64 splits、BLOCK_N 16）：

| 內核 | warps | 暫存器 | stack | shared |
|---|---|---:|---:|---:|
| 融合 reader（KV_NVFP4） | 4 | **255** | 0 B | 16,384 B |
| 融合 reader（KV_NVFP4） | 2 | 255 | **1,096 B（溢出）** | 16,384 B |
| gathered（步驟 D 的 FP16 split-K） | 4 | 166 | 0 | 24,576 B |
| gathered（步驟 D 的 FP16 split-K） | 2 | 255 | 0 | 24,576 B |

  fused 2 warps 的溢出只記為上限，不是目標。**注意：fused 4 warps 的 255 暫存器已經在上限，沒有餘裕**；occupancy 與之後任何成長會怎樣是 GPU 問題【沒數據】。
  既有的 KV_E4M3、FP16、gathered 啟動的 PTX 不變（測試守住）；禁用指令清單乾淨。
- **直譯器 parity**（`tests/models/qwen4_exp/test_nvfp4_kv_fused.py`）：fused 對 gather 對 FP16 內核，容差 rtol／atol 2e-3（FP16 捨入內；兩段 dot 的 QK 依設計不逐位元相同，見上方「不做 interleave」）。
  涵蓋 layer scale ≠ 1、非法與毒化項、共用 page 與重複項、output gate、LSE、單 split 路徑、旋鈕預設 = gather、admission 檢查。
- **步驟 E 腳本**：`benchmarks/sm70_nvfp4_kv_kernel_check.py` 多一個 `fused` arm（`--arms` 預設 `gather fused`）：
  `FUSED_MATCHES_GATHER`（對 gather 的 rel-L2 ≤ 2e-3，加精確零）、`FUSED_VS_FP16`、`FUSED_ROUNDTRIP_US`（每次 launch 的 fused／gather 比），以及走 fused 路由的突變對照。
- **測試數**（NVFP4 檔案：`test_nvfp4_kv_*.py`、`test_sm70_nvfp4_kv_kernel_check_cpu.py`、`test_sm70_moe_unique_experts_bench_cpu.py`）：
  直譯器 440 passed、32 skipped；預設 377 passed、95 skipped。
  ⚠ 不要對整個 `tests/models/qwen4_exp/` 跑 pytest：沒有 GPU 時 GPU-only 模組在收集階段就報 No CUDA；只跑上列檔案。
- **【沒數據】（剩下的，都要 GPU 窗；Q1.2 已量掉 `FUSED_MATCHES_GATHER`、`FUSED_ROUNDTRIP_US`，見下）**：fused 的 prefill 對 E4M3 的比（閘門 4 要 ≥ 0.9，M1 是 0.435／0.459；kernel 自檢只有 decode 列）；
  255 暫存器（4 warps，已在上限）對長 context occupancy 的影響；serving 層級的 decode `round_ms`（M1 的懲罰是 +0.898 ms/round，閘門 4 目標是回到約 E4M3）；fused 路由放進 serve 後的 12 案 parity 與 `kv-logprob-probe`；M=5（MTP4 verify）的 serving 層級。
  環境變數的轉發：serve script 經 `scripts/fence.sh`（`systemd-run --user --scope`，前景，繼承 cwd 與 env）啟動 `vllm serve`，呼叫端 export 的 `VLLM_SM70_QSA_NVFP4_FUSED_READER=1` 會到達 server 行程（Fable 讀 fence.sh 第 20 行確認，機制同 production 的 `VLLM_SM70_SAMPLING_CUDAGRAPH=1`）；server 上的實際行為仍未跑過【沒數據】。

#### GPU 結果（Q1.2，2026-10-05）

lead 的 GPU 窗（08:47，GPU 0，Tesla V100-SXM2-32GB，tip `19d328ea7`，venv `1cat-p070`，TP4 幾何 head 256／6 個 q head／1 個 kv head／top-k 2051）。
原始檔 `/data/bench/q12-20261005-084455/{SUMMARY.txt,kernel-check.txt,kernel-check.json}`；`benchmarks/sm70_nvfp4_kv_kernel_check.py --arms gather fused --rounds 4`，exit 0，`SELF_CHECK: PASS`
（8 項正向對照 PASS、10 項負向對照全 FLAGGED，含走 fused 路由的 `nibble_order_swapped`、`scale_placement_wrong`；9 個階段都 RAN，含 `decode_fused`）。

- **正確性【已量】**：`FUSED_MATCHES_GATHER: YES`。fused 對 gather：max|d| 4.883e-4（M=1）／1.953e-3（M=5），relL2 7.665e-6／1.667e-5，在 fp16 捨入量級（兩段 K=128 的 dot，依設計不逐位元相同）。
  `FUSED_VS_FP16` relL2 0.3478（M=1）／0.2232（M=5），與 gather 路徑的 `DECODE_VS_FP16` 完全相同；`NVFP4_BAND` 對 CPU 模型比值 1.000（帶 0.7–1.4）。
- **每次 launch 往返（µs，中位數，B2784／B2864）【已量】**：

| | gather 路徑 | fused | fused／gather（`FUSED_ROUNDTRIP_US`） | E4M3（B1616） | fused − E4M3 |
|---|---:|---:|---:|---:|---:|
| M=1 | 697.3／693.2 | 209.9／199.7 | 0.30／0.29 | 171.0 | +38.9／+28.7 |
| M=5 | 701.4／698.4 | 267.3／267.3 | 0.38／0.38 | 187.4 | +79.9／+79.9 |

  這個往返每次 launch 含約 22 µs host 間隙，比差、不比絕對值。E4M3 對照用 B1616，NVFP4 用 B2784／B2864，同為 context 8192、頁數不同。
  fused 拿掉 gather 路徑對 E4M3 的懲罰：M=1 (697.3−209.9)/(697.3−171.0) ≈ 93 %（B2864 ≈ 95 %）、M=5 ≈ 84 %（算式是用上表重算）。
  同次的 `decode_fp16_dequantized`（吃現成暫存的 FP16 內核）是 175.1／171.0（M=1）、184.3／182.3（M=5），所以 fused 比「已反量化的 FP16 reader」還貴約 +35／+29 µs（M=1）、+83／+85 µs（M=5）：殘差來自 fused 內核本身，不是 gather。
- **serving 層級殘差【估計】，不是量測**：M1 的 serving 懲罰 +0.898 ms/round（no-MTP、12 個 QSA 層）對應 harness 的 gather 路徑懲罰 (697.3−171.0)×12 ≈ 6.32 ms，serving／harness ≈ 0.142。
  (a) 殘差按同比例縮（host 間隙在 graph replay 消失）：M=1 約 +0.05–0.07 ms、M=5 約 +0.14 ms；(b) 殘差全是 GPU 時間、不縮：M=1 約 +0.34–0.47 ms、M=5 約 +0.96 ms。
  fused 與 E4M3 同為兩個 launch（split-K 加 merge；merge 內核沒改），host 間隙占殘差的比例【沒數據】，所以 (a) 只是下界。兩端都不低於閘門 4 的 decode 門檻（production + 2× spread，約 +0.04 ms）；verdict 要等 chain19 的 `nvfp4_fused` 臂（plan 072 queue Q1.4）。
- **仍然【沒數據】**：prefill 比值；255 暫存器的 occupancy 效應（長 context）；serve 內的 12 案 parity 與 `kv-logprob-probe`；M=5 的 serving 層級。

## 步驟 D：reader 接進稀疏注意力，先 gather 再走 FP16

沒有發現設計衝突：gather 出來的 FP16 張量可以直接當成分頁 cache 餵給現成的內核而不複製；K 與 V 的 layer scale 折疊位置
不同（K 乘在分數、V 乘在輸出），但參考把兩者都吸收進 `layer_scale` 後相乘，數學等價，差只在捨入，測試把兩種差異分開驗。

### 資料流

`qsa_sparse_paged_attention`（`ops/qsa.py:2403`）對 `kv_cache_dtype == "nvfp4"` 轉給 `qsa_sparse_attention_nvfp4`
（`ops/qsa_nvfp4.py:65`）。對每一組列：

1. `gather_dequant_nvfp4_sides_triton`（`nvfp4_kv_triton.py:227`）把每列選中的 token 解碼成 FP16，不乘 layer scale。
   這一步是精確的（步驟 C）。它收 cache 的兩個半邊，也就是 `forward_qsa` 從 `unbind(1)` 拿到的那兩個 4 維 view。
2. `nvfp4_entry_validity`（`nvfp4_kv.py:401`）算出內核會遮掉的位置；參考 gather 現在也呼叫它，所以規則只有一份定義。
3. 現成的 FP16 split-K 與 merge 內核照舊執行。gather 出來的 `[rows, topk, heads, 256]` 當成 `rows` 頁、每頁 `topk` 個
   token 的 cache：第 r 列是 request r，它的 block table 是單一頁 r，選中的第 c 項就是 token c，要遮掉的項設成 -1。
   沒有複製，內核本體逐行不變。
4. layer scale 照 E4M3 折疊：`k_scale` 乘在 QK 分數上，`v_scale` 乘在正規化後的輸出上，各一次、FP32。內核與 merge 內核
   的條件從 `KV_E4M3` 改成新的 constexpr `FOLD_SCALES`（`ops/qsa.py:695`、`:786`、`:892`）；解碼仍只在 `KV_E4M3` 下。
   既有路徑傳 `FOLD_SCALES == KV_E4M3`，PTX 不變（見測試）。

### 三道門怎麼放行

| 門 | 之前 | 現在 |
|---|---|---|
| `_run_qsa` 的 `NotImplementedError`（`qsa.py:930`） | 一律拒絕 nvfp4 | 只拒絕 DCP 分片的 cache |
| ops 入口對 dtype 字串 `nvfp4` 的 `ValueError`（`ops/qsa.py:2334`） | nvfp4 不在清單內 | nvfp4 分支：FP16 query、uint8 儲存、head size 是 16 的倍數、scale 有限為正；head size 由列位元組數 `×16/9` 還原 |
| `forward_qsa` 的 storage dtype `RuntimeError`（`qsa.py:381`） | uint8 與 query dtype 不同就拋 | nvfp4 分支：uint8、FP16 query、DCP 分片拒絕 |
| MTP draft | 沒有 | `_verify_nvfp4_kv_speculation`（`qsa.py:545`，owner 初始化時呼叫 `:658`）：有 speculative 設定就 `NotImplementedError` |
| page4 CUDA | 形狀檢查（末維 144）擋住 | nvfp4 分支在到達 page4 閘門前就回傳；gathered 的 FP16 呼叫另外明確排除 page4；測試把三個 page4 函式換成會拋例外的版本來證明沒呼叫 |

### 限制與成本

- `qsa_sparse_paged_attention` 的入口現在在 `TRITON_INTERPRET=1` 時也接受 CPU 張量（`ops/qsa.py:34`、`:2306`），只供測試；
  CUDA 張量的行為不變。
- 每次呼叫多寫再多讀 1,024 B/token 的 FP16 暫存，而融合 reader 只讀 288 B/token：M5 時各 10.5 MB 對 2.95 MB。
  以 900 GB/s 粗估每層約 23 µs 的暫存流量，12 個 QSA 層約 0.28 ms；這是算術估計，不是量測，不含 cache 命中。
  另有十幾個小的 torch 運算（validity）。
- 分組：`NVFP4_GATHER_SCRATCH_BYTES`（64 MiB）；列數不大於 8 時分組不改變結果，列數更大時 launch profile 可能不同，
  只有容差意義上的相等。

### 融合 reader：之後的最佳化，這次不建

> 2026-10-05 更新：融合 reader 已在 CPU 上建好（tip `e56be74f1`），見「編譯路徑與 reader」的「2026-10-05：融合 reader 已建（CPU）」；GPU 上仍沒有數據。以下為原文。

結構已在步驟 C 寫出（`KV_NVFP4` constexpr 分支、偶奇兩塊 tile、fp16 精確解碼、scale 位置同 E4M3）。相對於步驟 D 的差別：
省掉 1 KiB/token 的暫存寫讀與十幾個 torch 運算，少一次啟動；代價是 QK 變成兩段 K=128 的部分和，與 D 的路徑不再逐位元相同，
所以 D 是它的正確性基準，容許差距要用 `kv-logprob-probe` 對重跑組定。先決條件：步驟 E 在 GPU 上跑完，量到 gather 與 validity 在
decode round-trip 裡佔多少；佔比小就不值得做。

## 步驟 E：單 GPU 的內核數值與計時腳本

`benchmarks/sm70_nvfp4_kv_kernel_check.py`，結構仿 `sm70_ple_conv_cudnn_memory_check.py` 與
`sm70_gemm_alignment_memory_check.py`（在 `/data/src/1cat-wt-fable`）。**沒有執行過**。它做的事：

1. 真實幾何（從 `--config` 讀，TP4 為 head 256、6 個 query head、1 個 KV head、top-k 2051），NVFP4 block 2784 與 2864，
   E4M3 對照 block 1616，列數 1 與 5，每個 request 8192 個 token，K/V 與 top-k 都以種子產生。
2. store：Triton store 對參考 writer，data 與 scale 位元組分開比；E4M3 對照用 `reshape_and_cache_flash`。
3. gather：Triton gather 對 CPU 參考，用參考 cache 的位元組以免 store 的差異遮住它；top-k 含非法項。
4. decode：NVFP4 路徑對「FP16 內核跑在解碼後的 cache」在單位 scale 下逐位元比（路徑一致性）；真實 layer scale 下對未量化 FP16
   的 max |d|、mean |d| 與 relative L2；同樣的流程跑 E4M3 作為對照。
5. 計時：CUDA events，ABBA 交錯，store、gather、整條 NVFP4 decode、gather 之後的 FP16 內核、E4M3 decode。

印出：`STORE_MATCHES_REFERENCE`、`GATHER_MATCHES_REFERENCE`、`ZERO_FILL_OK`、`DECODE_PATH_IDENTICAL`（YES、NO 或 UNKNOWN，NO 附證據）、
`DECODE_VS_FP16`、`E4M3_CONTROL`、`NVFP4_VS_E4M3_REL_L2`、`TIMING_US`、`DECODE_ROUNDTRIP_US`。`--out` 寫 JSON。
結束碼：0 是 `SELF_CHECK: PASS`，1 是 `SELF_CHECK: FAIL`，2 是沒有 CUDA、GPU 已被佔用或參數不可用。

### 自檢

一行指令就能知道這次跑可不可信，最後一行是 `SELF_CHECK: PASS` 或 `SELF_CHECK: FAIL (<原因>)`：

- **正向對照必須 PASS**：上面四個逐位元的 verdict；`TIE_STORE_MATCHES_REFERENCE`，也就是 GPU store 在刻意造出的
  捨入平手上（E2M1 的中點與 E4M3 scale 的中點，單位 layer scale）等於參考；以及 `NVFP4_ERROR_IN_BAND` 與
  `E4M3_ERROR_IN_BAND`：經過這套 harness 量到的注意力輸出相對 L2，必須落在 CPU 模型預測值的 0.7 到 1.4 倍。
  模型是對同一份資料做 FP32 注意力，K/V 由參考量化器，或由 `reshape_and_cache_flash` 做的 `float8_e4m3fn` 轉換量化。
  E4M3 這一臂就是 harness 的已知答案檢查：它重現不出自己的誤差帶，比較本身就不可信。
  在直譯器的小 case 上，兩者的比值是 1.000。
- **負向對照必須被 FLAGGED**：三種突變拿同樣的比較去量，比較必須回報差異。`nibble_order_swapped`（每個資料位元組的兩個
  nibble 對調）與 `scale_placement_wrong`（scale 位元組照 SM100 的 V swizzle 重排，那不是 V100 的版面）在 store、gather、
  decode 三個環節各量一次；`tie_rule_removed_e2m1` 與 `tie_rule_removed_e4m3`（平手改成進位，分別在 nibble 與 scale 位元組）
  在平手 store 上量。MISSED 代表資料或比較看不見那一類錯誤；也可能是正向對照已經失敗的後果，例如 store 剛好錯成
  同一種樣子。
- **所有階段都要 RAN**：store、gather、zero_fill、decode、e4m3_control、tie_control、negative_controls、timing。
  跳過階段就是自檢失敗，所以 `--no-timing` 與 `--no-e4m3` 會讓 `SELF_CHECK` 變成 FAIL，case 丟例外也是。

自檢本身的驗證：未突變時全部 PASS；把 Triton store 的 E2M1 編碼改成 0.25 進位，或把 nibble 順序對調，`SELF_CHECK` 都變成 FAIL，
且 `STORE_MATCHES_REFERENCE` 與 `TIE_STORE_MATCHES_REFERENCE` 一起失敗。這些都在直譯器的小 case 上做，不是 GPU。

GPU 操作員要跑的：

```text
cd <worktree> && CUDA_VISIBLE_DEVICES=<idle gpu> PYTHONPATH=$PWD \
    /data/venvs/1cat-p070/bin/python benchmarks/sm70_nvfp4_kv_kernel_check.py \
    --out /data/bench/nvfp4_kv_kernel_check.json
```

回傳 stdout 與 JSON。看 `SOURCE` 三行確認匯入的是對的樹。**這台機器現在沒有閒置的 GPU**：四張都占約 31.5 GiB，使用率 61% 到 87%
（`nvidia-smi`，2026-10-04）。腳本在 GPU 已占用超過 1 GiB 時拒絕執行（結束碼 2），不要對服務正在用的 GPU 加 `--allow-busy`：
它會拿走服務的剩餘記憶體。`STORE_MATCHES_REFERENCE` 在 GPU 上為 NO 時先看差異是否只在 scale 位元組、是否在捨入平手點：
Triton 的 fp32 除法不一定是 IEEE，參考用的是 IEEE 除法。

### chain19 草稿

`scripts/c4140-ab/chain19-nvfp4-m1.sh`（llm-playground，未 commit，未對伺服器執行；`bash -n` 與 shellcheck 乾淨，`C19_DRY=1` 的乾跑只印計畫，
且不碰 `/data/bench/ab` 與 venv）：

- **閘門**：操作員呼叫；`TARGET_VENV` 不得是 `1cat-m589`；p071 tip 等於 `WANT_TIP`；serve script 要有 `C4140_KV_DTYPE`；視窗檔存在；沒有舊的 chain 在跑。
- **venv 閘門**：`TARGET_VENV`（`/data/venvs/1cat-p070`，不是 production 的 venv）裡每個 `vllm/**/*.py` 都必須與 `c4140-p070` 的 git blob 逐位元組相同，
  不得缺任何 p070 的檔案，不得有 venv 獨有的檔案（已知的打包產物除外：`_version.py`、`third_party/triton_kernels/`、兩個 `rotary.py`），
  p071 要新增的路徑也不得已經存在。這是為了擋住 p070-packs 測試留在 venv 裡的 Astra 修正：2026-10-04 的乾跑報出
  `sm70_fp16_gemv.py` 與 `sm70_profiles/acceleration.py` 兩個檔案不同。有任何差異就印出路徑並拒絕執行；乾跑改印結論。
  我用拋棄式的 venv 副本測過：還原後通過，改一個檔、p071 新增檔已存在、多一個陌生檔、少一個 p070 檔，各自被擋下。
- **流程**：備份 p071 會動到的 venv 檔案，同步並做 import smoke，對 nvfp4、fp8_e4m3、fp16 三種 KV 各自 settle（先 fresh compile，再重啟到連續兩個
  實例的 12 案 `--no-logprobs` parity 相同才算同一個數值狀態），量 prefill-probe（nvfp4 與 E4M3 對照）與探針（每臂加一個同實例重跑臂），最後 compare。
  全程無 MTP，同一個 `--max-model-len`（預設 131072），都帶 `--enable-prompt-tokens-details`。
- **KV dtype 旋鈕**：每臂設 `C4140_KV_DTYPE=nvfp4|fp8_e4m3|auto`，不再追加第二個 `--kv-cache-dtype`。這是 serve script 在 `C4140_STACK=p070` 分支裡新增的旋鈕，
  預設 `fp8_e4m3`，預設行為不變：我用一個只記錄 argv 的假 fence 比對了十種環境組合，預設、p069、p070 與 E4M3=0 的 argv 逐位元組相同，
  nvfp4 與 auto 只差那一個值。每臂啟動後，serve log 的 `kv_cache_dtype` 行必須等於該臂的 dtype，否則 chain 結束並還原。
- **還原**：先從備份還原 venv 並做 cmp 檢查，再把 production 寫死成 `scripts/c4140-1cat-flashnext-serve.sh restart`：沒有 `C4140_STACK`、沒有 `EXTRA_ARGS`，
  所有 `C4140_*` 變數先清掉，production 就是 serve script 的預設（c4140-p069 @25361ff5f、`/data/venvs/1cat-m589`）。重啟後要求 12 案 `--no-logprobs` parity
  與 `/data/bench/ab/parity-c11-P3-nolp.json` 相同；不同就再重啟一次，最多 2 次，仍不同就報 `RESTORE_PARITY_FAIL` 並讓 production 照現況跑著。
  這段邏輯用假 serve script 測過：第一次成功、第二次成功、兩次都失敗各一個情境，子行程看到的 `C4140_*` 只剩兩個我指定的變數。
- 被 `kill -9` 時用 `C19_RESTORE_ONLY=<備份目錄>` 還原 venv（沒有伺服器介入），在拋棄式 venv 副本上實測過。標了 `# UNVERIFIED:` 的參數沒有驗證過。

## 品質驗證：SGLang 的證據不是 1Cat 的上界

fleet SGLang 的品質數字，nvfp4 對 bf16 KV 的 ΔNLL 為 +0.0042 到 +0.0096 nats/token
（llm-playground repo 的 `engines/patches/sglang-qsa-nvfp4-kv/README.md:327-332`），**不能當 1Cat 的上界**。

SGLang 的第一個 4096-token prefill chunk 內，位置從不讀 pooled KV，所以 2K 視窗在所有 KV dtype 間逐位元相同，
較長視窗的平均 ΔNLL 也被這段零誤差的位置稀釋（同檔 `:345-346`；`notes/methodology/kv-cache-quality-probe.md` 規則三）。
1Cat 的 QSA 在同一個 forward 內先寫後讀：`_run_qsa` 先呼叫 `do_kv_cache_update`（`qsa.py:943`），
再呼叫 `forward_qsa`（`qsa.py:950`），讀的就是剛量化寫入的 cache。所以量化誤差從第一個 chunk 就存在。

`scripts/eval/kv-logprob-probe.py` 的計畫因此要這樣訂：

1. **第一個 prefill chunk 的視窗是必備的**，打分位置全落在第一個 chunk 內。保留 2K 視窗：它們在 SGLang 上沒有訊號，
   在 1Cat 上卻是第一個 chunk 的直接量測。chunk 邊界取 1Cat 的 scheduler chunk 大小，從 serve 設定讀，不沿用 SGLang 的 4096。
2. 第一個 chunk 的視窗、≥32K 視窗與 96K 桶分開報告，不併成一個 pooled 數字。
3. 每個比較都含未量化 KV 的重跑組，先看重跑組的 CI，小於它的差距無法解析（規則一）。1Cat 的每個 server 行程
   會落在兩種數值狀態之一（`notes/methodology/serving-determinism-floor.md`），所以每一組啟動後先過 12 案
   `--no-logprobs` parity，確認落在同一個狀態，否則重跑組量到的是啟動狀態差，不是 KV 精度。
4. 閾值沿用工具的提案規則：ΔNLL 95% CI 上界不超過 +0.005 nats 才算通過。
5. 工具現在有 vLLM 後端（見下），但沒有對真的 1Cat vLLM 跑過。

### 探針的 vLLM 後端

llm-playground 的 `scripts/eval/kv-logprob-probe.py capture --backend vllm`，在工作樹中，由 Fable 審閱後 commit；SGLang 路徑一行未動：

- 走 `/v1/completions`，`prompt_logprobs=20`、`max_tokens=1`。vLLM 對整個 prompt 都算 logprob，沒有 `logprob_start_len`，
  所以由客戶端保留視窗的最後 N 個位置，輸出 JSON 與 SGLang 的相同，`compare` 不用改。`max_tokens=0` 只在 `echo=true` 時被伺服器接受，
  所以一律送 1。
- 快取：p070 的 vLLM 對要求 prompt logprobs 的請求本來就不讀 prefix cache（`sampling_params.py:480-484`）。預設再用 nonce 前綴防守：
  每個視窗前面加 8 個由 salt 與視窗 id 決定的 token。salt 在所有臂（含重跑臂）相同，所以比較的內容相同；視窗 id 讓每個視窗的 nonce 都不同。
  所有位置因此平移 nonce 長度，存檔的 `start`、`pos0`、`length` 仍是視窗座標，打分的仍是視窗最後 N 個位置。
  `--flush-mode reset` 呼叫 `/reset_prefix_cache`，它只在 `VLLM_SERVER_DEV_MODE=1` 時存在，而 DEV_MODE 在 compile key 內，所以不是預設。
  `cached_tokens` 警告仍是偵測器，伺服器要加 `--enable-prompt-tokens-details` 才會回報，而且只在非零時回報，所以缺值記成 null。
- **探針跑的是 eager sampler**：要求 logprobs 的請求讓 1Cat 的 patch-3 sampling CUDA graph 退回 eager sampler。探針量的是 KV 品質，所以可以接受，
  但它不是 client 取樣路徑的證據；parity 檢查用 `--no-logprobs`，也是這個原因。
- 第一個 prefill chunk：serve 的 `--max-num-batched-tokens` 是 8192，2K 視窗加 8 個 nonce token 完全在第一個 chunk 內，所以 2K 桶就是第一個 chunk
  的桶，報告時與 ≥32K 桶分開看。
- 成本：每個位置 21 筆含解碼文字的 entry，約 2 KB，96K 視窗約 200 MB 的 JSON；例行臂用 2K、8K、32K，96K 要先過 smoke。
- 驗證：CPU 上 41 個測試（18 個新），用行程內的假 vLLM 伺服器，含「與 SGLang 解析器對同一份 logprob 給出相同結果」；
  對探針的 7 個突變都被測試抓到。

## 未完成與開放問題

1. 整條 NVFP4 路徑沒有在 V100 上執行過。Triton store、gather 與 gather 後的 FP16 路徑只在直譯器與 sm_70 編譯閘門驗證過；
   GPU 腳本（步驟 E）寫好但未執行，本機四張 GPU 都被服務占住。
2. 沒有放行的：page4 CUDA（目前也不會被呼叫）、DCP2、MTP draft，各有明確的 `NotImplementedError`。
3. store 的捨入平手無法與 PTX 對拍；GPU 上 `1.0 / x` 用的除法沒驗證，`STORE_MATCHES_REFERENCE` 會第一個告訴我們。
4. 現行 merge 內核在 Triton 直譯器跑不動，CPU 測試用 torch 替身；真的 merge 內核搭 `FOLD_SCALES=True` 只由編譯閘門與 GPU 腳本涵蓋。
5. gather 路徑的成本沒量：每次呼叫寫再讀 1 KiB/token 的暫存與十幾個小的 torch 運算；prefill 逐組處理，預期明顯慢於 E4M3 的
   CUDA page4。
6. 融合 reader（之後的最佳化）以 gather 路徑為正確性基準，QK 兩段部分和與它不逐位元相同；容許差距要用 `kv-logprob-probe`
   對重跑組定，不能用 SWE-34。
7. draft 的 scale 載入對 nvfp4 只由 `model.py` 的旗標放行，沒有 draft 實測，里程碑 2 要驗。
8. 首個 prefill chunk 也讀量化 KV（QSA 先寫後讀），SGLang 的 ΔNLL 證據不能當上界；驗證計畫見上一節。
9. packed 頁在 DCP1 下與 `KVBlockZeroer`、offload 的互動沒有驗證；測試只涵蓋 member view 的寫入不碰鄰居。
10. store 的 K、V 各一次啟動的實際成本，以及 `rows` 大時的 grid 形狀，沒有量。
11. `ops/qsa.py` 的內核簽名多了一個 constexpr。8 個既有組態的 PTX 在去掉行號後沒變，但服務重啟後仍應照
    `notes/methodology/serving-determinism-floor.md` 跑 12 案 parity。

## 2026-10-05 GPU 實測結果（GPU results：chain19 與 ABBA 窗）

取代文首與上一節第 1 點「沒有在任何 GPU 上跑過」的說法。tip `5db3e49b4`（相對 chain19 用的 `367094a11` 只多了 page4 plan 測試 fixture 的 import 修正）；
no-MTP、`--max-model-len 131072`、推測解碼關。判讀、原始檔路徑與閘門表在 llm-playground 的 `doc/plan/071-c4140-qsa-nvfp4-kv.md`（§3 預算 (d)、§4、§5、§8）。

- **kernel 自檢**：00:51 `SELF_CHECK: PASS`（7 個正向 PASS、8 個負向 FLAGGED、8 個階段 RAN）。
- **既有 GPU 測試**：PASS 附一個保留。`test_qsa_reference` 41 passed、3 failed，這 3 個在 base（`c4140-p070` @`1b5731f90`）上相同；`qsa_cache.py` 與該測試檔在 p070..p071 之間沒有改動，是既有問題，不是回歸。
  `test_qsa_e4m3` 9、`test_e4m3_mtp_gpu` 7、`test_e4m3_mtp_capture_gpu` 3、`test_qsa_dcp_attention` 4、`test_qsa_dcp_packed_cache` 2 全過。
  `test_sm70_qsa_page4_plan` 原本 38 skipped（fixture import），`5db3e49b4` 修好後 03:39 重跑 **38 passed、0 skipped**。
- **容量**：同一個預算 5.54 GiB/rank、131K：E4M3 772,338 tokens、NVFP4-A 1,161,251 tokens，**×1.5036**（位元組比是 1.778，差距來自 GDN state 的固定 ID）。
  serve log 與 `nvfp4-kv-capacity.py` 的輸出逐位相同，真 allocator 的 CPU 模型第一次在 NVFP4 格式本身上被 GPU 驗證。
- **prefill**（`prefill-probe.py`，冷、每個長度 2 次取第 2 次）：nvfp4 ÷ fp8_e4m3 = **0.435**（16K：2,424 對 5,567 tok/s）、**0.459**（64K：2,456 對 5,351 tok/s）。上 production 要 ≥ 90%，M1 預期內不過。
- **decode**（ABBA，4 個 fresh cell，每個 `GRAPH … FULL`，downgrade 0、skip 0）：nvfp4 − fp8_e4m3 = **+0.898 ms/round（約 +0.90，+7.87% 對 no-MTP 的約 11.4 ms）**；
  11 個 shape 各自 +0.895…+0.921 ms，1K 到 64K 持平，所以是每步固定成本（gather 加小 kernel），不是 KV 頻寬；約 75 µs/QSA 層。
  自檢 harness 的 699 µs/層含每次 launch 延遲，graph replay 下大半消失（推論，未用 profiler）。閘門 4 的 decode 判 FAIL（容許約 +0.04 ms），M1 預期內。M=5（MTP4 verify）沒量。
- **品質探針**（對 fp16，視窗平均 ΔNLL，nats/token，30 窗）：8K −0.00005 [−0.01048, +0.00789]、32K −0.00307 [−0.01155, +0.00394]，**PASS**（≤ +0.01）。
  首 chunk（2K）+0.0213 [−0.0043, +0.0509]，E4M3 對照臂也有 +0.0182 [+0.0014, +0.0369]，**未決**：量化 KV 在第一個 chunk 內先寫後讀是兩種格式共同的懲罰，16 窗解不開兩者之差（plan 071 §5）。
- **結論**：融合 reader（步驟 C）是下一個里程碑，prefill 與 decode 都要它（gather 加 FP16 reader 的路徑過不了閘門 4）；M2（MTP4 draft）排在它之後；3-bit（TQ3／TQ3.5）依 plan 071 §5.1 的規則**不排**。

## 2026-10-05 選項 B′：prefill 經 FP16 scratch 走 grouped page4（只有 CPU，所有 GPU 數字都是沒數據）

起因：W0（Q1.4b）量到 FP16 grouped 對 E4M3 grouped 的 prefill 比值 f = 0.976（16K）／0.948（64K），所以 llm-playground
`doc/plan/071-prefill-fused-reader-design.md` 的選項 B′ 成立。現況的 fused Triton reader 對 E4M3 只有 0.58–0.60，缺口是那個逐列、255 暫存器的 reader
本身，不是 gather 迴圈。B′ 的設計預估是 f × (1 − scratch 解碼成本) ≈ 0.92–0.95（**估計，沒有任何 GPU 量測**）。

### 資料流（每個 QSA 層、每個 prefill chunk）

1. `nvfp4_prefix_scratch_table`（`ops/qsa_nvfp4.py`）：每個 request 擁有 `ceil(seq_len / block)` 個連續 scratch 頁，起點是頁數的 exclusive cumsum；
   `compact[r, j] = offset[r] + j`，其餘 `-1`。全在裝置端，沒有同步。
2. `dequant_nvfp4_prefix_triton`（`ops/nvfp4_kv_triton.py`，kernel `_dequant_nvfp4_prefix_kernel`）把整段已寫入的前綴解碼成 FP16，
   寫進兩個連續張量 `[pages, block, heads, 256]`（K、V 分開，`stride(0) = block × 256`，正是 `_qsa_xqa_page4_shape_supported` 接受的 FP16 cache 版面）。
   解碼用 fused reader 的 `_nvfp4_tile_halves`，**不乘層級 scale**，所以和 gather 逐位元相同；
   序號 ≥ `seq_len` 的位置與沒有對映的頁一律寫精確 `+0.0`（grouped kernel 會把整個 4-token 微區塊載進 smem，頁尾殘留的 E4M3 NaN scale 會毒化 `P @ V`）。
   寫出用 int32 視圖（兩個 FP16 打成一個字），不是間隔 2 的 2 位元組 store。
3. 既有的 `grouped_sparse_page4_plan_fwd` 與 `grouped_sparse_page4_fwd` 照舊，只是 K/V 來源換成 scratch、block table 換成 `compact`、
   `kv_cache_dtype="auto"`（FP16 路徑**忽略** `k_scale`／`v_scale`，`fdp.cu` 的 `e4m3_kv ? k_scale : 1.0f`）。
   K 的層級 scale 折進 softmax scale（`256**-0.5 × k_scale`，在 host 端 double 算完再轉 float）；`rows % 8` 的餘列走既有的 XQA page4 batch，吃同一個 scale。
   兩條 CUDA 路徑一律傳 `k_scale = v_scale = 1.0`，所以不論它們會不會忽略 scale，都不會折兩次。
4. V 的層級 scale 與 output gate 由一個新的逐元素 kernel `_qsa_output_scale_gate_kernel` 一次套用（沒有 gate 時只乘 scale），一次 FP16 捨入。
   既有的 `_qsa_output_gate_kernel` 與 split-K、merge kernel 一個位元組都沒改。

### 開關與路由

| 環境變數 | 預設 | 作用 |
|---|---|---|
| `VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH` | 0 | 開 B′。關著時行為與先前逐位元相同（既有 NVFP4 測試全過，見下） |
| `VLLM_SM70_QSA_NVFP4_PREFILL_MIN_ROWS` | 64 | 小於此列數（decode、MTP verify）仍走 fused／gather reader |
| `VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH_TOKENS` | `max_model_len` | scratch 容量覆寫 |

`qsa_sparse_paged_attention` 在 fused reader 之前呼叫 `qsa_sparse_attention_nvfp4_prefill`，它回傳 `None` 時**什麼都沒寫**，呼叫端照原本會走的 reader 執行。
回退（`None`）的條件：列數不足、沒有 `query_positions`／`sequence_lengths`、scratch 沒預留、stream 正在 capture、cache 幾何與 scratch 不同、
`block_table.shape[0] × ceil(max_seq_len / block)` 超過 scratch 頁數、`max_sequence_length` 未知（`forward_qsa` 以 `attn_metadata.max_seq_len` 傳入）、
形狀超出 page4 契約、平台不是 SM70、擴充沒有 grouped／XQA page4。**回退一律是 fused（或 `FUSED_READER=0` 時的 gather），不是新路徑。**

serve log 的路由證據（每個 process 各一次，`logger.info_once`）：

```text
QSA NVFP4 KV read routes: decode and small batches use the fused Triton reader (VLLM_SM70_QSA_NVFP4_FUSED_READER=1); chunks of >=64 rows use the prefill scratch route (decode once to FP16 + grouped page4) (VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=1).
QSA NVFP4 prefill scratch reserved: 48 pages x 2784 tokens (133632 tokens, 130.5 MiB per rank) on cuda:0; reserved before the KV pool is sized.
QSA NVFP4 prefill scratch route active: prefix decoded once to FP16 (48 pages x 2784 tokens, 130.5 MiB) + grouped page4 (first chunk: 5568 rows).
QSA NVFP4 prefill scratch route declined (<reason>); using the NVFP4 reader.
```

第一行也修掉先前 decode fused knob 沒有路由證據的問題（FUSED_READER 的狀態現在一定會出現一次）。第四行每個不同原因各出現一次。

### scratch 的記帳（怎麼讓 KV 池定案時看得到它）

- 大小 = `ceil(capacity_tokens / block) × block × heads × head × 2（K、V）× 2 B`，每 rank、全部 12 個 QSA 層共用一份（層循序執行、各自解碼自己讀的內容）。
  block 2784：65536 → 65.25 MiB、131072 → **130.5 MiB**、262144 → 258.3 MiB（block 2864：64.3／128.7／257.3 MiB）。
- **profile run 在 `Qwen4ExpQSAAttention._run_qsa` 的 `metadata` 不是 dict 時直接 `output.zero_(); return`**（設計文件點出的 `qsa.py:937-939` 早退）。
  預留就放在這個早退分支裡（`_reserve_nvfp4_prefill_scratch` → `ensure_nvfp4_prefill_scratch`），所以這些位元組在 KV 池定案之前就是 torch 已配置的記憶體。
  層在 `__init__` 只保存 `cache_config`／`model_config` 的參考，在預留當下才讀 `block_size` 與 `max_model_len`（平台可能在層建好之後才定 block size）。
- 路由本身**絕不配置**：沒預留就回退並發一則 `warning_once`。這是刻意的：池已經定案後才配的 scratch 沒有被記帳，會吃掉池已經認領的記憶體。
- 用零初始化（不是 `empty`）：planner 的 padding 微區塊可能指到 scratch 第 0 頁，`0 × 垃圾` 必須有限。
- **沒數據**：「KV 池少了預期的位元組數」只能在 GPU 上用啟動 log 對照驗證（W2 的池大小對照）；CPU 只驗得到「預留發生在早退分支、且在路由之前」。

### 增量解碼的規則（watermark）

每層、每 chunk 把整段前綴重解碼一次是預設且唯一在 production 會發生的行為。kernel 有 `start_tokens`（每個 request 一個），
`Nvfp4ScratchWatermark` 只在能**證明**時才給非零的起點，條件全部成立才給：

- 上一次寫 scratch 的是同一個 owner（層）；
- request id 的 tuple 完全相同、順序相同（`None` 一律不信：cache hit、被搶佔重算、實體頁重用，沒有 id 就分不出來）；
- `query_start == 上次記錄的 seq_len`；
- 該 request 的 scratch 起始頁沒有移動（前面的 request 多一頁會推移後面的）。

scratch 是 12 層共用的，所以「上一次寫的是同一層」在 production 幾乎不成立，且 `forward_qsa` 目前不傳 request id（`None`），**實際上永遠是全量重解碼**。
每層一份 scratch 要 12 倍記憶體（約 1.5 GiB/rank），不做。設計文件估的全量解碼成本是 0.1–0.3 µs/token（**估計，沒量**），所以這個規則現在是正確性保險，不是效能功能。

### 數值

- scratch 內容對 gather 逐位元相同（含層級 scale ≠ 1 的快取、毒化位元組、共用實體頁、未對映／超出範圍的頁）：CPU 直譯器測過。
- attention 本體是 CUDA grouped kernel 自己的算術，與 Triton FP16 split-K 的歸約順序、P 的 FP16 捨入不同，**不逐位元相同**；
  V scale 在 kernel 輸出（已捨入 FP16）之後才套，比 gather 路由多一次 FP16 捨入（相對 2⁻¹¹ 量級）。
  起始容許 relL2 ≤ 2e-3（沿用 `FUSED_REL_L2_TOLERANCE`），那是借來的數，不是證據。
- 因為與 fused、gather 都不逐位元相同，nvfp4 的探針、prefill ABBA、SWE-34 都必須在 B′ 最終路徑上重量（plan 071 §8 owner 決定，沒變）。

### 驗證狀態

CPU（`TRITON_INTERPRET=1`、`CUDA_VISIBLE_DEVICES=`）：

- `tests/models/qwen4_exp/test_nvfp4_kv_prefill_scratch.py`：scratch 表手算、解碼對 gather 逐位元（單頁、整頁、單 token、多 request、頁尾、多 head）、層級 scale 不在解碼內、
  毒化、共用頁、不寫出自己的 scratch 槽、增量 = 全量、watermark 規則、突變案例（nibble 順序、scale 位置、頁尾不補零、compact 表差一頁、offset 沒加、層級 scale 乘進解碼）。
  另外對真 kernel 做過 4 個突變（nibble 偶奇互換、scale `//16`、去掉 `seq_len` 遮罩、K 解碼乘 0.5），第一個 parity 案例就失敗。
- `tests/models/qwen4_exp/test_nvfp4_kv_prefill_route.py`：以 torch 替身取代 `grouped_sparse_page4_*` 與 `decode_paged_xqa_fwd`，驗證接線與 scale 算術
  （K scale 折一次、V scale 折一次、`rows % 8` 餘列走 XQA 且同 scale、knob 關時不進入口、各種回退、預留位置、log 證據）。**替身不是 CUDA kernel**。
- 編譯閘門（`test_nvfp4_kv_sm70_compile.py`，ptxas sm_70，不用 GPU）：解碼 kernel 4 warps、每程式 8 token：88 暫存器、無 stack、4096 B shared、無 atomic、4 位元組 store；
  scale／gate kernel 10／15 暫存器；既有 split-K 的 PTX md5 不變。
- `benchmarks/sm70_nvfp4_kv_kernel_check.py --arms gather fused prefill_scratch`（需要 prefill 大小的 `--rows`）：`SCRATCH_MATCHES_GATHER`（逐位元＋頁尾為零）、
  兩個突變對 scratch 解碼必須被抓、GPU 上多做 route 對 fused／gather 的容許、計時臂 `prefill_scratch_decode`／`prefill_scratch_route`／`prefill_fused`／`prefill_e4m3`。CPU 自檢仍 PASS。

**沒數據（全部）**：任何 GPU 上的時間（解碼 kernel、整條路由、對 E4M3 的比值）、grouped kernel 吃外部 scratch 的實際結果與 relL2、
scratch 對池大小的實際影響、解碼 kernel 的 tile 大小（8 token）是否合適、1 位元組載入的有效頻寬、
多 request／混合 batch、128K 以上、`--max-num-batched-tokens 8352` 槓桿、聯集大小 U、MTP draft 層（M2）。

### 要先在 GPU 上看的（W1，閒置一張卡）

```bash
cd /data/src/1cat-wt-fable-bprime && CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=$PWD /data/venvs/1cat-p070/bin/python \
  benchmarks/sm70_nvfp4_kv_kernel_check.py --arms gather fused prefill_scratch --rows 1 5 512 2048 5568 \
  --context 65536 --out /data/bench/nvfp4_kv_kernel_check_bprime.json
```

先看 `SCRATCH_MATCHES_GATHER`、`SCRATCH_ROUTE_WITHIN_TOLERANCE`、`PREFILL_ROUNDTRIP_US`；再用 `VLLM_SM70_QSA_NVFP4_PREFILL_SCRATCH=1` 的 chain19 新臂量 prefill 對 E4M3，
並比對啟動池大小（預期少了上面的位元組數）。
