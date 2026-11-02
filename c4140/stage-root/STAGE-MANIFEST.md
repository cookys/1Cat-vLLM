# stage-develop-cs-g20-pool manifest (pool-serve-v1 arm B skeleton, hands-poolc)
- 來源: rsync -a --exclude __pycache__ /data/bench/explore/stage-develop-cs-g20-h1v3/ (真實檔案, 無 symlink, 無 hardlink 共用; vllm/_C.abi3.so 與 _s1_native .so 的 link count = 1)
- 建立時間: 2026-10-28T21:28:17+08:00
- 大小: 656M; symlink: 0
- vllm/_C.abi3.so (noD2E 預設) sha256: 72a56e4e17f9ae43b589a7b101fc853ba5741cc4b726a26564f6a9584f0f855e
  來源 /data/bench/explore/pool-serve-v1/build/noD2E/_C.abi3.so; D2E 變體 sha256 f8c32e78ef26ac12a8d808426c3883907b593ca60302edd723de853860ee895b (seal 時由 lead 依 RULING-d2-empty-v1 換, 見 /data/bench/explore/pool-serve-v1/build/SWAP-D2E.md)
- flash_attn_v100/_s1_native/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so sha256: 45158fcb1ccdd1f247574ddc42c41e574d4ea9283bedd1dd83c01f9026e016e0 (L26 r1 byte copy, sha8 45158fcb 已驗)
- 未動任何 .py (python bindings 由另一 lane 之後加入)
- diff -rq --exclude=__pycache__ vs h1v3: 恰為兩個 .so + 本 manifest (原 h1v3 manifest 被覆寫):
    Files stage-develop-cs-g20-h1v3/flash_attn_v100/_s1_native/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so and stage-develop-cs-g20-pool/flash_attn_v100/_s1_native/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so differ
    Files stage-develop-cs-g20-h1v3/STAGE-MANIFEST.md and stage-develop-cs-g20-pool/STAGE-MANIFEST.md differ
    Files stage-develop-cs-g20-h1v3/vllm/_C.abi3.so and stage-develop-cs-g20-pool/vllm/_C.abi3.so differ
- import 檢查 (CUDA_VISIBLE_DEVICES= PYTHONPATH=stage, python -B): vllm._C -> /data/bench/explore/stage-develop-cs-g20-pool/vllm/_C.abi3.so (成功, import 不需 CUDA)

## Integration (Fable, 2026-10-28 23:00, DESIGN-pool-serve-v1 §8)
- bind lane patches copied (base files verified equal to arm A first): vllm/model_executor/layers/vocab_parallel_embedding.py 1d20f1c4, vllm/model_executor/layers/linear.py 09891fd5, vllm/model_executor/models/qwen3_dflash.py 0b1aaf3a
- new modules + artifacts under vllm/model_executor/layers/: sm70_lmhead_pool.py f378373b, sm70_drafter_pool.py 938a34ce, sm70_lmhead_custom_v4.so f3dc07af, sm70_drafter_gemm.so 5cca2784, sm70_lmhead_lt_native.so b01182e4, sm70_lmhead_lt.py 7ac5465b (was a32175e3), sm70_lmhead_lt_table.json c445a44f
- diff -rq -x __pycache__ vs stage-develop-cs-g20-h1v3: 13 entries (3 differ .so/.md + 3 differ .py + 7 only-in-B); symlinks 0
- v3 (2026-10-29, DESIGN-pool-serve-v1 §10 R1, hands-poolfix): sm70_drafter_pool.py 79cc67a8 -> 938a34ce; SPLIT_TABLE 4 -> 3 shapes (gate_up (128, 8704, 5120) removed, 其餘不動); gate_up 走 static-guard F.linear 路徑, 記在 non_table。diff -rq vs h1v3 仍為 13 entries (僅該 .py 的 sha 與本 manifest 變)
