# BUILD-RECEIPT pool-serve-v1 poolc (hands-poolc, 2026-10-28T21:28:19+08:00)
DESIGN sha8 bd2a93d0. All nvcc/link under scripts/fence.sh --mem 24G --cpu 6, CUDA_VISIBLE_DEVICES= (no GPU touched).
## Sources (sha256)
- L31 awq src: 0901b789e21bb94fd8fe35c760ee9c9ff776d2ae8ee70bd1e14d081a6db5c6f3 (== l31 build/src copy)
- kgm13 awq src: 3eaad263c99eb16106c57ca7e5e788c31134d5da59a370f88f3fe05bd121223a (== d2e work/src copy); diff = 54-line L31 vec8 hunk only: receipts/l31-hunk.diff
## Commands
1. build/compile-awq.sh <variant> <W> (l31 build.sh argv, include roots -> W incl. ops; argv diff receipts/nvcc-argv-vs-l31.diff), launched by run-poolc-compiles.sh: noD2E W=/data/src/kgm112-v13 (136 s), D2E W=/data/bench/explore/d2-empty-v1/work/src (138 s), parallel, rc=0 both; ptxas logs out/<v>/awq_sm70_gemm.ptxas.log
2. link-noD2E.sh = link-kgm13.sh, link-D2E.sh = link-fix.sh; exactly 2 tokens changed each (receipts/<v>/link-token-diff.txt); fenced link, rc=0
3. sass-md5.py (cuobjdump -sass per Function: block md5) over 5 .so -> sass/*.json; compare normalizing anon-namespace TU hash; receipts/<v>/sass-parity.json
4. nm -D symbol-count receipts/<v>/sym-*.txt
5. rsync stage, cp --remove-destination the two .so, manifest, import check
## Objects / outputs
38e2c74fcaef26d0fd872026ba34cde8aa5cf5687472625d2f80a1807a9ba9b7  out/noD2E/awq_sm70_gemm.cu.o
593c823211cab60c5752d126d5942c4aa522132a104e8f5958af7a886d1f14a2  out/D2E/awq_sm70_gemm.cu.o
72a56e4e17f9ae43b589a7b101fc853ba5741cc4b726a26564f6a9584f0f855e  noD2E/_C.abi3.so
f8c32e78ef26ac12a8d808426c3883907b593ca60302edd723de853860ee895b  D2E/_C.abi3.so
L26 .so: 45158fcb1ccdd1f247574ddc42c41e574d4ea9283bedd1dd83c01f9026e016e0 (verified before copy)
stage _C: 72a56e4e17f9ae43b589a7b101fc853ba5741cc4b726a26564f6a9584f0f855e
stage _s1_native: 45158fcb1ccdd1f247574ddc42c41e574d4ea9283bedd1dd83c01f9026e016e0
base refs: stage-kgm13 _C 07f8ad05, d2e-fix _C 599addc4, l31v2 _C 38875ca1 (sha8 verified)
## Parity
noD2E: 4162 base functions all equal md5, 0 differing, 0 missing, 1 added (vec8, md5 29eafbfb... == l31v2 instance)
D2E: same counts vs 599addc4.
Raw (un-normalized) names: the anon-namespace TU hash (9e3ffcd8 suffix) differs because the awq TU source changed -> 42 awq kernels appear 'missing'/43 'added' by name only; md5 identical after normalization.
nm -D counts: noD2E 4664 (== base, no diff), D2E 4666 (== base); vec8 stub/kernel are local ('t') symbols, not in nm -D (nm shows 2 vec8 symbols in new, 0 in base).
## Import
vllm._C imports CPU-only: /data/bench/explore/stage-develop-cs-g20-pool/vllm/_C.abi3.so
