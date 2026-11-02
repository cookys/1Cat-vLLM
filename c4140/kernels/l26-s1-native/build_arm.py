#!/usr/bin/env python3
"""P0 of SUBTHRESHOLD BUNDLE v1: SMX3 softmax hunk (d1/k1-v1 arm-smx3 s1_verify.cuh) + D1 diet macro on the k7a develop source. Derived from d1-diet-build/build_arm.py. Build one `_s1_native` .so arm (S1_DEQUANT_DIET = 0|1|2) from the LIVE translation unit.

Recipe = flash_attn_v100/_s1_native/source/build-receipt/build.py (same argv, same common-object
link, same g++-14 link line) with exactly one change: -DS1_DEQUANT_DIET=<n> is appended, and the
source tree is the branch worktree (not a copy). All three arms use the same source path and the
same argv apart from that macro and -o, so the only difference is the macro.

usage: build_arm.py <arm: h1v3cs|h1v3csep> <outdir>  (l25-lut-fold-v1 copy; source = d1/h1-rskh-v1/src-h1v3, macro 2 = CSEP) ; run under scripts/fence.sh, CUDA_VISIBLE_DEVICES empty.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys

ARMS = {'ctrl': 3, 'l25': 4, 'l26': 5}
arm = sys.argv[1]
macro = ARMS[arm]
WT = Path('/data/bench/explore/d1/h1-rskh-v1')
SRC = Path('/data/bench/explore/l25-lut-fold-v1/build/src-l25')
ROOT = Path('/data/bench/explore/pv-hmma')
O = Path(sys.argv[2]).resolve()
O.mkdir(parents=True, exist_ok=True)
T = Path('/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/torch')
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
       'HF_HUB_DISABLE_TELEMETRY': '1', 'TRANSFORMERS_OFFLINE': '1'}
CC = '/usr/local/cuda-12.9/bin/' + 'nv' + 'cc'
obj = O / 'flash_decode_paged.o'
argv = [CC, '-c', str(SRC / 'kernel/flash_decode_paged.cu'),
        '-o', str(obj), '-O3', '-std=c++17', '-gencode', 'arch=compute_70,code=sm_70',
        '--use_fast_math', '--ptxas-options=-v', '--expt-relaxed-constexpr',
        '--expt-extended-lambda', '--compiler-options=-fPIC', '-ccbin', '/usr/bin/gcc-14',
        '-DTORCH_API_INCLUDE_EXTENSION_H', '-DTORCH_EXTENSION_NAME=flash_attn_v100_cuda',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
        '-DS1_DEQUANT_DIET=%d' % macro]
for path in (SRC / 'kernel', SRC / 'include', T / 'include', T / 'include/torch/csrc/api/include',
             Path('/home/cookys/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12')):
    argv += ['-I' + str(path)]
(O / 'compile.argv.json').write_text(json.dumps(argv, indent=2) + '\n')
with open(O / 'ptxas.log', 'w') as lg:
    subprocess.run(argv, check=True, env=ENV, stdout=lg, stderr=subprocess.STDOUT)
NS = ROOT / 'harness/native-build-base'
common = [NS / 'temp/kernel' / f'{n}.o' for n in (
    'fused_mha_api', 'fused_mha_forward', 'fused_mha_forward_paged', 'fp8_kv_bridge',
    'flash_decode_turboquant', 'fused_mha_backward')]
for p in common:
    assert p.exists(), f'Common object not ready: {p}'
    name = p.stem + ('.cpp' if p.stem == 'fused_mha_api' else '.cu')
    assert (ROOT / 'harness/native-source/kernel' / name).read_bytes() == (SRC / 'kernel' / name).read_bytes(), name
for file in (ROOT / 'harness/native-source').rglob('*'):
    rel = file.relative_to(ROOT / 'harness/native-source')
    if file.suffix not in ('.h', '.cuh', '.cu', '.cpp') or str(rel) in ('kernel/flash_decode_paged.cu', 'kernel/c4_partition.cuh', 'kernel/s1_verify.cuh'):
        continue
    if str(rel) in ('kernel/fp8_kv_utils.cuh', 'kernel/a_on_loader.cuh'):
        continue  # the only two headers this branch edits; common .o were built from the unedited ones (macro default 0 = identical text path)
    assert file.read_bytes() == (SRC / rel).read_bytes(), str(rel)
original_so = next((NS / 'lib').glob('flash_attn_v100_cuda*.so'))
lib = O / 'lib'
lib.mkdir(exist_ok=True)
so = lib / original_so.name
link = ['/usr/bin/g++-14', '-shared', str(obj), *map(str, common), '-L' + str(T / 'lib'),
        '-L/usr/local/cuda-12.9/lib64', '-lc10', '-ltorch', '-ltorch_cpu', '-ltorch_python',
        '-lcudart', '-lc10_cuda', '-ltorch_cuda', '-o', str(so)]
(O / 'link.argv.json').write_text(json.dumps(link, indent=2) + '\n')
subprocess.run(link, check=True, env=ENV)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


(O / 'build-receipt.json').write_text(json.dumps({
    'GPU_NOT_RUN': True, 'arm': arm, 'S1_DEQUANT_DIET': macro, 'so': str(so), 'sha256': sha(so),
    'worktree_copy': str(WT), 'translation_unit': str(SRC / 'kernel/flash_decode_paged.cu'),
    'translation_unit_sha256': sha(SRC / 'kernel/flash_decode_paged.cu'),
    'edited_headers': {k: sha(SRC / 'kernel' / k) for k in ('fp8_kv_utils.cuh', 'a_on_loader.cuh', 's1_verify.cuh')},
    'reused_unchanged_objects': [{'path': str(p), 'sha256': sha(p)} for p in common]}, indent=2) + '\n')
print(arm, str(so), sha(so))
