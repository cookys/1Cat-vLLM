"""CPU compile PVH translation unit, reuse identical harness common sources."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

D = Path(__file__).resolve().parent
ROOT = Path('/data/bench/explore/pv-hmma')
SRC = D/'native-source'
O = D/'build'
O.mkdir(exist_ok=True)
T = Path('/data/venvs/1cat-main-integp7pr-p8-gvB-integ/lib/python3.12/site-packages/torch')
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
       'HF_HUB_DISABLE_TELEMETRY': '1', 'TRANSFORMERS_OFFLINE': '1'}
obj = O / 'flash_decode_paged.o'
argv = ['/usr/local/cuda-12.9/bin/nvcc', '-c', str(SRC/'kernel/flash_decode_paged.cu'),
        '-o', str(obj), '-O3', '-std=c++17', '-gencode', 'arch=compute_70,code=sm_70',
        '--use_fast_math', '--ptxas-options=-v', '--expt-relaxed-constexpr',
        '--expt-extended-lambda', '--compiler-options=-fPIC', '-ccbin', '/usr/bin/gcc-14',
        '-DTORCH_API_INCLUDE_EXTENSION_H', '-DTORCH_EXTENSION_NAME=flash_attn_v100_cuda',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
for path in (SRC/'kernel', SRC/'include', T/'include', T/'include/torch/csrc/api/include',
             Path('/home/cookys/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12')):
    argv += ['-I' + str(path)]
(O/'compile.argv.json').write_text(json.dumps(argv, indent=2)+'\n')
subprocess.run(argv, check=True, env=ENV)
NS = ROOT/'harness/native-build-base'
common = [NS/'temp/kernel'/f'{n}.o' for n in (
    'fused_mha_api','fused_mha_forward','fused_mha_forward_paged','fp8_kv_bridge',
    'flash_decode_turboquant','fused_mha_backward')]
for p in common:
    assert p.exists(), f'Common object not ready: {p}'
    name = p.stem + ('.cpp' if p.stem == 'fused_mha_api' else '.cu')
    assert (ROOT/'harness/native-source/kernel'/name).read_bytes() == (SRC/'kernel'/name).read_bytes(), name
for file in (ROOT/'harness/native-source').rglob('*'):
    rel=file.relative_to(ROOT/'harness/native-source')
    if file.suffix not in ('.h','.cuh','.cu','.cpp') or str(rel) in ('kernel/flash_decode_paged.cu','kernel/c4_partition.cuh'):
        continue
    assert file.read_bytes()==(SRC/rel).read_bytes(), str(rel)
original_so = next((NS/'lib').glob('flash_attn_v100_cuda*.so'))
lib = O/'lib'; lib.mkdir(exist_ok=True)
so = lib / original_so.name
link = ['/usr/bin/g++-14', '-shared', str(obj), *map(str,common), '-L'+str(T/'lib'),
        '-L/usr/local/cuda-12.9/lib64', '-lc10', '-ltorch', '-ltorch_cpu', '-ltorch_python',
        '-lcudart', '-lc10_cuda', '-ltorch_cuda', '-o', str(so)]
(O/'link.argv.json').write_text(json.dumps(link, indent=2)+'\n')
subprocess.run(link, check=True, env=ENV)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
(O/'build-receipt.json').write_text(json.dumps({
    'GPU_NOT_RUN':True, 'so':str(so), 'sha256':sha(so),
    'reused_unchanged_objects':[{ 'path':str(p),'sha256':sha(p)} for p in common],
    'common_source_equality_verified':True,
    'variant':'s1-v4-A',
    'sources':{str(p.relative_to(SRC)):sha(p) for p in (SRC/'kernel').glob('pvb*')},
    'translation_unit_sha256':sha(SRC/'kernel/flash_decode_paged.cu')},indent=2)+'\n')
print(str(so),sha(so))
