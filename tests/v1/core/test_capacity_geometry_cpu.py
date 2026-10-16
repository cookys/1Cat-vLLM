"""CPU source-contract tests; do not import torch or initialize CUDA."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[3]


def extract(path, name, parent=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if parent:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == parent).body
    node = copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef)
                              and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), node],
                        type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(path), 'exec')


G = {'cdiv': lambda n, d: (n + d - 1) // d}
IFACE = 'vllm/v1/kv_cache_interface.py'
UTIL = 'vllm/v1/core/kv_cache_utils.py'
for name in ['FullAttentionSpec', 'MambaSpec']:
    exec(extract(IFACE, 'max_memory_usage_bytes', name), G)
    G[name] = type(name, (), {
        'max_memory_usage_bytes': G.pop('max_memory_usage_bytes')})
exec(extract(UTIL, 'get_kv_cache_group_capacity_rows'), G)
exec(extract(UTIL, '_report_kv_cache_config'), G)
MGR = 'vllm/v1/core/single_type_kv_cache_manager.py'
exec(extract(MGR, 'reachable_block_mask', 'MambaManager'), G)


def fixture(length=253953, mode='align', speculative=0, full_groups=1,
            mamba_groups=3):
    cfg = NS(model_config=NS(max_model_len=length),
             parallel_config=NS(decode_context_parallel_size=1,
                                prefill_context_parallel_size=1),
             cache_config=NS(mamba_cache_mode=mode))
    groups = []
    for kind, n in [('FullAttentionSpec', full_groups), ('MambaSpec', mamba_groups)]:
        for _ in range(n):
            spec = G[kind]()
            spec.page_size_bytes = 1179648
            spec.block_size = 4096
            if kind == 'MambaSpec':
                spec.num_speculative_blocks = speculative
                spec.mamba_cache_mode = mode
            groups.append(NS(kv_cache_spec=spec, layer_names=list(range(16))))
    return cfg, NS(kv_cache_groups=groups, num_blocks=1062)


class GeometryTests(unittest.TestCase):
    def test_nospec_has_one_length_dependent_group(self):
        rows = G['get_kv_cache_group_capacity_rows'](*fixture())
        self.assertEqual([r['request_blocks'] for r in rows], [63, 2, 2, 2])
        self.assertEqual(sum(r['request_blocks'] for r in rows), 69)
        self.assertEqual([r['kind'] for r in rows],
                         ['FullAttentionSpec'] + ['MambaSpec'] * 3)

    def test_startup_report_uses_max_len_70_not_138(self):
        cfg, kv = fixture(length=262144)
        rows = G['get_kv_cache_group_capacity_rows'](cfg, kv)
        blocks = sum(r['request_blocks'] for r in rows)
        self.assertEqual(blocks, 70)
        self.assertEqual(int(kv.num_blocks / blocks * 262144), 3977098)

    def test_all_mode_and_speculation_not_silently_sparse(self):
        rows = G['get_kv_cache_group_capacity_rows'](*fixture(mode='all'))
        self.assertEqual([r['request_blocks'] for r in rows], [63]*4)
        rows = G['get_kv_cache_group_capacity_rows'](*fixture(speculative=7,
                                                        full_groups=2,
                                                        mamba_groups=6))
        self.assertEqual([r['request_blocks'] for r in rows], [63]*2+[9]*6)

    def test_log_emit_calls_geometry_and_does_not_mutate(self):
        cfg, kv = fixture()
        before = repr(kv)
        calls = []
        G['logger'] = NS(info=lambda *a: calls.append(a), info_once=lambda *a: None)
        G['get_max_concurrency_for_kv_cache_config'] = lambda *_: 1062/69
        G['_report_kv_cache_config'](cfg, kv)
        self.assertEqual(len(calls), 4)
        self.assertEqual([c[6] for c in calls], [63, 2, 2, 2])
        self.assertEqual(repr(kv), before)

    def test_existing_sparse_mask_preserves_reachable_boundaries(self):
        cfg, kv = fixture()
        spec = kv.kv_cache_groups[1].kv_cache_spec
        fn = G['reachable_block_mask']
        mask = fn(None, 0, 63, 4096, spec, 0, (253952, 253953, 131072))
        self.assertEqual([i for i, value in enumerate(mask) if value], [31, 61])
        # Exact aligned identical resend must retain the preceding boundary too.
        mask = fn(None, 0, 63, 4096, spec, 0, (253951, 253952))
        self.assertEqual([i for i, value in enumerate(mask) if value], [60, 61])
        self.assertIsNone(fn(None, 0, 63, 4096, spec, None, ()))
        self.assertIsNone(fn(None, 0, 63, 2048, spec, 0, ()))


if __name__ == '__main__':
    unittest.main()
