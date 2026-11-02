"""jit-warm-v1 overlay hook, NEW boots (same mechanism as the p8-suffix-v2 overlay/new hook: one meta-path finder, chained to the next sitecustomize). It answers for exactly TWO module names:
vllm.v1.worker.gpu.model_runner (DESIGN 2(ii) sampler warm-up in _dummy_sampler_run) and vllm.triton_utils.jit_monitor (activate() logs the Triton cache dir). Any other import is untouched.
Why a hook and not a plain PYTHONPATH prepend: `vllm` is a regular package in the stage, so a directory `overlay/vllm/` without __init__.py cannot shadow a submodule."""
import importlib.abc, importlib.machinery, importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGETS = {
    'vllm.v1.worker.gpu.model_runner': os.path.join(_HERE, 'vllm', 'v1', 'worker', 'gpu', 'model_runner.py'),
    'vllm.triton_utils.jit_monitor': os.path.join(_HERE, 'vllm', 'triton_utils', 'jit_monitor.py'),
}


class _OverlayModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        f = _TARGETS.get(fullname)
        if f is None:
            return None
        return importlib.util.spec_from_file_location(fullname, f)


if not any(type(f).__name__ == '_OverlayModuleFinder' for f in sys.meta_path):
    sys.meta_path.insert(0, _OverlayModuleFinder())

# chain to the next sitecustomize (e.g. /usr/lib/python3.x/sitecustomize.py), exactly as if this file were absent
_rest = [p for p in sys.path if os.path.abspath(p or '.') != _HERE]
_spec = importlib.machinery.PathFinder.find_spec('sitecustomize', _rest)
if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
