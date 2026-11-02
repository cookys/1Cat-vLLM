"""tokcache site shim: install the RENEW-TOKCACHE hook, then chain to the next sitecustomize on sys.path (e.g. the jit-warm overlay's), as if this file were absent."""
import importlib.machinery, importlib.util, os, sys
if not getattr(sys, '_tokcache_shim_running', False):  # re-entry guard: another overlay's chain may find this file again
    sys._tokcache_shim_running = True
    _HERE = os.path.dirname(os.path.abspath(__file__)); _LANE = os.path.dirname(_HERE)
    if os.environ.get('TOKCACHE') != '0':
        if _LANE not in sys.path: sys.path.append(_LANE)
        import tokcache_overlay; tokcache_overlay.install_import_hook()
    sys._tok_shim_dirs = list(getattr(sys, '_tok_shim_dirs', [])) + [_HERE]  # PACK VENDOR PATCH (renew-tokcache-serve-v1): every chained shim hides ALL shim dirs from the next lookup, else shim2 re-finds shim1 (guard: no-op) and the chain never reaches the jit-warm sitecustomize
    _rest = [p for p in sys.path if os.path.abspath(p or '.') not in sys._tok_shim_dirs]
    _spec = importlib.machinery.PathFinder.find_spec('sitecustomize', _rest)
    if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
        _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
