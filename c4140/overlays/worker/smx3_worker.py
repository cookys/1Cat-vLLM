"""SMX3-ADOPT worker extension: one runtime-evidence line per worker process (RULING env-receipt-setproctitle-clobber: /proc/<pid>/environ of a renamed worker is unreliable).
Loaded through `--worker-extension-cls smx3_worker.TuneWorker` in BOTH arms. It reads the tune max M twice inside the worker: os.environ, and libc getenv() via ctypes -- the very call
turbomind's `nvfp4_dense_tune_max_m()` makes (awq_sm70_gemm.cu `std::getenv`). Print only; no RPC, no kernel code touched, no effect on serving."""
import ctypes
import os

KEYS = dict(tune_max_m='VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M', whole_m_tiles='VLLM_SM70_NVFP4_WHOLE_M_TILES')


def _libc_getenv(name):
    try:
        f = ctypes.CDLL(None).getenv
        f.restype = ctypes.c_char_p
        v = f(name.encode())
        return 'UNSET' if v is None else v.decode()
    except Exception as e:
        return 'ERR:' + type(e).__name__


def _announce():
    parts = ['%s=%s libc_%s=%s' % (k, os.environ.get(v, 'UNSET'), k, _libc_getenv(v)) for k, v in KEYS.items()]
    print('TUNEENV pid=%d ppid=%d %s' % (os.getpid(), os.getppid(), ' '.join(parts)), flush=True)


_announce()


class TuneWorker:
    """Empty worker mixin; the evidence line is printed at import."""
