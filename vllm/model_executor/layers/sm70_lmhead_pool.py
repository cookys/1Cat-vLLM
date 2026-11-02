# SPDX-License-Identifier: Apache-2.0
"""pool-serve-v1 lever: per-M lm_head dispatcher for the DFlash2 fp32-logits site (DESIGN-pool-serve-v1-fable-20261028 sec 2.1).

Env gate (set only in arm B's launcher): VLLM_SM70_LMHEAD_POOL=1. With the env unset the call site never imports this module and LIVE is byte-for-byte unchanged.
Route per call (M = rows of the 2-D input; both the target head and the drafter head share the weight, so both land here):
  (a) M in CUSTOM_TABLE -> lmhead_gemm_swap<64,1> from sm70_lmhead_custom_v4.so (f3dc07af), variant SWAP, fp16 x fp16 -> fp32;
  (b) else M in LT_TABLE -> the bundle-3 cuBLASLt wrapper (sm70_lmhead_lt.py: layer._sm70_lmhead_lt, pinned algo per M);
  (c) else -> None, the caller runs the unchanged torch.mm.
Table lookup is a pure function of M. Bias handling stays in the caller.

Capture safety: the .so is loaded and initialised in prepare(layer) (weight-prep time, before any capture) or lazily at the FIRST EAGER call; the first eager call per table M also validates the call contract and records the M as prepared.
A table M that arrives for the first time while a CUDA graph is being captured (torch.cuda.is_current_stream_capturing() or the mark_capture_begin() flag) is NOT routed: it falls back to torch.mm and is counted (capture_fallback).
No host sync, no .so load, no workspace allocation after capture begins. The SWAP kernel needs no workspace. The output tensor is allocated per call exactly like torch.mm does (a static per-M output would alias between the target head and the drafter head when both run at the same M before the first result is consumed).
Under torch.compile (torch.compiler.is_compiling()) the call is not routed (counted compiling_fallback): the lm_head forward is not inside a compiled region in the live stage, this only protects a future change.
Counters (per route, keyed by M) are dumped as JSON to $VLLM_SM70_POOL_RECEIPT_DIR/lmhead-hits-<pid>.json at interpreter exit and, as a safety net for workers that are killed, whenever the total call count reaches a power of two. Under CUDA-graph capture a counter counts captures, not replays."""
import atexit
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

ENV_FLAG = "VLLM_SM70_LMHEAD_POOL"
RECEIPT_ENV = "VLLM_SM70_POOL_RECEIPT_DIR"
HERE = Path(__file__).resolve().parent
CUSTOM_SO = HERE / "sm70_lmhead_custom_v4.so"
VARIANT_SWAP = 1          # kVariants order in the .so: CUSTOM-B 0, SWAP 1 (lmcu_wrap.VARIANTS)
N_VARIANTS = 2
EXPECTED_SWAP_NAME_TOKEN = "swap"

CUSTOM_TABLE = {   # M -> citation (the M set of lmcu_check.MS = the M values with N1 + T1 receipts; nothing else is certified)
    96: 'RULING-lmhead-custom-v4-fable-20261028.md:11 T1 M96 r 0.9847; N1 line 10 (bit-equal to CUSTOM-B, M96 seed 0 in the 14 pairs)',
    104: 'RULING-lmhead-custom-v4-fable-20261028.md:11 T1 M104 r 0.9922; N1 line 10 (14 pairs: seed 0 + seeds 1..3 headline M)',
    112: 'RULING-lmhead-custom-v4-fable-20261028.md:11 T1 M112 median r 0.9941, max round 0.9977; N1 line 10; replay == eager at M112',
    120: 'RULING-lmhead-custom-v4-fable-20261028.md:11 T1 M120 r 0.9847; N1 line 10 (14 pairs: seed 0 + seeds 1..3 headline M)',
    128: 'RULING-lmhead-custom-v4-fable-20261028.md:11 T1 M128 r 0.9905; N1 line 10 (M128 seed 0 in the 14 pairs)',
}

LT_TABLE = {   # M -> max_abs_LIVE (RESULT.json /max_abs_live_by_m[M]); keys == RESULT.json /accepted_ms (33 values), argmax_flips 0 of 336 rows
    96: 3.605546953622252e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[0] = 96; max_abs_live_by_m["96"]
    97: 3.616723734012339e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[1] = 97; max_abs_live_by_m["97"]
    98: 3.559583274181932e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[2] = 98; max_abs_live_by_m["98"]
    99: 3.361045855854172e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[3] = 99; max_abs_live_by_m["99"]
    100: 3.582025965442881e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[4] = 100; max_abs_live_by_m["100"]
    101: 3.932677782358951e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[5] = 101; max_abs_live_by_m["101"]
    102: 3.871855005854741e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[6] = 102; max_abs_live_by_m["102"]
    103: 4.021122003905475e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[7] = 103; max_abs_live_by_m["103"]
    104: 3.483074397081509e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[8] = 104; max_abs_live_by_m["104"]
    105: 3.6262135836295784e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[9] = 105; max_abs_live_by_m["105"]
    106: 3.774513606913388e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[10] = 106; max_abs_live_by_m["106"]
    107: 3.803100480581634e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[11] = 107; max_abs_live_by_m["107"]
    108: 3.477437621768331e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[12] = 108; max_abs_live_by_m["108"]
    109: 3.864774771500379e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[13] = 109; max_abs_live_by_m["109"]
    110: 4.039142231704318e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[14] = 110; max_abs_live_by_m["110"]
    111: 3.74996307073161e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[15] = 111; max_abs_live_by_m["111"]
    112: 3.664517862489447e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[16] = 112; max_abs_live_by_m["112"]
    113: 3.9397542423103005e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[17] = 113; max_abs_live_by_m["113"]
    114: 4.0031962271314114e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[18] = 114; max_abs_live_by_m["114"]
    115: 4.2356654375907965e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[19] = 115; max_abs_live_by_m["115"]
    116: 3.682715760078281e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[20] = 116; max_abs_live_by_m["116"]
    117: 4.345276465755887e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[21] = 117; max_abs_live_by_m["117"]
    118: 4.1071070882026106e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[22] = 118; max_abs_live_by_m["118"]
    119: 3.9892460335977376e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[23] = 119; max_abs_live_by_m["119"]
    120: 4.14302121498622e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[24] = 120; max_abs_live_by_m["120"]
    121: 3.908033932020771e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[25] = 121; max_abs_live_by_m["121"]
    122: 3.611818829085678e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[26] = 122; max_abs_live_by_m["122"]
    123: 3.976439984398894e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[27] = 123; max_abs_live_by_m["123"]
    124: 3.639102214947343e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[28] = 124; max_abs_live_by_m["124"]
    125: 3.985288640251383e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[29] = 125; max_abs_live_by_m["125"]
    126: 3.6004158573632594e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[30] = 126; max_abs_live_by_m["126"]
    127: 3.78765762434341e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[31] = 127; max_abs_live_by_m["127"]
    128: 3.5848686820827425e-05,   # out-lmhead-lt-v1-20261028/RESULT.json accepted_ms[32] = 128; max_abs_live_by_m["128"]
}

PAD_LO = 81               # LMHEAD-MPAD (DESIGN-lmhead-mpad-v1 sec 2): off-table M in [PAD_LO, 127] is zero-padded up to the next certified custom M
PAD_TO = {m: min(c for c in CUSTOM_TABLE if c >= m) for m in range(PAD_LO, 128) if m not in CUSTOM_TABLE}

ROUTES = ("custom", "pad", "lt", "mm", "capture_fallback", "compiling_fallback", "contract_fallback")

capture_mode = False      # set by mark_capture_begin() / cleared by mark_capture_end(); torch.cuda.is_current_stream_capturing() is checked as well
_lib = None
_lib_sha8 = None
_prepared_m = set()       # table Ms whose first eager call (contract check) has happened
_counters = {r: {} for r in ROUTES}
_total = 0
_lock = threading.Lock()


def mark_capture_begin():
    """hook for the pack's harness: from now on nothing may be loaded / prepared / allocated"""
    global capture_mode
    capture_mode = True


def mark_capture_end():
    global capture_mode
    capture_mode = False


def enabled():
    return os.environ.get(ENV_FLAG) == "1"


def _stream_ptr(x):
    """raw stream handle of the current stream of x's device; 0 for a non-CUDA tensor (CPU unit tests with a stub kernel)"""
    import torch
    return torch.cuda.current_stream(x.device).cuda_stream if x.is_cuda else 0


def _capturing():
    if capture_mode:
        return True
    try:
        import torch
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def lookup(m):
    """pure function of M: 'custom' | 'pad' | 'lt' | 'mm'"""
    if m in CUSTOM_TABLE:
        return "custom"
    if m in PAD_TO:
        return "pad"
    if m in LT_TABLE:
        return "lt"
    return "mm"


def _count(route, m):
    global _total
    d = _counters[route]
    d[m] = d.get(m, 0) + 1
    _total += 1
    if _total & (_total - 1) == 0:
        dump_receipt()


class _CustomLib:
    """ctypes wrapper of sm70_lmhead_custom_v4.so (same ABI as lmhead-custom-v4/pack/lmcu_wrap.py, SWAP variant only)"""

    def __init__(self, so_path):
        lib = self.lib = ctypes.CDLL(str(so_path))
        lib.lmcu_version.restype = ctypes.c_int
        lib.lmcu_nvariants.restype = ctypes.c_int
        lib.lmcu_name.restype = ctypes.c_char_p
        lib.lmcu_name.argtypes = [ctypes.c_int]
        lib.lmcu_run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        if lib.lmcu_nvariants() != N_VARIANTS:
            raise RuntimeError(f"lmcu_nvariants {lib.lmcu_nvariants()} != {N_VARIANTS}")
        self.swap_name = lib.lmcu_name(VARIANT_SWAP).decode()
        rc = lib.lmcu_init()
        if rc:
            raise RuntimeError(f"lmcu_init failed rc={rc}")

    def run(self, m, n, k, x_ptr, w_ptr, out_ptr, stream_ptr):
        rc = self.lib.lmcu_run(VARIANT_SWAP, m, n, k, ctypes.c_void_p(x_ptr), ctypes.c_void_p(w_ptr), ctypes.c_void_p(out_ptr), ctypes.c_void_p(stream_ptr))
        if rc:
            raise RuntimeError(f"lmcu_run(SWAP, m={m}, n={n}, k={k}) rc={rc}")


def _load_lib():
    """load + init once, outside capture only; fail loud when the .so is missing (a silent torch.mm would turn arm B into arm A)"""
    global _lib, _lib_sha8
    if _lib is None:
        if not CUSTOM_SO.exists():
            raise RuntimeError(f"{ENV_FLAG}=1 but {CUSTOM_SO} is missing")
        _lib = _CustomLib(CUSTOM_SO)
        _lib_sha8 = hashlib.sha256(CUSTOM_SO.read_bytes()).hexdigest()[:8]
    return _lib


def prepare(layer):
    """Called at weight-prep time (vocab_parallel_embedding process_weights_after_loading, before any capture): loads the custom .so and prepares the cuBLASLt wrapper. Raises on any mismatch."""
    _load_lib()
    if getattr(layer, "_sm70_lmhead_lt", None) is None:
        from vllm.model_executor.layers import sm70_lmhead_lt
        sm70_lmhead_lt.prepare(layer)


def _contract_ok(x_2d, w):
    import torch
    if x_2d.dim() != 2 or w.dim() != 2 or x_2d.dtype != torch.float16 or w.dtype != torch.float16:
        return False
    if w.shape[1] != x_2d.shape[1] or w.shape[0] % 128 or x_2d.shape[1] % 64:
        return False
    if not (x_2d.is_contiguous() and w.is_contiguous()):
        return False
    return x_2d.data_ptr() % 32 == 0 and w.data_ptr() % 32 == 0


def _run_custom(x_2d, w, m):
    import torch
    lib = _lib
    n, k = int(w.shape[0]), int(w.shape[1])
    out = torch.empty((m, n), dtype=torch.float32, device=x_2d.device)
    lib.run(m, n, k, x_2d.data_ptr(), w.data_ptr(), out.data_ptr(), _stream_ptr(x_2d))
    return out


def route(layer, x_2d):
    """-> fp32 (M, N) logits, or None when this call stays on the unchanged torch.mm"""
    import torch
    w = layer.weight
    m = int(x_2d.shape[0])
    kind = lookup(m)
    if torch.compiler.is_compiling():
        _count("compiling_fallback", m)
        return None
    if kind == "custom":
        if not _contract_ok(x_2d, w):
            _count("contract_fallback", m)
            return None
        if _capturing() and (_lib is None or m not in _prepared_m):
            _count("capture_fallback", m)
            return None
        _load_lib()                      # eager only: reached with _lib set when capturing
        _prepared_m.add(m)
        out = _run_custom(x_2d, w, m)
        _count("custom", m)
        return out
    if kind == "pad":
        m_pad = PAD_TO[m]
        if not _contract_ok(x_2d, w):
            _count("contract_fallback", m)
            return None
        if _capturing() and (_lib is None or m_pad not in _prepared_m):
            _count("capture_fallback", m)
            return None
        _load_lib()                      # eager only: reached with _lib set when capturing
        _prepared_m.add(m_pad)
        x_pad = torch.zeros((m_pad, int(x_2d.shape[1])), dtype=torch.float16, device=x_2d.device)
        x_pad[:m] = x_2d
        out = _run_custom(x_pad, w, m_pad)[:m]
        _count("pad", f"{m}->{m_pad}")
        return out
    if kind == "lt":
        lt = getattr(layer, "_sm70_lmhead_lt", None)
        if lt is None:
            if _capturing():
                _count("capture_fallback", m)
                return None
            from vllm.model_executor.layers import sm70_lmhead_lt
            lt = sm70_lmhead_lt.prepare(layer)
        out = lt.mm(x_2d, w)
        if out is not None:
            _count("lt", m)
            return out
        _count("contract_fallback", m)
        return None
    _count("mm", m)
    return None


def snapshot():
    return {r: {str(m): n for m, n in sorted(d.items())} for r, d in _counters.items()}


def dump_receipt(path=None):
    """write lmhead-hits-<pid>.json atomically into $VLLM_SM70_POOL_RECEIPT_DIR (no-op when the env is unset and no path is given)"""
    d = os.environ.get(RECEIPT_ENV)
    if path is None:
        if not d:
            return None
        path = Path(d) / f"lmhead-hits-{os.getpid()}.json"
    path = Path(path)
    rec = dict(schema="sm70-lmhead-pool-hits/2", pid=os.getpid(), env={ENV_FLAG: os.environ.get(ENV_FLAG)}, counters=snapshot(), total_calls=_total, custom_table=sorted(CUSTOM_TABLE), pad_table={str(k): v for k, v in sorted(PAD_TO.items())}, lt_table=sorted(LT_TABLE),
               custom_so=str(CUSTOM_SO), custom_so_sha8=_lib_sha8, swap_kernel_name=getattr(_lib, "swap_name", None), prepared_m=sorted(_prepared_m), capture_mode=capture_mode,
               note="under CUDA-graph capture a count is a capture, not a replay")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def reset_for_tests():
    global _lib, _lib_sha8, _total, capture_mode
    _lib = None
    _lib_sha8 = None
    _total = 0
    capture_mode = False
    _prepared_m.clear()
    for d in _counters.values():
        d.clear()


atexit.register(dump_receipt)
