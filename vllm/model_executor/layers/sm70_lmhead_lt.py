# SPDX-License-Identifier: Apache-2.0
"""bundle-3 lever 2: cublasLt lm_head GEMM (fp16 x fp16 -> fp32) with an algo pinned per M from a frozen table.
Env-gated by VLLM_SM70_LMHEAD_LT=1 at the call site (vocab_parallel_embedding._maybe_sm70_lm_head_forward); with the env unset nothing here is imported or called, so LIVE is byte-for-byte unchanged.
Contract: M in the table -> cublasLtMatmul with the table's algo (same layout/compute as the P0-LM CUBLASLT-BEST arm); any other M, dtype, shape or compile mode -> None (the caller then runs the LIVE torch.mm).
Capture safety: handle, descriptors, layouts and the static workspace are created in prepare() (weight-load time, before any capture); mm() only allocates the output (like torch.mm), calls cublasLtMatmul on the current stream and never syncs.
Fail loud: a table that does not match this cublasLt/device/shape, a missing .so, or a cublasLt error raises (no silent LIVE fallback that would turn arm B into arm A)."""
import ctypes
import hashlib
import json
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
HERE = Path(__file__).resolve().parent
SO_PATH = HERE / "sm70_lmhead_lt_native.so"
TABLE_PATH = HERE / "sm70_lmhead_lt_table.json"
SCHEMA = "sm70-lmhead-lt-table/1"


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_table(path=TABLE_PATH):
    """-> (table dict, sha256). Validates the schema and every algo string (128 hex chars = 8 x uint64 little endian)."""
    raw = Path(path).read_bytes()
    t = json.loads(raw)
    if t.get("schema") != SCHEMA:
        raise RuntimeError(f"lmhead-lt table schema {t.get('schema')!r} != {SCHEMA}")
    if not t.get("ms"):
        raise RuntimeError("lmhead-lt table has no M entries")
    for m, e in t["ms"].items():
        a = e.get("algo")
        if not (isinstance(a, str) and len(a) == 128 and all(c in "0123456789abcdef" for c in a)):
            raise RuntimeError(f"lmhead-lt table M={m}: malformed algo")
    return t, hashlib.sha256(raw).hexdigest()


class LmHeadLt:
    def __init__(self, n, k, table, table_sha, lib, device):
        self.n, self.k, self.table_sha, self.lib, self.device = n, k, table_sha, lib, device
        if int(table["n"]) != n or int(table["k"]) != k:
            raise RuntimeError(f"lmhead-lt table is for N={table['n']} K={table['k']}, layer is N={n} K={k}")
        cap = tuple(torch.cuda.get_device_capability(device))
        if list(cap) != list(table.get("device_cc", cap)):
            raise RuntimeError(f"lmhead-lt table device_cc {table.get('device_cc')} != {cap}")
        ver = int(lib.lt_version())
        if int(table.get("cublaslt_version", ver)) != ver:
            raise RuntimeError(f"lmhead-lt table was built with cublasLt {table.get('cublaslt_version')}, running {ver}")
        rc = lib.lt_init(n, k)
        if rc:
            raise RuntimeError(f"lt_init failed rc={rc}")
        self.ms, ws_max = set(), 0
        for m_s, e in sorted(table["ms"].items(), key=lambda kv: int(kv[0])):
            m = int(m_s)
            algo = (ctypes.c_uint64 * 8).from_buffer_copy(bytes.fromhex(e["algo"]))
            rc = lib.lt_add(m, algo)
            if rc:
                raise RuntimeError(f"lt_add M={m} failed rc={rc} (pinned algo rejected by this cublasLt)")
            ws = int(lib.lt_ws_bytes(m))
            if ws != int(e.get("ws", ws)):
                raise RuntimeError(f"lmhead-lt table M={m}: workspace {e.get('ws')} != validated {ws}")
            ws_max = max(ws_max, ws)
            self.ms.add(m)
        self.ws = torch.empty(ws_max, dtype=torch.uint8, device=device) if ws_max else None
        self.ws_cap = ws_max
        self.ws_ptr = self.ws.data_ptr() if ws_max else 0
        self.lt_calls, self.live_calls, self._logged = {}, {}, set()
        logger.info("[lmhead-lt] active: n=%d k=%d ms=%d..%d (%d) cublasLt=%d table_sha256=%s ws_bytes=%d", n, k, min(self.ms), max(self.ms), len(self.ms), ver, table_sha[:16], ws_max)

    def _count(self, d, m, tag):
        d[m] = d.get(m, 0) + 1
        c = d[m]
        if c & (c - 1) == 0:      # 1st, 2nd, 4th, 8th ... per M (under CUDA-graph capture this counts captures, not replays)
            logger.info("[lmhead-lt] route %s M=%d calls=%d", tag, m, c)

    def mm(self, x2d, weight):
        """-> fp32 (M, N) logits via cublasLt, or None when this call must stay on the LIVE path."""
        if torch.compiler.is_compiling():
            return None
        m = x2d.shape[0]
        if (m not in self.ms or x2d.dtype != torch.float16 or weight.dtype != torch.float16 or x2d.dim() != 2 or x2d.shape[1] != self.k
                or tuple(weight.shape) != (self.n, self.k) or not x2d.is_contiguous() or not weight.is_contiguous()):
            if m not in self._logged:
                self._logged.add(m)
                logger.info("[lmhead-lt] M=%d dtype=%s -> LIVE", m, x2d.dtype)
            self._count(self.live_calls, m, "live")
            return None
        out = torch.empty((m, self.n), dtype=torch.float32, device=x2d.device)
        rc = self.lib.lt_run(m, ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(x2d.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(self.ws_ptr),
                             ctypes.c_size_t(self.ws_cap), ctypes.c_void_p(torch.cuda.current_stream(x2d.device).cuda_stream))
        if rc:
            raise RuntimeError(f"lt_run M={m} failed rc={rc}")
        self._count(self.lt_calls, m, "lt")
        return out


def _load_lib():
    lib = ctypes.CDLL(str(SO_PATH))
    lib.lt_version.restype = ctypes.c_int
    lib.lt_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.lt_add.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    lib.lt_ws_bytes.argtypes = [ctypes.c_int]
    lib.lt_ws_bytes.restype = ctypes.c_size_t
    lib.lt_has.argtypes = [ctypes.c_int]
    lib.lt_run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return lib


def prepare(layer):
    """Called at weight-prep time when VLLM_SM70_LMHEAD_LT=1 (before any CUDA-graph capture). Sets layer._sm70_lmhead_lt. Raises on any mismatch."""
    if getattr(layer, "_sm70_lmhead_lt", None) is not None:
        return layer._sm70_lmhead_lt
    w = layer.weight
    if not w.is_cuda or w.dtype != torch.float16 or w.dim() != 2:
        raise RuntimeError(f"VLLM_SM70_LMHEAD_LT=1 but the lm_head weight is {w.dtype} on {w.device} dim={w.dim()} (fp16 CUDA 2-D required)")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("lmhead-lt prepare() must run before CUDA graph capture")
    table, sha = load_table()
    layer._sm70_lmhead_lt = LmHeadLt(int(w.shape[0]), int(w.shape[1]), table, sha, _load_lib(), w.device)
    return layer._sm70_lmhead_lt
