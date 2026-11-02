# SPDX-License-Identifier: Apache-2.0
"""pool-serve-v1 member DRAFTER-SK: split-K SWAP GEMM for four DFlash2 drafter linears (DESIGN-pool-serve-v1-fable-20261028 sec 2.2, RULING-drafter-gemm-v1-fable-20261028 sec 3).

Env gate (set only in arm B's launcher): VLLM_SM70_DRAFTER_GEMM=1. With the env unset linear.py / qwen3_dflash.py never import this module and LIVE is byte-for-byte unchanged.
Table (M, N, K) -> split S, N = out features = weight.shape[0], K = in features = x.shape[-1] (the dg_plan.py convention: X [M,K] fp16 x W [N,K] fp16, K-contiguous, no bias, fp16 out):
  gate_up (128, 8704, 5120) S1, down (128, 5120, 4352) S2, fc (120, 1280, 25600) S8, ctxkv (120, 2560, 5120) S4. conv / qkv / o_proj stay on F.linear (they lose, ruling sec 4).
Kernel = sm70_drafter_gemm.so (5cca2784) dg_run(m, n, k, s, x, w, out, ws, stream): S == 1 one kernel; S > 1 fp32 partials to the workspace + the fixed-order reduce kernel, both launched INSIDE dg_run (dg_wrap.DrafterGemm.mm does exactly this call), so there is no separate reduce call to bind.

Why a torch custom op: the live serve runs mode=VLLM_COMPILE (inductor, FULL_AND_PIECEWISE) and DFlashQwen3Model is @support_torch_compile, so UnquantizedLinearMethod.apply of gate_up / down is TRACED. A raw ctypes call and a dict lookup on the symbolic batch dim cannot be traced. The static guards (env, prefix, dtype, bias, (N, K) of the weight) are shape-independent and are evaluated at trace time; the M-dependent table decision and the kernel call live inside the opaque op sm70_pool::drafter_linear, whose fake impl only declares the output shape. Outside the table the op runs F.linear, which is the same cuBLAS call the live graph makes (fp16, no bias).

Capture safety: the .so is loaded and the per-(M,N,K) fp32 workspace is allocated at the FIRST EAGER call of that shape (vLLM's warm-up precedes capture). A table shape that arrives first while a CUDA graph is being captured (torch.cuda.is_current_stream_capturing() or the mark_capture_begin() flag) falls back to F.linear and is counted (capture_fallback). No host sync. The output is allocated per call like F.linear does; the workspace is only an intermediate consumed inside the same stream-ordered dg_run, so sharing it across the five layers is safe.
Counters: drafter-hits-<pid>.json in $VLLM_SM70_POOL_RECEIPT_DIR (at exit, plus whenever the total reaches a power of two): per (M,N,K) routed / fallback / capture_fallback / contract_fallback, per table shape a hit rate, and the first 20 non-table draft shapes seen by the static guard. Under CUDA-graph capture a count is a capture, not a replay."""
import atexit
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

ENV_FLAG = "VLLM_SM70_DRAFTER_GEMM"
RECEIPT_ENV = "VLLM_SM70_POOL_RECEIPT_DIR"
PREFIX_ENV = "VLLM_SM70_DRAFTER_PREFIXES"      # comma separated override of DRAFT_PREFIXES
HERE = Path(__file__).resolve().parent
DRAFTER_SO = HERE / "sm70_drafter_gemm.so"
N_VARIANTS = 3                                  # S1-F16, SPLIT-PARTIAL, SPLIT-REDUCE (dg_wrap.VARIANTS)

# Draft module name prefixes (see README sec "prefix pattern"): the DFlash2 drafter is loaded by v1/worker/gpu/spec_decode/dflash/utils.py:load_dflash_model with get_model(prefix="") and
# DFlashQwen3ForCausalLM wraps it as maybe_prefix(prefix, "model"), so its linears are "model.layers.<target_layers + i>.mlp.{gate_up,down}_proj" and "model.fc"; the legacy draft-model proposer
# (v1/spec_decode/draft_model.py:76) uses "draft_model". The target's modules start with "language_model.", so "model.layers." does not match them; the exact-shape table is the second guard.
DRAFT_PREFIXES = ("draft_model.", "model.layers.", "model.fc")

# pool-serve-v3 re-key (RULING-pool-serve-v2 sec 4.2, DESIGN-pool-serve-v3 sec 1.2): the table is keyed by (N, K) and holds ranges (m_lo, m_hi, S); a call routes when m_lo <= M <= m_hi (first matching range), else it falls back.
# down / gate_up are exact M = 128 (in-graph, padded to the capture size; RULING-drafter-gateup-live-v1). fc / ctxkv are EMPTY in this source: compose_stage_v3.py fills the two marked lines verbatim from the table.json of drafter-table-live-m-v1
# (every range = the contiguous measured-PASS M values of that cell, never widened by hand). An empty list never routes.
RANGE_TABLE = {
    (5120, 4352): [(96, 128, 2)],    # down    S2 (M 96..128, drafter-table-m104-v1 table.json ee7e660f)
    (8704, 5120): [(96, 128, 1)],    # gate_up S1 (M 96..128, drafter-table-m104-v1 table.json ee7e660f)
    (1280, 25600): [(96, 128, 8)],    # fc      S8 (M 96..128, drafter-table-m104-v1 table.json ee7e660f)
    (2560, 5120): [(96, 128, 4)],    # ctxkv   S4 (M 96..128, drafter-table-m104-v1 table.json ee7e660f)
}
SHAPE_NAMES = {(8704, 5120): "gate_up", (5120, 4352): "down", (1280, 25600): "fc", (2560, 5120): "ctxkv"}      # (N, K) -> name
# derived views (receipt / slot_check): one entry per range, keyed at the range's m_hi
SPLIT_TABLE = {(hi, n, k): s for (n, k), rs in RANGE_TABLE.items() for (lo, hi, s) in rs}
NK_TABLE = frozenset(RANGE_TABLE)
NON_TABLE_KEEP = 20

capture_mode = False
_lib = None
_lib_sha8 = None
_ws = {}                  # (M,N,K) -> preallocated fp32 workspace (uint8 tensor), created at the first eager call of that shape
_ws_allocs = 0
_stats = {}               # (M,N,K) -> dict(routed, fallback, capture_fallback, contract_fallback)
_capture_sizes = {}       # (N,K) -> set of M seen while a CUDA graph was being captured (a capture is not a replay)
_non_table = []           # first NON_TABLE_KEEP (prefix tail, M or -1, N, K) seen by the static guard for draft-prefix layers outside the table
_total = 0
_lock = threading.Lock()


def mark_capture_begin():
    global capture_mode
    capture_mode = True


def mark_capture_end():
    global capture_mode
    capture_mode = False


def enabled():
    return os.environ.get(ENV_FLAG) == "1"


def draft_prefixes():
    v = os.environ.get(PREFIX_ENV)
    return tuple(p for p in v.split(",") if p) if v else DRAFT_PREFIXES


def is_draft_prefix(prefix):
    return isinstance(prefix, str) and prefix.startswith(draft_prefixes())


def split_for(m, n, k):
    """pure function of (M, N, K): the split S of the first range of (N, K) with m_lo <= M <= m_hi, or None when the shape / M is not in the table"""
    for lo, hi, s in RANGE_TABLE.get((n, k), ()):
        if lo <= m <= hi:
            return s
    return None


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


def _stat(key):
    s = _stats.get(key)
    if s is None:
        s = _stats[key] = dict(routed=0, routed_capture=0, fallback=0, capture_fallback=0, contract_fallback=0)
    return s


def _bump(key, field):
    global _total
    _stat(key)[field] += 1
    _total += 1
    if _total & (_total - 1) == 0:
        dump_receipt()


class _DrafterLib:
    """ctypes wrapper of sm70_drafter_gemm.so (same ABI as drafter-gemm-v1/pack/dg_wrap.py)"""

    def __init__(self, so_path):
        lib = self.lib = ctypes.CDLL(str(so_path))
        lib.dg_nvariants.restype = ctypes.c_int
        lib.dg_ws_bytes.restype = ctypes.c_ulonglong
        lib.dg_ws_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.dg_run.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 5
        if lib.dg_nvariants() != N_VARIANTS:
            raise RuntimeError(f"dg_nvariants {lib.dg_nvariants()} != {N_VARIANTS}")
        rc = lib.dg_init()
        if rc:
            raise RuntimeError(f"dg_init failed rc={rc}")

    def ws_bytes(self, m, n, s):
        return int(self.lib.dg_ws_bytes(m, n, s))

    def run(self, m, n, k, s, x_ptr, w_ptr, out_ptr, ws_ptr, stream_ptr):
        rc = self.lib.dg_run(m, n, k, s, ctypes.c_void_p(x_ptr), ctypes.c_void_p(w_ptr), ctypes.c_void_p(out_ptr), ctypes.c_void_p(ws_ptr), ctypes.c_void_p(stream_ptr))
        if rc:
            raise RuntimeError(f"dg_run(m={m}, n={n}, k={k}, s={s}) rc={rc}")


def _load_lib():
    global _lib, _lib_sha8
    if _lib is None:
        if not DRAFTER_SO.exists():
            raise RuntimeError(f"{ENV_FLAG}=1 but {DRAFTER_SO} is missing")
        _lib = _DrafterLib(DRAFTER_SO)
        _lib_sha8 = hashlib.sha256(DRAFTER_SO.read_bytes()).hexdigest()[:8]
    return _lib


def _contract_ok(x2d, w, s):
    import torch
    m, k = x2d.shape
    n = w.shape[0]
    if x2d.dtype != torch.float16 or w.dtype != torch.float16 or w.dim() != 2 or w.shape[1] != k:
        return False
    if not (x2d.is_contiguous() and w.is_contiguous()):
        return False
    if x2d.data_ptr() % 32 or w.data_ptr() % 32:
        return False
    return 1 <= m <= 128 and n % 128 == 0 and k % 64 == 0 and k % s == 0 and (k // s) % 64 == 0


def _linear_fallback(x, w):
    import torch.nn.functional as F
    return F.linear(x, w)


def drafter_linear_impl(x, w):
    """runtime body of the op: x [..., K] fp16, w [N, K] fp16 -> [..., N] fp16"""
    import torch
    k = int(x.shape[-1])
    n = int(w.shape[0])
    x2d = x.reshape(-1, k)
    m = int(x2d.shape[0])
    key = (m, n, k)
    s = split_for(m, n, k)
    cap = _capturing()
    if cap:
        _capture_sizes.setdefault((n, k), set()).add(m)
    if s is None:
        _bump(key, "fallback")
        return _linear_fallback(x, w)
    if not _contract_ok(x2d, w, s):
        _bump(key, "contract_fallback")
        return _linear_fallback(x, w)
    if cap and (_lib is None or (s > 1 and key not in _ws)):
        _bump(key, "capture_fallback")
        return _linear_fallback(x, w)
    global _ws_allocs
    lib = _load_lib()
    ws_ptr = 0
    if s > 1:
        ws = _ws.get(key)
        if ws is None:
            ws = _ws[key] = torch.empty(lib.ws_bytes(m, n, s), dtype=torch.uint8, device=x.device)
            _ws_allocs += 1
        ws_ptr = ws.data_ptr()
    out = torch.empty((m, n), dtype=torch.float16, device=x.device)
    lib.run(m, n, k, s, x2d.data_ptr(), w.data_ptr(), out.data_ptr(), ws_ptr, _stream_ptr(x))
    _bump(key, "routed_capture" if cap else "routed")
    return out.reshape(*x.shape[:-1], n)


def _register_op():
    import torch
    try:
        @torch.library.custom_op("sm70_pool::drafter_linear", mutates_args=())
        def drafter_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return drafter_linear_impl(x, w)

        @drafter_linear.register_fake
        def _(x, w):
            return x.new_empty((*x.shape[:-1], w.shape[0]))

        return drafter_linear
    except RuntimeError:        # already registered (module reloaded in the same process)
        return torch.ops.sm70_pool.drafter_linear


_op = _register_op()


def _note_non_table(prefix, x, w):
    if len(_non_table) < NON_TABLE_KEEP:
        m = x.shape[0] if x.dim() >= 2 else -1
        _non_table.append((prefix.rsplit(".", 1)[-1] if isinstance(prefix, str) else "", m if isinstance(m, int) else -1, int(w.shape[0]), int(w.shape[1])))


def maybe_linear(layer, x):
    """Called from UnquantizedLinearMethod.apply when VLLM_SM70_DRAFTER_GEMM == "1" and bias is None. Static (shape-independent) guards only; returns the op result or None (the caller then runs dispatch_unquantized_gemm unchanged)."""
    import torch
    w = layer.weight
    if not is_draft_prefix(getattr(layer, "prefix", "")):
        return None
    if x.dtype != torch.float16 or w.dtype != torch.float16 or w.dim() != 2 or x.dim() < 2:
        return None
    if (int(w.shape[0]), int(w.shape[1])) not in NK_TABLE or x.shape[-1] != w.shape[1]:
        if not torch.compiler.is_compiling():
            _note_non_table(layer.prefix, x, w)
        return None
    return _op(x, w)


def maybe_linear_weight(x, w, bias):
    """Context K/V site (qwen3_dflash.py _project_context_kv F.linear: the fused [L*2*kv, K] matrix is not a Linear module, so there is no prefix; the site only exists in the draft model). Returns None -> caller runs F.linear."""
    import torch
    if bias is not None or x.dtype != torch.float16 or w.dtype != torch.float16 or w.dim() != 2 or x.dim() < 2:
        return None
    if (int(w.shape[0]), int(w.shape[1])) not in NK_TABLE or x.shape[-1] != w.shape[1]:
        return None
    return _op(x, w)


def _table_summary():
    """per table (N, K) (DESIGN-pool-serve-v3 sec 1.2): eager calls whose M is inside a range (routed + fallback + contract_fallback; capture-time calls are not eager), eager routed, hit_rate = routed / in-range eager calls
    (None when there are none), capture_sizes = sorted M seen under capture, in_graph = capture_sizes inside a range. calls_at_nk / routed are kept for the v2 readers (calls_at_nk = every call at (N, K), routed = eager routed)."""
    out = {}
    for (n, k), rs in RANGE_TABLE.items():
        inr = lambda m: any(lo <= m <= hi for lo, hi, _s in rs)
        calls = [(mm, v) for (mm, nn, kk), v in _stats.items() if (nn, kk) == (n, k)]
        in_eager = sum(v["routed"] + v["fallback"] + v["contract_fallback"] for mm, v in calls if inr(mm))
        routed = sum(v["routed"] for mm, v in calls if inr(mm))
        caps = sorted(_capture_sizes.get((n, k), ()))
        out[SHAPE_NAMES[(n, k)]] = dict(n=n, k=k, ranges=[list(r) for r in rs], calls_at_nk=sum(sum(v.values()) for _mm, v in calls), eager_in_range=in_eager, routed=routed, hit_rate=(routed / in_eager if in_eager else None),
                                        capture_sizes=caps, in_graph=[m for m in caps if inr(m)], routed_capture=sum(v["routed_capture"] for _mm, v in calls))
    return out


def snapshot():
    return {f"{m}x{n}x{k}": dict(v) for (m, n, k), v in sorted(_stats.items())}


def dump_receipt(path=None):
    d = os.environ.get(RECEIPT_ENV)
    if path is None:
        if not d:
            return None
        path = Path(d) / f"drafter-hits-{os.getpid()}.json"
    path = Path(path)
    rec = dict(schema="sm70-drafter-pool-hits/1", pid=os.getpid(), env={ENV_FLAG: os.environ.get(ENV_FLAG)}, per_shape=snapshot(), table_summary=_table_summary(), total_calls=_total,
               non_table_draft_shapes=[dict(layer=a, m=b, n=c, k=e) for (a, b, c, e) in _non_table], draft_prefixes=list(draft_prefixes()), ws_allocs=_ws_allocs,
               drafter_so=str(DRAFTER_SO), drafter_so_sha8=_lib_sha8, split_table={f"{n}x{k}": [list(r) for r in rs] for (n, k), rs in RANGE_TABLE.items()}, capture_sizes={f"{n}x{k}": sorted(v) for (n, k), v in sorted(_capture_sizes.items())}, capture_mode=capture_mode,
               note="a count under CUDA-graph capture is a capture, not a replay; non_table_draft_shapes is filled at eager/trace time (M = -1 when symbolic)")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def reset_for_tests():
    global _lib, _lib_sha8, _ws_allocs, _total, capture_mode
    _lib = None
    _lib_sha8 = None
    _ws_allocs = 0
    _total = 0
    capture_mode = False
    _ws.clear()
    _stats.clear()
    _capture_sizes.clear()
    _non_table.clear()


atexit.register(dump_receipt)
