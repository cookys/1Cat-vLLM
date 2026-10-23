"""Cold-boot scoped dispatch to measured S1-v4-A5, retaining original native ABI."""
import logging
import os
import torch

LOG = logging.getLogger(__name__)
_seen = set()


def eligible(q, k, v, bt, sl, dtype, one_pass):
    return (dtype == 'nvfp4' and one_pass is True
            and q.ndim == 3 and q.dtype == torch.float16
            and bt.ndim == 2 and 1 <= bt.shape[0] <= (20 if os.getenv("VLLM_TEAM_MIXED") == "1" else 16)
            and q.shape == (bt.shape[0]*8, 6, 256)
            and sl.shape == (bt.shape[0],)
            and sl.dtype == bt.dtype == torch.int32
            and all(t.is_contiguous() for t in (q, bt, sl))
            and k.ndim == v.ndim == 4 and k.shape == v.shape
            and k.shape[1:] == (4096, 1, 144)
            and k.dtype == v.dtype == torch.uint8
            and all(t.stride()[1:] == (144, 144, 1)
                    and t.stride(0) % 16 == 0 and t.data_ptr() % 16 == 0
                    for t in (k, v)))


def select_native(base, q, k, v, bt, sl, dtype, one_pass, capture):
    requested = os.getenv('FLASH_V100_VERIFY_ARM')
    name = 'A1B' if requested == 'A1' else requested
    use = name in ('S1', 'A1B') and eligible(q, k, v, bt, sl, dtype, one_pass)
    native = base
    if use and name == 'S1':
        from ._s1_native import flash_attn_v100_cuda
        native = flash_attn_v100_cuda
    elif use:
        from ._a1b_native import flash_attn_v100_cuda
        native = flash_attn_v100_cuda
    selected = name if use else 'FIX'
    key = (bool(capture), tuple(q.shape), tuple(bt.shape), selected)
    if key not in _seen:
        _seen.add(key)
        LOG.warning('D1_ROUTE phase=%s B=%s requested=%s selected=%s native=%s',
                    'capture' if capture else 'eager', bt.shape[0], name, selected, native.__file__)
    from .team_route import selection
    selection(int(bt.shape[0]), selected, bool(capture))
    _team_batch = int(bt.shape[0])
    if _team_batch <= 4 and _team_batch in (1, 2, 4):
        from .team_route_smallb import selection as smallb_selection
        smallb_selection(_team_batch, selected, bool(capture))
    return native
