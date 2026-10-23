"""Cold-boot scoped dispatch to measured A+A1b, retaining original native ABI."""
import logging
import os
import torch

LOG = logging.getLogger(__name__)
_seen = set()


def eligible(q, k, v, bt, sl, dtype, one_pass):
    return (dtype == 'nvfp4' and one_pass is True
            and q.ndim == 3 and q.dtype == torch.float16
            and bt.ndim == 2 and 1 <= bt.shape[0] <= 16
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
    on = os.getenv('FLASH_V100_VERIFY_ARM') == 'A1'
    use = on and eligible(q, k, v, bt, sl, dtype, one_pass)
    native = base
    if use:
        from ._a1b_native import flash_attn_v100_cuda
        native = flash_attn_v100_cuda
    key = (bool(capture), tuple(q.shape), tuple(bt.shape), use)
    if key not in _seen:
        _seen.add(key)
        LOG.warning('D1_ROUTE phase=%s B=%s qrows=%s requested=%s selected=%s native=%s',
                    'capture' if capture else 'eager', bt.shape[0], q.shape[0],
                    'A1B' if on else 'FIX', 'A1B' if use else 'FIX', native.__file__)
    return native
