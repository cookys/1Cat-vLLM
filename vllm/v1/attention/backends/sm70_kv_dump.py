# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated dump of the pre-quantization fp16 K/V (diagnostic, plan 072 Q1.23).

VLLM_FLASH_V100_KV_DUMP_DIR (default "" = disabled) turns on a hook in the
SM70 Flash-V100 ``do_kv_cache_update``. It runs *before* any quantizing store
and copies the fresh K/V of the first request (request 0 of the batch, only
while its context grows contiguously) into per-layer CPU buffers until
VLLM_FLASH_V100_KV_DUMP_MAX_TOKENS tokens were collected. Output:

    <dir>/rank<r>/layer<L>_k.npy   fp16 [tokens, head_dim]  ([tokens, heads, head_dim] if heads > 1)
    <dir>/rank<r>/layer<L>_v.npy
    <dir>/rank<r>/meta.json

Disabled cost: one module-level bool read (``ENABLED``) per call, no env
parsing. The hook never mutates key/value.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Cached once at import; env is fixed before the engine process starts.
ENABLED: bool = bool(envs.VLLM_FLASH_V100_KV_DUMP_DIR)

_LAYER_RE = re.compile(r"layers?\.(\d+)")
_DUMPER: KVDumper | None = None


class KVDumper:
    def __init__(self, dump_dir: str, max_tokens: int, rank: int = 0) -> None:
        import os

        self.dir = os.path.join(dump_dir, f"rank{rank}")
        self.root = dump_dir
        self.max_tokens = int(max_tokens)
        self.rank = rank
        self._state: dict[str, dict[str, Any]] = {}
        self._layer_ids: dict[str, int] = {}  # layer_name -> file index
        self._written: dict[str, dict[str, Any]] = {}
        self._log_pending = False
        self._logged_written = False
        self._meta_extra: dict[str, Any] = {}
        logger.info(
            "FLASH_ATTN_V100 KV dump active dir=%s max_tokens=%d rank=%d",
            dump_dir,
            self.max_tokens,
            rank,
        )

    def _file_index(self, layer_name: str, fallback: int) -> int:
        if layer_name in self._layer_ids:
            return self._layer_ids[layer_name]
        m = _LAYER_RE.search(layer_name or "")
        idx = int(m.group(1)) if m else fallback
        used = set(self._layer_ids.values())
        while idx in used:  # e.g. draft-model layers reusing indices
            idx += 10000
        self._layer_ids[layer_name] = idx
        return idx

    def collect(
        self,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str | None = None,
        block_size: int | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> None:
        if layer_name in self._written:
            if self._log_pending:
                self._emit_written_log()
            return
        st = self._state.setdefault(layer_name, {"k": [], "v": [], "n": 0})
        if len(query_start_loc) < 2:
            return
        q0 = int(query_start_loc[1]) - int(query_start_loc[0])
        s0 = int(seq_lens[0])
        collected = st["n"]
        contiguous = q0 >= 1 and (s0 - q0) == collected
        decode_like = q0 == 1 and collected > 0
        if collected == 0 and q0 < 2:
            return  # warmup / dummy decode, not a prompt
        if not contiguous or decode_like:
            if collected > 0:
                self._finalize(layer_name, kv_cache_dtype, block_size, k_scale, v_scale,
                               num_kv_heads, head_dim)
            return
        take = min(q0, self.max_tokens - collected)
        start = int(query_start_loc[0])
        k = key[start : start + take].detach()
        v = value[start : start + take].detach()
        k = k.reshape(take, num_kv_heads, head_dim)
        v = v.reshape(take, num_kv_heads, head_dim)
        if k.dtype != torch.float16:
            k = k.to(torch.float16)
        if v.dtype != torch.float16:
            v = v.to(torch.float16)
        st["k"].append(k.to("cpu", non_blocking=False).numpy().copy())
        st["v"].append(v.to("cpu", non_blocking=False).numpy().copy())
        st["n"] += take
        if st["n"] >= self.max_tokens:
            self._finalize(layer_name, kv_cache_dtype, block_size, k_scale, v_scale,
                           num_kv_heads, head_dim)

    def _finalize(self, layer_name, kv_cache_dtype, block_size, k_scale, v_scale,
                  num_kv_heads, head_dim) -> None:
        import os

        st = self._state.pop(layer_name, None)
        if st is None or st["n"] == 0:
            self._written[layer_name] = {"tokens": 0}
            return
        idx = self._file_index(layer_name, len(self._written))
        os.makedirs(self.dir, exist_ok=True)
        k = np.concatenate(st["k"], axis=0)
        v = np.concatenate(st["v"], axis=0)
        if num_kv_heads == 1:
            k, v = k[:, 0, :], v[:, 0, :]
        np.save(os.path.join(self.dir, f"layer{idx}_k.npy"), k)
        np.save(os.path.join(self.dir, f"layer{idx}_v.npy"), v)
        self._written[layer_name] = {
            "index": idx,
            "tokens": int(st["n"]),
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
        self._meta_extra.update(
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
        )
        self._write_meta()
        self._log_pending = True

    def _write_meta(self) -> None:
        import os

        done = {n: w for n, w in self._written.items() if w.get("tokens")}
        tokens = max((w["tokens"] for w in done.values()), default=0)
        meta = {
            "layers": sorted(w["index"] for w in done.values()),
            "layer_names": {str(w["index"]): n for n, w in done.items()},
            "tokens": tokens,
            "tokens_per_layer": {str(w["index"]): w["tokens"] for w in done.values()},
            "max_tokens": self.max_tokens,
            "k_scale": {str(w["index"]): w["k_scale"] for w in done.values()},
            "v_scale": {str(w["index"]): w["v_scale"] for w in done.values()},
            "tp_rank": self.rank,
            "dtype": "float16",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **self._meta_extra,
        }
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    def _emit_written_log(self) -> None:
        if self._logged_written:
            return
        self._logged_written = True
        self._log_pending = False
        done = [w for w in self._written.values() if w.get("tokens")]
        logger.info(
            "FLASH_ATTN_V100 KV dump written dir=%s rank=%d layers=%d tokens=%d",
            self.root,
            self.rank,
            len(done),
            max((w["tokens"] for w in done), default=0),
        )


def _metadata_for_layer(layer_name: str) -> Any:
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        return raw.get(layer_name)
    return raw


def _get_dumper() -> KVDumper:
    global _DUMPER
    if _DUMPER is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = int(get_tensor_model_parallel_rank())
        except Exception:
            rank = 0
        _DUMPER = KVDumper(
            envs.VLLM_FLASH_V100_KV_DUMP_DIR,
            envs.VLLM_FLASH_V100_KV_DUMP_MAX_TOKENS,
            rank,
        )
    return _DUMPER


def _scalar(x: Any) -> float | None:
    try:
        return float(x.item() if hasattr(x, "item") else x)
    except Exception:
        return None


def on_kv_update(layer, impl, key: torch.Tensor, value: torch.Tensor) -> None:
    """Hook called from do_kv_cache_update before any store. Never raises."""
    try:
        if key.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        layer_name = getattr(layer, "layer_name", None) or f"layer{id(layer)}"
        md = _metadata_for_layer(layer_name)
        if md is None:
            return
        qsl = getattr(md, "query_start_loc_cpu", None)
        if qsl is None:
            qsl = md.query_start_loc
        qsl = qsl.to("cpu")
        seq_lens = md.seq_lens.to("cpu")
        _get_dumper().collect(
            layer_name,
            key,
            value,
            qsl,
            seq_lens,
            num_kv_heads=int(impl.num_kv_heads),
            head_dim=int(key.shape[-1]) if key.dim() == 3 else int(impl.head_size),
            kv_cache_dtype=getattr(impl, "kv_cache_dtype", None),
            block_size=getattr(md, "block_size", None),
            k_scale=_scalar(getattr(layer, "_k_scale_float", None)),
            v_scale=_scalar(getattr(layer, "_v_scale_float", None)),
        )
    except Exception:  # diagnostic path must never break serving
        logger.exception("FLASH_ATTN_V100 KV dump hook failed")
