# SPDX-License-Identifier: Apache-2.0
"""MF12-I memory trace (information cell, instrumentation only).

Active only when MF12I_TRACE_DIR is set; otherwise every call is a no-op.  One JSON line per
snapshot: torch.cuda.mem_get_info() free/total, memory_reserved, memory_allocated, written to
$MF12I_TRACE_DIR/memtrace-pid<pid>.jsonl (append, one file per worker process).  Never called
while a CUDA graph is being captured on the current stream.
"""

import json
import os
import time

import torch

_DIR = os.environ.get("MF12I_TRACE_DIR")
_FH = None


def enabled() -> bool:
    return bool(_DIR)


def _rank() -> int:
    try:
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def snap(tag: str, **fields) -> None:
    global _FH
    if not _DIR:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
        free, total = torch.cuda.mem_get_info()
        rec = {
            "t": time.time(),
            "tag": tag,
            "rank": _rank(),
            "dev": torch.cuda.current_device(),
            "pid": os.getpid(),
            "free": int(free),
            "total": int(total),
            "reserved": int(torch.cuda.memory_reserved()),
            "allocated": int(torch.cuda.memory_allocated()),
        }
        rec.update(fields)
        if _FH is None:
            os.makedirs(_DIR, exist_ok=True)
            _FH = open(
                os.path.join(_DIR, f"memtrace-pid{os.getpid()}.jsonl"), "a", buffering=1
            )
        _FH.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:  # tracing must never break the server
        try:
            print(f"MF12I trace error at {tag}: {e!r}", flush=True)
        except Exception:
            pass
