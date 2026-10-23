"""Small-B runtime route evidence. B=8 stays on the original team_route module."""
import hashlib
import json
import logging
import math
import os
import stat
import time
from pathlib import Path

from .team_route import NATIVES, expected, enabled

LOG = logging.getLogger("flash_attn_v100.team_route")
BATCHES = (1, 2, 4)
CONTROL_MAX_AGE_S = 15441.0
CONTROL_POLL_S = 0.25
_active = None
_graphs = {}
_last = {}
_control_cache = {}


def _control_path():
    trace = os.getenv("VLLM_S1_MAL_TRACE_DIR")
    if not trace:
        return None
    return Path(trace).parent / "route-smallb-control.json"


def _control(B, *, force_poll=False):
    # This guard must stay before path construction, stat/open, or cache lookup.
    if type(B) is not int or B > 4 or B not in (1, 2, 4):
        return None
    if not enabled():
        return None
    path = _control_path()
    if path is None:
        return None
    now = time.monotonic()
    cached = _control_cache.get(B)
    if not force_poll and cached and now - cached[0] < CONTROL_POLL_S:
        return cached[2]
    value = None
    cache_key = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size > 1024:
                raise ValueError("control-file-type-owner-or-size")
            identity = (st.st_ino, st.st_size, st.st_mtime_ns)
            raw = bytearray()
            while len(raw) <= 1024:
                block = os.read(fd, 1025 - len(raw))
                if not block:
                    break
                raw.extend(block)
            if len(raw) > 1024:
                raise ValueError("control-file-too-large")
        finally:
            os.close(fd)
        item = json.loads(raw.decode("utf-8"))
        if type(item) is not dict or set(item) != {"schema", "arm", "batch", "issued_mono", "token"}:
            raise ValueError("control-schema")
        if item["schema"] != "chain126c-smallb-control/1":
            raise ValueError("control-version")
        if item["arm"] != f"c{B}" or type(item["batch"]) is not int or item["batch"] != B:
            raise ValueError("control-arm-batch")
        issued = item["issued_mono"]
        if type(issued) not in (int, float) or not math.isfinite(issued):
            raise ValueError("control-time-type")
        age = now - issued
        if age < -0.05 or age > CONTROL_MAX_AGE_S:
            raise ValueError("control-stale-or-future")
        token = item["token"]
        if type(token) is not str or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("control-token-format")
        digest = hashlib.sha256(raw).hexdigest()
        cache_key = (type(item["arm"]).__name__, item["arm"], type(item["batch"]).__name__, B,
                     token, identity[0], identity[1], identity[2])
        value = dict(arm=item["arm"], batch=B, token_sha256=digest,
                     control_identity=dict(inode=identity[0], size=identity[1], mtime_ns=identity[2]))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        value = None
    _control_cache[B] = (now, cache_key, value)
    return value


def _emit(row, state, *, force=False):
    key = (row["phase"], row.get("graph_id"), row["B"], row["selected"])
    now = time.monotonic()
    if force or now - _last.get(key, -1e9) >= 2:
        _last[key] = now
        LOG.warning("TEAM_ROUTE %s", json.dumps(dict(
            row, mono=now, pid=os.getpid(),
            native_sha256=NATIVES.get(row["selected"]),
            arm_control=state["arm"], arm_control_sha256=state["token_sha256"],
            arm_control_identity=state["control_identity"],
        ), sort_keys=True))


def capture_begin(graph, B):
    global _active
    if type(B) is not int or B not in BATCHES or not enabled():
        return
    if _active is not None:
        raise RuntimeError("smallb-team-route-nested-capture")
    _active = (graph, B, [])


def capture_end(graph):
    global _active
    if _active is None:
        return
    if _active[0] is not graph:
        raise RuntimeError("smallb-team-route-capture-object")
    _graphs[id(graph)] = (graph, _active[1], tuple(_active[2]))
    _active = None


def selection(B, selected, capture):
    if type(B) is not int or B not in BATCHES or not enabled():
        return
    if _active is not None:
        if _active[1] != B:
            raise RuntimeError("smallb-team-route-capture-batch")
        _active[2].append((B, selected))
    elif not capture:
        state = _control(B)
        if state:
            _emit(dict(phase="eager", B=B, graph_B=B, graph_id=None, selected=selected), state,
                  force=selected != expected())


def replay(graph, live_B, graph_B):
    if type(live_B) is not int or live_B not in BATCHES or not enabled():
        return
    state = _control(live_B)
    if not state:
        return
    item = _graphs.get(id(graph))
    calls = item[2] if item and item[0] is graph else ()
    good = graph_B == live_B and len(calls) == 16 and all(b == live_B and s == expected() for b, s in calls)
    _emit(dict(phase="replay", B=live_B, graph_B=graph_B, graph_id=id(graph),
               captured_calls=len(calls), selected=expected() if good else "FIX"), state, force=not good)


def control_cache_clear():
    """Test-only explicit cache reset; production transitions wait out the poll TTL."""
    _control_cache.clear()
