"""Host-only evidence tied to actual graph object; no tensor reads or CUDA calls."""
import json
import logging
import os
import time
LOG = logging.getLogger(__name__)
_active = None
_graphs = {}
_last = {}
NATIVES = {'S1': 'fe35ca278fa28dec4d82fc1878801dd23fddffed9bd531446dc3bf311d79b58b', 'A1B': 'c3cf94613d8a9d0ec27fb46d88534c70ac04f13108e2ad7ad023fa9ba0069e31'}
BATCHES = (8, 10, 12, 20)

def expected():
    return 'A1B' if os.getenv('FLASH_V100_VERIFY_ARM') == 'A1' else 'S1'


def enabled():
    return os.getenv('VLLM_TEAM_MIXED') == '1'


def capture_begin(graph):
    global _active
    if enabled():
        if _active is not None:
            raise RuntimeError('team-route-nested-capture')
        _active = (graph, [])


def capture_end(graph):
    global _active
    if enabled():
        if _active is None or _active[0] is not graph:
            raise RuntimeError('team-route-capture-object')
        _graphs[id(graph)] = (graph, tuple(_active[1]))
        _active = None


def emit(row, *, force=False):
    key = (row['phase'], row.get('graph_id'), row['B'], row['selected'])
    now = time.monotonic()
    if force or now - _last.get(key, -1e9) >= 2:
        _last[key] = now
        LOG.warning('TEAM_ROUTE %s', json.dumps(dict(row, mono=now, pid=os.getpid(), native_sha256=NATIVES.get(row['selected'])), sort_keys=True))


def selection(B, selected, capture):
    if not enabled():
        return
    if _active is not None:
        _active[1].append((B, selected))
    elif not capture and B in BATCHES:
        emit(dict(phase='eager', B=B, graph_B=B, graph_id=None, selected=selected), force=selected != expected())


def replay(graph, live_B, graph_B):
    if not enabled() or live_B not in BATCHES:
        return
    item = _graphs.get(id(graph))
    calls = item[1] if item and item[0] is graph else ()
    good = graph_B == live_B and len(calls) == 16 and all(b == live_B and s == expected() for b, s in calls)
    emit(dict(phase='replay', B=live_B, graph_B=graph_B, graph_id=id(graph),
              captured_calls=len(calls), selected=expected() if good else 'FIX'), force=not good)
