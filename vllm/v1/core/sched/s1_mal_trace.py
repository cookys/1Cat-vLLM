"""CPU-only observation of existing scheduler outputs; never touches a tensor."""
import json
import os
import re
import time
from pathlib import Path

_TARGET = re.compile(r'^chatcmpl-s1mal-([a-zA-Z0-9]+)-g(\d+)-u(\d+)(?:-[a-f0-9]{8})?$')
_writer = None
_counter = 0
_failed = False
_previous_us = 0.0


def snapshot(requests, scheduled, output, *, mono=None, wall=None, step=0):
    rows = []
    for rid in scheduled.num_scheduled_tokens:
        match = _TARGET.fullmatch(rid)
        if not match:
            continue
        req = requests.get(rid)
        if req is None or req.is_finished():
            continue
        index = output.req_id_to_index[rid]
        ids = output.sampled_token_ids[index] if output.sampled_token_ids else []
        drafts = scheduled.scheduled_spec_decode_tokens.get(rid) or []
        n = len(ids)
        rows.append(dict(request_id=rid, run=match[1], generation=int(match[2]),
                         user=int(match[3]), output_tokens_before=len(req._output_token_ids),
                         generated=n, draft=len(drafts), accepted=n-1 if drafts and n else None,
                         prompt_tokens=req.num_prompt_tokens, total_ctx_before=req.num_prompt_tokens+len(req._output_token_ids),
                         cached_tokens=(req.prefill_stats.num_cached_tokens if req.prefill_stats is not None and req.prefill_stats.num_prompt_tokens > 0 else None)))
    if not rows:
        return None
    return dict(schema=1, pid=os.getpid(), step=step, mono=time.monotonic() if mono is None else mono,
                wall=time.time() if wall is None else wall,
                scheduled_requests=len(scheduled.num_scheduled_tokens), rows=rows)


def trace_step(requests, scheduled, output):
    global _writer, _counter, _failed, _previous_us
    dest = os.getenv('VLLM_S1_MAL_TRACE_DIR', '')
    if not dest or _failed:
        return
    try:
        started = time.monotonic()
        _counter += 1
        row = snapshot(requests, scheduled, output, step=_counter)
        if row is None:
            return
        if _writer is None:
            Path(dest).mkdir(parents=True, exist_ok=True)
            _writer = (Path(dest)/f'steps-{os.getpid()}.jsonl').open('x', buffering=1)
        row['previous_observe_us'] = _previous_us
        _writer.write(json.dumps(row, separators=(',', ':'))+'\n')
        _previous_us = (time.monotonic()-started)*1e6
    except Exception as exc:
        # Telemetry must not alter the algorithm. The harness treats any error
        # or missing/invalid stream as undetermined, never fills missing counts.
        _failed = True
        try:
            (Path(dest)/f'ERROR-{os.getpid()}.json').write_text(json.dumps(dict(error=repr(exc))))
        except OSError:
            pass
