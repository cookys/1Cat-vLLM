"""RENEW-TOKCACHE overlay v3 (v2 defect: when the longest common prefix was shorter than GUARD, safe_boundary got a negative limit, str.rfind counted it from the END of the text and the splice reused another prompt's tokens; verify_fail was never incremented). Wraps AsyncMicrobatchTokenizer.encode (vllm/utils/async_utils.py:62; called from renderers/base.py:420) without editing the stage.
Cache: rendered prompt text -> (token ids, token END char offsets). A new text reuses the tokens of the longest COMMON prefix of a cached entry, cut back to a SAFE
boundary p <= common_len - GUARD, and tokenizes only text[p:]. Safe boundary: p = start of a `<|im_start|>` (special token, exact split) OR text[p-1]=='\\n' and text[p:] is
non-newline blanks then a non-space char (Qwen pretokenize regex: `\\s*[\\r\\n]+` ends at the last newline of a whitespace run), AND p coincides with a stored token end.
Offsets: fast HF tokenizers via return_offsets_mapping; if that is unavailable (e.g. fastokens shim) a byte-length fallback is used for byte-level BPE (token byte lengths
from the vocab, validated sum == len(utf8(text)), else the request is skipped). Anything unusual -> plain path (counted as skip).
Thread safety: one lock guards the LRU and counters (renderer_num_workers > 1 runs encode on several executor threads); entries are immutable once inserted.
Receipt: $VLLM_SM70_POOL_RECEIPT_DIR/tokcache-hits-<pid>.json (atexit + at every power-of-two request count). Install logs one line with the overlay sha8.
v3 guards: plan() returns None unless bl > GUARD; safe_boundary refuses limit <= 0; encode() splices only when p <= bl - GUARD and e.text[:p] == text[:p] (else skip:invariant, plain path); every TOKCACHE_VERIFY_EVERY-th hit (default 8) is re-tokenized in full on the executor after the spliced ids are returned, verify_checked/verify_fail count it, a mismatch writes one VERIFY-FAIL stderr line.
Env: TOKCACHE_ENTRIES(32) TOKCACHE_MAX_TOKENS(8e6) TOKCACHE_GUARD(256) TOKCACHE_MIN_TEXT(20000) TOKCACHE_VERIFY_EVERY(8) TOKCACHE=0 disables TOKCACHE_FORCE_FALLBACK=1 forces byte fallback."""
import atexit, bisect, hashlib, importlib.abc, importlib.machinery, importlib.util, json, os, re, sys, threading
from array import array
from collections import OrderedDict

ENTRIES = int(os.environ.get('TOKCACHE_ENTRIES', '32'))
MAX_TOKENS = int(float(os.environ.get('TOKCACHE_MAX_TOKENS', '8000000')))
GUARD = int(os.environ.get('TOKCACHE_GUARD', '256'))
MIN_TEXT = int(os.environ.get('TOKCACHE_MIN_TEXT', '20000'))
VERIFY_EVERY = int(os.environ.get('TOKCACHE_VERIFY_EVERY', '8'))
RECEIPT_ENV = 'VLLM_SM70_POOL_RECEIPT_DIR'
MARK = '<|im_start|>'
_OK = re.compile(r'[^\S\r\n]*\S')
SHA8 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:8]
_LOCK = threading.Lock()
STATS = {'requests': 0, 'hit': 0, 'miss': 0, 'skip': 0, 'skip_reasons': {}, 'reused_tokens': 0, 'new_tokens': 0, 'fallback_offsets': 0, 'verify_checked': 0, 'verify_fail': 0}
_BYTE_TABLES = {}


def _skip(reason):
    with _LOCK:
        STATS['skip'] += 1; STATS['skip_reasons'][reason] = STATS['skip_reasons'].get(reason, 0) + 1


class _Entry:
    __slots__ = ('text', 'ids', 'ends')
    def __init__(self, text, ids, ends): self.text, self.ids, self.ends = text, ids, ends
    def nbytes(self): return len(self.text) + 4 * len(self.ids) + 4 * len(self.ends)


class PrefixTokCache:
    def __init__(self, entries=ENTRIES, max_tokens=MAX_TOKENS, guard=GUARD):
        self.entries, self.max_tokens, self.guard = entries, max_tokens, guard
        self.lru = OrderedDict(); self.tokens = 0; self.peak_bytes = 0; self.lock = _LOCK

    @staticmethod
    def common_len(a, b):
        n = min(len(a), len(b))
        if (a.startswith(b[:n]) if len(a) >= len(b) else b.startswith(a)): return n
        lo, hi = 0, n
        while hi - lo > 4096:
            mid = (lo + hi) // 2
            if a[lo:mid] == b[lo:mid]: lo = mid
            else: hi = mid
        while lo < hi and a[lo] == b[lo]: lo += 1
        return lo

    @staticmethod
    def safe_boundary(text, limit):
        if limit <= 0: return 0          # a negative end would make rfind count from the END of the text (v2 defect)
        n = len(text); best = 0
        j = text.rfind(MARK, 0, limit + len(MARK))
        if 0 < j <= limit: best = j
        i = text.rfind('\n', 0, limit)
        while i >= 0 and i + 1 > best:
            q = i + 1
            if q < n and _OK.match(text, q): best = max(best, q); break
            i = text.rfind('\n', 0, i)
        return best

    def nbytes(self): return sum(e.nbytes() for e in self.lru.values())

    def plan(self, text):
        with self.lock:
            entries = list(self.lru.values())
        best, bl = None, 0
        for e in entries:  # compares run outside the lock; entries are immutable
            c = self.common_len(e.text, text)
            if c > bl: best, bl = e, c
        if best is None or bl <= self.guard: return None
        p = self.safe_boundary(text, bl - self.guard)
        if p <= 0: return None
        k = bisect.bisect_right(best.ends, p)
        if k == 0 or best.ends[k - 1] != p: return None
        with self.lock:
            if id(best) in self.lru: self.lru.move_to_end(id(best))
        return best, p, k

    def insert(self, text, ids, ends):
        e = _Entry(text, ids, ends)
        with self.lock:
            self.lru[id(e)] = e; self.tokens += len(ids)
            while len(self.lru) > self.entries or self.tokens > self.max_tokens:
                _, old = self.lru.popitem(last=False); self.tokens -= len(old.ids)
            self.peak_bytes = max(self.peak_bytes, self.nbytes())
        return e


def _bytes_to_unicode():  # GPT-2 byte-level alphabet (same table as transformers' removed import path)
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def _byte_table(tok):
    """per-id UTF-8 byte length for byte-level BPE (fallback offsets). None if the vocab is not byte-level."""
    key = id(tok)
    if key in _BYTE_TABLES: return _BYTE_TABLES[key]
    import numpy as np
    try:
        dec = {v: k for k, v in _bytes_to_unicode().items()}
        vocab = tok.get_vocab(); size = max(max(vocab.values()) + 1, len(tok))
        lens = np.zeros(size, dtype=np.int64)
        added = {}
        try: added = {t.content: i for i, t in tok.added_tokens_decoder.items()}
        except Exception: pass
        for s, i in vocab.items():
            if s in added: lens[i] = len(s.encode('utf-8'))
            else: lens[i] = len(s)  # every char of a byte-level token string maps to exactly one byte
            if s not in added and not all(c in dec for c in s): _BYTE_TABLES[key] = None; return None
        for s, i in added.items(): lens[i] = len(s.encode('utf-8'))
    except Exception:
        _BYTE_TABLES[key] = None; return None
    _BYTE_TABLES[key] = lens
    return lens


def _ends_fallback(tok, text, ids):
    import numpy as np
    lens = _byte_table(tok)
    if lens is None: return None
    ends_b = np.cumsum(lens[np.asarray(ids, dtype=np.int64)])
    b = text.encode('utf-8')
    if int(ends_b[-1]) != len(b): return None  # normalizer changed the text: cannot trust
    if len(b) != len(text):  # non-ASCII: convert byte offsets to char offsets
        start = (np.frombuffer(b, dtype=np.uint8) & 0xC0) != 0x80
        ends_b = np.concatenate(([0], np.cumsum(start)))[ends_b]
    out = array('i'); out.frombytes(ends_b.astype(np.int32).tobytes())
    return out


def _tokenize(tok, txt, kwargs, force_fallback):
    """-> (ids list, ends array|None)"""
    kw = dict(kwargs); kw.pop('truncation', None); kw.pop('max_length', None)
    if not force_fallback:
        try:
            be = tok(txt, return_offsets_mapping=True, **kw)
            om = be['offset_mapping']
            return list(be['input_ids']), array('i', (b for _, b in om))
        except Exception:
            pass
    ids = list(tok(txt, **kw)['input_ids'])
    with _LOCK: STATS['fallback_offsets'] += 1
    return ids, _ends_fallback(tok, txt, ids)


def dump_receipt(cache=None, path=None):
    d = os.environ.get(RECEIPT_ENV)
    if path is None:
        if not d: return None
        path = os.path.join(d, f'tokcache-hits-{os.getpid()}.json')
    with _LOCK:
        rec = dict(schema='tokcache-hits/2', pid=os.getpid(), overlay_sha8=SHA8, overlay=os.path.abspath(__file__), counters={k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()},
                   config=dict(entries=ENTRIES, max_tokens=MAX_TOKENS, guard=GUARD, min_text=MIN_TEXT, verify_every=VERIFY_EVERY))
    c = cache or _CACHE[0]
    if c is not None:
        rec['cache'] = dict(entries=len(c.lru), tokens=c.tokens, bytes=c.nbytes(), peak_bytes=c.peak_bytes)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp{threading.get_ident()}'
        open(tmp, 'w').write(json.dumps(rec, sort_keys=True) + '\n'); os.replace(tmp, path)
    except OSError:
        return None
    return path


_CACHE = [None]


_VERIFY_TASKS = set()


def _spawn_verify(loop, executor, tok, prompt, kwargs, ff, spliced, req_n):
    """sampled verify, off the request path: full tokenization on the executor vs the spliced ids. Never raises into the request."""
    async def run():
        try:
            full, _ = await loop.run_in_executor(executor, _tokenize, tok, prompt, kwargs, ff)
            div = next((i for i, (a, b) in enumerate(zip(spliced, full)) if a != b), min(len(spliced), len(full)))
            with _LOCK:
                STATS['verify_checked'] += 1
                bad = list(spliced) != list(full)
                if bad: STATS['verify_fail'] += 1
            if bad:
                sys.stderr.write(f'[tokcache-overlay] VERIFY-FAIL request={req_n} spliced={len(spliced)} full={len(full)} first_div={div}\n'); sys.stderr.flush()
        except Exception:
            _skip('verify_error')
    t = loop.create_task(run()); _VERIFY_TASKS.add(t); t.add_done_callback(_VERIFY_TASKS.discard)


def make_encode(orig_encode, cache=None):
    cache = cache or PrefixTokCache(); _CACHE[0] = cache
    async def encode(self, prompt, **kwargs):
        with _LOCK:
            STATS['requests'] += 1; n = STATS['requests']
        if n & (n - 1) == 0: dump_receipt(cache)
        if os.environ.get('TOKCACHE') == '0': _skip('disabled'); return await orig_encode(self, prompt, **kwargs)
        if not isinstance(prompt, str): _skip('non_str'); return await orig_encode(self, prompt, **kwargs)
        if len(prompt) < MIN_TEXT: _skip('short'); return await orig_encode(self, prompt, **kwargs)
        if kwargs.get('add_special_tokens', True) is not False: _skip('add_special_tokens'); return await orig_encode(self, prompt, **kwargs)
        maxlen = kwargs.get('max_length') if kwargs.get('truncation') else None
        ff = os.environ.get('TOKCACHE_FORCE_FALLBACK') == '1'
        loop = self._loop; tok = self.tokenizer
        pl = cache.plan(prompt)
        if pl is None:
            ids, ends = await loop.run_in_executor(self._executor, _tokenize, tok, prompt, kwargs, ff)
            if ends is None: _skip('no_offsets'); return await orig_encode(self, prompt, **kwargs)
            cache.insert(prompt, array('i', ids), ends)
            with _LOCK: STATS['miss'] += 1; STATS['new_tokens'] += len(ids)
        else:
            e, p, k = pl
            bl = cache.common_len(e.text, prompt)
            if not (0 < p <= bl - cache.guard and e.text[:p] == prompt[:p]): _skip('invariant'); return await orig_encode(self, prompt, **kwargs)          # v3: never splice across a prefix that is not really shared
            rid, rends = await loop.run_in_executor(self._executor, _tokenize, tok, prompt[p:], kwargs, ff)
            if rends is None: _skip('no_offsets'); return await orig_encode(self, prompt, **kwargs)
            ids = e.ids[:k].tolist() + rid
            ends = e.ends[:k] + array('i', (b + p for b in rends))
            cache.insert(prompt, array('i', ids), ends)
            with _LOCK:
                STATS['hit'] += 1; STATS['reused_tokens'] += k; STATS['new_tokens'] += len(rid); hn = STATS['hit']
            if VERIFY_EVERY > 0 and hn % VERIFY_EVERY == 0 and not (maxlen is not None and len(ids) > maxlen):
                _spawn_verify(loop, self._executor, tok, prompt, kwargs, ff, ids, n)
        if maxlen is not None and len(ids) > maxlen:
            _skip('truncation'); return await orig_encode(self, prompt, **kwargs)
        return ids
    return encode


def install(cls):
    if getattr(cls, '_tokcache_installed', False): return
    cls.encode = make_encode(cls.encode); cls._tokcache_installed = True
    atexit.register(dump_receipt)
    sys.stderr.write(f'[tokcache-overlay] installed sha8={SHA8} pid={os.getpid()} entries={ENTRIES} max_tokens={MAX_TOKENS} guard={GUARD} min_text={MIN_TEXT} verify_every={VERIFY_EVERY} receipt_dir={os.environ.get(RECEIPT_ENV)}\n'); sys.stderr.flush()


def install_import_hook(target='vllm.utils.async_utils'):
    """wrap AsyncMicrobatchTokenizer right after `target` is imported (stage file untouched); installs immediately if already imported."""
    if target in sys.modules and hasattr(sys.modules[target], 'AsyncMicrobatchTokenizer'):
        install(sys.modules[target].AsyncMicrobatchTokenizer); return
    class F(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, tgt=None):
            if name != target: return None
            sys.meta_path.remove(self)
            try: spec = importlib.machinery.PathFinder.find_spec(name, path)
            finally: pass
            if spec is None or spec.loader is None: return None
            loader = spec.loader; real = loader.exec_module
            def exec_module(mod, _real=real):
                _real(mod); install(mod.AsyncMicrobatchTokenizer)
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, F())
