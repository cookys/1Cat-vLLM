"""Real-hook tests: stage vllm.utils.async_utils + real tokenizer object (get_tokenizer -> CachedHfTokenizer, copy + maybe_make_thread_pool as HfRenderer.__init__ does, hf.py:923-945)."""
import asyncio, copy, glob, hashlib, json, os, random, statistics, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
STAGE = '/data/bench/explore/stage-develop-cs-g20-pool-v3-pf'; LANE = os.path.dirname(os.path.abspath(__file__))  # PACK VENDOR PATCH: was the cpu lane path, so the vendored overlay (not the cpu lane copy) is the one imported
sys.path.insert(0, STAGE); sys.path.insert(1, LANE)
RDIR = tempfile.mkdtemp(prefix='tokcache-rcpt-', dir=os.environ.get('TMPDIR')); os.environ['VLLM_SM70_POOL_RECEIPT_DIR'] = RDIR
import tokcache_overlay as T
T.install_import_hook()
import vllm.utils.async_utils as AU
assert getattr(AU.AsyncMicrobatchTokenizer, '_tokcache_installed', False), 'hook did not install'
from vllm.tokenizers.registry import get_tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
import test_tokcache as TT
M = TT.M; KW = TT.KW
base = get_tokenizer(M); tok = copy.copy(base); maybe_make_thread_pool(tok, 4 + 1)
CLS = {'tokenizer_class': type(tok).__name__, 'is_fast': tok.is_fast, 'mro1': [c.__name__ for c in type(tok).__mro__[:4]]}
TT.tk = tok  # text generators use tok.apply_chat_template (thread pool wrapper)

def fresh(workers=1):
    ex = ThreadPoolExecutor(max_workers=workers)
    a = AU.AsyncMicrobatchTokenizer(tok, executor=ex)
    return a, ex
async def plain(a, text):
    return await orig(a, text, **KW)
orig = AU.AsyncMicrobatchTokenizer.encode.__closure__  # placeholder replaced below
# recover the original encode: closure var orig_encode of the installed wrapper
orig = [c.cell_contents for c in AU.AsyncMicrobatchTokenizer.encode.__closure__ if callable(c.cell_contents) and getattr(c.cell_contents, '__name__', '') == 'encode'][0]

def reset(): 
    T._CACHE[0].lru.clear(); T._CACHE[0].tokens = 0
    with T._LOCK:
        for k in ('requests','hit','miss','skip','reused_tokens','new_tokens','fallback_offsets'): T.STATS[k] = 0
        T.STATS['skip_reasons'] = {}

async def seq(texts):
    reset(); a, ex = fresh(); res = []
    for i, t in enumerate(texts):
        s = time.perf_counter(); pl = await orig(a, t, **KW); tp = time.perf_counter() - s
        h0 = T.STATS['hit']
        s = time.perf_counter(); got = await AU.AsyncMicrobatchTokenizer.encode(a, t, **KW); tc = time.perf_counter() - s
        res.append(dict(i=i, tokens=len(pl), plain_s=round(tp, 4), cached_s=round(tc, 4), hit=T.STATS['hit'] > h0, equal=got == pl,
                        first_mismatch=None if got == pl else next((j for j, (x, y) in enumerate(zip(got, pl)) if x != y), None)))
    ex.shutdown(); return res
def summ(res):
    h = [r for r in res if r['hit']]
    return dict(n=len(res), hits=len(h), all_equal=all(r['equal'] for r in res), mismatches=[r for r in res if not r['equal']][:5],
                median_plain_s=statistics.median(r['plain_s'] for r in (h or res)), median_cached_s=statistics.median(r['cached_s'] for r in (h or res)),
                cold_cached_s=res[0]['cached_s'], cold_plain_s=res[0]['plain_s'])

async def concurrent4(rounds=1):
    """4 workers (renderer_num_workers=4): 4 different conversations renewed concurrently, 20 renewals each; all checked vs plain."""
    reset(); a, ex = fresh(4)
    convs = []
    for c in range(4):
        TT.rnd.seed(100 + c)
        msgs0 = [{'role': 'user', 'content': f'conv {c}\n' + TT.slice_chars(400000)}]
        out = [TT.render(msgs0)]; msgs = list(msgs0)
        for k in range(20):
            msgs += [{'role': 'assistant', 'content': TT.slice_chars(TT.rnd.randint(200, 800) * 3)}, {'role': 'user', 'content': f'f{k} ' + TT.slice_chars(200).replace('\n', ' ')}]
            out.append(TT.render(msgs))
        convs.append(out)
    expect = [[list((await orig(a, t, **KW))) for t in c] for c in convs]
    reset()
    bad = []; times = []
    async def user(c):
        for i, t in enumerate(convs[c]):
            s = time.perf_counter(); got = await AU.AsyncMicrobatchTokenizer.encode(a, t, **KW); times.append(time.perf_counter() - s)
            if got != expect[c][i]: bad.append((c, i, next((j for j, (x, y) in enumerate(zip(got, expect[c][i])) if x != y), None)))
    s = time.perf_counter(); await asyncio.gather(*[user(c) for c in range(4)]); wall = time.perf_counter() - s
    ex.shutdown()
    return dict(conversations=4, renewals_each=21, requests=84, mismatches=bad, hits=T.STATS['hit'], miss=T.STATS['miss'], skip=T.STATS['skip'], wall_s=round(wall, 3), median_req_s=round(statistics.median(times), 4))

def receipt_unit():
    import numpy as np
    d = tempfile.mkdtemp(dir=os.environ.get('TMPDIR')); os.environ['VLLM_SM70_POOL_RECEIPT_DIR'] = d
    p = T.dump_receipt(); r = json.load(open(p))
    ok = (os.path.basename(p) == f'tokcache-hits-{os.getpid()}.json' and r['schema'] == 'tokcache-hits/2' and r['overlay_sha8'] == T.SHA8 and r['pid'] == os.getpid()  # PACK VENDOR PATCH: schema was tokcache-hits/1
          and all(k in r['counters'] for k in ('requests', 'hit', 'miss', 'skip', 'reused_tokens', 'new_tokens')) and 'bytes' in r['cache'] and r['counters']['hit'] == T.STATS['hit'])
    os.environ.pop('VLLM_SM70_POOL_RECEIPT_DIR'); none = T.dump_receipt() is None
    os.environ['VLLM_SM70_POOL_RECEIPT_DIR'] = RDIR
    return dict(path=p, ok=bool(ok), noenv_noop=none, counters=r['counters'], cache=r['cache'])

async def main():
    out = dict(tokenizer=CLS, overlay_sha8=T.SHA8, fastokens_installed=False, fastokens_env=os.environ.get('VLLM_USE_FASTOKENS'))
    try: import fastokens; out['fastokens_installed'] = True
    except ImportError: pass
    TT.rnd.seed(7)
    texts = TT.chat_texts()
    out['real_chat20'] = summ(await seq(texts)); out['real_chat20_stats'] = {k: v for k, v in T.STATS.items()}
    out['real_variant21'] = summ(await seq(TT.variant_texts()))
    os.environ['TOKCACHE_FORCE_FALLBACK'] = '1'
    out['fallback_chat20'] = summ(await seq(texts)); out['fallback_chat20_stats'] = {k: v for k, v in T.STATS.items()}
    out['fallback_cjk'] = summ(await seq([TT.render([{'role': 'user', 'content': ('世界é🙂 ' * 3000 + '\n' + TT.slice_chars(60000))[:90000] + f'\n{k}' + TT.slice_chars(200) }]) for k in range(4)]))
    del os.environ['TOKCACHE_FORCE_FALLBACK']
    out['threads4'] = await concurrent4()
    out['receipt_unit'] = receipt_unit()
    T.dump_receipt(); out['receipt_files'] = {os.path.basename(f): json.load(open(f))['counters'] for f in glob.glob(RDIR + '/tokcache-hits-*.json')}
    json.dump(out, open(os.path.join(os.environ.get('TOKCACHE_REAL_OUT') or tempfile.gettempdir(), 'REAL.json'), 'w'), indent=1)  # PACK VENDOR PATCH: was LANE/REAL.json
    for k in ('real_chat20', 'real_variant21', 'fallback_chat20', 'fallback_cjk'): print(k, {x: out[k][x] for x in ('n', 'hits', 'all_equal', 'median_plain_s', 'median_cached_s', 'cold_plain_s', 'cold_cached_s')})
    print('threads4', out['threads4']); print('receipt', out['receipt_unit']['ok'], out['receipt_unit']['noenv_noop']); print(CLS, out['fastokens_installed'])
asyncio.run(main())
