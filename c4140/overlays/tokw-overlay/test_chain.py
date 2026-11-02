import asyncio, copy, glob, json, os, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/data/bench/explore/stage-develop-cs-g20-pool-v3-pf')
import vllm.utils.async_utils as AU
from vllm.tokenizers.registry import get_tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
T = sys.modules.get('tokcache_overlay')
M = '/data/models/Qwen3.8-27B-QUASAR-NVFP4'
KW = dict(add_special_tokens=False, truncation=True, max_length=262145)
tok = copy.copy(get_tokenizer(M)); maybe_make_thread_pool(tok, 5)
corpus = json.load(open('/data/bench/explore/out-d2-padtune-serve-v1-20261031/A1/c14/load/prompt-0.json'))['variant1']['messages'][0]['content']
rnd = random.Random(11)
def sl(n):
    s = rnd.randrange(0, len(corpus) - n); return corpus[s:s + n]
def render(m): return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
def reset():
    T._CACHE[0].lru.clear(); T._CACHE[0].tokens = 0
    with T._LOCK:
        for k in ('requests','hit','miss','skip','reused_tokens','new_tokens','fallback_offsets'): T.STATS[k] = 0
        T.STATS['skip_reasons'] = {}
async def main():
    out = dict(order=os.environ.get('ORDER'), tokw_installed=getattr(AU.AsyncMicrobatchTokenizer, '_tokw_installed', False),
               tokcache_installed=getattr(AU.AsyncMicrobatchTokenizer, '_tokcache_installed', False), tokcache_sha8=T.SHA8 if T else None)
    # (a) 4 staggered distinct prompts, 4 workers, 3 reps
    texts = [f'distinct {i} {rnd.random()}\n' + sl(800000) for i in range(4)]
    reps = []
    for _ in range(3):
        reset(); a = AU.AsyncMicrobatchTokenizer(tok, executor=ThreadPoolExecutor(4)); lat = [0]*4
        async def one(i):
            await asyncio.sleep(i * 0.1); s = time.perf_counter(); await a.encode(texts[i], **KW); lat[i] = time.perf_counter() - s
        t0 = time.perf_counter(); await asyncio.gather(*[one(i) for i in range(4)]); w = time.perf_counter() - t0
        reps.append(dict(wall=round(w, 3), lat=[round(x, 3) for x in lat], hit=T.STATS['hit'], miss=T.STATS['miss'], skip=T.STATS['skip']))
    out['a_staggered'] = dict(reps=reps, median_wall=statistics.median(r['wall'] for r in reps))
    # (b) 4 conversations x 20 renewals concurrently
    convs = []
    for c in range(4):
        msgs = [{'role': 'user', 'content': f'conv {c}\n' + sl(400000)}]; o = [render(msgs)]
        for k in range(20):
            msgs += [{'role': 'assistant', 'content': sl(rnd.randint(200, 800) * 3)}, {'role': 'user', 'content': f'f{k} ' + sl(200).replace('\n', ' ')}]
            o.append(render(msgs))
        convs.append(o)
    expect = [[tok(t, **KW)['input_ids'] for t in c] for c in convs]
    reset(); a = AU.AsyncMicrobatchTokenizer(tok, executor=ThreadPoolExecutor(4)); bad = []; tm = []
    async def user(c):
        for i, t in enumerate(convs[c]):
            h0 = T.STATS['hit']; s = time.perf_counter(); got = await a.encode(t, **KW); d = time.perf_counter() - s
            tm.append((c, i, d)); 
            if got != expect[c][i]: bad.append((c, i))
    t0 = time.perf_counter(); await asyncio.gather(*[user(c) for c in range(4)]); wall = time.perf_counter() - t0
    st = {k: T.STATS[k] for k in ('requests','hit','miss','skip','reused_tokens','new_tokens')}
    warm = [d for c, i, d in tm if i > 0]
    out['b_conversations'] = dict(requests=len(tm), equal=len(tm) - len(bad), mismatches=bad, wall=round(wall, 3), stats=st,
        median_first_req_s=round(statistics.median(d for c, i, d in tm if i == 0), 3), median_renewal_req_s=round(statistics.median(warm), 4), p90_renewal_req_s=round(sorted(warm)[int(len(warm)*0.9)], 4))
    p = T.dump_receipt(); out['receipt'] = dict(path=p, counters=json.load(open(p))['counters'] if p else None, matches_stats=bool(p) and all(json.load(open(p))['counters'][k] == st[k] for k in st))
    print('RESULT ' + json.dumps(out))
asyncio.run(main())
