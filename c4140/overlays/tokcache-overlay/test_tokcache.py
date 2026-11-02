import asyncio, json, os, random, statistics, sys, time, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TOKCACHE_GUARD', '256')
import tokcache_overlay as T
from transformers import AutoTokenizer
M = '/data/models/Qwen3.8-27B-QUASAR-NVFP4'
tk = AutoTokenizer.from_pretrained(M)
d = json.load(open('/data/bench/explore/out-d2-padtune-serve-v1-20261031/A1/c14/load/prompt-0.json'))
corpus = d['variant1']['messages'][0]['content']
KW = dict(add_special_tokens=False, truncation=True, max_length=262145)  # as params.get_encode_kwargs (params.py:299+), max_input_tokens+1

class FakeAsyncTok:  # same shape as vllm AsyncMicrobatchTokenizer.encode: run tokenizer on the single executor
    def __init__(self):
        self.tokenizer = tk; self._executor = ThreadPoolExecutor(1); self._loop = asyncio.get_running_loop()
    async def encode(self, prompt, **kw):
        return (await self._loop.run_in_executor(self._executor, lambda: self.tokenizer(prompt, **kw)))['input_ids']

def mk(guard):
    T.GUARD = guard
    cache = T.PrefixTokCache(guard=guard)
    return cache, T.make_encode(FakeAsyncTok.encode, cache)

def render(msgs): return tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
rnd = random.Random(7)
def slice_chars(n):
    s = rnd.randrange(0, len(corpus) - n); return corpus[s:s + n]

async def run_seq(name, texts, guard=256):
    cache, enc = mk(guard); ft = FakeAsyncTok(); res = []
    for i, t in enumerate(texts):
        s = time.perf_counter(); plain = await ft.encode(t, **KW); tp = time.perf_counter() - s
        T.STATS.update(hit=0, miss=0, reused_tokens=0)
        s = time.perf_counter(); got = await enc(ft, t, **KW); tc = time.perf_counter() - s
        hit = T.STATS['hit'] == 1
        eq = got == plain
        first = None if eq else next((j for j, (a, b) in enumerate(zip(got, plain)) if a != b), min(len(got), len(plain)))
        res.append(dict(i=i, chars=len(t), tokens=len(plain), plain_s=round(tp, 4), cached_s=round(tc, 4), hit=hit, reused_tokens=T.STATS['reused_tokens'], equal=eq, first_mismatch=first))
    return res

def summarize(res):
    hits = [r for r in res if r['hit']]
    return dict(n=len(res), hits=len(hits), all_equal=all(r['equal'] for r in res), mismatches=[r for r in res if not r['equal']],
                median_plain_s=statistics.median(r['plain_s'] for r in (hits or res)), median_cached_s=statistics.median(r['cached_s'] for r in (hits or res)))

def chat_texts():
    msgs = [{'role': 'user', 'content': corpus}]; out = [render(msgs)]
    for k in range(20):
        n = rnd.randint(200, 800) * 3
        ans = slice_chars(n)
        if k % 5 == 4: ans = '<think>\nlet me think about ' + slice_chars(300) + '\n</think>\n\n' + ans
        msgs += [{'role': 'assistant', 'content': ans}, {'role': 'user', 'content': f'Follow-up {k}: ' + slice_chars(rnd.randint(60, 400)).replace('\n', ' ')}]
        out.append(render(msgs))
    return out

def variant_texts():  # driver shape: fixed corpus, only the tail changes (cohort_v2.py:22-26)
    OLD = 'Write a new well-documented Python module unrelated to the files above. Output only code.'
    base = corpus[:-len(OLD)] if corpus.endswith(OLD) else corpus
    return [render([{'role': 'user', 'content': base + f'Now write a new, original, well-documented Python module (variant {u}-{g}) that is unrelated to the files above. Output only code.'}]) for u, g in [(0, 1)] + [(0, k) for k in range(2, 22)]]

def midword_texts():
    # entry text cut mid-word, then extended; then cut mid-line / mid-number / mid-CJK
    out = []
    for cut in [400001, 500003, 600007, 700013]:
        c = corpus[:cut]
        out += [render([{'role': 'user', 'content': c}]), render([{'role': 'user', 'content': corpus[:cut + 5000]}])]
    return out

def cold_texts():
    lines = corpus.split('\n'); rnd.shuffle(lines); other = '\n'.join(lines)
    return [render([{'role': 'user', 'content': corpus}]), render([{'role': 'user', 'content': other}]), render([{'role': 'user', 'content': 'x' * 30000 + 'é世界 ' * 5000}])]

async def fuzz(guard, trials=300):
    cache, enc = mk(guard); ft = FakeAsyncTok(); bad = []; hits = 0
    pool = corpus[:150000] + ' é世界🙂 ' * 50 + '\r\n\t  \n\n   x́\ńy ' * 50 + corpus[150000:300000]
    for t_ in range(trials):
        a = pool[:rnd.randrange(30000, len(pool))]
        b = a[:rnd.randrange(len(a) // 2, len(a))] + pool[rnd.randrange(0, len(pool) - 3000):][:rnd.randrange(1, 3000)]
        for txt in (a, b):
            T.STATS['hit'] = 0
            got = await enc(ft, txt, **KW); plain = await ft.encode(txt, **KW)
            hits += T.STATS['hit']
            if got != plain: bad.append(dict(trial=t_, len=len(txt), first=next((j for j, (x, y) in enumerate(zip(got, plain)) if x != y), None)))
    return dict(guard=guard, trials=trials * 2, hits=hits, mismatches=bad[:10], n_mismatch=len(bad))

async def main():
    out = {'tokenizer': type(tk).__name__, 'kw': KW, 'guard': T.GUARD, 'overlay_sha8': hashlib.sha256(open(T.__file__, 'rb').read()).hexdigest()[:8]}
    out['probe_no_special_added'] = tk('hello world', add_special_tokens=True)['input_ids'] == tk('hello world', add_special_tokens=False)['input_ids']
    out['chat20'] = {'detail': (r := await run_seq('chat', chat_texts()))}; out['chat20'].update(summarize(r))
    out['variant21'] = {'detail': (r := await run_seq('var', variant_texts()))}; out['variant21'].update(summarize(r))
    out['midword'] = {'detail': (r := await run_seq('mid', midword_texts()))}; out['midword'].update(summarize(r))
    out['cold'] = {'detail': (r := await run_seq('cold', cold_texts()))}; out['cold'].update(summarize(r))
    out['chat20_guard0'] = summarize(await run_seq('chat0', chat_texts(), guard=0))
    out['fuzz_guard0'] = await fuzz(0); out['fuzz_guard256'] = await fuzz(256)
    out['stats'] = dict(T.STATS)
    json.dump(out, open('REPORT.json', 'w'), indent=1)
    for k in ('chat20', 'variant21', 'midword', 'cold', 'chat20_guard0'):
        v = out[k]; print(k, {x: v[x] for x in ('n', 'hits', 'all_equal', 'median_plain_s', 'median_cached_s')})
    print({k: out[k] for k in ('fuzz_guard0', 'fuzz_guard256')}); print(out['probe_no_special_added'])
if __name__ == '__main__': asyncio.run(main())
