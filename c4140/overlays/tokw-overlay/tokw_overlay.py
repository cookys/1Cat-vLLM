"""Overlay: non-blocking _batch_encode_loop for vllm.utils.async_utils.AsyncMicrobatchTokenizer (stage file untouched).
Original loop awaits run_in_executor inline, so one loop task = at most one executor call in flight per kwargs-key.
Patched loop submits the executor call, hands completion to a separate task, and goes straight back to queue.get().
Batching semantics (2 ms window, identical kwargs -> one batched call, max_batch_size) are unchanged.  TOKW_OVERLAY=0 disables."""
import asyncio, importlib.abc, importlib.machinery, os, sys
from functools import partial

def install(cls):
    if getattr(cls, '_tokw_installed', False): return
    try:
        from transformers import BatchEncoding
    except Exception:
        from transformers.tokenization_utils_base import BatchEncoding

    async def _batch_encode_loop(self, queue, can_batch):
        pending = self.__dict__.setdefault('_tokw_pending', set())
        while True:
            prompt, kwargs, result_future = await queue.get()
            prompts = [prompt]; kwargs_list = [kwargs]; result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s
            while len(prompts) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0: break
                try:
                    prompt, kwargs, result_future = await asyncio.wait_for(queue.get(), timeout)
                    prompts.append(prompt); result_futures.append(result_future)
                    if not can_batch: kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break
            try:
                if can_batch and len(prompts) > 1:
                    efut = self._loop.run_in_executor(self._executor, partial(self.tokenizer, prompts, **kwargs))
                    batched = True
                else:
                    fn = lambda prompts=prompts, kwargs=kwargs_list: [self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs)]
                    efut = self._loop.run_in_executor(self._executor, fn)
                    batched = False
            except Exception as e:
                for f in result_futures:
                    if not f.done(): f.set_exception(e)
                continue
            async def _finish(efut=efut, result_futures=result_futures, batched=batched):
                try:
                    results = await efut
                    if batched:
                        for i, f in enumerate(result_futures):
                            if not f.done(): f.set_result(BatchEncoding({k: v[i] for k, v in results.items()}))
                    else:
                        for f, r in zip(result_futures, results):
                            if not f.done(): f.set_result(r)
                except Exception as e:
                    for f in result_futures:
                        if not f.done(): f.set_exception(e)
            t = self._loop.create_task(_finish()); pending.add(t); t.add_done_callback(pending.discard)

    cls._batch_encode_loop = _batch_encode_loop; cls._tokw_installed = True
    sys.stderr.write('[tokw-overlay] installed pid=%d\n' % os.getpid())

def install_import_hook(target='vllm.utils.async_utils'):
    """Order-independent: patch SourceFileLoader.exec_module at CLASS level, so it still fires when another overlay's meta_path
    finder (which resolves the spec itself via PathFinder and never consults later finders) sits at meta_path[0]. v1 used a meta_path
    finder and was silently skipped when the tokcache overlay's finder was inserted after it."""
    if target in sys.modules and hasattr(sys.modules[target], 'AsyncMicrobatchTokenizer'):
        install(sys.modules[target].AsyncMicrobatchTokenizer); return
    L = importlib.machinery.SourceFileLoader
    if getattr(L.exec_module, '_tokw_wrapped', False): return
    real = L.exec_module
    def exec_module(self, module, _real=real):
        _real(self, module)
        if getattr(module, '__name__', None) == target and hasattr(module, 'AsyncMicrobatchTokenizer'):
            install(module.AsyncMicrobatchTokenizer)
    exec_module._tokw_wrapped = True
    L.exec_module = exec_module
