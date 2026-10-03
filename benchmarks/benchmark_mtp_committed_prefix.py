# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-run HTTP probe: exact input lengths, warm prefixes and MTP guard misses.

Does not start/stop/configure the server. Recording DOES use the serving GPU;
--compare is offline. Use identical --alignment and --seed in the two cells.
TTFT means first streamed token ID (not the empty stream header).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


def post(base, route, body):
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=1800)


def completion(base, body):
    start = time.perf_counter()
    first = None
    ids, logprobs = [], []
    usage, finish = {}, None
    with post(base, "/v1/completions", body) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                new_ids = choice.get("token_ids") or []
                if new_ids and first is None:
                    first = time.perf_counter() - start
                ids.extend(new_ids)
                lp = choice.get("logprobs") or {}
                logprobs.extend(lp.get("token_logprobs") or [])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not ids or len(logprobs) != len(ids):
        raise ValueError("server must return token_ids and logprobs for every token")
    return {
        "token_ids": ids,
        "logprobs": logprobs,
        "ttft_s": first,
        "wall_s": time.perf_counter() - start,
        "usage": usage,
        "finish_reason": finish,
    }


def cases(vocab, alignment, seed):
    """A/B interleaving exercises slot reuse; changed token[B] must miss B."""
    rng = random.Random(seed)
    for multiplier in (2, 10):
        size = multiplier * alignment + 17
        prompts = {key: rng.choices(vocab, k=size) for key in ("a", "b")}
        for key in prompts:
            yield f"m{multiplier}-{key}-producer", key, prompts[key]
        for append in (1024, 2048, 4096):
            for key in prompts:
                prompt = prompts[key] + rng.choices(vocab, k=append)
                yield f"m{multiplier}-{key}-append{append}", key, prompt
        for key in prompts:
            prompt = prompts[key].copy()
            boundary = multiplier * alignment
            prompt[boundary] = next(x for x in vocab if x != prompt[boundary])
            yield f"m{multiplier}-{key}-guard-miss", key, prompt


def compare(a, b):
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    if left["alignment"] != right["alignment"] or left["seed"] != right["seed"]:
        raise ValueError("alignment and seed must match")
    if left["cases"].keys() != right["cases"].keys():
        raise ValueError("case names differ")
    failed = False
    for name, expected in left["cases"].items():
        observed = right["cases"][name]
        if expected["prompt_sha256"] != observed["prompt_sha256"]:
            raise ValueError(f"prompt differs: {name}")
        equal_ids = expected["token_ids"] == observed["token_ids"]
        # API JSON float equality is stricter than the existing four-decimal
        # recorder, but neither is a comparison of full device logits/state.
        equal_lp = expected["logprobs"] == observed["logprobs"]
        failed |= not equal_ids or not equal_lp
        print(f"{name}: numeric_ids_equal={equal_ids} API_logprobs_equal={equal_lp}")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--alignment", type=int, default=1616)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.label or not args.out or args.alignment < 32 or args.max_tokens < 1:
        parser.error("recording needs --label, --out, alignment>=32, max-tokens>=1")
    with post(
        args.base_url,
        "/tokenize",
        {
            "model": args.model,
            "prompt": "def class return import self value index buffer kernel thread "
            "queue lock cache token layer stream graph tensor shard expert route",
            "add_special_tokens": False,
        },
    ) as response:
        vocab = list(dict.fromkeys(json.load(response)["tokens"]))
    if len(vocab) < 2:
        raise ValueError("tokenizer returned fewer than two distinct tokens")
    result = {
        "label": args.label,
        "alignment": args.alignment,
        "seed": args.seed,
        "cases": {},
    }
    run_salt = f"committed-prefix-{args.label}-{time.time_ns()}"
    for sampling in ("greedy", "sampled"):
        for name, family, prompt in cases(vocab, args.alignment, args.seed):
            name = f"{sampling}-{name}"
            body = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0 if sampling == "greedy" else 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "seed": args.seed,
                "logprobs": 1,
                "return_token_ids": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_salt": f"{run_salt}-{sampling}-{family}",
                "request_id": f"snapshot-{args.label}-{name}",
            }
            row = completion(args.base_url, body)
            row["prompt_sha256"] = hashlib.sha256(
                json.dumps(prompt).encode()
            ).hexdigest()
            row["prompt_length"] = len(prompt)
            result["cases"][name] = row
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(name, f"TTFT={row['ttft_s']:.3f}s", row["usage"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
