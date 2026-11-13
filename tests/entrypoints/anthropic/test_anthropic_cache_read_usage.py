# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Anthropic /v1/messages usage.cache_read_input_tokens."""

import asyncio
from types import SimpleNamespace

from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    StreamOptions,
    UsageInfo,
)
from vllm.outputs import CompletionOutput, RequestOutput


def _make_handler(*, enable_prompt_tokens_details: bool) -> AnthropicServingMessages:
    handler = object.__new__(AnthropicServingMessages)
    handler.stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    handler.enable_prompt_tokens_details = enable_prompt_tokens_details
    handler.enable_force_include_usage = False
    handler.use_harmony = False
    handler.tool_call_id_type = "random"
    handler.parser_cls = None
    handler.tool_parser = None
    handler.reasoning_parser_cls = None
    handler.enable_auto_tools = False
    handler.exclude_tools_when_tool_choice_none = False
    handler.enable_log_outputs = False
    handler.enable_log_deltas = False
    handler.request_logger = None
    handler.system_fingerprint = None
    handler.response_role = "assistant"
    handler.default_sampling_params = {}
    handler._should_stream_with_auto_tool_parsing = lambda _req: False
    handler._should_check_for_unstreamed_tool_arg_tokens = lambda *_a: False
    handler.get_chat_request_role = lambda _req: "assistant"
    return handler


def _engine_outputs(num_cached_tokens: int | None) -> list[RequestOutput]:
    prompt_ids = list(range(12000))
    step1 = RequestOutput(
        request_id="req-cache-test",
        prompt="hello",
        prompt_token_ids=prompt_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="Hel",
                token_ids=[101],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=None,
            )
        ],
        finished=False,
        num_cached_tokens=num_cached_tokens,
    )
    step2 = RequestOutput(
        request_id="req-cache-test",
        prompt="hello",
        prompt_token_ids=prompt_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="lo!",
                token_ids=[102],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        num_cached_tokens=num_cached_tokens,
    )
    final_full = RequestOutput(
        request_id="req-cache-test",
        prompt="hello",
        prompt_token_ids=prompt_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="Hello!",
                token_ids=[101, 102],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        num_cached_tokens=num_cached_tokens,
    )
    return [step1, step2, final_full]


def _parse_sse(events: list[str]) -> dict[str, dict]:
    import json

    out: dict[str, dict] = {}
    for ev in events:
        lines = [ln for ln in ev.strip().splitlines() if ln]
        etype = next(ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("event:"))
        data = next(ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("data:"))
        out[etype] = json.loads(data)
    return out


def test_non_streaming_cache_read_input_tokens():
    handler = _make_handler(enable_prompt_tokens_details=True)
    _, _, final_ro = _engine_outputs(num_cached_tokens=8192)

    async def _gen():
        yield final_ro

    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    meta = RequestResponseMetadata(request_id="req-cache-test")
    chat_resp = asyncio.run(
        handler.chat_completion_full_generator(
            request=req,
            result_generator=_gen(),
            request_id="req-cache-test",
            model_name="test-model",
            conversation=[{"role": "user", "content": "hi"}],
            tokenizer=SimpleNamespace(),
            request_metadata=meta,
        )
    )
    assert isinstance(chat_resp, ChatCompletionResponse)
    anth_resp = handler.messages_full_converter(chat_resp)
    assert anth_resp.usage.cache_read_input_tokens == 8192
    assert anth_resp.model_dump(exclude_none=True)["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 2,
        "cache_read_input_tokens": 8192,
    }


def test_non_streaming_without_details_preserves_wire_shape():
    handler = _make_handler(enable_prompt_tokens_details=False)
    chat_resp = ChatCompletionResponse(
        id="chatcmpl-no-details",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=12000,
            completion_tokens=2,
            total_tokens=12002,
            prompt_tokens_details=None,
        ),
    )
    anth_resp = handler.messages_full_converter(chat_resp)
    assert anth_resp.usage.cache_read_input_tokens is None
    assert anth_resp.model_dump(exclude_none=True)["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 2,
    }


def test_streaming_message_start_and_delta_match_engine_cached_tokens():
    handler = _make_handler(enable_prompt_tokens_details=True)
    step1, step2, _ = _engine_outputs(num_cached_tokens=8192)

    async def _run():
        async def _engine_gen():
            yield step1
            yield step2

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
        )
        meta = RequestResponseMetadata(request_id="req-cache-test")
        openai_sse = handler.chat_completion_stream_generator(
            request=req,
            result_generator=_engine_gen(),
            request_id="req-cache-test",
            model_name="test-model",
            conversation=[{"role": "user", "content": "hi"}],
            tokenizer=SimpleNamespace(),
            request_metadata=meta,
        )
        out = []
        async for ev in handler.message_stream_converter(openai_sse):
            out.append(ev)
        return out

    events = _parse_sse(asyncio.run(_run()))
    assert events["message_start"]["message"]["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 0,
        "cache_read_input_tokens": 8192,
    }
    assert events["message_delta"]["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 2,
        "cache_read_input_tokens": 8192,
    }


def test_streaming_without_details_omits_cache_read_field():
    handler = _make_handler(enable_prompt_tokens_details=False)
    step1, step2, _ = _engine_outputs(num_cached_tokens=8192)

    async def _run():
        async def _engine_gen():
            yield step1
            yield step2

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
        )
        meta = RequestResponseMetadata(request_id="req-cache-test")
        openai_sse = handler.chat_completion_stream_generator(
            request=req,
            result_generator=_engine_gen(),
            request_id="req-cache-test",
            model_name="test-model",
            conversation=[{"role": "user", "content": "hi"}],
            tokenizer=SimpleNamespace(),
            request_metadata=meta,
        )
        out = []
        async for ev in handler.message_stream_converter(openai_sse):
            out.append(ev)
        return out

    events = _parse_sse(asyncio.run(_run()))
    assert events["message_start"]["message"]["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 0,
    }
    assert events["message_delta"]["usage"] == {
        "input_tokens": 12000,
        "output_tokens": 2,
    }
    assert PromptTokenUsageInfo(cached_tokens=None).cached_tokens is None
    assert ChatCompletionStreamResponse is not None
    assert ChatCompletionResponseStreamChoice is not None
    assert DeltaMessage is not None
