"""候选工厂：每个固定 kwarg 钉一条断言（登记表逐行有测试）；fake 开关只在工厂生效。

上游以 httpx2.MockTransport 注入 http_async_client 拦截（探针 probe_m12_a），零网络：
请求真的到达我们的 handler，本身就是"注入客户端被使用"的证据。
"""

import json

import httpx2
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import Settings
from app.engine.fakes import FakeReplyChatModel
from app.engine.gateway.candidates import (
    CANDIDATE_TIMEOUT,
    STREAM_CHUNK_TIMEOUT_S,
    build_candidates,
    make_candidate,
)
from app.engine.gateway.routing import Candidate

SSE = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen",'
    '"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
    "data: [DONE]\n\n"
)
TOOL = {
    "type": "function",
    "function": {
        "name": "order_query",
        "description": "查订单",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
        },
    },
}
QWEN_PLUS = Candidate("bailian", "qwen-plus")


@pytest.fixture
def seen() -> list[dict]:
    return []


@pytest.fixture
def client(seen) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=SSE
        )

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


@pytest.fixture
def model(client) -> ChatOpenAI:
    return make_candidate(
        QWEN_PLUS,
        base_url="http://test/v1",
        api_key=SecretStr("sk-test-key"),
        http_async_client=client,
    )


async def drain(model, **kwargs):
    return [c async for c in model.astream([HumanMessage("hi")], **kwargs)]


# ---------------------------------------------------------------- 固定 kwargs 逐行


def test_sdk_retries_are_disabled(model):
    # 全场最重要的一行：双重重试防线只剩纪律保证，由本测试钉住
    assert model.root_async_client.max_retries == 0


async def test_request_body_carries_usage_and_thinking_keys(model, seen, client):
    chunks = await drain(model)
    (req,) = seen
    assert req["url"] == "http://test/v1/chat/completions"
    assert req["body"]["stream"] is True
    assert req["body"]["stream_options"] == {"include_usage": True}
    assert req["body"]["enable_thinking"] is False
    assert "max_completion_tokens" not in req["body"]
    assert "max_tokens" not in req["body"]
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c
    assert full.usage_metadata["total_tokens"] == 10  # usage 真的流回来了


async def test_max_completion_tokens_only_when_caller_passes_max_tokens(model, seen):
    await drain(model, max_tokens=64)
    (req,) = seen
    assert req["body"]["max_completion_tokens"] == 64
    assert "max_tokens" not in req["body"]


def test_cache_off_and_timeouts_passed_through(model):
    assert model.cache is False
    assert model.root_async_client.timeout == CANDIDATE_TIMEOUT
    assert model.stream_chunk_timeout == STREAM_CHUNK_TIMEOUT_S
    assert STREAM_CHUNK_TIMEOUT_S >= 25.0  # 不得小于 M1.3 的首块窗口，否则抢首块窗


async def test_tools_forwarded_via_astream_kwargs(model, seen):
    # 网关 _astream 的转发形态（ADR-005）：不 bind_tools，直接 kwargs 透传给候选
    await drain(model, tools=[TOOL], tool_choice="auto")
    (req,) = seen
    assert req["body"]["tools"] == [TOOL]
    assert req["body"]["tool_choice"] == "auto"


async def test_two_candidates_share_the_injected_client(client, seen):
    a = make_candidate(
        Candidate("bailian", "qwen-flash"),
        base_url="http://test/v1",
        api_key=SecretStr("sk-test-key"),
        http_async_client=client,
    )
    b = make_candidate(
        QWEN_PLUS,
        base_url="http://test/v1",
        api_key=SecretStr("sk-test-key"),
        http_async_client=client,
    )
    await drain(a)
    await drain(b)
    assert [r["body"]["model"] for r in seen] == ["qwen-flash", "qwen-plus"]


def test_secret_never_appears_in_model_repr(model):
    assert "sk-test-key" not in repr(model)


def test_empty_api_key_fails_loud_at_build_time(client):
    with pytest.raises(ValueError, match="API key 未配置"):
        make_candidate(
            QWEN_PLUS,
            base_url="http://test/v1",
            api_key=SecretStr(""),
            http_async_client=client,
        )


# ---------------------------------------------------------------- fake 开关


def test_fake_switch_replaces_every_candidate_without_touching_secrets(client):
    settings = Settings(_env_file=None, aegis_fake_llm=True, aegis_fake_llm_delay_s=0.3)
    table = build_candidates(
        settings,
        [QWEN_PLUS, Candidate("bailian", "qwen-flash")],
        http_async_client=client,
    )
    assert set(table) == {QWEN_PLUS, Candidate("bailian", "qwen-flash")}
    for m in table.values():
        assert isinstance(m, FakeReplyChatModel)
        assert m.delay_s == 0.3


def test_real_switch_builds_chat_openai_per_candidate(client):
    settings = Settings(
        _env_file=None,
        aegis_fake_llm=False,
        dashscope_api_key="sk-test-key",
        providers={"bailian": "http://test/v1"},
    )
    table = build_candidates(
        settings,
        [QWEN_PLUS, Candidate("bailian", "qwen-flash")],
        http_async_client=client,
    )
    assert {c.model: type(m) for c, m in table.items()} == {
        "qwen-plus": ChatOpenAI,
        "qwen-flash": ChatOpenAI,
    }
    assert table[QWEN_PLUS].model_name == "qwen-plus"
    assert table[QWEN_PLUS].openai_api_base == "http://test/v1"
