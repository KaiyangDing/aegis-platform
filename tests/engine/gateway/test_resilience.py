"""单候选韧性：v1 resilience 12 条平移 + v2 新增（翻译落点、首块判据、截断/零块流、块间空闲、出站闸缝、401 端到端打码）。

候选替身 ScriptedCandidate（tests/engine/gateway/doubles.py）走真实的 BaseChatModel.astream 外壳
（回调、末块补齐、零块 ValueError 都是框架行为）；端到端条目用 ChatOpenAI + httpx2.MockTransport 真走 SDK
（探针 probe_m13_*），零网络。时序接缝（_sleep/_uniform/_monotonic）全部替换：除首块窗口的几条 ≤0.2s 真等外，测试不真睡。
"""

import asyncio
import dataclasses
import json
import time
import traceback
from email.utils import formatdate

import httpx2
import openai
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI, StreamChunkTimeoutError
from structlog.testing import capture_logs

from app.engine.gateway import resilience
from app.engine.gateway.candidates import STREAM_CHUNK_TIMEOUT_S
from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    GatewayOverloadedError,
    OutboundGateTimeout,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
)
from app.engine.gateway.resilience import (
    RetryPolicy,
    complete_with_retry,
    compute_backoff,
)
from tests.engine.gateway.doubles import (
    HANG,
    REQ,
    finish,
    ok,
    scripted,
    status_error,
    text,
)

MSGS = [HumanMessage("x")]
KEY = "sk-abc123DEF456ghi789"


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """替换测试接缝：记录每次退避时长，且不真的睡。"""
    recorded: list[float] = []

    async def fake_sleep(d: float) -> None:
        recorded.append(d)

    monkeypatch.setattr(resilience, "_sleep", fake_sleep)
    return recorded


@pytest.fixture
def no_jitter(monkeypatch) -> None:
    """满抖动取上界，让退避序列变得可断言。"""
    monkeypatch.setattr(resilience, "_uniform", lambda a, b: b)


async def collect(cand, policy=None, **kw) -> list[str]:
    return [
        str(c.content)
        async for c in complete_with_retry(
            cand, MSGS, provider="p", policy=policy, **kw
        )
    ]


# ---------------------------------------------------------------- v1 十二条平移


async def test_no_failure_passthrough(sleeps):
    p = scripted(ok())
    assert await collect(p) == ["好", ""]
    assert p.calls == 1
    assert sleeps == []


async def test_retries_then_succeeds(sleeps, no_jitter):
    p = scripted(
        [ProviderTimeoutError("x", "t")], [ProviderServerError("x", "5xx")], ok()
    )
    assert await collect(p) == ["好", ""]
    assert p.calls == 3  # 失败 2 次 + 成功 1 次
    assert len(sleeps) == 2


async def test_honors_retry_after(sleeps):
    p = scripted([RateLimitedError("x", "busy", retry_after=3.0)], ok())
    await collect(p)
    assert sleeps == [3.0]  # 服务端说等 3 秒，就等 3 秒，不套抖动


async def test_backoff_is_exponential_with_cap(sleeps, no_jitter):
    p = scripted(*[[ProviderServerError("x", "e")]] * 3, ok())
    policy = RetryPolicy(max_attempts=4, base_backoff=0.5, max_backoff=1.5)
    await collect(p, policy)
    assert sleeps == [0.5, 1.0, 1.5]  # 0.5 → 1.0 → (2.0 被削顶到) 1.5


async def test_non_retryable_fails_immediately(sleeps):
    p = scripted([AuthError("x", "bad key")], ok())
    with pytest.raises(AuthError):
        await collect(p)
    assert p.calls == 1
    assert sleeps == []


async def test_gives_up_after_max_attempts(sleeps, no_jitter):
    p = scripted(*[[ProviderTimeoutError("x", "t")]] * 3, ok())
    with pytest.raises(ProviderTimeoutError):
        await collect(p, RetryPolicy(max_attempts=3))
    assert p.calls == 3
    assert len(sleeps) == 2


async def test_no_retry_after_first_chunk(sleeps):
    p = scripted([text("half"), ProviderServerError("boom", "mid-stream")], ok())
    got = []
    with pytest.raises(ProviderServerError, match="mid-stream"):
        async for c in complete_with_retry(p, MSGS, provider="p"):
            got.append(str(c.content))
    assert got == ["half"]  # 半截已流出，绝不能重试造成重复输出
    assert p.calls == 1
    assert sleeps == []


async def test_total_timeout_budget_stops_retrying(sleeps, no_jitter):
    p = scripted([ProviderTimeoutError("x", "t")], ok())
    with pytest.raises(ProviderTimeoutError):
        await collect(p, RetryPolicy(total_timeout=0.0))  # 预算为零：一次都不许等
    assert p.calls == 1
    assert sleeps == []


async def test_first_chunk_timeout_cuts_hang(sleeps):
    """挂起被首块超时切断，翻译成 ProviderTimeoutError——可重试、记熔断账的形态。"""
    p = scripted([HANG])
    with pytest.raises(ProviderTimeoutError, match="首块超时") as info:
        await collect(p, RetryPolicy(max_attempts=1, first_chunk_timeout=0.05))
    assert p.calls == 1
    assert p.closed == 1  # 切断即关闭候选流（探针⑧：随取消展开同步释放）
    assert isinstance(info.value.__cause__, TimeoutError)


async def test_hang_walks_standard_retry_path(sleeps, no_jitter):
    """挂起两次后恢复：走的是与 5xx 完全相同的重试路径，零新分支。"""
    p = scripted([HANG], [HANG], ok())
    got = await collect(p, RetryPolicy(max_attempts=3, first_chunk_timeout=0.05))
    assert got == ["好", ""]
    assert p.calls == 3
    assert len(sleeps) == 2


async def test_deadline_blocks_further_attempts(sleeps):
    """deadline 剩余不足 min_attempt_budget：不再开新尝试，真实死因原样上抛。"""
    p = scripted(
        [ProviderServerError("x", "real-cause")], [ProviderServerError("x", "second")]
    )
    with pytest.raises(ProviderServerError, match="real-cause"):
        await collect(p, RetryPolicy(max_attempts=3), deadline=time.monotonic() + 0.5)
    assert p.calls == 1  # 剩余 0.5s < 8s 门槛：第二次尝试没资格开始
    assert sleeps == []


async def test_deadline_caps_first_chunk_wait():
    """剩余预算比 first_chunk_timeout 更紧时取小值等首块——等的是预算不是参数。"""
    p = scripted([HANG])
    start = time.monotonic()
    policy = RetryPolicy(
        max_attempts=1, first_chunk_timeout=30.0, min_attempt_budget=0.0
    )
    with pytest.raises(ProviderTimeoutError):
        await collect(p, policy, deadline=start + 0.05)
    assert time.monotonic() - start < 5.0  # 若没取小值，这里要等 30s 才红


# ---------------------------------------------------------------- 翻译落点（v2 新增）


async def test_raw_sdk_error_is_translated_before_retry_decision(sleeps, no_jitter):
    """候选抛的是 SDK 原生异常：先翻译再判可重试；最终抛出的是家族成员且 __cause__ 保留 SDK 异常。"""
    p = scripted([status_error(503)], [status_error(502)], [status_error(500)])
    with pytest.raises(ProviderServerError, match="HTTP 500") as info:
        await collect(p, RetryPolicy(max_attempts=3))
    assert p.calls == 3
    assert len(sleeps) == 2
    assert isinstance(info.value.__cause__, openai.InternalServerError)


async def test_retry_after_header_from_real_429(sleeps):
    p = scripted([status_error(429, headers={"retry-after": "3"})], ok())
    await collect(p)
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    ("header", "expect"),
    [
        ("120", 8.0),  # 超上限：封顶 max_backoff
        ("garbage", 0.5),  # 解析失败：退化为指数退避（no_jitter 取上界）
        ("-5", 0.0),  # 负数：当作"立刻可重试"，不许把预算判定变成负数
        ("nan", 0.5),  # 非有限值：当作没给
    ],
)
async def test_real_429_retry_after_variants(sleeps, no_jitter, header, expect):
    p = scripted([status_error(429, headers={"retry-after": header})], ok())
    await collect(p)
    assert sleeps == [expect]


async def test_real_429_http_date_retry_after(sleeps):
    when = formatdate(time.time() + 3, usegmt=True)
    p = scripted([status_error(429, headers={"retry-after": when})], ok())
    await collect(p)
    assert 2.0 <= sleeps[0] <= 3.0


async def test_negative_retry_after_cannot_bypass_deadline_precheck(sleeps):
    """退避为负会让 `now + delay + min_attempt_budget > deadline` 永远不成立——归一后预检照常生效。"""
    p = scripted([status_error(429, headers={"retry-after": "-5"})], ok())
    with pytest.raises(RateLimitedError):
        await collect(p, deadline=time.monotonic() + 4.0)  # 剩余 4s < 8s 门槛
    assert p.calls == 1
    assert sleeps == []


async def test_unknown_exception_is_not_disguised_as_upstream_fault(sleeps):
    p = scripted([KeyError("programming error")], ok())
    with pytest.raises(KeyError):
        await collect(p)
    assert p.calls == 1
    assert sleeps == []


async def test_pool_timeout_is_overloaded_not_retried_not_provider_error(sleeps):
    try:
        raise openai.APITimeoutError(REQ) from httpx2.PoolTimeout("pool", request=REQ)
    except openai.APITimeoutError as err:
        p = scripted([err], ok())
    with pytest.raises(GatewayOverloadedError):
        await collect(p)
    assert p.calls == 1
    assert sleeps == []


async def test_framework_chunk_timeout_before_first_chunk_is_not_mislabelled(sleeps):
    """stream_chunk_timeout < 首块窗的配置错误形态（探针⑥#4）：走翻译表，不冒充自家首块超时。"""
    p = scripted([StreamChunkTimeoutError(0.2, chunks_received=0)])
    with pytest.raises(ProviderTimeoutError) as info:
        await collect(p, RetryPolicy(max_attempts=1))
    assert "首块超时" not in str(info.value)
    assert isinstance(info.value.__cause__, StreamChunkTimeoutError)


async def test_stream_chunk_timeout_never_steals_first_chunk_window():
    assert STREAM_CHUNK_TIMEOUT_S >= RetryPolicy().first_chunk_timeout


def test_retry_policy_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="first_chunk_timeout"):
        RetryPolicy(first_chunk_timeout=0)
    with pytest.raises(ValueError, match="total_timeout"):
        RetryPolicy(total_timeout=-1)


async def test_retry_emits_structured_log(sleeps, no_jitter):
    p = scripted([status_error(503)], ok())
    with capture_logs() as logs:
        await collect(p)
    (evt,) = [e for e in logs if e["event"] == "gateway.retry"]
    assert (evt["attempt"], evt["error"], evt["delay_s"]) == (
        1,
        "ProviderServerError",
        0.5,
    )


# ---------------------------------------------------------------- 首块判据：只认可见块


async def test_leading_empty_chunks_do_not_satisfy_first_chunk_window(sleeps):
    """思考型模型的 reasoning delta 被框架吐成空块：不喂首块计时器，思考期的挂起照样被切断。"""
    p = scripted([text(""), text(""), HANG])
    with pytest.raises(ProviderTimeoutError, match="首块超时"):
        await collect(p, RetryPolicy(max_attempts=1, first_chunk_timeout=0.05))
    assert p.calls == 1


async def test_leading_empty_chunks_are_dropped_before_first_visible_chunk():
    p = scripted([text(""), text(""), text("好"), finish()])
    assert await collect(p) == ["好", ""]


async def test_finish_only_reply_counts_as_visible():
    p = scripted([finish()])  # 空回答：只有终止信号也是完整流
    assert await collect(p) == [""]


# ---------------------------------------------------------------- 首块后：翻译不重试、截断、弃流


async def test_mid_stream_transport_error_is_translated_not_retried(sleeps):
    p = scripted([text("好"), httpx2.ReadTimeout("read", request=REQ)], ok())
    got = []
    with pytest.raises(ProviderTimeoutError) as info:
        async for c in complete_with_retry(p, MSGS, provider="p"):
            got.append(str(c.content))
    assert got == ["好"]
    assert p.calls == 1
    assert sleeps == []
    assert isinstance(info.value.__cause__, httpx2.ReadTimeout)


async def test_mid_stream_error_event_is_translated(sleeps):
    boom = openai.APIError("stream", REQ, body={"message": "x", "code": "E1"})
    p = scripted([text("好"), boom])
    with pytest.raises(ProviderServerError, match="流内错误 E1"):
        await collect(p)
    assert p.calls == 1
    assert sleeps == []


async def test_truncated_stream_is_not_success(sleeps):
    """无 finish_reason 的流：块照常流出（含框架补的空末块），耗尽处抛截断；首块后不重试。"""
    p = scripted([text("好")], ok())
    got = []
    with pytest.raises(ProviderServerError, match="流被截断"):
        async for c in complete_with_retry(p, MSGS, provider="p"):
            got.append(str(c.content))
    assert got == ["好", ""]
    assert p.calls == 1
    assert sleeps == []


async def test_truncated_tool_call_stream_is_not_success(sleeps):
    p = scripted(
        [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "f", "args": '{"a":1}', "id": "c1", "index": 0}
                ],
            )
        ]
    )
    with pytest.raises(ProviderServerError, match="流被截断"):
        await collect(p)
    assert (p.calls, sleeps) == (1, [])


async def test_finish_reason_without_usage_is_complete():
    p = scripted([text("好"), finish(usage=False)])
    assert await collect(p) == ["好", ""]


@pytest.mark.parametrize("reason", ["length", "tool_calls"])
async def test_any_finish_reason_counts_as_complete(reason):
    p = scripted([text("好"), finish(reason)])
    assert await collect(p) == ["好", ""]


async def test_empty_stream_is_retryable_server_error(sleeps, no_jitter):
    """候选一块都没吐：框架抛 ValueError，翻译成零块流（可重试）——不是成功也不是编程错误。"""
    p = scripted([], ok())
    assert await collect(p) == ["好", ""]
    assert p.calls == 2
    assert len(sleeps) == 1


async def test_consumer_abandon_closes_candidate_within_one_loop_tick():
    """弃流：GeneratorExit 同步进候选外壳，候选 _astream 在下一个循环周期被事件循环的终结钩子关闭；不翻译不重试。"""
    p = scripted([text("好"), HANG])
    gen = complete_with_retry(p, MSGS, provider="p")
    assert str((await anext(gen)).content) == "好"
    await gen.aclose()
    await asyncio.sleep(0)
    assert p.closed == 1
    assert p.calls == 1


async def test_cancellation_during_first_chunk_is_not_translated():
    p = scripted([HANG])

    async def consume():
        await anext(complete_with_retry(p, MSGS, provider="p"))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert (p.calls, p.closed) == (1, 1)


async def test_cancellation_during_backoff_propagates(monkeypatch):
    monkeypatch.setattr(resilience, "_uniform", lambda a, b: b)
    p = scripted([ProviderServerError("x", "e")], ok())
    task = asyncio.create_task(collect(p, RetryPolicy(base_backoff=5.0)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert p.calls == 1


async def test_cancellation_mid_stream_is_not_translated():
    p = scripted([text("好"), HANG])
    got = []

    async def consume():
        async for c in complete_with_retry(p, MSGS, provider="p"):
            got.append(str(c.content))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert (got, p.closed) == (["好"], 1)


async def test_whole_stream_has_no_upper_bound(monkeypatch):
    """整流不设上限：首块后即使时钟跨过 total_timeout 与 deadline，流照常完成（假时钟）。"""
    clock = {"now": 1000.0}
    monkeypatch.setattr(resilience, "_monotonic", lambda: clock["now"])

    def jump():
        clock["now"] += 300.0

    p = scripted([text("一"), jump, text("二"), jump, finish()])
    got = await collect(p, RetryPolicy(total_timeout=60.0), deadline=1010.0)
    assert got == ["一", "二", ""]


async def test_kwargs_are_forwarded_to_candidate():
    p = scripted(ok())
    await collect(p, tools=[{"type": "function"}], tool_choice="auto")
    assert p.seen_kwargs[0]["tools"] == [{"type": "function"}]
    assert p.seen_kwargs[0]["tool_choice"] == "auto"


# ---------------------------------------------------------------- 出站闸缝（C8）


async def test_gate_wait_is_outside_first_chunk_window():
    """闸内等待 0.2s > 首块窗 0.05s，仍成功：排队不计入首块计时、不会被冤枉成供应商超时。"""
    waits: list[float | None] = []

    async def slow_gate(max_wait):
        waits.append(max_wait)
        await asyncio.sleep(0.2)
        return True

    p = scripted(ok())
    policy = RetryPolicy(max_attempts=1, first_chunk_timeout=0.05)
    assert await collect(p, policy, acquire=slow_gate) == ["好", ""]
    assert waits[0] == pytest.approx(
        52.0, abs=0.5
    )  # 无 deadline：total_timeout 60 − 下限 8


async def test_gate_budget_is_derived_from_deadline_and_total_timeout():
    waits: list[float | None] = []

    async def gate(max_wait):
        waits.append(max_wait)
        return True

    p = scripted(ok())
    policy = RetryPolicy(min_attempt_budget=8.0)
    await collect(p, policy, deadline=time.monotonic() + 20.0, acquire=gate)
    assert waits[0] == pytest.approx(12.0, abs=0.5)  # deadline 剩余 20 更紧：20 − 8
    p2 = scripted(ok())
    await collect(
        p2, RetryPolicy(total_timeout=15.0, min_attempt_budget=8.0), acquire=gate
    )
    assert waits[1] == pytest.approx(7.0, abs=0.5)  # 无 deadline：total_timeout 15 − 8


async def test_gate_refusal_before_any_attempt_is_outbound_gate_timeout(sleeps):
    async def full(max_wait):
        return False

    p = scripted(ok())
    with pytest.raises(OutboundGateTimeout, match="出站闸") as info:
        await collect(p, acquire=full)
    assert isinstance(info.value, RateLimitedError)  # 候选环按 429 待遇
    assert p.calls == 0
    assert sleeps == []


async def test_gate_refusal_on_retry_raises_real_cause(sleeps, no_jitter):
    answers = iter([True, False])

    async def gate(max_wait):
        return next(answers)

    p = scripted([ProviderServerError("x", "real-cause")], ok())
    with capture_logs() as logs, pytest.raises(ProviderServerError, match="real-cause"):
        await collect(p, acquire=gate)
    assert p.calls == 1
    assert len(sleeps) == 1
    assert [e["event"] for e in logs] == ["gateway.retry", "gateway.gate_refused"]


async def test_gate_is_taken_once_per_attempt(sleeps, no_jitter):
    taken = 0

    async def gate(max_wait):
        nonlocal taken
        taken += 1
        return True

    p = scripted([status_error(503)], [status_error(503)], ok())
    await collect(p, acquire=gate)
    assert (p.calls, taken) == (3, 3)


async def test_gate_exception_is_not_translated_or_retried(sleeps):
    async def broken(max_wait):
        raise RuntimeError("gate bug")

    p = scripted(ok())
    with pytest.raises(RuntimeError, match="gate bug"):
        await collect(p, acquire=broken)
    assert (p.calls, sleeps) == (0, [])


async def test_probe_policy_is_single_attempt(sleeps):
    p = scripted([ProviderServerError("x", "e")], ok())
    with pytest.raises(ProviderServerError):
        await collect(p, dataclasses.replace(RetryPolicy(), max_attempts=1))
    assert p.calls == 1


# ---------------------------------------------------------------- 纯函数


def test_compute_backoff_prefers_capped_retry_after(monkeypatch):
    monkeypatch.setattr(resilience, "_uniform", lambda a, b: b)
    policy = RetryPolicy(base_backoff=0.5, max_backoff=8.0)
    assert compute_backoff(1, policy, retry_after=3.0) == 3.0
    assert compute_backoff(1, policy, retry_after=120.0) == 8.0
    assert compute_backoff(1, policy, retry_after=-1.0) == 0.0
    assert compute_backoff(1, policy, retry_after=float("nan")) == 0.5
    assert compute_backoff(3, policy, None) == 2.0


# ---------------------------------------------------------------- 端到端：真走 SDK + MockTransport


def sse(delta: dict, finish_reason: str | None = None) -> str:
    body = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"


SSE_HEAD = sse({"role": "assistant", "content": "你好"})
SSE_TAIL = sse({}, "stop") + "data: [DONE]\n\n"


class Paced(httpx2.AsyncByteStream):
    def __init__(self, script: list[tuple[float, str]]) -> None:
        self.script = script

    async def __aiter__(self):
        for delay, piece in self.script:
            await asyncio.sleep(delay)
            yield piece.encode()

    async def aclose(self) -> None:
        pass


def chat_openai(handler, *, stream_chunk_timeout: float = 60.0) -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key="sk-test-key",
        base_url="http://test/v1",
        max_retries=0,
        stream_usage=True,
        stream_chunk_timeout=stream_chunk_timeout,
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def event_stream(text_body: str):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=text_body
        )

    return handler


async def test_401_body_with_key_is_masked_end_to_end_including_traceback():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {KEY}"}}
        )

    with pytest.raises(AuthError) as info:
        await collect(chat_openai(handler))
    rendered = "".join(traceback.format_exception(info.value))
    assert KEY not in str(info.value)
    assert KEY not in rendered  # SDK 异常不进 __cause__ 链：日志 exc_info 渲染也不泄露
    assert "sk-***" in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None  # 在 except 块外抛出：解释器也没机会挂上


async def test_inter_chunk_stall_trips_stream_chunk_timeout_after_first_chunk():
    """块间空闲：首块已流出，框架的已解析块级计时器触发 → 翻译为超时、不重试。"""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Paced([(0.0, SSE_HEAD), (0.5, SSE_TAIL)]),
        )

    model = chat_openai(handler, stream_chunk_timeout=0.1)
    got = []
    with pytest.raises(ProviderTimeoutError) as info:
        async for c in complete_with_retry(model, MSGS, provider="bailian"):
            got.append(str(c.content))
    assert got == ["你好"]
    cause = info.value.__cause__
    assert isinstance(cause, StreamChunkTimeoutError)
    assert cause.chunks_received == 1


async def test_empty_body_end_to_end_pins_framework_message(sleeps, no_jitter):
    """空正文 200：框架抛 ValueError（消息前缀被翻译表识别）；两次都空则抛零块流、__cause__ 是 ValueError。"""
    hits = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal hits
        hits += 1
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=""
        )

    with pytest.raises(ProviderServerError, match="流为空") as info:
        await collect(chat_openai(handler), RetryPolicy(max_attempts=2))
    assert hits == 2
    assert isinstance(info.value.__cause__, ValueError)


async def test_truncated_sse_end_to_end(sleeps):
    hits = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal hits
        hits += 1
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=SSE_HEAD
        )

    got = []
    with pytest.raises(ProviderServerError, match="流被截断"):
        async for c in complete_with_retry(chat_openai(handler), MSGS, provider="p"):
            got.append(str(c.content))
    assert got == ["你好", ""]
    assert (hits, sleeps) == (1, [])


async def test_pool_timeout_end_to_end_is_overloaded(sleeps):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.PoolTimeout("pool", request=request)

    with pytest.raises(GatewayOverloadedError):
        await collect(chat_openai(handler))
    assert sleeps == []


async def test_sse_error_event_end_to_end(sleeps):
    err = 'data: {"error":{"message":"x","type":"server_error","code":"E1"}}\n\n'
    got = []
    with pytest.raises(ProviderServerError, match="流内错误 E1"):
        async for c in complete_with_retry(
            chat_openai(event_stream(SSE_HEAD + err)), MSGS, provider="p"
        ):
            got.append(str(c.content))
    assert (got, sleeps) == (["你好"], [])


async def test_in_stream_context_overflow_is_bad_request_not_retried(sleeps):
    """流内 error 事件含 'exceeds the context window'：框架重包为 ContextOverflowError（无状态码）→ 确定性拒绝。"""
    err = 'data: {"error":{"message":"input exceeds the context window","code":"ctx"}}\n\n'
    hits = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal hits
        hits += 1
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=err
        )

    with pytest.raises(BadRequestError, match="上下文超长"):
        await collect(chat_openai(handler))
    assert (hits, sleeps) == (1, [])


async def test_reasoning_deltas_end_to_end_do_not_count_as_first_chunk():
    """框架把只含 reasoning_content 的 delta 吐成空块（探针 B）：网关不把它们当首块、也不透传。"""
    body = (
        sse({"role": "assistant", "reasoning_content": "让我想想"})
        + sse({"reasoning_content": "……"})
        + SSE_HEAD
        + SSE_TAIL
    )
    got = [
        str(c.content)
        async for c in complete_with_retry(
            chat_openai(event_stream(body)), MSGS, provider="p"
        )
    ]
    assert got[0] == "你好"
    assert got == ["你好", "", ""]  # 内容块、finish 块、框架补的空末块
