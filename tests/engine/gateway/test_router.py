"""AegisGateway 候选环与终局：v1 test_router 平移（替身 StubBreaker/StubLimiter）+ v2 新增
（BaseChatModel 骨架形态、出站闸缝按尝试计、弃流/取消不进账只归还试探锁）。

熔断/出站闸是 Protocol 形状替身；候选是走真实框架外壳的 ScriptedCandidate（tests/engine/gateway/doubles.py）。
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.engine.gateway import resilience
from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    GatewayError,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    OutboundGateTimeout,
    ProviderError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
)
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate
from tests.engine.gateway.doubles import (
    HANG,
    ScriptedCandidate,
    StubBreaker,
    StubLimiter,
    ok,
    scripted,
    text,
)

MSGS = [HumanMessage("x")]


def cand(name: str) -> Candidate:
    return Candidate(name, f"model-{name}")


def make_gw(
    named: dict[str, ScriptedCandidate],
    breaker: StubBreaker | None = None,
    limiter: StubLimiter | None = None,
    *,
    tier: str = "fast",
    **kw,
) -> tuple[AegisGateway, StubBreaker, StubLimiter]:
    breaker = breaker or StubBreaker()
    limiter = limiter or StubLimiter()
    gw = AegisGateway(
        routes={tier: [cand(n) for n in named]},
        models={cand(n): m for n, m in named.items()},
        breaker=breaker,
        limiter=limiter,
        default_tier=tier,
        **kw,
    )
    return gw, breaker, limiter


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def nosleep(d: float) -> None: ...

    monkeypatch.setattr(resilience, "_sleep", nosleep)


async def collect(gw: AegisGateway, **kw) -> list[str]:
    return [str(c.content) async for c in gw.astream(MSGS, **kw)]


async def settle() -> None:
    """让事件循环跑几拍：框架内层生成器由终结钩子在下一周期关闭（探针⑧）。"""
    for _ in range(5):
        await asyncio.sleep(0)


# ---------------------------------------------------------------- 候选环三待遇（v1 平移）


async def test_first_candidate_success_short_circuits():
    p1, p2 = scripted(ok()), scripted(ok())
    gw, breaker, _ = make_gw({"p1": p1, "p2": p2})
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (1, 0)
    assert breaker.successes == ["p1:model-p1"]


async def test_breaker_deny_skips_candidate_without_calling_it_or_its_gate():
    p1, p2 = scripted(ok()), scripted(ok())
    gw, _, limiter = make_gw(
        {"p1": p1, "p2": p2}, breaker=StubBreaker({"p1:model-p1": "deny"})
    )
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (0, 1)  # 秒拒：一次都没被打扰
    assert [k for k, _ in limiter.asked] == ["p2"]  # 连出站闸的队都不排


async def test_server_error_counts_to_breaker_then_falls_back():
    p1, p2 = scripted([ProviderServerError("p1", "boom")]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, retry_policy=RetryPolicy(max_attempts=1)
    )
    assert await collect(gw) == ["好", ""]
    assert breaker.failures == ["p1:model-p1"]
    assert breaker.successes == ["p2:model-p2"]
    assert (p1.calls, p2.calls) == (1, 1)  # 备选用的是自己的模型实例


async def test_retry_happens_inside_candidate_before_fallback():
    p1, p2 = scripted([ProviderServerError("p1", "blip")], ok()), scripted(ok())
    gw, breaker, limiter = make_gw({"p1": p1, "p2": p2})
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (2, 0)
    assert breaker.failures == []  # 站内自愈，不记熔断账
    assert [k for k, _ in limiter.asked] == ["p1", "p1"]  # 出站闸按尝试计


async def test_gate_refusal_skips_to_next_with_zero_breaker_count():
    p1, p2 = scripted(ok()), scripted(ok())
    gw, breaker, _ = make_gw({"p1": p1, "p2": p2}, limiter=StubLimiter(deny={"p1"}))
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (0, 1)  # 这家连排队都排不上，换下一站
    assert breaker.failures == []
    assert breaker.successes == ["p2:model-p2"]


async def test_gate_refusal_does_not_override_real_cause():
    """p1 真 5xx、p2 被本地闸拒：终局死因仍是 p1 的上游错误（v1 wait_take False 不写 last_error）。"""
    p1, p2 = scripted([ProviderServerError("p1", "real-cause")]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2},
        limiter=StubLimiter(deny={"p2"}),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(GatewayExhausted) as ei:
        await collect(gw)
    assert isinstance(ei.value.__cause__, ProviderServerError)
    assert "real-cause" in str(ei.value.__cause__)
    assert breaker.failures == ["p1:model-p1"]


async def test_rate_limited_falls_back_without_breaker_count():
    p1, p2 = scripted([RateLimitedError("p1", "busy")]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, retry_policy=RetryPolicy(max_attempts=1)
    )
    assert await collect(gw) == ["好", ""]
    assert breaker.failures == []  # 429 不是"上游死了"


async def test_local_overload_neither_counted_nor_rerouted():
    p1, p2 = (
        scripted([GatewayOverloadedError("[p1] 本地连接池排队超时")]),
        scripted(ok()),
    )
    gw, breaker, _ = make_gw({"p1": p1, "p2": p2})
    with pytest.raises(GatewayOverloadedError):
        await collect(gw)
    assert breaker.failures == []
    assert p2.calls == 0  # 所有候选共用同一个连接池，换路无意义


async def test_midstream_failure_never_falls_back():
    p1 = scripted([text("half"), ProviderServerError("p1", "mid-stream")])
    p2 = scripted(ok())
    gw, breaker, _ = make_gw({"p1": p1, "p2": p2})
    got = []
    with pytest.raises(GatewayStreamInterrupted, match="流中断于 p1:model-p1") as ei:
        async for c in gw.astream(MSGS):
            got.append(str(c.content))
    assert isinstance(ei.value.__cause__, ProviderServerError)  # 死因在 __cause__
    assert got == ["half"]
    assert p2.calls == 0  # 红线一：绝不换路重放
    assert breaker.failures == ["p1:model-p1"]  # 但账照记


@pytest.mark.parametrize(
    "exc",
    [
        AuthError("p1", "k"),
        BadRequestError("p1", "b"),
        GatewayOverloadedError("[p1] pool"),
        OutboundGateTimeout("p1", "gate"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_midstream_non_counted_errors_interrupt_and_release_probe(exc):
    """首块后无论哪种无裁决错误都是 StreamInterrupted：不换路、不进账、试探锁归还。"""
    p1, p2 = scripted([text("半"), exc]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, breaker=StubBreaker({"p1:model-p1": "probe"})
    )
    with pytest.raises(GatewayStreamInterrupted) as ei:
        await collect(gw)
    assert ei.value.__cause__ is exc
    assert p2.calls == 0
    assert breaker.failures == [] and breaker.releases == ["p1:model-p1"]


FAMILY = [
    RateLimitedError("p1", "m"),
    OutboundGateTimeout("p1", "m"),
    ProviderTimeoutError("p1", "m"),
    ProviderServerError("p1", "m"),
    BadRequestError("p1", "m"),
    AuthError("p1", "m"),
]


@pytest.mark.parametrize("exc", FAMILY, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("mid", [False, True], ids=["pre", "mid"])
async def test_no_provider_error_escapes_gateway(exc, mid):
    """DoD#1 遍历式：任何剧本下 ProviderError 家族不从网关穿出；首块后一律 StreamInterrupted 且不换路。"""
    p1 = scripted([text("半"), exc] if mid else [exc])
    p2 = scripted([ProviderServerError("p2", "down")])
    gw, _, _ = make_gw({"p1": p1, "p2": p2}, retry_policy=RetryPolicy(max_attempts=1))
    with pytest.raises(GatewayError) as ei:
        await collect(gw)
    assert not isinstance(ei.value, ProviderError)
    if mid:
        assert isinstance(ei.value, GatewayStreamInterrupted)
        assert p2.calls == 0


async def test_midstream_rate_limit_is_stream_interrupted_without_count():
    p1 = scripted([text("half"), RateLimitedError("p1", "busy")])
    gw, breaker, _ = make_gw({"p1": p1})
    with pytest.raises(GatewayStreamInterrupted):
        await collect(gw)
    assert breaker.failures == []


async def test_hang_is_cut_counted_and_fails_over():
    """C1 主路径：挂起 → 首块超时切断 → 记熔断账 → 换路成功。"""
    p1, p2 = scripted([HANG]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2},
        retry_policy=RetryPolicy(max_attempts=1, first_chunk_timeout=0.05),
    )
    assert await collect(gw) == ["好", ""]
    assert breaker.failures == ["p1:model-p1"]


# ---------------------------------------------------------------- 终局三段


async def test_all_candidates_fail_raises_exhausted_with_cause():
    p1 = scripted([ProviderServerError("p1", "a")])
    p2 = scripted([ProviderServerError("p2", "b")])
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, retry_policy=RetryPolicy(max_attempts=1)
    )
    with pytest.raises(GatewayExhausted, match="档位 fast 的所有候选均不可用") as ei:
        await collect(gw)
    assert isinstance(ei.value.__cause__, ProviderServerError)
    assert breaker.failures == ["p1:model-p1", "p2:model-p2"]


async def test_all_rejected_raises_gateway_rejected():
    p1 = scripted([AuthError("p1", "bad key")])
    p2 = scripted([BadRequestError("p2", "schema 转换非法")])
    gw, breaker, _ = make_gw({"p1": p1, "p2": p2})
    with pytest.raises(GatewayRejected, match="全部候选均被确定性拒绝") as ei:
        await collect(gw)
    assert isinstance(ei.value.__cause__, BadRequestError)
    assert breaker.failures == []  # 确定性拒绝不记熔断账


async def test_mixed_rejection_and_transient_stays_exhausted():
    p1 = scripted([ProviderServerError("p1", "boom")])
    p2 = scripted([AuthError("p2", "bad key")])
    gw, _, _ = make_gw({"p1": p1, "p2": p2}, retry_policy=RetryPolicy(max_attempts=1))
    with pytest.raises(GatewayExhausted):
        await collect(gw)


async def test_breaker_deny_counts_as_transient():
    p1, p2 = scripted(ok()), scripted([AuthError("p2", "bad key")])
    gw, _, _ = make_gw(
        {"p1": p1, "p2": p2}, breaker=StubBreaker({"p1:model-p1": "deny"})
    )
    with pytest.raises(GatewayExhausted):
        await collect(gw)


async def test_deadline_gate_before_any_candidate():
    """预算开局就不足：一个候选都不骚扰，熔断入口与出站闸都没被问。"""
    p1 = scripted(ok())
    gw, breaker, limiter = make_gw({"p1": p1})
    with pytest.raises(GatewayExhausted, match="首块预算 0.001s 耗尽"):
        await collect(gw, deadline_s=0.001)
    assert p1.calls == 0
    assert breaker.allowed == []
    assert limiter.asked == []


async def test_unrouted_tier_fails_cleanly_before_touching_anything():
    p1 = scripted(ok())
    gw, breaker, limiter = make_gw({"p1": p1})  # 只有 fast
    with pytest.raises(GatewayExhausted, match="档位 strong 没有配置任何候选"):
        await collect(gw, tier="strong")
    assert breaker.allowed == []
    assert limiter.asked == []


# ---------------------------------------------------------------- 半开试探


async def test_probe_decision_gets_single_attempt():
    p1, p2 = scripted([ProviderServerError("p1", "still down")], ok()), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, breaker=StubBreaker({"p1:model-p1": "probe"})
    )
    assert await collect(gw) == ["好", ""]
    assert p1.calls == 1  # 若走默认策略会是 2（重试后成功）
    assert breaker.failures == ["p1:model-p1"]
    assert breaker.releases == []  # 失败上报本身归还试探锁，不重复归还


async def test_probe_success_is_adjudicated_without_extra_release():
    p1 = scripted(ok())
    gw, breaker, _ = make_gw({"p1": p1}, breaker=StubBreaker({"p1:model-p1": "probe"}))
    assert await collect(gw) == ["好", ""]
    assert breaker.successes == ["p1:model-p1"]
    assert breaker.releases == []


async def test_probe_token_released_when_result_is_no_verdict():
    p1, p2 = scripted([RateLimitedError("p1", "busy")]), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2}, breaker=StubBreaker({"p1:model-p1": "probe"})
    )
    assert await collect(gw) == ["好", ""]
    assert breaker.releases == ["p1:model-p1"]  # 429 不构成裁决：令牌归还
    assert breaker.failures == []


async def test_probe_token_released_when_gate_blocks_the_probe():
    p1, p2 = scripted(ok()), scripted(ok())
    gw, breaker, _ = make_gw(
        {"p1": p1, "p2": p2},
        breaker=StubBreaker({"p1:model-p1": "probe"}),
        limiter=StubLimiter(deny={"p1"}),
    )
    assert await collect(gw) == ["好", ""]
    assert p1.calls == 0
    assert breaker.releases == ["p1:model-p1"]  # 领了令牌没打出去 → 还回去


async def test_probe_token_released_on_overload_passthrough():
    p1 = scripted([GatewayOverloadedError("[p1] pool")])
    gw, breaker, _ = make_gw({"p1": p1}, breaker=StubBreaker({"p1:model-p1": "probe"}))
    with pytest.raises(GatewayOverloadedError):
        await collect(gw)
    assert breaker.releases == ["p1:model-p1"]
    assert breaker.failures == []


# ---------------------------------------------------------------- 簿记在流尾：弃流/取消不进账


async def test_consumer_abandon_reports_nothing_and_releases_probe():
    p1 = scripted([text("好"), HANG])
    gw, breaker, _ = make_gw({"p1": p1}, breaker=StubBreaker({"p1:model-p1": "probe"}))
    agen = gw.astream(MSGS)
    assert str((await anext(agen)).content) == "好"
    await agen.aclose()
    await settle()
    assert breaker.successes == []
    assert breaker.failures == []
    assert breaker.releases == ["p1:model-p1"]  # 半开试探被弃：不闭合，只归还锁


async def test_cancellation_during_first_chunk_reports_nothing():
    p1 = scripted([HANG])
    gw, breaker, _ = make_gw({"p1": p1}, breaker=StubBreaker({"p1:model-p1": "probe"}))

    async def consume():
        await anext(gw.astream(MSGS))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert breaker.successes == [] and breaker.failures == []
    assert breaker.releases == ["p1:model-p1"]


# ---------------------------------------------------------------- BaseChatModel 骨架形态


def order_query(order_id: str) -> str:
    """查订单状态。"""
    return order_id


def test_cache_is_off_by_class_and_on_bound_view():
    gw, _, _ = make_gw({"p1": scripted(ok())})
    assert AegisGateway.model_fields["cache"].default is False
    assert gw.cache is False
    assert (
        gw.bind(tier="fast").bound is gw
    )  # 绑定视图不复制模型：缓存开关只在原实例上判定


async def test_bind_tools_returns_binding_and_forwards_openai_format():
    p1 = scripted(ok())
    gw, _, _ = make_gw({"p1": p1})
    bound = gw.bind_tools([order_query])
    assert bound is not gw
    await bound.ainvoke(MSGS)
    kw = p1.seen_kwargs[0]
    assert kw["tools"][0]["type"] == "function"
    assert kw["tools"][0]["function"]["name"] == "order_query"
    assert "tool_choice" not in kw  # None 不上线


async def test_gateway_own_kwargs_never_reach_candidate():
    p1 = scripted(ok())
    gw, _, _ = make_gw({"p1": p1})
    await gw.bind(tier="fast", deadline_s=30.0).ainvoke(MSGS)
    assert not {"tier", "deadline_s"} & p1.seen_kwargs[0].keys()


async def test_bind_tools_accepts_pydantic_class_tools():
    from pydantic import BaseModel

    class OrderQuery(BaseModel):
        """查订单状态。"""

        order_id: str

    p1 = scripted(ok())
    gw, _, _ = make_gw({"p1": p1})
    await gw.bind_tools([OrderQuery]).ainvoke(MSGS)
    tool = p1.seen_kwargs[0]["tools"][0]
    assert tool["function"]["name"] == "OrderQuery"
    assert "order_id" in tool["function"]["parameters"]["properties"]


async def test_unknown_exception_during_probe_passes_through_and_releases():
    p1 = scripted([KeyError("bug")])
    gw, breaker, _ = make_gw({"p1": p1}, breaker=StubBreaker({"p1:model-p1": "probe"}))
    with pytest.raises(KeyError):
        await collect(gw)
    assert breaker.failures == [] and breaker.releases == ["p1:model-p1"]


async def test_bind_tools_forwards_tool_choice_when_given():
    p1 = scripted(ok())
    gw, _, _ = make_gw({"p1": p1})
    await gw.bind_tools([order_query], tool_choice="auto").ainvoke(MSGS)
    assert p1.seen_kwargs[0]["tool_choice"] == "auto"


async def test_ainvoke_aggregates_stream_into_plain_ai_message():
    gw, breaker, _ = make_gw({"p1": scripted(ok())})
    reply = await gw.ainvoke(MSGS)
    assert type(reply) is AIMessage  # 不是 AIMessageChunk
    assert reply.content == "好"
    assert reply.usage_metadata["total_tokens"] == 2
    assert reply.response_metadata["finish_reason"] == "stop"
    assert breaker.successes == ["p1:model-p1"]  # 整段回走的是同一条候选环


def test_sync_invoke_is_unsupported():
    gw, _, _ = make_gw({"p1": scripted(ok())})
    with pytest.raises(NotImplementedError, match="只支持异步"):
        gw.invoke(MSGS)


async def test_default_tier_and_bind_override():
    pf, ps = scripted(ok()), scripted(ok())
    gw = AegisGateway(
        routes={"fast": [cand("pf")], "standard": [cand("ps")]},
        models={cand("pf"): pf, cand("ps"): ps},
        breaker=StubBreaker(),
        limiter=StubLimiter(),
    )
    await gw.ainvoke(MSGS)
    assert (pf.calls, ps.calls) == (0, 1)  # 默认 standard
    await gw.bind(tier="fast").ainvoke(MSGS)
    assert (pf.calls, ps.calls) == (1, 1)


async def test_stop_and_extra_kwargs_reach_the_candidate():
    p1 = scripted(ok())
    gw, _, _ = make_gw({"p1": p1})
    await collect(gw, stop=["END"], max_tokens=64)
    assert p1.seen_kwargs[0]["stop"] == ["END"]
    assert p1.seen_kwargs[0]["max_tokens"] == 64


async def test_gate_budget_is_passed_from_deadline():
    p1 = scripted(ok())
    gw, _, limiter = make_gw(
        {"p1": p1}, retry_policy=RetryPolicy(min_attempt_budget=8.0)
    )
    await collect(gw, deadline_s=20.0)
    ((provider, max_wait),) = limiter.asked
    assert provider == "p1"
    assert max_wait == pytest.approx(12.0, abs=0.5)
