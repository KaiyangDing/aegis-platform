"""故障注入器：三模式各一 + 只命中 target + 满足与真候选同一协议（kwargs/bind_tools 透传）。"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from app.engine.gateway import faults, resilience
from app.engine.gateway.errors import (
    GatewayStreamInterrupted,
    ProviderServerError,
    ProviderTimeoutError,
)
from app.engine.gateway.faults import FaultInjector, inject_faults
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate
from tests.engine.gateway.doubles import StubBreaker, StubLimiter, ok, scripted

MSGS = [HumanMessage("x")]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {}}}
C1, C2 = Candidate("p1", "m1"), Candidate("p2", "m2")


@pytest.fixture
def always(monkeypatch):
    monkeypatch.setattr(faults, "_random", lambda: 0.0)


@pytest.fixture
def never(monkeypatch):
    monkeypatch.setattr(faults, "_random", lambda: 1.0)


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def nosleep(d: float) -> None: ...

    monkeypatch.setattr(resilience, "_sleep", nosleep)


def injector(inner, mode: str = "error", **kw) -> FaultInjector:
    return FaultInjector(inner=inner, provider="p1", rate=1.0, mode=mode, **kw)


# ---------------------------------------------------------------- 点名与三模式


def test_inject_faults_wraps_only_targets_and_is_identity_at_rate_zero():
    a, b = scripted(ok()), scripted(ok())
    table = inject_faults(
        {C1: a, C2: b}, rate=1.0, targets={"p1:m1"}, mode="error", hang_s=1.0
    )
    assert isinstance(table[C1], FaultInjector)
    assert table[C1].inner is a and table[C1].provider == "p1"
    assert table[C2] is b
    off = inject_faults({C1: a}, rate=0.0, targets={"p1:m1"}, mode="error", hang_s=1.0)
    assert off[C1] is a


async def test_error_mode_raises_before_inner_is_called(always):
    inner = scripted(ok())
    with pytest.raises(ProviderServerError, match="故障注入（error）"):
        [c async for c in injector(inner).astream(MSGS)]
    assert inner.calls == 0


async def test_hang_mode_sleeps_via_seam_then_raises_fallback(always, monkeypatch):
    slept: list[float] = []

    async def record(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(faults, "_hang_sleep", record)
    inner = scripted(ok())
    with pytest.raises(ProviderServerError, match="hang 兜底"):
        [c async for c in injector(inner, "hang", hang_s=30.0).astream(MSGS)]
    assert slept == [30.0]
    assert inner.calls == 0


async def test_midstream_mode_yields_first_chunk_then_dies(always):
    inner = scripted(ok())
    got = []
    with pytest.raises(ProviderTimeoutError, match="midstream"):
        async for c in injector(inner, "midstream").astream(MSGS):
            got.append(str(c.content))
    assert got == ["好"]
    assert inner.calls == 1
    await asyncio.sleep(0)
    assert inner.closed == 1  # aclosing：内层流随之关闭（下一个循环周期）


async def test_midstream_with_empty_inner_stream_is_a_pre_first_chunk_failure(always):
    """内层一块都没吐：框架抛零块 ValueError（首块前），经 complete_with_retry 翻译为可重试的零块流。"""
    inner = scripted([])
    with pytest.raises(ValueError, match="No generation chunks"):
        [c async for c in injector(inner, "midstream").astream(MSGS)]


def test_inject_faults_strips_whitespace_in_targets():
    a = scripted(ok())
    table = inject_faults(
        {C1: a}, rate=1.0, targets={" p1:m1 "}, mode="error", hang_s=1.0
    )
    assert isinstance(table[C1], FaultInjector)


# ---------------------------------------------------------------- 与真候选同一协议


async def test_passthrough_when_not_injected_forwards_kwargs(never):
    inner = scripted(ok())
    got = [str(c.content) async for c in injector(inner).astream(MSGS, tools=[TOOL])]
    assert got == ["好", ""]
    assert inner.seen_kwargs[0]["tools"] == [TOOL]


async def test_bind_tools_passthrough(never):
    inner = scripted(ok())
    await injector(inner).bind_tools([TOOL], tool_choice="auto").ainvoke(MSGS)
    kw = inner.seen_kwargs[0]
    assert kw["tools"] == [TOOL] and kw["tool_choice"] == "auto"


def test_injector_cache_is_off():
    assert injector(scripted(ok())).cache is False
    assert FaultInjector.model_fields["cache"].default is False


# ---------------------------------------------------------------- 经网关走完整真实路径


def gateway(models, **kw):
    breaker = StubBreaker()
    gw = AegisGateway(
        routes={"fast": [C1, C2]},
        models=models,
        breaker=breaker,
        limiter=StubLimiter(),
        default_tier="fast",
        retry_policy=RetryPolicy(max_attempts=1, first_chunk_timeout=0.05),
        **kw,
    )
    return gw, breaker


async def test_through_gateway_error_mode_hits_only_target(always):
    p1, p2 = scripted(ok()), scripted(ok())
    models = inject_faults(
        {C1: p1, C2: p2}, rate=1.0, targets={"p1:m1"}, mode="error", hang_s=1.0
    )
    gw, breaker = gateway(models)
    assert [str(c.content) async for c in gw.astream(MSGS)] == ["好", ""]
    assert (p1.calls, p2.calls) == (0, 1)  # p1 被注入器拦在门外，p2 不受影响
    assert breaker.failures == ["p1:m1"]  # 注入的故障走完整的真实路径


async def test_through_gateway_hang_mode_is_cut_and_fails_over(always):
    p1, p2 = scripted(ok()), scripted(ok())
    models = inject_faults(
        {C1: p1, C2: p2}, rate=1.0, targets={"p1:m1"}, mode="hang", hang_s=30.0
    )
    gw, breaker = gateway(models)
    assert [str(c.content) async for c in gw.astream(MSGS)] == ["好", ""]
    assert (p1.calls, p2.calls) == (0, 1)  # 替身在挂起，真身 p1 从未被调用
    assert breaker.failures == ["p1:m1"]  # 挂起被首块超时切断并记账


async def test_through_gateway_midstream_raises_stream_interrupted(always):
    p1, p2 = scripted(ok()), scripted(ok())
    models = inject_faults(
        {C1: p1, C2: p2}, rate=1.0, targets={"p1:m1"}, mode="midstream", hang_s=1.0
    )
    gw, breaker = gateway(models)
    got = []
    with pytest.raises(GatewayStreamInterrupted) as ei:
        async for c in gw.astream(MSGS):
            got.append(str(c.content))
    assert got == ["好"]
    assert isinstance(ei.value.__cause__, ProviderTimeoutError)
    assert breaker.failures == ["p1:m1"]
    assert p2.calls == 0  # 半截不换路
