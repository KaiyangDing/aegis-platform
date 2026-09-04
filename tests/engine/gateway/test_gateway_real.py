"""候选环 + 真件（MemoryBreaker + ProviderLimiter）：M1.4b 卡的集成条目。
半开/到期用假时钟推进（clocks.py）；出站闸真等 ≤0.2s。"""

import asyncio
import gc

import pytest
from langchain_core.messages import HumanMessage

from app.engine.gateway import resilience
from app.engine.gateway.breakers import BreakerPolicy, MemoryBreaker
from app.engine.gateway.errors import (
    GatewayOverloadedError,
    GatewayStreamInterrupted,
    ProviderServerError,
    RateLimitedError,
)
from app.engine.gateway.outbound import ProviderLimiter
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate
from tests.engine.gateway.clocks import fake_clock
from tests.engine.gateway.doubles import HANG, ScriptedCandidate, ok, scripted, text

MSGS = [HumanMessage("x")]
C1, C2 = Candidate("p1", "m1"), Candidate("p2", "m2")
RESET = 30.0


class RecordingBreaker(MemoryBreaker):
    """记录上报，方便断言"零上报"；状态判定仍是真件。"""

    def __init__(self, *, fail_max: int) -> None:
        super().__init__(BreakerPolicy(fail_max=fail_max, reset_timeout=RESET))
        self.reported: list[tuple[str, str]] = []

    async def report_success(self, key, *, probe):
        self.reported.append(("success", key))
        await super().report_success(key, probe=probe)

    async def report_failure(self, key, *, probe):
        self.reported.append(("failure", key))
        await super().report_failure(key, probe=probe)

    def failures(self, key: str) -> int:
        return self._slot(key).fails


@pytest.fixture
def clock(monkeypatch):
    return fake_clock(monkeypatch)


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    async def nosleep(d: float) -> None: ...

    monkeypatch.setattr(resilience, "_sleep", nosleep)


def gateway(
    p1: ScriptedCandidate,
    p2: ScriptedCandidate,
    *,
    fail_max: int = 2,
    limiter: ProviderLimiter | None = None,
    first_chunk_timeout: float = 0.05,
    max_attempts: int = 1,
) -> tuple[AegisGateway, RecordingBreaker]:
    breaker = RecordingBreaker(fail_max=fail_max)
    gw = AegisGateway(
        routes={"fast": [C1, C2]},
        models={C1: p1, C2: p2},
        breaker=breaker,
        limiter=limiter or ProviderLimiter(rate=100.0, burst=100.0, max_wait=0.5),
        default_tier="fast",
        retry_policy=RetryPolicy(
            max_attempts=max_attempts, first_chunk_timeout=first_chunk_timeout
        ),
    )
    return gw, breaker


async def collect(gw) -> list[str]:
    return [str(c.content) async for c in gw.astream(MSGS)]


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def open_p1(gw, breaker) -> None:
    """让 p1 连续失败到开路（p2 兜底成功）。"""
    for _ in range(2):
        assert await collect(gw) == ["好", ""]
    assert await breaker.state(C1.key) == "open"


async def cancel_midstream(gw) -> None:
    async def consume():
        async for _ in gw.astream(MSGS):
            await asyncio.sleep(999)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    del task
    for _ in range(
        10
    ):  # 取消不会立刻关闭生成器：候选环的 finally 由异步生成器终结器晚几拍跑
        gc.collect()
        await asyncio.sleep(0)


# ---------------------------------------------------------------- 熔断真件经候选环


async def test_consecutive_failures_open_and_stop_calling_backend():
    p1, p2 = scripted([ProviderServerError("p1", "boom")]), scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    assert p1.calls == 2
    assert await collect(gw) == ["好", ""]
    assert p1.calls == 2  # open 秒拒：后端不再被打
    assert p2.calls == 3


async def test_half_open_probe_gets_single_attempt_then_reopens(clock):
    """试探只尝试一次：策略给 3 次重试预算，试探仍只打一下（无试探规则会是 3 + 3 = 6）。"""
    booms = [[ProviderServerError("p1", "boom")] for _ in range(6)]
    p1, p2 = scripted(*booms, ok()), scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=1, max_attempts=3)
    assert await collect(gw) == ["好", ""]  # 3 次尝试耗尽 → 跳闸 → p2 兜底
    assert p1.calls == 3
    assert await breaker.state(C1.key) == "open"
    clock(RESET + 1)
    assert await collect(gw) == ["好", ""]
    assert p1.calls == 4  # 试探恰好一次（不重试）
    assert await breaker.state(C1.key) == "open"  # 试探失败立即重开


async def test_half_open_probe_success_closes(clock):
    p1, p2 = (
        scripted(
            [ProviderServerError("p1", "boom")],
            [ProviderServerError("p1", "boom")],
            ok(),
        ),
        scripted(ok()),
    )
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    clock(RESET + 1)
    assert await collect(gw) == ["好", ""]
    assert await breaker.state(C1.key) == "closed"


async def test_429_neither_counts_nor_resets():
    p1, p2 = (
        scripted(
            [ProviderServerError("p1", "boom")], [RateLimitedError("p1", "busy")], ok()
        ),
        scripted(ok()),
    )
    gw, breaker = gateway(p1, p2, fail_max=3)
    await collect(gw)  # 5xx：计 1
    await collect(gw)  # 429：不进账也不清账
    assert breaker.failures(C1.key) == 1
    assert breaker.reported == [
        ("failure", C1.key),
        ("success", C2.key),
        ("success", C2.key),
    ]


async def test_half_open_429_keeps_state_and_falls_back(clock):
    p1, p2 = (
        scripted(
            [ProviderServerError("p1", "boom")],
            [ProviderServerError("p1", "boom")],
            [RateLimitedError("p1", "busy")],
        ),
        scripted(ok()),
    )
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    clock(RESET + 1)
    assert await collect(gw) == ["好", ""]  # 试探吃了 429 → 换路
    assert await breaker.state(C1.key) == "half-open"  # 无裁决：不闭合
    assert await breaker.allow(C1.key) == "probe"  # 令牌已归还，下一个请求可再探


async def test_half_open_overload_passes_through_and_keeps_state(clock):
    p1 = scripted(
        [ProviderServerError("p1", "boom")],
        [ProviderServerError("p1", "boom")],
        [GatewayOverloadedError("[p1] pool")],
    )
    p2 = scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    clock(RESET + 1)
    with pytest.raises(GatewayOverloadedError):
        await collect(gw)
    assert await breaker.state(C1.key) == "half-open"
    assert await breaker.allow(C1.key) == "probe"


async def test_abandoned_probe_does_not_close_and_releases_token(clock):
    p1 = scripted(
        [ProviderServerError("p1", "boom")],
        [ProviderServerError("p1", "boom")],
        [text("好"), HANG],
    )
    p2 = scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    clock(RESET + 1)
    agen = gw.astream(MSGS)
    assert str((await anext(agen)).content) == "好"  # 半开试探已在流
    await agen.aclose()
    await settle()
    assert await breaker.state(C1.key) == "half-open"  # 弃流不闭合
    assert ("success", C1.key) not in breaker.reported
    assert await breaker.allow(C1.key) == "probe"  # 令牌已归还


async def test_cancelled_probe_keeps_half_open_and_releases_token(clock):
    p1 = scripted(
        [ProviderServerError("p1", "boom")],
        [ProviderServerError("p1", "boom")],
        [text("好"), HANG],
    )
    p2 = scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=2)
    await open_p1(gw, breaker)
    clock(RESET + 1)
    before = list(breaker.reported)
    await cancel_midstream(gw)
    await settle()
    assert breaker.reported == before
    assert await breaker.state(C1.key) == "half-open"
    assert await breaker.allow(C1.key) == "probe"


@pytest.mark.parametrize("how", ["aclose", "cancel"])
async def test_abandon_or_cancel_leaves_failure_counter_untouched(how):
    """DoD#3 真件版：弃流/取消 → 零上报、失败计数不变、状态仍 closed。"""
    p1, p2 = (
        scripted([ProviderServerError("p1", "boom")], [text("好"), HANG]),
        scripted(ok()),
    )
    gw, breaker = gateway(p1, p2, fail_max=3)
    await collect(gw)  # 计 1
    before = list(breaker.reported)
    if how == "aclose":
        agen = gw.astream(MSGS)
        await anext(agen)
        await agen.aclose()
    else:
        await cancel_midstream(gw)
    await settle()
    assert breaker.reported == before
    assert breaker.failures(C1.key) == 1
    assert await breaker.state(C1.key) == "closed"


async def test_midstream_failure_counts_and_interrupts():
    p1, p2 = scripted([text("half"), ProviderServerError("p1", "mid")]), scripted(ok())
    gw, breaker = gateway(p1, p2, fail_max=1)
    with pytest.raises(GatewayStreamInterrupted):
        await collect(gw)
    assert await breaker.state(C1.key) == "open"
    assert p2.calls == 0


# ---------------------------------------------------------------- 出站闸真件经候选环


async def test_gate_timeout_skips_candidate_with_zero_breaker_count():
    lim = ProviderLimiter(rate=0.01, burst=1.0, max_wait=0.05)
    assert await lim.acquire("p1", 0)  # 把 p1 的唯一令牌先用掉
    p1, p2 = scripted(ok()), scripted(ok())
    gw, breaker = gateway(p1, p2, limiter=lim)
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (0, 1)
    assert breaker.reported == [("success", C2.key)]  # p1 零记账


async def test_gate_wait_is_outside_first_chunk_window():
    """闸内排队 ≈0.2s > 首块窗 0.05s 仍成功：排队既不算首块超时也不进熔断账。"""
    lim = ProviderLimiter(rate=5.0, burst=1.0, max_wait=1.0)
    assert await lim.acquire("p1", 0)
    p1, p2 = scripted(ok()), scripted(ok())
    gw, breaker = gateway(p1, p2, limiter=lim, first_chunk_timeout=0.05)
    assert await collect(gw) == ["好", ""]
    assert (p1.calls, p2.calls) == (1, 0)
    assert breaker.reported == [("success", C1.key)]
