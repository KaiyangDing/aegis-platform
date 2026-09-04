"""出站闸：每 provider 一桶、冷启动满桶、限时取令牌（自家 max_wait 与 deadline 反推的 max_wait 取小）、
等不到就不等（预判）。真等 ≤0.2s；InMemoryRateLimiter 自带时钟无接缝，登记为真等例外。"""

import asyncio
import time

import pytest

from app.engine.gateway.outbound import ProviderLimiter


def limiter(**kw) -> ProviderLimiter:
    kw.setdefault("rate", 10.0)
    kw.setdefault("burst", 2.0)
    kw.setdefault("max_wait", 0.5)
    return ProviderLimiter(**kw)


async def test_cold_start_has_burst_then_refuses_without_wait():
    lim = limiter(burst=2.0, rate=0.01)
    assert await lim.acquire("p", 0) is True
    assert await lim.acquire("p", 0) is True  # 冷启动满桶：开局 burst 个立刻可用
    assert await lim.acquire("p", 0) is False  # 第三个：max_wait=0 只试不排


async def test_waits_within_budget_until_token_refills():
    lim = limiter(burst=1.0, rate=20.0)
    assert await lim.acquire("p", 0) is True
    start = time.monotonic()
    assert await lim.acquire("p", 0.5) is True  # 1/20s 后补一个令牌
    assert time.monotonic() - start < 0.4


async def test_deadline_budget_caps_the_wait():
    lim = limiter(burst=1.0, rate=0.01, max_wait=10.0)
    assert await lim.acquire("p", 0) is True
    start = time.monotonic()
    assert await lim.acquire("p", 0.1) is False  # deadline 反推的 0.1s 比自家 10s 更紧
    assert time.monotonic() - start < 0.4


async def test_none_budget_uses_own_max_wait():
    lim = limiter(burst=1.0, rate=20.0, max_wait=0.5)
    assert await lim.acquire("p", None) is True
    start = time.monotonic()
    assert (
        await lim.acquire("p", None) is True
    )  # 无 deadline 约束：自家 0.5s 内等到补给
    assert time.monotonic() - start < 0.4


async def test_own_max_wait_caps_even_without_budget():
    lim = limiter(burst=1.0, rate=5.0, max_wait=0.1)  # 补给要 0.2s > 自家上限 0.1s
    assert await lim.acquire("p", None) is True
    start = time.monotonic()
    assert await lim.acquire("p", None) is False
    assert time.monotonic() - start < 0.05  # 预判等不到：立刻放弃，不白等


async def test_unreachable_token_is_refused_without_waiting():
    """v1 wait_take 语义：桶按补给模型算出预算内根本等不到，就不烧预算。"""
    lim = limiter(burst=1.0, rate=0.01, max_wait=10.0)
    assert await lim.acquire("p", 0) is True
    start = time.monotonic()
    assert await lim.acquire("p", 0.3) is False
    assert time.monotonic() - start < 0.05


async def test_tiny_budget_still_gets_one_retry_before_deadline():
    """预算小于轮询周期也不能必败：到期前至少再试一次（库的 blocking=True 有 50ms 地板）。"""
    lim = limiter(burst=1.0, rate=1000.0)
    assert await lim.acquire("p", 0) is True
    assert await lim.acquire("p", 0.01) is True  # 1ms 后就有令牌


async def test_concurrent_acquires_never_overissue():
    lim = limiter(burst=3.0, rate=0.01)
    results = await asyncio.gather(*(lim.acquire("p", 0) for _ in range(10)))
    assert results.count(True) == 3


async def test_negative_or_zero_budget_is_non_blocking():
    lim = limiter(burst=1.0, rate=0.01)
    assert await lim.acquire("p", -1.0) is True
    assert await lim.acquire("p", -1.0) is False


async def test_providers_have_independent_buckets():
    lim = limiter(burst=1.0, rate=0.01)
    assert await lim.acquire("a", 0) is True
    assert await lim.acquire("a", 0) is False
    assert await lim.acquire("b", 0) is True  # 百炼挤了不该连累第二家


@pytest.mark.parametrize(
    "bad",
    [
        {"rate": 0.0},
        {"burst": 0.0},
        {"burst": 0.5},  # 库要求 available_tokens >= 1 才发放：<1 的桶永远拒绝
        {"max_wait": -1.0},
        {"check_every_n_seconds": 0.0},
    ],
)
def test_invalid_parameters_fail_loud(bad):
    with pytest.raises(ValueError):
        limiter(**bad)
