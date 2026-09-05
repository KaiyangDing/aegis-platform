"""熔断：自研三键状态机（ADR-007）。

内存版用假时钟（clocks.py）走完整状态机：确定性、零真等。
Redis 版用真 Redis db1 + 毫秒级 TTL 验证只有共享存储才有的事实：跨实例同视野、全集群单探针、
令牌 TTL 自愈、失败窗、迟到上报；断连降级用开关代理（FlakyClient）驱动，零真等。真等：遗忘窗一条 ≈0.85s，其余 ≤0.55s。
"""

import asyncio
import math

import pytest
import redis
from structlog.testing import capture_logs

from app.engine.gateway import utterances as u
from app.engine.gateway.breakers import BreakerPolicy, MemoryBreaker, RedisBreaker
from app.engine.gateway.protocols import BreakerLike
from tests.engine.gateway.clocks import fake_clock

FAIL_MAX = 3
MEM = BreakerPolicy(
    fail_max=FAIL_MAX, reset_timeout=30.0, probe_ttl=120.0, fail_window=300.0
)
# Redis 版：open/令牌 0.2s 真等；失败窗放宽到 3s，让"必须仍存活"的断言有 ≥2s 余量
FAST = BreakerPolicy(
    fail_max=FAIL_MAX, reset_timeout=0.2, probe_ttl=0.2, fail_window=3.0
)
FORGET = BreakerPolicy(
    fail_max=FAIL_MAX, reset_timeout=0.2, probe_ttl=0.2, fail_window=0.8
)
# 备胎热态断言用：open 窗放宽到 2s——首次降级的 warning 带 exc_info，dev 模式 structlog 渲染
# traceback 约 0.4s，比 FAST 的 0.2s open 窗还长，备胎会在被问到之前先半开（2026-09-05 修复的假红）
WARM = BreakerPolicy(
    fail_max=FAIL_MAX, reset_timeout=2.0, probe_ttl=0.2, fail_window=3.0
)
KEY = "p:m"


@pytest.fixture
def clock(monkeypatch):
    return fake_clock(monkeypatch)


@pytest.fixture
def mem() -> MemoryBreaker:
    return MemoryBreaker(MEM)


async def fail(b, key: str, times: int) -> None:
    for _ in range(times):
        await b.report_failure(key, probe=False)


async def open_it(b, key: str) -> None:
    await fail(b, key, FAIL_MAX)
    assert await b.allow(key) == "deny"


def changes(logs) -> list[tuple[str, str]]:
    return [
        (e["old"], e["new"]) for e in logs if e["event"] == u.LOG_BREAKER_STATE_CHANGED
    ]


# ---------------------------------------------------------------- 内存版：状态机


def test_both_implementations_satisfy_the_protocol(mem):
    assert isinstance(mem, BreakerLike)
    assert isinstance(RedisBreaker(object(), MEM), BreakerLike)  # type: ignore[arg-type]


async def test_closed_allows_by_default(mem):
    assert await mem.allow(KEY) == "allow"
    assert await mem.state(KEY) == "closed"


async def test_failures_below_threshold_keep_allowing(mem):
    await fail(mem, KEY, FAIL_MAX - 1)
    assert await mem.allow(KEY) == "allow"
    assert await mem.state(KEY) == "closed"


async def test_threshold_failures_open_and_deny(mem):
    await open_it(mem, KEY)
    assert await mem.state(KEY) == "open"


async def test_success_resets_failure_count(mem):
    await fail(mem, KEY, FAIL_MAX - 1)
    await mem.report_success(KEY, probe=False)
    await fail(mem, KEY, FAIL_MAX - 1)
    assert await mem.allow(KEY) == "allow"


async def test_open_expires_into_half_open_with_exactly_one_probe(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout - 0.1)
    assert await mem.allow(KEY) == "deny"
    clock(0.2)
    assert await mem.state(KEY) == "half-open"
    assert await mem.allow(KEY) == "probe"
    assert await mem.allow(KEY) == "deny"  # 令牌已被拿走


async def test_probe_failure_reopens_and_clears_token(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"
    await mem.report_failure(KEY, probe=True)
    assert await mem.state(KEY) == "open"
    assert await mem.allow(KEY) == "deny"
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"  # 重开时令牌已清：不必等 probe_ttl


async def test_probe_success_closes_fully(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"
    await mem.report_success(KEY, probe=True)
    assert await mem.state(KEY) == "closed"
    await fail(mem, KEY, FAIL_MAX - 1)
    assert await mem.allow(KEY) == "allow"  # 失败账也清了


async def test_release_probe_frees_token_immediately(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"
    await mem.release_probe(KEY)
    assert await mem.state(KEY) == "half-open"
    assert await mem.allow(KEY) == "probe"


async def test_release_without_verdict_leaves_counter_untouched(mem):
    await fail(mem, KEY, FAIL_MAX - 1)
    await mem.release_probe(KEY)
    await mem.report_failure(KEY, probe=False)
    assert await mem.allow(KEY) == "deny"


async def test_probe_token_ttl_self_heals_after_holder_vanishes(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"
    clock(MEM.probe_ttl - 0.1)
    assert await mem.allow(KEY) == "deny"
    clock(0.2)
    assert await mem.allow(KEY) == "probe"


async def test_stale_failures_are_forgotten_after_fail_window(mem, clock):
    await fail(mem, KEY, FAIL_MAX - 1)
    clock(MEM.fail_window + 1)
    await mem.report_failure(KEY, probe=False)
    assert await mem.allow(KEY) == "allow"


async def test_failure_inside_window_extends_the_window(mem, clock):
    await fail(mem, KEY, 1)
    clock(MEM.fail_window - 1)
    await fail(mem, KEY, 1)
    clock(2)  # 距首次失败已超窗，但第二次失败刷新了窗
    await mem.report_failure(KEY, probe=False)
    assert await mem.allow(KEY) == "deny"


async def test_half_open_without_any_verdict_is_forgotten_after_fail_window(mem, clock):
    """第二条自愈路径：试探方消失且窗内再无裁决 → 整体遗忘回 closed（晚于令牌自愈）。"""
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    assert await mem.allow(KEY) == "probe"
    clock(MEM.fail_window - MEM.reset_timeout)  # 距最后一次失败恰好 fail_window + 1
    assert await mem.state(KEY) == "closed"
    assert await mem.allow(KEY) == "allow"


async def test_late_failure_while_open_extends_open_without_trip_log(mem, clock):
    """迟到上报（ADR-007 决策 5）：跳闸前在飞的请求在 open 期间失败 → 续期，不是再次跳闸。"""
    await open_it(mem, KEY)
    clock(MEM.reset_timeout - 1)
    with capture_logs() as logs:
        await mem.report_failure(KEY, probe=False)
    assert changes(logs) == []
    clock(2)  # 原 open 已过期；续期后仍 open
    assert await mem.allow(KEY) == "deny"
    clock(MEM.reset_timeout)
    assert await mem.allow(KEY) == "probe"


async def test_late_success_while_open_closes(mem, clock):
    await open_it(mem, KEY)
    with capture_logs() as logs:
        await mem.report_success(KEY, probe=False)
    assert changes(logs) == [("open", "closed")]
    assert await mem.state(KEY) == "closed"
    assert await mem.allow(KEY) == "allow"


async def test_probe_failure_after_account_was_forgotten_only_returns_token(mem, clock):
    """半开窗后段才领到的令牌可能比失败账活得久：此时试探失败从 1 计、只归还令牌。"""
    await open_it(mem, KEY)
    clock(MEM.fail_window - 50)  # 半开很久没人来探
    assert await mem.allow(KEY) == "probe"  # 令牌至 +120
    clock(51)  # 失败账过期，令牌仍在
    await mem.report_failure(KEY, probe=True)
    assert mem._slot(KEY).probe_until == 0.0
    assert await mem.state(KEY) == "closed"
    await fail(mem, KEY, FAIL_MAX - 2)
    assert await mem.allow(KEY) == "allow"  # 从 1 计：1 + (FAIL_MAX-2) < FAIL_MAX
    await fail(mem, KEY, 1)
    assert await mem.allow(KEY) == "deny"


async def test_keys_are_independent(mem):
    await open_it(mem, KEY)
    assert await mem.allow("other:m") == "allow"


async def test_concurrent_allow_grants_exactly_one_probe(mem, clock):
    await open_it(mem, KEY)
    clock(MEM.reset_timeout + 1)
    decisions = await asyncio.gather(*(mem.allow(KEY) for _ in range(8)))
    assert sorted(decisions) == ["deny"] * 7 + ["probe"]


async def test_state_changes_are_logged(mem, clock):
    with capture_logs() as logs:
        await open_it(mem, KEY)
        clock(MEM.reset_timeout + 1)
        await mem.allow(KEY)
        await mem.report_failure(KEY, probe=True)  # half-open → open
        clock(MEM.reset_timeout + 1)
        await mem.allow(KEY)
        await mem.report_success(KEY, probe=True)  # half-open → closed
    assert changes(logs) == [
        ("closed", "open"),
        ("half-open", "open"),
        ("half-open", "closed"),
    ]
    assert all(
        e["breaker"] == KEY for e in logs if e["event"] == u.LOG_BREAKER_STATE_CHANGED
    )


async def test_plain_success_is_silent(mem):
    with capture_logs() as logs:
        await mem.report_success(KEY, probe=False)
    assert logs == []


async def test_quiet_memory_breaker_logs_nothing():
    quiet = MemoryBreaker(MEM, quiet=True)
    with capture_logs() as logs:
        await fail(quiet, KEY, FAIL_MAX)
        await quiet.report_success(KEY, probe=False)
    assert logs == []
    assert await quiet.state(KEY) == "closed"


@pytest.mark.parametrize(
    "bad",
    [
        {"fail_max": 0},
        {"reset_timeout": 0},
        {"probe_ttl": 0},
        {"fail_window": -1},
        {"fail_window": math.inf},  # 过得了 > 0，却让 Redis 版 TTL 换算溢出
        {"reset_timeout": math.nan},
        {"probe_interval": 0},
        {
            "reset_timeout": 30.0,
            "probe_ttl": 120.0,
            "fail_window": 150.0,
        },  # 遗忘不得早于令牌自愈
    ],
)
def test_policy_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="熔断参数非法"):
        BreakerPolicy(**bad)


# ---------------------------------------------------------------- Redis 版：只有共享存储才有的事实


@pytest.fixture
def key(namespace: str) -> str:
    return f"{namespace}:p:m"


@pytest.fixture
def two(redis_async) -> tuple[RedisBreaker, RedisBreaker]:
    """两个实例 = 两个进程（同一状态机、同一 Redis）。"""
    return RedisBreaker(redis_async, FAST), RedisBreaker(redis_async, FAST)


async def test_redis_state_is_shared_across_instances(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)
    assert await b.allow(key) == "deny"
    assert await b.state(key) == "open"
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await b.state(key) == "half-open"
    assert await b.allow(key) == "probe"
    await b.report_success(key, probe=True)
    assert await a.allow(key) == "allow"
    assert await a.state(key) == "closed"


async def test_redis_grants_a_single_probe_cluster_wide(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await a.allow(key) == "probe"
    assert await b.allow(key) == "deny"
    await a.release_probe(key)
    assert await b.allow(key) == "probe"


async def test_redis_concurrent_allow_grants_exactly_one_probe(two, key):
    a, _ = two
    await fail(a, key, FAIL_MAX)
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    with capture_logs() as logs:
        decisions = await asyncio.gather(*(a.allow(key) for _ in range(8)))
    assert sorted(decisions) == ["deny"] * 7 + ["probe"]  # fail-open 的 allow 冒充不了
    assert not [e for e in logs if e["event"] == u.LOG_BREAKER_UNAVAILABLE]


async def test_redis_probe_token_self_heals(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await a.allow(key) == "probe"  # a 拿到令牌后"崩溃"，从不上报
    await asyncio.sleep(FAST.probe_ttl + 0.05)
    assert await b.allow(key) == "probe"


async def test_redis_probe_failure_reopens_and_clears_token(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await a.allow(key) == "probe"
    await a.report_failure(key, probe=True)
    assert await b.allow(key) == "deny"
    assert await b.state(key) == "open"
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await b.allow(key) == "probe"


async def test_redis_late_success_while_open_closes(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)
    assert await b.allow(key) == "deny"
    with capture_logs() as logs:
        await a.report_success(key, probe=False)  # 跳闸前在飞的请求迟到成功
    assert changes(logs) == [("open", "closed")]
    assert await b.allow(key) == "allow"
    assert await b.state(key) == "closed"


async def test_redis_late_failure_while_open_extends_open(two, key):
    a, b = two
    await fail(a, key, FAIL_MAX)  # open 至 +0.2
    await asyncio.sleep(0.15)
    with capture_logs() as logs:
        await a.report_failure(key, probe=False)  # 续期至 +0.35
    assert changes(logs) == []
    await asyncio.sleep(0.10)  # +0.25：原 open 已过期
    assert await b.allow(key) == "deny"
    await asyncio.sleep(0.15)  # +0.40
    assert await b.allow(key) == "probe"


async def test_redis_fail_window_forgets_stale_failures(redis_async, key):
    a = RedisBreaker(redis_async, FORGET)
    await fail(a, key, FAIL_MAX - 1)
    await asyncio.sleep(FORGET.fail_window + 0.05)
    with capture_logs() as logs:
        await a.report_failure(key, probe=False)
        assert await a.allow(key) == "allow"
    assert not [e for e in logs if e["event"] == u.LOG_BREAKER_UNAVAILABLE]
    assert await redis_async.get(f"aegis:cb:{key}:fails") == "1"


async def test_redis_success_clears_all_three_keys(two, key, redis_async):
    a, _ = two
    await fail(a, key, FAIL_MAX)
    await asyncio.sleep(FAST.reset_timeout + 0.05)
    assert await a.allow(key) == "probe"
    await a.report_success(key, probe=True)
    assert await redis_async.keys(f"aegis:cb:{key}:*") == []


# ---------------------------------------------------------------- Redis 版：断连降级与恢复


class FlakyClient:
    """真客户端的开关代理：dead=True 时每个触点抛 ConnectionError；触点计数用来证明"没碰 Redis"。"""

    def __init__(self, real) -> None:
        self.real, self.dead, self.touches = real, False, 0

    def _touch(self) -> None:
        self.touches += 1
        if self.dead:
            raise redis.exceptions.ConnectionError("dead")

    def pipeline(self, transaction: bool = True):
        self._touch()
        return self.real.pipeline(transaction=transaction)

    async def set(self, *args, **kwargs):
        self._touch()
        return await self.real.set(*args, **kwargs)

    async def exists(self, *args):
        self._touch()
        return await self.real.exists(*args)

    async def delete(self, *args):
        self._touch()
        return await self.real.delete(*args)


@pytest.fixture
def flaky(redis_async) -> FlakyClient:
    return FlakyClient(redis_async)


def degraded(logs) -> list[str]:
    return [e["op"] for e in logs if e["event"] == u.LOG_BREAKER_DEGRADED]


async def test_dead_redis_degrades_to_local_state_machine(dead_redis_async, key):
    """真死端口：首触失败即降级，之后备胎就是熔断器，跳闸照样记日志。"""
    b = RedisBreaker(dead_redis_async, MEM)
    with capture_logs() as logs:
        assert await b.allow(key) == "allow"  # 备胎 closed
        await fail(b, key, FAIL_MAX)  # 降级期上报只进备胎
        assert await b.allow(key) == "deny"
        assert await b.state(key) == "open"
    assert degraded(logs) == ["allow"]  # 只喊一次，且只撞了一次 Redis
    assert changes(logs) == [("closed", "open")]
    assert all(
        e.get("store") == "memory"
        for e in logs
        if e["event"] == u.LOG_BREAKER_STATE_CHANGED
    )


async def test_degradation_is_sticky_and_probes_after_interval(flaky, key, clock):
    b = RedisBreaker(flaky, FAST)
    flaky.dead = True
    assert await b.allow(key) == "allow"
    n = flaky.touches
    assert await b.allow(key) == "allow"  # 粘滞窗内：一次都不碰
    await b.report_failure(key, probe=False)
    await b.report_success(key, probe=False)
    await b.release_probe(key)
    assert flaky.touches == n
    clock(FAST.probe_interval + 0.1)
    assert await b.allow(key) == "allow"  # 顺路探针：碰一次，仍失败
    assert flaky.touches == n + 1
    assert await b.allow(key) == "allow"  # 探针失败顺延窗口：又不碰了
    assert flaky.touches == n + 1


async def test_recovery_switches_back_to_redis(flaky, key, clock, redis_async):
    b = RedisBreaker(flaky, FAST)
    flaky.dead = True
    with capture_logs() as logs:
        await b.allow(key)  # 降级
        flaky.dead = False
        assert await b.allow(key) == "allow"  # 窗内仍走备胎
        before = flaky.touches
        clock(FAST.probe_interval + 0.1)
        assert await b.allow(key) == "allow"  # 探针成功 → 切回
        assert flaky.touches == before + 1
        await b.report_failure(key, probe=False)  # 恢复后上报回到 Redis
        assert flaky.touches == before + 2
    events = [e["event"] for e in logs]
    assert events.count(u.LOG_BREAKER_DEGRADED) == 1
    assert events.count(u.LOG_BREAKER_RECOVERED) == 1
    assert await redis_async.get(f"aegis:cb:{key}:fails") == "1"


async def test_local_fallback_is_warm(flaky, key):
    """双写：主路健康时备胎也在记账，Redis 一断就接手，不必重新攒失败。"""
    b = RedisBreaker(flaky, WARM)
    await fail(b, key, FAIL_MAX)  # 走 Redis，同时进备胎
    assert await b.allow(key) == "deny"
    flaky.dead = True
    assert await b.allow(key) == "deny"  # 备胎已是 open
    assert await b.state(key) == "open"


async def test_fallback_stays_quiet_while_redis_is_healthy(flaky, key):
    b = RedisBreaker(flaky, FAST)
    with capture_logs() as logs:
        await fail(b, key, FAIL_MAX)
    assert changes(logs) == [("closed", "open")]  # 只有主路喊，备胎不重复
    assert [e["store"] for e in logs if e["event"] == u.LOG_BREAKER_STATE_CHANGED] == [
        "redis"
    ]
