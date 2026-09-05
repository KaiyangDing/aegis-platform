"""入站限流原语（M1.6；ADR-010）：临时 FastAPI 应用 + 真 Redis db1 走契约（前 N 次 200、第 N+1 次 429 且 detail 逐字、
Retry-After ≥ 1、键格式、租户/scope 互不连累、窗口到期归零、不信 X-Forwarded-For、NoScriptError 重载、未挂载与摘闸后 fail-open）；
Redis 故障用可编程替身 + 假时钟（不真等）钉降级本地窗、粘滞、探针与恢复、迟到成功不恢复；死端口一条走真异常路径。

限流器是挂在 app.state 上的实例：每个临时应用各挂各的，测试之间没有进程级状态要清。
script_flush 只清 Redis 脚本缓存（服务器级），本仓其它组件不用 Lua，无连坐。
"""

import asyncio
import math
import time

import httpx
import pytest
import structlog
from fastapi import Depends, FastAPI, Request
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import NoScriptError

pytest.importorskip("app.core.limits", reason="M1.6 未敲：app/core/limits.py 不存在")

from app.core import limits as lm
from app.core.config import Settings

if not hasattr(lm, "InboundLimiter"):
    pytest.skip(
        "M1.6 注入版未敲：limits.py 尚无 InboundLimiter", allow_module_level=True
    )

from app.core.limits import (
    DEFAULT_DETAIL,
    LOG_LIMITS_DEGRADED,
    LOG_LIMITS_RECOVERED,
    PREFIX,
    InboundLimiter,
    client_ip,
    identity,
    limiter_of,
    rate_limit,
    retry_after_seconds,
)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubRedis:
    """可编程 Redis 替身：只实现 script_load / evalsha；fail 开关模拟连接故障。"""

    SHA = "sha-stub"

    def __init__(self, *, fail: bool = False, fail_load: bool = False) -> None:
        self.fail = fail
        self.fail_load = fail_load
        self.reply = 0
        self.evalsha_calls = 0
        self.load_calls = 0

    async def script_load(self, script: str) -> str:
        self.load_calls += 1
        if self.fail_load:
            raise RedisConnectionError("down")
        return self.SHA

    async def evalsha(self, sha: str, numkeys: int, *args: str) -> int:
        self.evalsha_calls += 1
        if self.fail:
            raise RedisConnectionError("down")
        if sha != self.SHA:
            raise NoScriptError("No matching script. Please use EVAL.")
        return self.reply


async def tenant_header(request: Request) -> str:
    return identity("tenant", request.headers["X-Tenant-Id"])


@pytest.fixture
async def limiter(redis_async) -> InboundLimiter:
    lim = InboundLimiter(redis_async)
    await lim.start()
    return lim


async def started(stub: StubRedis, **kw) -> InboundLimiter:
    lim = InboundLimiter(stub, **kw)
    await lim.start()
    return lim


def make_app(
    scope: str, times: int, seconds: int = 60, *, limiter=None, **kw
) -> FastAPI:
    """limiter=None 时不挂任何东西：app.state 上根本没有这个属性（未挂载形态）。"""
    app = FastAPI()
    app.state.handled = 0
    if limiter is not None:
        app.state.inbound_limiter = limiter

    @app.post("/chat", dependencies=[Depends(rate_limit(scope, times, seconds, **kw))])
    async def chat(request: Request) -> dict[str, bool]:
        request.app.state.handled += 1
        return {"ok": True}

    return app


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def hit(c: httpx.AsyncClient, tenant: str | None = None) -> httpx.Response:
    return await c.post("/chat", headers={"X-Tenant-Id": tenant} if tenant else {})


def key_of(tenant: str, scope: str) -> str:
    return f"{PREFIX}:tenant:{tenant}:{scope}"


# ---------------------------------------------------------------- 契约（真 Redis db1）


async def test_first_n_pass_then_429_with_verbatim_detail(limiter, namespace):
    app = make_app(namespace, 3, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        resps = [await hit(c, "tA") for _ in range(4)]
    assert [r.status_code for r in resps] == [200, 200, 200, 429]
    assert resps[-1].json() == {"detail": DEFAULT_DETAIL}
    assert 1 <= int(resps[-1].headers["Retry-After"]) <= 60
    assert app.state.handled == 3  # 429 先于 handler：超限探测不出任何业务事实


async def test_key_format_count_and_fixed_window_live_in_redis(
    redis_async, limiter, namespace
):
    assert PREFIX == "aegis-limit"
    assert limiter.key("tenant:tA", namespace) == key_of("tA", namespace)
    app = make_app(namespace, 5, 60, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        for _ in range(3):
            await hit(c, "tA")
    key = key_of("tA", namespace)
    assert await redis_async.keys(f"{PREFIX}:*:{namespace}") == [key]
    assert await redis_async.get(key) == "3"
    assert 0 < await redis_async.pttl(key) <= 60_000


async def test_tenants_have_separate_windows(limiter, namespace):
    app = make_app(namespace, 1, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        assert (await hit(c, "tA")).status_code == 200
        assert (await hit(c, "tA")).status_code == 429
        assert (await hit(c, "tB")).status_code == 200


async def test_scopes_have_separate_windows(limiter, namespace):
    """两个应用共用一个限流器（同一进程的两个路由组）：scope 不同则各数各的。"""
    chat = make_app(f"{namespace}-chat", 1, limiter=limiter, identify=tenant_header)
    kb = make_app(f"{namespace}-kb", 1, limiter=limiter, identify=tenant_header)
    async with client(chat) as c1, client(kb) as c2:
        assert (await hit(c1, "tA")).status_code == 200
        assert (await hit(c1, "tA")).status_code == 429
        assert (await hit(c2, "tA")).status_code == 200


async def test_window_expiry_resets_the_count(redis_async, limiter, namespace):
    app = make_app(namespace, 1, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        assert (await hit(c, "tA")).status_code == 200
        assert (await hit(c, "tA")).status_code == 429
        await redis_async.delete(key_of("tA", namespace))  # = 窗口到期
        assert (await hit(c, "tA")).status_code == 200
    assert await redis_async.get(key_of("tA", namespace)) == "1"


async def test_default_identity_is_client_ip(redis_async, limiter, namespace):
    app = make_app(namespace, 1, limiter=limiter)
    async with client(app) as c:
        assert (await hit(c)).status_code == 200
        assert (await hit(c)).status_code == 429
    assert await redis_async.exists(f"{PREFIX}:ip:127.0.0.1:{namespace}") == 1


async def test_client_ip_ignores_x_forwarded_for(redis_async, limiter, namespace):
    """探针⑲：库默认 identifier 信 XFF，伪造即换身份；本仓只认直连对端，换 XFF 换不出新窗。"""
    app = make_app(namespace, 1, limiter=limiter)
    async with client(app) as c:
        first = await c.post("/chat", headers={"X-Forwarded-For": "10.0.0.1"})
        second = await c.post("/chat", headers={"X-Forwarded-For": "10.0.0.2"})
    assert (first.status_code, second.status_code) == (200, 429)
    assert await redis_async.keys(f"{PREFIX}:ip:10.*:{namespace}") == []
    assert await redis_async.exists(f"{PREFIX}:ip:127.0.0.1:{namespace}") == 1


async def test_detail_is_injectable_per_mount(limiter, namespace):
    app = make_app(namespace, 1, limiter=limiter, identify=tenant_header, detail="慢点")
    async with client(app) as c:
        await hit(c, "tA")
        assert (await hit(c, "tA")).json() == {"detail": "慢点"}


async def test_fail_open_when_nothing_is_mounted(redis_async, namespace):
    """app.state 上没有属性、或属性为 None（lifespan 无 Redis 时的形态），都放行且不碰 Redis。"""
    absent = make_app(namespace, 1, identify=tenant_header)
    assert limiter_of(absent) is None
    explicit_none = make_app(namespace, 1, identify=tenant_header)
    explicit_none.state.inbound_limiter = None
    for app in (absent, explicit_none):
        async with client(app) as c:
            assert [(await hit(c, "tA")).status_code for _ in range(4)] == [200] * 4
    assert await redis_async.keys(f"{PREFIX}:*:{namespace}") == []


async def test_unmounting_restores_fail_open(redis_async, limiter, namespace):
    """关停时 lifespan 把挂点置 None：之后的请求放行且不再碰 Redis。"""
    app = make_app(namespace, 1, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        await hit(c, "tA")
        assert (await hit(c, "tA")).status_code == 429
        app.state.inbound_limiter = None
        assert (await hit(c, "tA")).status_code == 200
    assert await redis_async.get(key_of("tA", namespace)) == "1"


async def test_noscript_reloads_script_and_keeps_counting(
    redis_async, limiter, namespace
):
    app = make_app(namespace, 2, limiter=limiter, identify=tenant_header)
    async with client(app) as c:
        assert (await hit(c, "tA")).status_code == 200
        await redis_async.script_flush()  # Redis 重启 / SCRIPT FLUSH
        assert (await hit(c, "tA")).status_code == 200  # 重载后照常计数：第 2 次仍放行
        assert (await hit(c, "tA")).status_code == 429
    assert not limiter.degraded  # NoScriptError 是重载路径，不是故障降级


async def test_dead_port_degrades_at_start_and_serves_locally(
    dead_redis_async, namespace
):
    lim = InboundLimiter(dead_redis_async)
    with structlog.testing.capture_logs() as logs:  # 不渲染堆栈：计时只量 Redis 触点
        t0 = time.perf_counter()
        await lim.start()  # 不炸
        elapsed = time.perf_counter() - t0
    # 一次短超时（fixture 0.05s），不是重试叠加或系统默认的 2s 连接等待
    assert elapsed < 0.5
    assert lim.degraded and [e["event"] for e in logs] == [LOG_LIMITS_DEGRADED]
    app = make_app(namespace, 1, limiter=lim, identify=tenant_header)
    async with client(app) as c:
        assert (await hit(c, "tA")).status_code == 200
        assert (await hit(c, "tA")).status_code == 429  # 本地窗裁决


# ---------------------------------------------------------------- 降级 / 粘滞 / 探针 / 恢复（替身 + 假时钟）


async def test_runtime_failure_degrades_to_local_window_and_sticks(
    monkeypatch, namespace
):
    monkeypatch.setattr(lm, "_monotonic", Clock())
    stub = StubRedis()
    lim = await started(stub, probe_interval=5.0)
    stub.fail = True
    app = make_app(namespace, 2, 60, limiter=lim, identify=tenant_header)
    with structlog.testing.capture_logs() as logs:
        async with client(app) as c:
            codes = [(await hit(c, "tA")).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert stub.evalsha_calls == 1  # 首次触点降级，之后粘滞：不再碰 Redis
    assert lim.degraded and app.state.handled == 2
    assert [e["event"] for e in logs] == [LOG_LIMITS_DEGRADED]  # 只喊一次


async def test_local_retry_after_is_the_remaining_window(monkeypatch, namespace):
    clock = Clock()
    monkeypatch.setattr(lm, "_monotonic", clock)
    lim = await started(StubRedis(fail=True))  # script_load 成功、evalsha 故障
    app = make_app(namespace, 1, 5, limiter=lim, identify=tenant_header)
    async with client(app) as c:
        await hit(c, "tA")
        clock.advance(2.0)  # 窗还剩 3s
        resp = await hit(c, "tA")
    assert resp.status_code == 429 and resp.headers["Retry-After"] == "3"
    assert resp.json() == {"detail": DEFAULT_DETAIL}


async def test_probe_after_interval_recovers_and_clears_local_windows(
    monkeypatch, namespace
):
    clock = Clock()
    monkeypatch.setattr(lm, "_monotonic", clock)
    stub = StubRedis()
    lim = await started(stub, probe_interval=5.0)
    stub.fail = True
    app = make_app(namespace, 1, 60, limiter=lim, identify=tenant_header)
    async with client(app) as c:
        await hit(c, "tA")
        assert (await hit(c, "tA")).status_code == 429
        clock.advance(4.9)
        assert (await hit(c, "tA")).status_code == 429 and stub.evalsha_calls == 1
        clock.advance(0.2)  # 过了 probe_interval：顺路探针
        stub.fail = False
        with structlog.testing.capture_logs() as logs:
            assert (
                await hit(c, "tA")
            ).status_code == 200  # 探针成功：共享计数裁决（stub 回 0）
    assert stub.evalsha_calls == 2 and not lim.degraded
    assert lim._local.windows == {}  # 旧窗作废
    assert [e["event"] for e in logs] == [LOG_LIMITS_RECOVERED]


async def test_probe_failure_extends_the_sticky_window(monkeypatch, namespace):
    clock = Clock()
    monkeypatch.setattr(lm, "_monotonic", clock)
    stub = StubRedis()
    lim = await started(stub, probe_interval=5.0)
    stub.fail = True
    app = make_app(namespace, 10, 60, limiter=lim, identify=tenant_header)
    async with client(app) as c:
        await hit(c, "tA")  # 降级
        clock.advance(5.1)
        await hit(c, "tA")  # 探针失败 → 顺延
        assert stub.evalsha_calls == 2 and lim.degraded
        clock.advance(4.9)
        await hit(c, "tA")  # 新窗内：不碰 Redis
        assert stub.evalsha_calls == 2
        clock.advance(0.2)
        await hit(c, "tA")  # 再探
        assert stub.evalsha_calls == 3


async def test_stale_in_flight_success_does_not_undo_degradation(monkeypatch):
    """健康期发出、降级后才返回的成功不是探针：只返回自己的裁决，不恢复、不清窗、不引发再降级。"""
    monkeypatch.setattr(lm, "_monotonic", Clock())
    gate = asyncio.Event()

    class SlowThenBroken(StubRedis):
        async def evalsha(self, sha: str, numkeys: int, *args: str) -> int:
            self.evalsha_calls += 1
            if self.evalsha_calls == 1:
                await gate.wait()  # 第一次：挂着，等别人先失败
                return 0
            raise RedisConnectionError("down")

    stub = SlowThenBroken()
    lim = await started(stub, probe_interval=5.0)
    stale = asyncio.create_task(lim.check("k1", 1, 60_000))
    await asyncio.sleep(0)  # 让 stale 真正发出去
    with structlog.testing.capture_logs() as logs:
        assert await lim.check("k2", 1, 60_000) == 0  # 失败 → 降级，本地窗开 k2
        assert lim.degraded
        gate.set()
        assert await stale == 0  # 迟到的成功：裁决照返
    assert lim.degraded and list(lim._local.windows) == ["k2"]  # 不恢复、不清窗
    assert stub.evalsha_calls == 2
    assert [e["event"] for e in logs] == [LOG_LIMITS_DEGRADED]


async def test_start_with_unreachable_redis_degrades_and_loads_script_on_first_probe(
    monkeypatch, namespace
):
    clock = Clock()
    monkeypatch.setattr(lm, "_monotonic", clock)
    stub = StubRedis(fail_load=True)
    with structlog.testing.capture_logs() as logs:
        lim = await started(stub, probe_interval=5.0)
    assert [e["event"] for e in logs] == [LOG_LIMITS_DEGRADED]
    assert lim.degraded
    app = make_app(namespace, 1, 60, limiter=lim, identify=tenant_header)
    async with client(app) as c:
        assert (await hit(c, "tA")).status_code == 200
        assert (await hit(c, "tA")).status_code == 429  # 本地窗
        assert (
            stub.evalsha_calls == 0
        )  # 没有 sha 不能 evalsha（探针⑳：sha=None 是 DataError）
        clock.advance(5.1)
        stub.fail_load = False
        assert (
            await hit(c, "tB")
        ).status_code == 200  # 探针：先 script_load 再 evalsha
    assert stub.load_calls == 2 and stub.evalsha_calls == 1 and not lim.degraded


async def test_two_limiters_do_not_share_degraded_state(monkeypatch, namespace):
    """注入而非全局：两个应用各挂各的限流器，一个降级不连累另一个。"""
    monkeypatch.setattr(lm, "_monotonic", Clock())
    broken, healthy = StubRedis(fail=True), StubRedis()
    lim_a, lim_b = await started(broken), await started(healthy)
    app_a = make_app(namespace, 1, limiter=lim_a, identify=tenant_header)
    app_b = make_app(namespace, 1, limiter=lim_b, identify=tenant_header)
    async with client(app_a) as ca, client(app_b) as cb:
        await hit(ca, "tA")
        assert (await hit(cb, "tA")).status_code == 200
    assert lim_a.degraded and not lim_b.degraded
    assert broken.evalsha_calls == 1 and healthy.evalsha_calls == 1


# ---------------------------------------------------------------- 本地固定窗（Lua 直译）


def test_local_window_mirrors_lua_semantics(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(lm, "_monotonic", clock)
    lw = lm._LocalWindows()
    assert lw.check("k", 2, 5000) == 0  # SET PX 分支：开窗
    assert lw.check("k", 2, 5000) == 0  # INCR 分支：不续期
    assert lw.check("k", 2, 5000) == 5000  # PTTL 分支：剩余毫秒
    clock.advance(4.0)
    assert lw.check("k", 2, 5000) == 1000
    clock.advance(0.9999)
    assert lw.check("k", 2, 5000) == 1  # 窗末前一瞬：向上取整，绝不返回放行哨兵 0
    clock.advance(0.0001)
    assert lw.check("k", 2, 5000) == 0  # 窗到期：新窗
    assert lw.check("other", 1, 5000) == 0 and lw.check("k", 2, 5000) == 0


def test_local_windows_are_lru_bounded(monkeypatch):
    monkeypatch.setattr(lm, "LOCAL_WINDOWS_MAX", 2)
    lw = lm._LocalWindows()
    for k in ("a", "b", "c"):
        lw.check(k, 1, 60_000)
    assert list(lw.windows) == ["b", "c"]  # 最久未用的 a 被淘汰
    lw.check("b", 1, 60_000)  # 被拒也算触碰：move_to_end
    lw.check("d", 1, 60_000)
    assert list(lw.windows) == ["b", "d"]


# ---------------------------------------------------------------- 纯函数与参数校验


@pytest.mark.parametrize(
    ("remaining_ms", "seconds"),
    [(1, 1), (999, 1), (1000, 1), (4796, 5), (0, 1), (-2, 1)],
)
def test_retry_after_is_ceil_seconds_at_least_one(remaining_ms, seconds):
    assert retry_after_seconds(remaining_ms) == seconds


@pytest.mark.parametrize(
    ("scope", "times", "seconds"),
    [
        ("", 1, 1),
        ("a:b", 1, 1),
        ("x" * 65, 1, 1),
        ("chat", 0, 1),
        ("chat", 1, 0),
        ("chat", 1, 1.5),  # 浮点秒会让 PX 参数变成 "1500.0"，Redis 拒收
        ("chat", 2.0, 1),
        ("chat", "3", 1),
        ("chat", True, 1),
    ],
)
def test_rate_limit_rejects_bad_params_at_mount(scope, times, seconds):
    with pytest.raises(ValueError, match="入站限流参数非法"):
        rate_limit(scope, times, seconds)


async def test_identity_helpers():
    assert identity("tenant", "tA") == "tenant:tA"
    assert identity("user", "u1") == "user:u1"
    no_client = Request({"type": "http", "headers": [], "method": "GET", "path": "/"})
    assert await client_ip(no_client) == "ip:unknown"


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan])
def test_limiter_rejects_bad_probe_interval(bad):
    with pytest.raises(ValueError, match="probe_interval"):
        InboundLimiter(StubRedis(), probe_interval=bad)


def test_config_probe_interval_field():
    if "inbound_probe_interval_s" not in Settings.model_fields:
        pytest.skip("M1.6 未敲：Settings 尚无 inbound_probe_interval_s")
    assert Settings(_env_file=None).inbound_probe_interval_s == 5.0
    for bad in (0, math.inf):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, inbound_probe_interval_s=bad)
