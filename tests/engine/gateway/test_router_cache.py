"""候选环的缓存侧（M1.5a ③⑧）：命中短路一切 / 完整流才入库 / 半截与弃流不入库 / 传输参数不进 key /
入库在熔断上报之后 / 网关构造必带合法 tenant_id。缓存是 Protocol 形状替身（StubCache）。"""

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import ValidationError

pytest.importorskip(
    "app.engine.gateway.cache", reason="M1.5a 未敲：app/engine/gateway/cache.py 不存在"
)

from app.engine.gateway import resilience
from app.engine.gateway.cache import CACHED_MARK, CachedReply
from app.engine.gateway.errors import GatewayStreamInterrupted, ProviderServerError
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate
from tests.engine.gateway.doubles import (
    ScriptedCandidate,
    StubBreaker,
    StubLimiter,
    finish,
    ok,
    scripted,
    text,
)

MSGS = [HumanMessage("x")]


def cand(name: str) -> Candidate:
    return Candidate(name, f"model-{name}")


class StubCache:
    """记录 get 的 key 与 put 的 (key, reply)；hit 固定。"""

    def __init__(self, hit: CachedReply | None = None, events: list | None = None):
        self.hit = hit
        self.gets: list[str] = []
        self.puts: list[tuple[str, CachedReply]] = []
        self.events = events if events is not None else []

    async def get(self, key: str) -> CachedReply | None:
        self.gets.append(key)
        return self.hit

    async def put(self, key: str, reply: CachedReply) -> None:
        self.puts.append((key, reply))
        self.events.append("put")


class EventBreaker(StubBreaker):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    async def report_success(self, key, *, probe):
        self.events.append("report_success")
        await super().report_success(key, probe=probe)


def stamped_hit() -> CachedReply:
    return CachedReply(
        "p9",
        "model-p9",
        [
            AIMessageChunk(content="缓存答案", response_metadata={CACHED_MARK: True}),
            AIMessageChunk(
                content="",
                response_metadata={CACHED_MARK: True, "finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
                chunk_position="last",
            ),
        ],
    )


def make_gw(
    named: dict[str, ScriptedCandidate],
    *,
    cache: StubCache | None,
    breaker: StubBreaker | None = None,
    tier: str = "fast",
    **kw,
) -> tuple[AegisGateway, StubBreaker, StubLimiter]:
    breaker = breaker or StubBreaker()
    limiter = StubLimiter()
    gw = AegisGateway(
        tenant_id="t1",
        routes={tier: [cand(n) for n in named]},
        models={cand(n): m for n, m in named.items()},
        breaker=breaker,
        limiter=limiter,
        reply_cache=cache,
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


# ---------------------------------------------------------------- 命中


async def test_cache_hit_short_circuits_everything():
    p1 = scripted(ok())
    gw, breaker, limiter = make_gw({"p1": p1}, cache=StubCache(hit=stamped_hit()))
    assert await collect(gw) == ["缓存答案", ""]
    assert p1.calls == 0
    assert breaker.allowed == []  # 连熔断入口都没问——最外圈短路
    assert limiter.asked == []


async def test_cache_hit_stamp_survives_ainvoke_aggregation():
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=StubCache(hit=stamped_hit()))
    msg = await gw.ainvoke(MSGS)
    assert msg.content == "缓存答案"
    assert msg.response_metadata[CACHED_MARK] is True


# ---------------------------------------------------------------- 入库


async def test_cache_miss_stores_full_stream_under_serving_candidate():
    cache = StubCache()
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=cache)
    await collect(gw)
    assert len(cache.puts) == 1
    key, reply = cache.puts[0]
    assert key == cache.gets[0]  # 同一把 key
    assert (reply.provider, reply.model) == ("p1", "model-p1")
    assert [str(c.content) for c in reply.chunks] == ["好", ""]


async def test_fallback_success_is_stored_under_second_candidate():
    cache = StubCache()
    gw, _, _ = make_gw(
        {"p1": scripted([ProviderServerError("p1", "boom")]), "p2": scripted(ok())},
        cache=cache,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    await collect(gw)
    assert cache.puts[0][1].provider == "p2"


async def test_midstream_failure_is_never_cached():
    cache = StubCache()
    gw, _, _ = make_gw(
        {"p1": scripted([text("半"), ProviderServerError("p1", "boom")])}, cache=cache
    )
    with pytest.raises(GatewayStreamInterrupted):
        await collect(gw)
    assert cache.puts == []  # 事故绝不能变成可重放的事故


async def test_abandoned_stream_is_never_cached():
    cache = StubCache()
    gw, breaker, _ = make_gw(
        {"p1": scripted([text("一"), text("二"), finish()])}, cache=cache
    )
    agen = gw.astream(MSGS)
    await anext(agen)
    await agen.aclose()  # 消费者提前挂断：簿记在流尾，一个都不发生
    assert cache.puts == []
    assert breaker.successes == []


async def test_put_happens_after_breaker_report():
    events: list[str] = []
    cache = StubCache(events=events)
    gw, _, _ = make_gw(
        {"p1": scripted(ok())}, cache=cache, breaker=EventBreaker(events)
    )
    await collect(gw)
    assert events == ["report_success", "put"]


async def test_cache_disabled_means_plain_path():
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=None)
    assert gw.reply_cache is None
    assert await collect(gw) == ["好", ""]


# ---------------------------------------------------------------- key 的边界


async def test_deadline_is_not_in_key():
    # session_id 到 M1.5c 才成为具名参数，对应断言在 test_router_budget.py
    cache = StubCache(hit=stamped_hit())
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=cache)
    await collect(gw)
    await collect(gw, deadline_s=5)
    await collect(gw, deadline_s=90)
    assert len(set(cache.gets)) == 1  # 传输参数不是语义：同一把 key


async def test_tier_tools_and_stop_change_key():
    cache = StubCache(hit=stamped_hit())
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=cache)
    gw.routes["strong"] = gw.routes["fast"]
    await collect(gw)
    await collect(gw, tier="strong")
    await collect(gw, stop=["\n"])
    await collect(gw, tools=[{"type": "function", "function": {"name": "f"}}])
    assert len(set(cache.gets)) == 4


# ---------------------------------------------------------------- 构造契约


def test_gateway_requires_tenant_id():
    with pytest.raises(ValidationError):
        AegisGateway(routes={}, models={}, breaker=StubBreaker(), limiter=StubLimiter())


@pytest.mark.parametrize("bad", ["tA:evil", "", "租户甲", "a" * 65])
def test_gateway_rejects_illegal_tenant_id(bad):
    with pytest.raises(ValidationError, match="tenant_id"):
        AegisGateway(
            tenant_id=bad,
            routes={},
            models={},
            breaker=StubBreaker(),
            limiter=StubLimiter(),
        )


def test_framework_cache_stays_off_and_reply_cache_is_separate():
    gw, _, _ = make_gw({"p1": scripted(ok())}, cache=StubCache())
    # 框架缓存（ainvoke 内部流式会查，探针㉑）静态关闭
    assert AegisGateway.model_fields["cache"].default is False
    assert gw.cache is False
    assert gw.reply_cache is not None  # 我们的租户缓存是另一个字段
