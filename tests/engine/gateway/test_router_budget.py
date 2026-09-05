"""候选环的预算与记账侧（M1.5c ④⑤⑥⑧）：月度闸 / 单请求估算闸 / 租户桶 / resolver 三态 / 记账行字段 /
簿记在流尾负向（弃流、取消、半截零记账）。计量/预算源/租户桶都是 Protocol 形状替身。"""

import asyncio
import gc

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

pytest.importorskip("app.engine.gateway.cache", reason="M1.5a 未敲")
from app.engine.gateway import router as router_mod

if not hasattr(router_mod, "CACHE_PROVIDER"):
    pytest.skip("M1.5c 未敲：router 尚无记账侧", allow_module_level=True)

from app.engine.gateway import resilience
from app.engine.gateway.cache import CACHED_MARK, CachedReply
from app.engine.gateway.errors import (
    BudgetExceeded,
    GatewayStreamInterrupted,
    ProviderServerError,
    TenantQuotaExceeded,
)
from app.engine.gateway.protocols import MeterLike
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import CACHE_PROVIDER, AegisGateway
from app.engine.gateway.routing import Candidate
from tests.engine.gateway.doubles import (
    HANG,
    ScriptedCandidate,
    StubBreaker,
    StubLimiter,
    finish,
    ok,
    scripted,
    text,
)

pytestmark = pytest.mark.usefixtures("no_backoff_sleep")

MSGS = [HumanMessage("x")]


def cand(name: str) -> Candidate:
    return Candidate(name, f"model-{name}")


class StubMeter:
    def __init__(self, spent: int = 0, events: list | None = None) -> None:
        self.spent = spent
        self.records: list[dict] = []
        self.events = events if events is not None else []

    async def record(self, **fields) -> None:
        self.records.append(fields)
        self.events.append("record")

    async def month_spend(self, tenant_id: str) -> int:
        return self.spent


class ExplodingMeter(StubMeter):
    async def record(self, **fields) -> None:
        raise ConnectionError("ledger db down")

    async def month_spend(self, tenant_id: str) -> int:
        raise ConnectionError("ledger db down")


class StubResolver:
    def __init__(self, budget: int | None) -> None:
        self.budget = budget
        self.asked: list[str] = []

    async def __call__(self, tenant_id: str) -> int | None:
        self.asked.append(tenant_id)
        return self.budget


class StubCache:
    def __init__(self, hit: CachedReply | None = None, events: list | None = None):
        self.hit = hit
        self.gets: list[str] = []
        self.puts: list[CachedReply] = []
        self.events = events if events is not None else []

    async def get(self, key: str) -> CachedReply | None:
        self.gets.append(key)
        return self.hit

    async def put(self, key: str, reply: CachedReply) -> None:
        self.puts.append(reply)
        self.events.append("put")


class EventBreaker(StubBreaker):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.events = events

    async def report_success(self, key, *, probe):
        self.events.append("report_success")
        await super().report_success(key, probe=probe)


def make_gw(
    named: dict[str, ScriptedCandidate],
    *,
    meter: StubMeter | None = None,
    tenant_limiter: StubLimiter | None = None,
    breaker: StubBreaker | None = None,
    tier: str = "fast",
    **kw,
) -> tuple[AegisGateway, StubLimiter]:
    limiter = StubLimiter()
    gw = AegisGateway(
        tenant_id="t1",
        routes={tier: [cand(n) for n in named]},
        models={cand(n): m for n, m in named.items()},
        breaker=breaker or StubBreaker(),
        limiter=limiter,
        meter=meter,
        tenant_limiter=tenant_limiter,
        default_tier=tier,
        **kw,
    )
    return gw, limiter


@pytest.fixture
def no_backoff_sleep(monkeypatch):
    async def nosleep(d: float) -> None: ...

    monkeypatch.setattr(resilience, "_sleep", nosleep)


async def collect(gw: AegisGateway, **kw) -> list[str]:
    return [str(c.content) async for c in gw.astream(MSGS, **kw)]


async def settle() -> None:
    gc.collect()
    for _ in range(10):
        await asyncio.sleep(0)


# ---------------------------------------------------------------- 记账行


async def test_success_records_one_row_with_identity_and_usage():
    meter = StubMeter()
    gw, _ = make_gw({"p1": scripted(ok())}, meter=meter)
    await collect(gw)
    assert len(meter.records) == 1
    row = meter.records[0]
    assert row["tenant_id"] == "t1"
    assert (row["provider"], row["model"], row["tier"]) == ("p1", "model-p1", "fast")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (
        1,
        1,
    )  # doubles.finish 的 usage
    assert (row["cached"], row["usage_missing"], row["session_id"]) == (
        False,
        False,
        None,
    )
    assert len(row["request_id"]) == 32  # uuid4().hex


async def test_session_id_and_tier_flow_into_the_row():
    meter = StubMeter()
    gw, _ = make_gw({"p1": scripted(ok())}, meter=meter, tier="strong")
    await collect(gw, session_id="s-9", tier="strong")
    assert (meter.records[0]["session_id"], meter.records[0]["tier"]) == (
        "s-9",
        "strong",
    )


async def test_session_id_is_not_in_cache_key():
    """session_id 是传输参数不是语义（M1.5c 起为 _astream 具名参数，不进 **kwargs）。"""
    cache = StubCache()
    gw, _ = make_gw({"p1": scripted(ok(), ok(), ok())}, reply_cache=cache)
    await collect(gw)
    await collect(gw, session_id="s-1")
    await collect(gw, session_id="s-2")
    assert len(set(cache.gets)) == 1


async def test_each_call_gets_a_fresh_request_id():
    meter = StubMeter()
    gw, _ = make_gw({"p1": scripted(ok(), ok())}, meter=meter)
    await collect(gw)
    await collect(gw)
    assert meter.records[0]["request_id"] != meter.records[1]["request_id"]


async def test_missing_upstream_usage_is_flagged_not_faked():
    meter = StubMeter()
    gw, _ = make_gw({"p1": scripted([text("好"), finish(usage=False)])}, meter=meter)
    await collect(gw)
    row = meter.records[0]
    assert row["usage_missing"] is True
    assert (row["prompt_tokens"], row["completion_tokens"]) == (0, 0)


async def test_cache_hit_records_zero_cost_row():
    meter = StubMeter()
    hit = CachedReply(
        "p9",
        "model-p9",
        [
            AIMessageChunk(content="缓存", response_metadata={CACHED_MARK: True}),
            AIMessageChunk(
                content="",
                response_metadata={CACHED_MARK: True, "finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
                chunk_position="last",
            ),
        ],
    )
    gw, _ = make_gw({"p1": scripted(ok())}, meter=meter, reply_cache=StubCache(hit=hit))
    await collect(gw)
    row = meter.records[0]
    assert (row["provider"], row["model"], row["cached"]) == (
        CACHE_PROVIDER,
        "model-p9",
        True,
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (
        7,
        3,
    )  # 记了账，成本由 cached 归零


async def test_record_happens_after_breaker_report_and_cache_put():
    events: list[str] = []
    gw, _ = make_gw(
        {"p1": scripted(ok())},
        meter=StubMeter(events=events),
        reply_cache=StubCache(events=events),
        breaker=EventBreaker(events),
    )
    await collect(gw)
    assert events == ["report_success", "put", "record"]


async def test_no_meter_means_no_budget_gate_and_no_record():
    gw, _ = make_gw({"p1": scripted(ok())}, meter=None, monthly_token_budget=1)
    assert await collect(gw) == ["好", ""]


# ---------------------------------------------------------------- 月度预算


async def test_budget_exceeded_blocks_before_everything():
    p1 = scripted(ok())
    tenant_gate = StubLimiter()
    gw, limiter = make_gw(
        {"p1": p1},
        meter=StubMeter(spent=100),
        tenant_limiter=tenant_gate,
        monthly_token_budget=100,
    )
    with pytest.raises(BudgetExceeded, match="本月"):
        await collect(gw)
    assert p1.calls == 0
    assert limiter.asked == [] and tenant_gate.asked == []  # 预算闸在一切闸门之前


async def test_monthly_budget_zero_means_disabled():
    gw, _ = make_gw(
        {"p1": scripted(ok())}, meter=StubMeter(spent=10**9), monthly_token_budget=0
    )
    assert await collect(gw) == ["好", ""]


async def test_meter_failures_never_break_request():
    gw, _ = make_gw(
        {"p1": scripted(ok())}, meter=ExplodingMeter(), monthly_token_budget=100
    )
    # month_spend 挂 → fail-open 放行；record 挂 → 只告警。请求全程无感
    assert await collect(gw) == ["好", ""]


async def test_resolver_takes_precedence_and_blocks():
    resolver = StubResolver(budget=100)
    p1 = scripted(ok())
    gw, _ = make_gw(
        {"p1": p1},
        meter=StubMeter(spent=100),
        monthly_token_budget=10_000,
        budget_resolver=resolver,
    )
    with pytest.raises(BudgetExceeded, match="预算 100"):
        await collect(gw)
    assert resolver.asked == ["t1"]  # 按网关绑定的租户查表
    assert p1.calls == 0


async def test_resolver_relaxes_static_budget():
    gw, _ = make_gw(
        {"p1": scripted(ok())},
        meter=StubMeter(spent=150),
        monthly_token_budget=100,
        budget_resolver=StubResolver(budget=1_000),
    )
    assert await collect(gw) == ["好", ""]


async def test_resolver_zero_disables_gate_without_reading_ledger():
    meter = ExplodingMeter()  # budget≤0 不该打 SUM：账本挂着也无感
    gw, _ = make_gw(
        {"p1": scripted(ok())},
        meter=meter,
        monthly_token_budget=100,
        budget_resolver=StubResolver(budget=0),
    )
    assert await collect(gw) == ["好", ""]


async def test_resolver_none_and_error_both_fail_open():
    async def broken(tenant_id: str) -> int:
        raise ConnectionError("tenants db down")

    gw, _ = make_gw(
        {"p1": scripted(ok(), ok())},
        meter=StubMeter(spent=10**9),
        monthly_token_budget=100,
        budget_resolver=broken,
    )
    assert await collect(gw) == ["好", ""]
    gw.budget_resolver = StubResolver(budget=None)  # None = 读挂
    assert await collect(gw) == ["好", ""]


# ---------------------------------------------------------------- 单请求预算


async def test_request_budget_gate_blocks_oversized_prompt():
    p1 = scripted(ok())
    tenant_gate = StubLimiter()
    gw, limiter = make_gw(
        {"p1": p1}, tenant_limiter=tenant_gate, request_token_budget=10
    )
    with pytest.raises(BudgetExceeded, match="单请求"):
        [c async for c in gw.astream([HumanMessage("验" * 50)])]
    assert p1.calls == 0
    assert limiter.asked == [] and tenant_gate.asked == []  # 闸门在限流之前


async def test_request_budget_counts_tool_schema():
    gw, _ = make_gw({"p1": scripted(ok())}, request_token_budget=10)
    big = {"type": "function", "function": {"name": "f", "description": "说" * 40}}
    with pytest.raises(BudgetExceeded, match="单请求"):
        await collect(gw, tools=[big])


async def test_request_budget_zero_means_disabled():
    gw, _ = make_gw({"p1": scripted(ok())}, request_token_budget=0)
    assert [c async for c in gw.astream([HumanMessage("验" * 500)])]


async def test_monthly_gate_is_checked_before_request_gate():
    gw, _ = make_gw(
        {"p1": scripted(ok())},
        meter=StubMeter(spent=100),
        monthly_token_budget=100,
        request_token_budget=1,
    )
    with pytest.raises(BudgetExceeded, match="本月"):
        [c async for c in gw.astream([HumanMessage("验" * 50)])]


# ---------------------------------------------------------------- 租户出站闸


async def test_tenant_quota_exhausted_fails_before_any_provider():
    p1 = scripted(ok())
    gw, limiter = make_gw({"p1": p1}, tenant_limiter=StubLimiter(deny={"t1"}))
    with pytest.raises(TenantQuotaExceeded, match="t1"):
        await collect(gw)
    assert p1.calls == 0  # 红线二：租户配额环外把关
    assert limiter.asked == []


async def test_tenant_gate_wait_is_bounded_by_deadline():
    tenant_gate = StubLimiter()
    gw, _ = make_gw(
        {"p1": scripted(ok(), ok())},
        tenant_limiter=tenant_gate,
        retry_policy=RetryPolicy(min_attempt_budget=8.0),
    )
    await collect(gw)
    await collect(gw, deadline_s=20)
    assert tenant_gate.asked[0] == ("t1", None)
    key, wait = tenant_gate.asked[1]
    assert key == "t1" and 11.5 < wait <= 12.0  # deadline 剩余 − min_attempt_budget


async def test_tenant_gate_is_asked_after_budget_gates_pass():
    tenant_gate = StubLimiter()
    gw, _ = make_gw(
        {"p1": scripted(ok())},
        meter=StubMeter(spent=1),
        tenant_limiter=tenant_gate,
        monthly_token_budget=100,
        request_token_budget=100,
    )
    await collect(gw)
    assert tenant_gate.asked == [("t1", None)]


# ---------------------------------------------------------------- 簿记在流尾负向（DoD#3）


async def test_abandoned_stream_records_nothing():
    meter, cache = StubMeter(), StubCache()
    gw, _ = make_gw(
        {"p1": scripted([text("一"), text("二"), finish()])},
        meter=meter,
        reply_cache=cache,
    )
    agen = gw.astream(MSGS)
    await anext(agen)
    await agen.aclose()
    await settle()
    assert meter.records == [] and cache.puts == []


async def test_cancelled_stream_records_nothing():
    meter, cache = StubMeter(), StubCache()
    gw, _ = make_gw(
        {"p1": scripted([text("一"), HANG])}, meter=meter, reply_cache=cache
    )

    async def consume():
        async for _ in gw.astream(MSGS):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    del task
    await settle()
    assert meter.records == [] and cache.puts == []


async def test_midstream_failure_records_nothing():
    meter = StubMeter()
    gw, _ = make_gw(
        {"p1": scripted([text("半"), ProviderServerError("p1", "boom")])}, meter=meter
    )
    with pytest.raises(GatewayStreamInterrupted):
        await collect(gw)
    assert meter.records == []


def test_stub_meter_matches_protocol():
    assert isinstance(StubMeter(), MeterLike)
