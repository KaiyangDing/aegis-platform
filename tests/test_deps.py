"""组合根（M1.5c）：配置 → 共享件 → 按租户装配的搬运正确性（其它测试全用替身，只有这里看得见）；
fake 模式端到端流出回复；两租户网关共享同一批共享件、各绑各的身份；真 Redis + 真 PG 的端到端：
第一遍打上游并记真账，第二遍命中缓存并记零成本账。"""

import httpx2
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import select

pytest.importorskip("app.domain.usage", reason="M1.5b 未敲：app/domain/usage.py 不存在")
pytest.importorskip("app.engine.gateway.cache", reason="M1.5a 未敲")
from app import deps as deps_mod

if not hasattr(deps_mod, "build_gateway_parts"):
    pytest.skip("M1.5c 未敲：deps.py 尚无组合根", allow_module_level=True)

from app.core.config import Settings
from app.deps import build_gateway_parts, gateway_for
from app.domain.usage import CACHE_PROVIDER, UsageRecord
from app.engine.fakes import FAKE_REPLY, FakeReplyChatModel
from app.engine.gateway.breakers import MemoryBreaker, RedisBreaker
from app.engine.gateway.cache import CacheStore
from app.engine.gateway.faults import FaultInjector
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate


def settings(**overrides) -> Settings:
    overrides.setdefault("aegis_fake_llm", True)
    return Settings(_env_file=None, **overrides)


@pytest.fixture
async def http_client():
    client = httpx2.AsyncClient()
    yield client
    await client.aclose()


def parts_without_infra(s: Settings, http_client):
    return build_gateway_parts(
        s, http_client=http_client, redis=None, session_factory=None
    )


# ---------------------------------------------------------------- 装配


async def test_fake_mode_streams_reply_end_to_end(http_client):
    gw = gateway_for(parts_without_infra(settings(), http_client), "tA")
    chunks = [str(c.content) async for c in gw.astream([HumanMessage("你好")])]
    assert "".join(chunks) == FAKE_REPLY
    assert (await gw.ainvoke([HumanMessage("你好")])).content == FAKE_REPLY


def test_no_redis_means_memory_breaker_and_no_cache(http_client):
    parts = parts_without_infra(settings(), http_client)
    assert isinstance(parts.breaker, MemoryBreaker)
    assert parts.cache_store is None
    assert parts.meter is None
    gw = gateway_for(parts, "tA")
    assert gw.reply_cache is None and gw.meter is None


def test_redis_means_redis_breaker_and_cache_store(http_client, redis_async):
    parts = build_gateway_parts(
        settings(), http_client=http_client, redis=redis_async, session_factory=None
    )
    assert isinstance(parts.breaker, RedisBreaker)
    assert isinstance(parts.cache_store, CacheStore)


def test_cache_ttl_zero_disables_reply_cache(http_client, redis_async):
    parts = build_gateway_parts(
        settings(cache_ttl_seconds=0),
        http_client=http_client,
        redis=redis_async,
        session_factory=None,
    )
    assert parts.cache_store is None
    assert gateway_for(parts, "tA").reply_cache is None
    # 框架缓存与租户缓存是两回事，前者永远关（ainvoke 内部流式会查它，探针㉑）
    assert AegisGateway.model_fields["cache"].default is False


def test_two_tenants_share_components_but_bind_their_own_identity(
    http_client, redis_async
):
    parts = build_gateway_parts(
        settings(), http_client=http_client, redis=redis_async, session_factory=None
    )
    a, b = gateway_for(parts, "tA"), gateway_for(parts, "tB")
    assert (
        a.breaker is b.breaker
        and a.limiter is b.limiter
        and a.tenant_limiter is b.tenant_limiter
    )
    assert a.meter is b.meter
    # pydantic 校验会重建 dict 容器，但候选实例（连接池/注入器）是同一批对象
    assert a.models.keys() == b.models.keys()
    assert all(a.models[c] is b.models[c] for c in a.models)
    assert a.routes == b.routes
    assert (a.tenant_id, b.tenant_id) == ("tA", "tB")
    assert a.reply_cache is not b.reply_cache
    assert a.reply_cache.key("d").startswith("aegis:cache:v1:tA:")
    assert b.reply_cache.key("d").startswith("aegis:cache:v1:tB:")


@pytest.mark.parametrize("bad", ["tA:evil", "", "租户甲", "a" * 65])
def test_illegal_tenant_ids_rejected_at_entry(http_client, bad):
    parts = parts_without_infra(settings(), http_client)
    with pytest.raises(ValueError, match="tenant_id"):
        gateway_for(parts, bad)


@pytest.mark.parametrize("good", ["tA", "tenant-1", "T_x-9", "a" * 64])
def test_legal_tenant_ids_pass_at_entry(http_client, good):
    assert (
        gateway_for(parts_without_infra(settings(), http_client), good).tenant_id
        == good
    )


def test_settings_are_wired_into_gateway(http_client):
    parts = parts_without_infra(
        settings(tenant_monthly_token_budget=123, request_token_budget=45), http_client
    )
    gw = gateway_for(parts, "tA")
    assert (gw.monthly_token_budget, gw.request_token_budget) == (123, 45)
    assert gw.budget_resolver is None  # M1：静态配置；M3 接租户目录
    assert set(gw.routes) == {"fast", "standard", "strong"}
    assert gw.routes["fast"][0] == Candidate("bailian", "qwen-flash")


def test_fake_switch_selects_candidate_type(http_client):
    on = parts_without_infra(settings(aegis_fake_llm=True), http_client)
    assert all(isinstance(m, FakeReplyChatModel) for m in on.models.values())
    off = parts_without_infra(
        settings(aegis_fake_llm=False, dashscope_api_key="sk-x"), http_client
    )
    assert all(isinstance(m, ChatOpenAI) for m in off.models.values())


def test_fault_targets_are_wrapped_only_where_named(http_client):
    parts = parts_without_infra(
        settings(
            fault_injection_rate=0.5, fault_injection_targets=["bailian:qwen-plus"]
        ),
        http_client,
    )
    assert isinstance(parts.models[Candidate("bailian", "qwen-plus")], FaultInjector)
    assert isinstance(
        parts.models[Candidate("bailian", "qwen-flash")], FakeReplyChatModel
    )


# ---------------------------------------------------------------- 真件端到端


async def test_end_to_end_real_redis_and_ledger(
    http_client, redis_async, db_session_factory, namespace
):
    """fake 上游 + 真缓存 + 真账本：第一遍打上游记真账，第二遍命中缓存记零成本账，两行都带租户。"""
    parts = build_gateway_parts(
        settings(),
        http_client=http_client,
        redis=redis_async,
        session_factory=db_session_factory,
    )
    gw = gateway_for(parts, "tA")
    msgs = [HumanMessage(namespace)]
    first = await gw.ainvoke(msgs)
    second = await gw.ainvoke(msgs)
    assert first.content == second.content == FAKE_REPLY
    assert "aegis_cached" not in first.response_metadata
    assert second.response_metadata["aegis_cached"] is True
    async with db_session_factory() as s:
        rows = (
            (
                await s.execute(
                    select(UsageRecord)
                    .where(UsageRecord.tenant_id == "tA")
                    .order_by(UsageRecord.id)
                )
            )
            .scalars()
            .all()
        )
    assert [(r.provider, r.cached) for r in rows] == [
        ("bailian", False),
        (CACHE_PROVIDER, True),
    ]
    assert (
        rows[0].prompt_tokens == 120 and rows[1].prompt_tokens == 120
    )  # 命中行照记 token，成本归零
    assert rows[1].cost == 0
    assert rows[0].model == rows[1].model == "qwen-plus"  # 默认档 standard 的首选
    # 别的租户同一问题必 miss：再打一遍上游、再记一行真账
    other = gateway_for(parts, "tB")
    assert "aegis_cached" not in (await other.ainvoke(msgs)).response_metadata
