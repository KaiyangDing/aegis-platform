"""计量账本（M1.5b）：价目表 Decimal / 成本纯函数 / ORM 往返与 DB 时钟 / 回滚夹具 / 记账员四种行 / 月度聚合 /
迁移与 ORM 零漂移。DB 测试需本机 Postgres（aegis_test 由 conftest 经 alembic 建表）。"""

from decimal import Decimal

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import select
from structlog.testing import capture_logs

pytest.importorskip("app.domain.usage", reason="M1.5b 未敲：app/domain/usage.py 不存在")

from app.core.db import Base
from app.domain.usage import (
    CACHE_PROVIDER,
    EMBEDDING_TIER,
    LOG_PRICE_MISSING,
    MeteringRecorder,
    PriceTable,
    UsageRecord,
    compute_cost,
    price_table,
)
from app.engine.gateway.protocols import MeterLike

PRICES: PriceTable = {"qwen-flash": (Decimal("0.00015"), Decimal("0.0015"))}


def make_record(request_id: str, tenant: str = "t1") -> UsageRecord:
    return UsageRecord(
        request_id=request_id,
        tenant_id=tenant,
        tier="fast",
        provider="bailian",
        model="qwen-flash",
        prompt_tokens=10,
        completion_tokens=5,
        cached=False,
        usage_missing=False,
        cost=Decimal("0.000123"),
    )


# ---------------------------------------------------------------- 纯函数


def test_price_table_converts_via_str_not_binary_float():
    table = price_table({"m": [0.1, 0.2]})
    assert table == {"m": (Decimal("0.1"), Decimal("0.2"))}
    assert Decimal(0.1) != Decimal("0.1")  # noqa: RUF032  直接 Decimal(float) 会把二进制尾差带进账本


def test_compute_cost_known_model():
    assert compute_cost(
        "qwen-flash", 1000, 2000, cached=False, prices=PRICES
    ) == Decimal("0.00315")  # 1k×0.00015 + 2k×0.0015


def test_compute_cost_cached_is_free():
    assert compute_cost(
        "qwen-flash", 1000, 2000, cached=True, prices=PRICES
    ) == Decimal(0)


def test_compute_cost_unknown_model_zero_but_loud():
    with capture_logs() as logs:
        cost = compute_cost("gpt-999", 1000, 1000, cached=False, prices=PRICES)
    assert cost == Decimal(0)
    assert [log["event"] for log in logs] == [
        LOG_PRICE_MISSING
    ]  # 静默记零是财务事故，必须喊
    assert logs[0]["model"] == "gpt-999"


def test_recorder_matches_meter_like_protocol():
    assert isinstance(MeteringRecorder(None, PRICES), MeterLike)  # type: ignore[arg-type]


# ---------------------------------------------------------------- ORM 与夹具


async def test_usage_record_roundtrip(db_session):
    rec = make_record("req-roundtrip")
    db_session.add(rec)
    await db_session.flush()  # 发 INSERT 拿自增 id，但不 commit（夹具最终回滚）
    await db_session.refresh(rec)  # 回读服务端填充的列
    assert rec.id is not None
    assert (
        rec.created_at is not None and rec.created_at.tzinfo is not None
    )  # DB 时钟、带时区
    got = (
        await db_session.execute(
            select(UsageRecord).where(UsageRecord.request_id == "req-roundtrip")
        )
    ).scalar_one()
    assert got.cost == Decimal("0.000123")  # Decimal 无损往返——用 float 这里会开始出鬼


async def test_rollback_isolation_first(db_session):
    # 与下一个测试插入完全相同的 request_id：两个都过 = 回滚夹具真的在工作
    db_session.add(make_record("iso-proof"))
    await db_session.flush()


async def test_rollback_isolation_second(db_session):
    db_session.add(make_record("iso-proof"))
    await db_session.flush()


async def test_migration_matches_orm_metadata(db_conn):
    """迁移是被测物：手写的 0001 与 ORM 元数据零漂移（列/类型/索引）。"""

    def diff(sync_conn):
        return compare_metadata(MigrationContext.configure(sync_conn), Base.metadata)

    assert await db_conn.run_sync(diff) == []


# ---------------------------------------------------------------- 记账员


async def fetch(factory, request_id: str) -> UsageRecord:
    async with factory() as s:
        return (
            await s.execute(
                select(UsageRecord).where(UsageRecord.request_id == request_id)
            )
        ).scalar_one()


async def test_recorder_writes_priced_row(db_session_factory):
    rec = MeteringRecorder(db_session_factory, PRICES)
    await rec.record(
        tenant_id="t-meter",
        request_id="r1",
        session_id="s1",
        tier="fast",
        provider="bailian",
        model="qwen-flash",
        prompt_tokens=1000,
        completion_tokens=2000,
        cached=False,
        usage_missing=False,
    )
    row = await fetch(db_session_factory, "r1")
    assert row.cost == Decimal("0.00315")
    assert (row.tenant_id, row.provider, row.tier, row.session_id) == (
        "t-meter",
        "bailian",
        "fast",
        "s1",
    )
    assert (row.cached, row.usage_missing) == (False, False)


async def test_recorder_cached_row_costs_zero(db_session_factory):
    rec = MeteringRecorder(db_session_factory, PRICES)
    await rec.record(
        tenant_id="t-meter",
        request_id="r2",
        session_id=None,
        tier="fast",
        provider=CACHE_PROVIDER,
        model="qwen-flash",
        prompt_tokens=1000,
        completion_tokens=2000,
        cached=True,
        usage_missing=False,
    )
    row = await fetch(db_session_factory, "r2")
    assert row.cost == Decimal(0)
    assert row.cached is True


async def test_recorder_marks_usage_missing(db_session_factory):
    rec = MeteringRecorder(db_session_factory, PRICES)
    await rec.record(
        tenant_id="t-meter",
        request_id="r3",
        session_id=None,
        tier="fast",
        provider="bailian",
        model="qwen-flash",
        prompt_tokens=0,
        completion_tokens=0,
        cached=False,
        usage_missing=True,
    )
    row = await fetch(db_session_factory, "r3")
    assert row.usage_missing is True  # 0 是"缺"不是"免费"
    assert row.cost == Decimal(0)


async def test_month_spend_counts_only_real_calls_of_this_tenant(db_session_factory):
    rec = MeteringRecorder(db_session_factory, PRICES)
    common = {
        "session_id": None,
        "tier": "fast",
        "model": "qwen-flash",
        "usage_missing": False,
    }
    await rec.record(
        tenant_id="t-x",
        request_id="a",
        provider="bailian",
        prompt_tokens=1000,
        completion_tokens=2000,
        cached=False,
        **common,
    )
    await rec.record(  # 缓存回放：不计入预算
        tenant_id="t-x",
        request_id="b",
        provider=CACHE_PROVIDER,
        prompt_tokens=500,
        completion_tokens=500,
        cached=True,
        **common,
    )
    await rec.record(  # 别的租户：不计入
        tenant_id="t-y",
        request_id="c",
        provider="bailian",
        prompt_tokens=9000,
        completion_tokens=0,
        cached=False,
        **common,
    )
    assert await rec.month_spend("t-x") == 3000
    assert await rec.month_spend("t-nobody") == 0


async def test_record_embedding_row(db_session_factory):
    rec = MeteringRecorder(db_session_factory, price_table({"emb": [0.5, 0.0]}))
    await rec.record_embedding(
        tenant_id="t-emb",
        request_id="e1",
        model="emb",
        prompt_tokens=2000,
        provider="bailian",
    )
    row = await fetch(db_session_factory, "e1")
    assert (row.tier, row.completion_tokens, row.cached) == (EMBEDDING_TIER, 0, False)
    assert row.cost == Decimal(1)
    assert await rec.month_spend("t-emb") == 2000  # 真实花销，计入月度预算
