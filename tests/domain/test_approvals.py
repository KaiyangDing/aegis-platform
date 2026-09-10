"""审批单五态 CAS（M2.7 ApprovalStore；ADR-013 决策 4）：幂等开单、双坐席恰一赢家、到期批准被拒（fail-closed）、cancel 不查过期、
expire_due 只碰到期 pending 且可注入时钟、attach_event 恰一次、读单 / 按会话列单、非 uuid 拒绝。真 PG（回滚夹具）。"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, StatementError

pytest.importorskip(
    "app.domain.approvals", reason="M2.2 未敲：app/domain/approvals.py 不存在"
)

from app.domain import approvals as approvals_mod

if not hasattr(approvals_mod, "ApprovalStore"):
    pytest.skip("M2.7 未敲：approvals.py 尚无 ApprovalStore", allow_module_level=True)

from app.domain.approvals import APPROVAL_STATUSES, ApprovalStore


def _aid() -> str:
    return str(uuid.uuid4())


async def _open(
    store: ApprovalStore,
    aid: str | None = None,
    *,
    ttl_s: float = 3600.0,
    session_id: str = "s-ap",
    tenant_id: str = "t-a",
) -> dict:
    return await store.create(
        approval_id=aid or _aid(),
        tenant_id=tenant_id,
        session_id=session_id,
        run_id="r-1",
        tool_name="demo_refund_apply",
        args={"order_id": "1024", "amount": 350},
        ttl_s=ttl_s,
    )


async def _db_now(factory) -> datetime:
    async with factory() as s:
        return (await s.execute(select(func.now()))).scalar_one()


def test_statuses_snapshot():
    assert APPROVAL_STATUSES == (
        "pending",
        "approved",
        "rejected",
        "cancelled",
        "expired",
    )


async def test_create_is_idempotent_and_uses_db_clock(db_session_factory):
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    first = await _open(store, aid, ttl_s=3600)
    assert first["created"] is True and first["status"] == "pending"
    assert first["operator_id"] is None and first["event_id"] is None
    assert first["decided_at"] is None and first["args"] == {
        "order_id": "1024",
        "amount": 350,
    }
    expires = datetime.fromisoformat(first["expires_at"])
    assert expires.tzinfo is not None
    assert abs(
        (expires - (await _db_now(db_session_factory))) - timedelta(hours=1)
    ) < timedelta(seconds=5)
    second = await _open(
        store, aid, ttl_s=1
    )  # 重放：同 id 再开 → 命中既有单，TTL 不重算
    assert second["created"] is False and second["expires_at"] == first["expires_at"]
    assert len(await store.list_for_session("t-a", "s-ap")) == 1


async def test_get_returns_view_or_none(db_session_factory):
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    await _open(store, aid)
    got = await store.get(aid)
    assert got is not None and got["id"] == aid and got["status"] == "pending"
    assert "created" not in got
    assert await store.get(_aid()) is None


async def test_decide_approves_and_rejects_pending(db_session_factory):
    store = ApprovalStore(db_session_factory)
    a, b = _aid(), _aid()
    await _open(store, a)
    await _open(store, b)
    assert await store.decide(a, approved=True, operator_id="op-1") is True
    assert await store.decide(b, approved=False, operator_id="op-2") is True
    ra, rb = await store.get(a), await store.get(b)
    assert (
        ra["status"] == "approved" and ra["operator_id"] == "op-1" and ra["decided_at"]
    )
    assert rb["status"] == "rejected" and rb["operator_id"] == "op-2"


async def test_second_decision_loses_and_never_overwrites(db_session_factory):
    """双坐席同点：赢家恰一个，输家的决定绝不覆盖赢家。"""
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    await _open(store, aid)
    assert await store.decide(aid, approved=True, operator_id="op-快") is True
    assert await store.decide(aid, approved=False, operator_id="op-慢") is False
    row = await store.get(aid)
    assert row["status"] == "approved" and row["operator_id"] == "op-快"


async def test_decide_refuses_expired_fail_closed(db_session_factory):
    """到期 fail-closed：过期单拒绝翻转——哪怕坐席点了批准，归宿只有 expire_due。"""
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    await _open(store, aid, ttl_s=-1.0)  # 已过期的 pending
    assert await store.decide(aid, approved=True, operator_id="op-1") is False
    row = await store.get(aid)
    assert row["status"] == "pending" and row["decided_at"] is None


async def test_cancel_pending_ignores_expiry_and_loses_after_decision(
    db_session_factory,
):
    store = ApprovalStore(db_session_factory)
    expired, decided = _aid(), _aid()
    await _open(store, expired, ttl_s=-1.0)
    assert await store.cancel(expired) is True  # 撤回不查过期
    assert (await store.get(expired))["status"] == "cancelled"
    await _open(store, decided)
    assert await store.decide(decided, approved=True, operator_id="op-1") is True
    assert await store.cancel(decided) is False  # 终态不许被撤回改写
    assert (await store.get(decided))["status"] == "approved"


async def test_expire_due_flips_only_due_pending(db_session_factory):
    """到期扫描只碰"pending 且已到期"：未到期的、已终态的都不许动（过滤式断言：只对本测试的单下结论）。"""
    store = ApprovalStore(db_session_factory)
    due, future, gone = _aid(), _aid(), _aid()
    await _open(store, due, ttl_s=-1.0)
    await _open(store, future, ttl_s=36000)
    await _open(store, gone, ttl_s=-1.0)
    await store.cancel(gone)
    flipped = await store.expire_due()
    assert due in flipped and future not in flipped and gone not in flipped
    assert (await store.get(due))["status"] == "expired"
    assert (await store.get(due))["decided_at"] is not None
    assert (await store.get(future))["status"] == "pending"
    assert (await store.get(gone))["status"] == "cancelled"


async def test_expire_due_with_injected_clock(db_session_factory):
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    await _open(store, aid, ttl_s=3600)
    assert aid not in await store.expire_due()  # DB 时钟视角：还活着
    flipped = await store.expire_due(now=datetime.now(UTC) + timedelta(hours=2))
    assert aid in flipped
    assert (await store.get(aid))["status"] == "expired"


async def test_attach_event_exactly_once(db_session_factory):
    store = ApprovalStore(db_session_factory)
    aid = _aid()
    await _open(store, aid)
    eid = _aid()
    assert await store.attach_event(aid, event_id=eid) is True
    assert (
        await store.attach_event(aid, event_id=_aid()) is False
    )  # 重放回填拿 False，不覆盖
    assert (await store.get(aid))["event_id"] == eid
    assert await store.attach_event(_aid(), event_id=eid) is False  # 无单


async def test_list_for_session_orders_and_filters_tenant(db_session_factory):
    store = ApprovalStore(db_session_factory)
    sid = f"s-{uuid.uuid4().hex[:8]}"
    # 回滚夹具里所有语句在同一外层事务内：created_at（now() 是事务时钟）相同，次序退回 id——生产里每次开单各自成事务
    first, second = sorted((_aid(), _aid()))
    await _open(store, first, session_id=sid)
    await _open(store, second, session_id=sid)
    await _open(store, session_id=sid, tenant_id="t-b")
    rows = await store.list_for_session("t-a", sid)
    assert [r["id"] for r in rows] == [first, second]
    assert await store.list_for_session("t-c", sid) == []


async def test_non_uuid_id_rejected(db_session_factory):
    store = ApprovalStore(db_session_factory)
    with pytest.raises((StatementError, DBAPIError)):
        await _open(store, "not-a-uuid")
