"""会话调度状态（M2.2）：值快照、建行与读身份、五翻转 CAS（含输家 False、无行 False、非法值拒绝）、恢复计数、updated_at 随翻转移动。
真 PG（回滚夹具）。"""

import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

pytest.importorskip(
    "app.domain.sessions", reason="M2.2 未敲：app/domain/sessions.py 不存在"
)

from app.domain.sessions import (
    RECOVERY_LIMIT,
    RUN_STATES,
    SessionRecord,
    SessionStateStore,
)


def _sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


async def _force(factory, sid: str, state: str) -> None:
    """测试用后门：直接把行摆到某个状态（生产代码没有"直接 SET"的入口）。"""
    async with factory.begin() as s:
        await s.execute(
            update(SessionRecord).where(SessionRecord.id == sid).values(run_state=state)
        )


def test_run_states_and_recovery_limit_snapshot():
    assert RUN_STATES == ("idle", "running", "awaiting_approval", "failed")
    assert RECOVERY_LIMIT == 3


async def test_create_then_get_defaults(db_session_factory):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")
    assert await store.get(sid) == {
        "id": sid,
        "tenant_id": "t-a",
        "user_id": "u-1",
        "run_state": "idle",
        "recovery_count": 0,
    }
    assert await store.get("no-such-session") is None
    async with db_session_factory() as s:
        row = (
            await s.execute(select(SessionRecord).where(SessionRecord.id == sid))
        ).scalar_one()
    assert row.created_at.tzinfo is not None and row.updated_at is not None


async def test_create_duplicate_rejected(db_session_factory):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")
    with pytest.raises(IntegrityError):
        await store.create(sid, tenant_id="t-a", user_id="u-1")


@pytest.mark.parametrize(
    ("expected", "to"),
    [
        ("idle", "running"),  # T1 起跑
        ("running", "awaiting_approval"),  # T2 挂起
        ("awaiting_approval", "running"),  # T3 恢复
        ("running", "idle"),  # T4 终止
        ("running", "failed"),  # T5 恢复放弃
    ],
)
async def test_five_transitions_succeed_from_expected_state(
    db_session_factory, expected: str, to: str
):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")
    await _force(db_session_factory, sid, expected)
    assert await store.transition(sid, expected=expected, to=to) is True
    assert (await store.get(sid))["run_state"] == to


async def test_cas_loser_gets_false_and_state_untouched(db_session_factory):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")
    assert await store.transition(sid, expected="idle", to="running") is True
    assert await store.transition(sid, expected="idle", to="running") is False
    assert (await store.get(sid))["run_state"] == "running"
    assert await store.transition(sid, expected="running", to="idle") is True


async def test_transition_on_missing_session_is_false(db_session_factory):
    store = SessionStateStore(db_session_factory)
    assert await store.transition("no-such", expected="idle", to="running") is False


@pytest.mark.parametrize("bad", ["flying", "", "IDLE"])
async def test_unknown_state_rejected_before_touching_db(db_session_factory, bad: str):
    store = SessionStateStore(db_session_factory)
    with pytest.raises(ValueError, match="run_state 非法"):
        await store.transition("x", expected=bad, to="running")
    with pytest.raises(ValueError, match="run_state 非法"):
        await store.transition("x", expected="idle", to=bad)


async def test_recovery_count_bump_and_reset(db_session_factory):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")
    assert await store.bump_recovery(sid) == 1
    assert await store.bump_recovery(sid) == 2
    assert (await store.get(sid))["recovery_count"] == 2
    await store.reset_recovery(sid)
    assert (await store.get(sid))["recovery_count"] == 0
    assert await store.bump_recovery("no-such") is None


async def test_updated_at_moves_on_transition(db_session_factory):
    store = SessionStateStore(db_session_factory)
    sid = _sid()
    await store.create(sid, tenant_id="t-a", user_id="u-1")

    async def stamp():
        async with db_session_factory() as s:
            return (
                await s.execute(
                    select(SessionRecord.updated_at).where(SessionRecord.id == sid)
                )
            ).scalar_one()

    before = await stamp()
    await store.transition(sid, expected="idle", to="running")
    assert await stamp() >= before


async def test_list_ids_by_tenant(db_session_factory):
    store = SessionStateStore(db_session_factory)
    tenant = f"t-{uuid.uuid4().hex[:6]}"
    a, b = _sid(), _sid()
    await store.create(a, tenant_id=tenant, user_id="u-1")
    await store.create(b, tenant_id=tenant, user_id="u-2")
    await store.create(_sid(), tenant_id="t-other", user_id="u-9")
    assert set(await store.list_ids(tenant)) == {a, b}
