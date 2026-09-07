"""事件事实源（M2.2；ADR-011）：append 的 seq 播种与原文往返、DB 时钟与框架坐标、同 id 去重返回既有 seq、跨 run 接续、
按会话计数、围栏终态零重试、白名单退避重试与耗尽、bug 类异常裸抛、read 的 after_seq / 租户双过滤 / limit、非 uuid id 拒绝。
真 PG（回滚夹具）；故障用可编程会话工厂，不真睡。"""

import uuid
from typing import Any

import pytest
from sqlalchemy import func, literal, select
from sqlalchemy.exc import (
    DBAPIError,
    OperationalError,
    ProgrammingError,
    StatementError,
)

pytest.importorskip(
    "app.domain.events", reason="M2.2 未敲：app/domain/events.py 不存在"
)

from app.domain.events import (
    RETRY_BACKOFF_S,
    EventRecord,
    EventStore,
    EventStoreUnavailable,
    EventWriteFenced,
)


def _eid() -> str:
    return str(uuid.uuid4())


def _sid() -> str:
    return f"s-{uuid.uuid4().hex[:8]}"


async def _append(store: EventStore, sid: str, eid: str | None = None, **kw: Any):
    kw.setdefault("tenant_id", "t-a")
    kw.setdefault("run_id", "r-1")
    kw.setdefault("event_type", "llm_call")
    kw.setdefault("payload", {})
    return await store.append(event_id=eid or _eid(), session_id=sid, **kw)


async def _count(factory, sid: str) -> int:
    async with factory() as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(EventRecord)
                .where(EventRecord.session_id == sid)
            )
        ).scalar_one()


def _sleep_recorder(into: list[float]):
    async def _sleep(delay: float) -> None:
        into.append(delay)

    return _sleep


class _FlakyFactory:
    """前 fail_times 次 begin() 抛连接级故障，之后透传真工厂——重试路径的手术刀。"""

    def __init__(self, real, fail_times: int, exc: type[Exception] = OperationalError):
        self.real = real
        self.remaining = fail_times
        self.exc = exc
        self.calls = 0

    def _raise(self) -> None:
        if issubclass(self.exc, DBAPIError):
            raise self.exc("boom", None, RuntimeError("connection refused"))
        raise self.exc("connection refused")

    def begin(self):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            self._raise()
        return self.real.begin()

    def __call__(self):
        return self.real()


class _StaleSeqStore(EventStore):
    """模拟"读到旧 max"的并发写者：seq 恒算成 1。"""

    def _next_seq(self, session_id: str):
        return literal(1)


# ---------------------------------------------------------------- append


async def test_append_assigns_sequential_seq_and_persists_verbatim(db_session_factory):
    store = EventStore(db_session_factory)
    sid, eid = _sid(), _eid()
    payload = {"content": "你好", "nested": {"金额": 30, "items": [1, 2]}}
    assert await _append(
        store,
        sid,
        eid,
        event_type="user_message",
        payload=payload,
        task_id="task-1",
        checkpoint_id="ck-1",
    ) == (1, True)
    assert (await _append(store, sid))[0] == 2
    assert (await _append(store, sid))[0] == 3
    async with db_session_factory() as s:
        row = (
            await s.execute(select(EventRecord).where(EventRecord.id == eid))
        ).scalar_one()
    assert (row.seq, row.type, row.schema_version) == (1, "user_message", 1)
    assert row.payload == payload  # JSONB 原文往返
    assert (row.tenant_id, row.run_id) == ("t-a", "r-1")
    assert (row.task_id, row.checkpoint_id) == ("task-1", "ck-1")
    assert row.created_at is not None and row.created_at.tzinfo is not None  # DB 时钟


async def test_same_id_twice_is_a_dedupe_hit(db_session_factory):
    """重放去重（ADR-011 决策 3）：同 id 第二次不落行、返回既有 seq、created=False，后续 seq 不跳号。"""
    store = EventStore(db_session_factory)
    sid, eid = _sid(), _eid()
    assert await _append(store, sid, eid, payload={"n": 1}) == (1, True)
    assert await _append(store, sid, eid, payload={"n": 1}) == (1, False)
    assert await _append(store, sid) == (2, True)
    assert await _count(db_session_factory, sid) == 2


async def test_seq_continues_across_runs(db_session_factory):
    store = EventStore(db_session_factory)
    sid = _sid()
    await _append(store, sid, run_id="r-1")
    await _append(store, sid, run_id="r-1")
    assert await _append(store, sid, run_id="r-2") == (3, True)


async def test_seq_is_scoped_per_session(db_session_factory):
    store = EventStore(db_session_factory)
    a, b = _sid(), _sid()
    assert await _append(store, a) == (1, True)
    assert await _append(store, b) == (1, True)
    assert await _append(store, a) == (2, True)


async def test_fence_is_terminal_without_retry(db_session_factory):
    """围栏：算出的 seq 已被别的 id 占用 → EventWriteFenced，零退避零重试（唯一约束兜底，探针⑸）。"""
    slept: list[float] = []
    store = _StaleSeqStore(db_session_factory, sleep=_sleep_recorder(slept))
    sid = _sid()
    assert await _append(store, sid) == (1, True)
    with pytest.raises(EventWriteFenced, match=sid):
        await _append(store, sid)
    assert slept == []


async def test_transient_failure_retries_then_succeeds(db_session_factory):
    slept: list[float] = []
    flaky = _FlakyFactory(db_session_factory, fail_times=2)
    store = EventStore(flaky, sleep=_sleep_recorder(slept))  # type: ignore[arg-type]
    assert await _append(store, _sid()) == (1, True)
    assert slept == [0.1, 0.2]


async def test_retry_exhaustion_raises_unavailable(db_session_factory):
    slept: list[float] = []
    flaky = _FlakyFactory(db_session_factory, fail_times=99)
    store = EventStore(flaky, sleep=_sleep_recorder(slept))  # type: ignore[arg-type]
    with pytest.raises(EventStoreUnavailable, match="终止本次 run"):
        await _append(store, _sid())
    assert slept == list(RETRY_BACKOFF_S)
    assert flaky.calls == 1 + len(RETRY_BACKOFF_S)


async def test_os_level_connection_error_retries_like_transient(db_session_factory):
    """池建连期的 OS 级错误未经 SQLAlchemy 包装裸穿——与包装后的连接级故障同待遇。"""
    slept: list[float] = []
    flaky = _FlakyFactory(db_session_factory, fail_times=2, exc=ConnectionRefusedError)
    store = EventStore(flaky, sleep=_sleep_recorder(slept))  # type: ignore[arg-type]
    assert await _append(store, _sid()) == (1, True)
    assert slept == [0.1, 0.2]


async def test_bug_class_errors_propagate_without_retry(db_session_factory):
    """白名单哲学：ProgrammingError 是 bug 信号——SQL 写错了重试三次不会试对。"""
    slept: list[float] = []
    flaky = _FlakyFactory(db_session_factory, fail_times=99, exc=ProgrammingError)
    store = EventStore(flaky, sleep=_sleep_recorder(slept))  # type: ignore[arg-type]
    with pytest.raises(ProgrammingError):
        await _append(store, _sid())
    assert slept == [] and flaky.calls == 1


async def test_non_uuid_id_rejected_loudly(db_session_factory):
    """id 列是 uuid：非派生 id 进不来（图外写入也必须走 event_id）。"""
    store = EventStore(db_session_factory)
    with pytest.raises(StatementError):
        await _append(store, _sid(), eid="not-a-uuid")


# ---------------------------------------------------------------- read


async def test_read_orders_by_seq_and_returns_agent_event_shape(db_session_factory):
    store = EventStore(db_session_factory)
    sid = _sid()
    ids = [_eid() for _ in range(3)]
    for i, eid in enumerate(ids):
        await _append(store, sid, eid, payload={"i": i}, task_id=f"task-{i}")
    rows = await store.read("t-a", sid)
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert [r["id"] for r in rows] == ids
    assert set(rows[0]) == {
        "id",
        "tenant_id",
        "session_id",
        "run_id",
        "seq",
        "type",
        "payload",
        "schema_version",
        "task_id",
        "checkpoint_id",
    }
    assert rows[1]["payload"] == {"i": 1} and rows[1]["task_id"] == "task-1"


async def test_read_after_seq_and_limit(db_session_factory):
    store = EventStore(db_session_factory)
    sid = _sid()
    for _ in range(4):
        await _append(store, sid)
    assert [r["seq"] for r in await store.read("t-a", sid, after_seq=2)] == [3, 4]
    assert [r["seq"] for r in await store.read("t-a", sid, limit=2)] == [1, 2]
    assert await store.read("t-a", sid, after_seq=4) == []


async def test_read_is_tenant_scoped(db_session_factory):
    """会话不属于该租户 → 空列表，不是别人的事件（每个边界调用点核对 tenant_id）。"""
    store = EventStore(db_session_factory)
    sid = _sid()
    await _append(store, sid, tenant_id="t-a")
    assert len(await store.read("t-a", sid)) == 1
    assert await store.read("t-b", sid) == []
