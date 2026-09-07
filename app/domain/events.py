"""事件事实源：events 表 ORM + EventStore（ADR-011 决策 3/4/6/7）。不 import engine。

写路径三条纪律：
- 事件 id 由调用方派生（engine 的 event_id：稳定任务身份 → uuid5），本模块只做 `ON CONFLICT (id) DO NOTHING`——
  重放同一步骤命中既有行即去重，返回既有 seq 并标记 created=False；
- seq 由同一条 INSERT 里的标量子查询 `max(seq)+1` 播种（单写者；会话锁归后续里程碑），`(session_id, seq)` 唯一约束是
  并发写入的最后防线：撞上即围栏异常，终态不重试；
- 连接级故障退避 (0.1, 0.2, 0.4) 三次，耗尽抛不可用；其余异常（ProgrammingError / DataError…）是 bug 信号裸抛。
payload 存原文（JSONB）；created_at 用数据库时钟（事件不带应用侧时间戳）；全表 tenant_id。
engine 的 EventSink / EventSource 协议（签名只用内建类型）由本类靠结构匹配实现。
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    ScalarSelect,
    String,
    UniqueConstraint,
    Uuid,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

RETRY_BACKOFF_S: tuple[float, ...] = (0.1, 0.2, 0.4)
"""瞬态故障的退避序列：共 3 次重试，总额外等待约 0.7s。不做抖动：单写者没有雷群竞争面。"""


class EventStoreUnavailable(RuntimeError):
    """PG 瞬态故障重试耗尽：事实源不可用 = 服务不可用，终止本次 run。"""


class EventWriteFenced(RuntimeError):
    """围栏信号：(session_id, seq) 被别的写者占用——本写者的会话所有权已旁落。终态，绝不退避重试。"""


class EventRecord(Base):
    """events：审计 / 回放 / SSE 游标的事实源（图恢复归 checkpoint，ADR-011）。"""

    __tablename__ = "events"
    __table_args__ = (
        # 并发写入的最后防线：会话锁是第一防线，锁失效时数据库物理兜底；也是按会话顺序读的索引
        UniqueConstraint("session_id", "seq", name="uq_events_session_seq"),
        # 租户维度审计（按时间翻页）
        Index("ix_events_tenant_created", "tenant_id", "created_at"),
    )

    # 派生 uuid5（engine.runtime.events.event_id）：幂等键在副作用之前就存在，自增 id 给不了
    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(64))
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    schema_version: Mapped[int] = mapped_column(Integer)
    # 框架坐标：回指 checkpoint（图外写入的事件为空）
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def _row_dict(row: EventRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "session_id": row.session_id,
        "run_id": row.run_id,
        "seq": row.seq,
        "type": row.type,
        "payload": row.payload,
        "schema_version": row.schema_version,
        "task_id": row.task_id,
        "checkpoint_id": row.checkpoint_id,
    }


class EventStore:
    """事件落盘与读取。append 返回即已 durably committed——write-ahead"落盘是副作用的前置"由此成立。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sf = session_factory
        self._sleep = sleep

    def _next_seq(self, session_id: str) -> ScalarSelect[int]:
        """seq 播种：会话内 max(seq)+1，与 INSERT 同一语句求值（测试用子类改写它来制造围栏）。"""
        return (
            select(func.coalesce(func.max(EventRecord.seq), 0) + 1)
            .where(EventRecord.session_id == session_id)
            .scalar_subquery()
        )

    async def append(
        self,
        *,
        event_id: str,
        tenant_id: str,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> tuple[int, bool]:
        """写一条事实。返回 (seq, created)：created=False 即同 id 已在（重放去重命中）。"""
        stmt = (
            insert(EventRecord)
            .values(
                id=event_id,
                tenant_id=tenant_id,
                session_id=session_id,
                run_id=run_id,
                seq=self._next_seq(session_id),
                type=event_type,
                payload=dict(payload),
                schema_version=1,
                task_id=task_id,
                checkpoint_id=checkpoint_id,
            )
            .on_conflict_do_nothing(index_elements=["id"])
            .returning(EventRecord.seq)
        )
        attempt = 0
        while True:
            try:
                async with self._sf.begin() as s:
                    seq = (await s.execute(stmt)).scalar_one_or_none()
                    if seq is not None:
                        return seq, True
                    # 命中既有 id：DO NOTHING 不返回行（探针⑸），二次 SELECT 取既有 seq
                    existing = (
                        await s.execute(
                            select(EventRecord.seq).where(EventRecord.id == event_id)
                        )
                    ).scalar_one()
                    return existing, False
            except IntegrityError as e:
                # 能撞的只剩 (session_id, seq)：id 冲突已被 DO NOTHING 吸收——别的写者占了我的 seq
                raise EventWriteFenced(
                    f"session={session_id} 的 seq 已被其他写者占用——所有权旁落，本 run 应自毁"
                ) from e
            except (OperationalError, InterfaceError, OSError) as e:
                # 可重试白名单：连接级故障才配重试；OSError 族在池建连期未经 SQLAlchemy 包装裸穿
                if attempt >= len(RETRY_BACKOFF_S):
                    raise EventStoreUnavailable(
                        f"事件写入重试 {len(RETRY_BACKOFF_S)} 次仍失败——事实源不可用，终止本次 run"
                    ) from e
                await self._sleep(RETRY_BACKOFF_S[attempt])
                attempt += 1

    async def read(
        self,
        tenant_id: str,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按 seq 升序读会话事件（SSE 的 after_seq 游标语义；D8 种子与恢复分诊也走这里）。

        tenant_id 与 session_id 双过滤：会话不属于该租户时得到空列表而不是别人的事件。
        """
        stmt = (
            select(EventRecord)
            .where(
                EventRecord.tenant_id == tenant_id,
                EventRecord.session_id == session_id,
                EventRecord.seq > after_seq,
            )
            .order_by(EventRecord.seq)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._sf() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [_row_dict(row) for row in rows]
