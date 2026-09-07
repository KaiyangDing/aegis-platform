"""会话调度状态：sessions 表 ORM + SessionStateStore（run_state 五翻转 CAS + recovery_count）。不 import engine。

run_state 存字符串列，合法值在 RUN_STATES 元组（engine 侧的 SessionRunState 枚举与之同值，测试互钉）；
不用 PG 原生 ENUM——加值只改代码不 ALTER TYPE。
翻转全部走 CAS：条件进 WHERE，输赢看 rowcount；没有"直接 SET"的入口。合法迁移图（T1 idle→running / T2 running→
awaiting_approval / T3 awaiting→running / T4 running→idle / T5 running→failed）由调用方（engine）以 expected 参数表达。
锁 / 租约 / reaper 归后续里程碑；本表只有恢复计数。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

RUN_STATES: tuple[str, ...] = ("idle", "running", "awaiting_approval", "failed")
"""sessions.run_state 合法值全集（值快照由测试钉死）。"""

RECOVERY_LIMIT = 3
"""同一会话崩溃恢复次数上限（前作 C9）：超过即 T5 failed + recovery_abandoned 事件（M2.9 消费）。"""


class SessionRecord(Base):
    """sessions：身份（tenant_id / user_id）+ 调度状态 + 恢复计数。"""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64))
    run_state: Mapped[str] = mapped_column(String(32), default="idle")
    recovery_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


def _check_state(value: str) -> str:
    if value not in RUN_STATES:
        raise ValueError(f"run_state 非法：{value!r}（合法值 {RUN_STATES}）")
    return value


class SessionStateStore:
    """sessions 行的原语层：建行 / 读身份 / CAS 翻转 / 恢复计数。实现 engine 的 SessionStateLike。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def create(self, session_id: str, *, tenant_id: str, user_id: str) -> None:
        """建会话行（M2 测试与 M3 API 的入口）；重复 id 由主键拒绝。"""
        async with self._sf.begin() as s:
            s.add(SessionRecord(id=session_id, tenant_id=tenant_id, user_id=user_id))

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """读身份与状态；无行返回 None（run 开头读行取身份，无行拒绝起跑）。"""
        async with self._sf() as s:
            row = await s.get(SessionRecord, session_id)
        if row is None:
            return None
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "user_id": row.user_id,
            "run_state": row.run_state,
            "recovery_count": row.recovery_count,
        }

    async def transition(self, session_id: str, *, expected: str, to: str) -> bool:
        """CAS：只有当前是 expected 才翻到 to；返回是否翻成（False = 输家或无行）。"""
        _check_state(expected)
        _check_state(to)
        stmt = (
            update(SessionRecord)
            .where(SessionRecord.id == session_id, SessionRecord.run_state == expected)
            .values(run_state=to)
        )
        async with self._sf.begin() as s:
            result = await s.execute(stmt)
        return result.rowcount == 1

    async def bump_recovery(self, session_id: str) -> int | None:
        """恢复计数 +1 并返回新值（原子 UPDATE … RETURNING）；无行返回 None。"""
        stmt = (
            update(SessionRecord)
            .where(SessionRecord.id == session_id)
            .values(recovery_count=SessionRecord.recovery_count + 1)
            .returning(SessionRecord.recovery_count)
        )
        async with self._sf.begin() as s:
            return (await s.execute(stmt)).scalar_one_or_none()

    async def reset_recovery(self, session_id: str) -> None:
        """正常终止即清零：上限管的是"连续"崩溃。"""
        stmt = (
            update(SessionRecord)
            .where(SessionRecord.id == session_id)
            .values(recovery_count=0)
        )
        async with self._sf.begin() as s:
            await s.execute(stmt)

    async def list_ids(self, tenant_id: str) -> list[str]:
        """租户下的会话 id（测试与后续管理端用）。"""
        stmt = select(SessionRecord.id).where(SessionRecord.tenant_id == tenant_id)
        async with self._sf() as s:
            return list((await s.execute(stmt)).scalars().all())
