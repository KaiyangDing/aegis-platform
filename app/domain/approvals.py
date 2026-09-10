"""审批单：approvals 表 ORM + ApprovalStore（五态 CAS；ADR-013 决策 4 / 7）。不 import engine。

不挂工具调用外键：审批发生在 write-ahead 之前，审批的是参数快照；执行后以 event_id 回填审计链（attach_event 恰一次）。
status 存字符串列，合法值在 APPROVAL_STATUSES（超时与撤回是一等状态，不是 rejected 的变体）。
三条纪律：
- 开单幂等：id 由 engine 派生自稳定任务身份（与事件 id 同一机制），INSERT … ON CONFLICT (id) DO NOTHING——
  节点重放命中既有单即返回其现状，绝不重复开单；
- 翻转全部 CAS：条件进 WHERE、输赢看 rowcount，没有"直接 SET"的入口；decide 查 pending 且未过期（到期 fail-closed），
  cancel 只查 pending（撤回已到期未清扫的单无害且语义更干净）；双坐席同点恰一赢家；
- 时钟一律数据库 now()：expires_at 的写入钟与 decide 的比较钟同源，无应用侧漂移；expire_due 可注入时钟（测试不等真实时间）。
engine 的 ApprovalStoreLike 协议（签名只用内建类型）由本类靠结构匹配实现；返回字典里的时间字段是 ISO 字符串
（进事件 payload 的值不带 datetime）。
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import DateTime, Index, String, Uuid, func, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

APPROVAL_STATUSES: tuple[str, ...] = (
    "pending",
    "approved",
    "rejected",
    "cancelled",
    "expired",
)
"""approvals.status 五态（值快照由测试钉死；engine 侧 ApprovalStatus 与之同值，测试互钉）。"""

PENDING, APPROVED, REJECTED, CANCELLED, EXPIRED = APPROVAL_STATUSES


class ApprovalRecord(Base):
    """approvals：一次风险闸门命中开一张单；expires_at 由 LoopPolicy.approval_ttl_s 与数据库时钟生成。"""

    __tablename__ = "approvals"
    __table_args__ = (
        # 到期扫描键：pending 且 expires_at 已过（expire_due 消费）
        Index("ix_approvals_expiry", "status", "expires_at"),
    )

    # 派生 uuid5（与事件 id 同一机制）：节点重放不重复开单
    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)  # 坐席同租户校验
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(64))
    # 参数快照：批准后前置校验重跑防 TOCTOU
    args: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    operator_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 执行后回填的 tool_call 事件 id："批准已兑现"的唯一凭证
    event_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _row_dict(row: ApprovalRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "session_id": row.session_id,
        "run_id": row.run_id,
        "tool_name": row.tool_name,
        "args": row.args,
        "status": row.status,
        "operator_id": row.operator_id,
        "event_id": row.event_id,
        "expires_at": _iso(row.expires_at),
        "decided_at": _iso(row.decided_at),
        "created_at": _iso(row.created_at),
    }


class ApprovalStore:
    """审批单状态机的原语层：只管一张表的真相。事件写入与 run_state 翻转不在此层——那是 Approvals 中间件与恢复入口的事。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def create(
        self,
        *,
        approval_id: str,
        tenant_id: str,
        session_id: str,
        run_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        ttl_s: float,
    ) -> dict[str, Any]:
        """幂等开单：返回该单现状（created 标记本次是否新建）。expires_at = now() + ttl_s，数据库时钟。"""
        stmt = (
            insert(ApprovalRecord)
            .values(
                id=approval_id,
                tenant_id=tenant_id,
                session_id=session_id,
                run_id=run_id,
                tool_name=tool_name,
                args=dict(args),
                status=PENDING,
                expires_at=func.now() + timedelta(seconds=ttl_s),
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        async with self._sf.begin() as s:
            created = (await s.execute(stmt)).rowcount == 1
            row = (
                await s.execute(
                    select(ApprovalRecord).where(ApprovalRecord.id == approval_id)
                )
            ).scalar_one()
            return {**_row_dict(row), "created": created}

    async def get(self, approval_id: str) -> dict[str, Any] | None:
        """读单现状；无单返回 None（恢复入口据此拒绝挂起载荷与审批表不一致的会话）。"""
        async with self._sf() as s:
            row = await s.get(ApprovalRecord, approval_id)
        return None if row is None else _row_dict(row)

    async def decide(
        self, approval_id: str, *, approved: bool, operator_id: str
    ) -> bool:
        """坐席决策：pending 且未过期才翻转（到期 fail-closed）——过期单一律拒绝，归宿只有 expire_due。"""
        stmt = (
            update(ApprovalRecord)
            .where(
                ApprovalRecord.id == approval_id,
                ApprovalRecord.status == PENDING,
                ApprovalRecord.expires_at > func.now(),
            )
            .values(
                status=APPROVED if approved else REJECTED,
                operator_id=operator_id,
                decided_at=func.now(),
            )
        )
        async with self._sf.begin() as s:
            return (await s.execute(stmt)).rowcount == 1

    async def cancel(self, approval_id: str) -> bool:
        """用户撤回：pending 即可翻转，不查过期。"""
        stmt = (
            update(ApprovalRecord)
            .where(ApprovalRecord.id == approval_id, ApprovalRecord.status == PENDING)
            .values(status=CANCELLED, decided_at=func.now())
        )
        async with self._sf.begin() as s:
            return (await s.execute(stmt)).rowcount == 1

    async def expire_due(self, *, now: datetime | None = None) -> list[str]:
        """把 pending 且已到期的单批量翻 expired，返回翻转的单号（调度归 M3 的 reaper）。

        now 可注入：单测不必等真实时钟走到 expires_at；生产不传 → 落回 func.now()，与 decide 同一口钟。
        """
        cutoff = func.now() if now is None else now
        stmt = (
            update(ApprovalRecord)
            .where(
                ApprovalRecord.status == PENDING, ApprovalRecord.expires_at <= cutoff
            )
            .values(status=EXPIRED, decided_at=func.now())
            .returning(ApprovalRecord.id)
        )
        async with self._sf.begin() as s:
            return list((await s.execute(stmt)).scalars().all())

    async def attach_event(self, approval_id: str, *, event_id: str) -> bool:
        """执行后回填审计链：CAS（WHERE event_id IS NULL）——回填恰一次，重复调用（重放）拿 False。"""
        stmt = (
            update(ApprovalRecord)
            .where(ApprovalRecord.id == approval_id, ApprovalRecord.event_id.is_(None))
            .values(event_id=event_id)
        )
        async with self._sf.begin() as s:
            return (await s.execute(stmt)).rowcount == 1

    async def list_for_session(
        self, tenant_id: str, session_id: str
    ) -> list[dict[str, Any]]:
        """会话下的审批单（按开单序；租户双过滤）：坐席页与测试用。"""
        stmt = (
            select(ApprovalRecord)
            .where(
                ApprovalRecord.tenant_id == tenant_id,
                ApprovalRecord.session_id == session_id,
            )
            .order_by(ApprovalRecord.created_at, ApprovalRecord.id)
        )
        async with self._sf() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [_row_dict(row) for row in rows]
