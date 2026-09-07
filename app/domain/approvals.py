"""审批单：approvals 表 ORM（M2.2 只立表，五态 CAS 的 ApprovalStore 随 M2.7；ADR-013）。不 import engine。

不挂工具调用外键：审批发生在 write-ahead 之前，审批的是参数快照；执行后以 event_id 回填审计链。
status 存字符串列，合法值在 APPROVAL_STATUSES（超时与撤回是一等状态，不是 rejected 的变体）。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

APPROVAL_STATUSES: tuple[str, ...] = (
    "pending",
    "approved",
    "rejected",
    "cancelled",
    "expired",
)
"""approvals.status 五态（值快照由测试钉死）。"""


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
