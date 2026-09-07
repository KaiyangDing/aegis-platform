"""审批单表（M2.2 只立表；五态 CAS 随 M2.7）：五态值快照、ORM 往返与默认值、JSONB 参数快照、非 uuid id 拒绝。"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import StatementError

pytest.importorskip(
    "app.domain.approvals", reason="M2.2 未敲：app/domain/approvals.py 不存在"
)

from app.domain.approvals import APPROVAL_STATUSES, ApprovalRecord


def _record(**overrides) -> ApprovalRecord:
    base = {
        "id": str(uuid.uuid4()),
        "tenant_id": "t-a",
        "session_id": "s-1",
        "run_id": "r-1",
        "tool_name": "demo_refund_apply",
        "args": {"order_id": "1024", "amount": 350},
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
    }
    return ApprovalRecord(**{**base, **overrides})


def test_statuses_snapshot():
    """超时与撤回是一等状态，不是 rejected 的变体。"""
    assert APPROVAL_STATUSES == (
        "pending",
        "approved",
        "rejected",
        "cancelled",
        "expired",
    )


async def test_roundtrip_and_defaults(db_session):
    rec = _record()
    db_session.add(rec)
    await db_session.flush()
    await db_session.refresh(rec)
    assert rec.status == "pending"
    assert rec.operator_id is None and rec.event_id is None and rec.decided_at is None
    assert rec.created_at.tzinfo is not None
    got = (
        await db_session.execute(
            select(ApprovalRecord).where(ApprovalRecord.id == rec.id)
        )
    ).scalar_one()
    assert got.args == {"order_id": "1024", "amount": 350}
    assert got.expires_at.tzinfo is not None


async def test_non_uuid_id_rejected(db_session):
    db_session.add(_record(id="not-a-uuid"))
    with pytest.raises(StatementError):
        await db_session.flush()
