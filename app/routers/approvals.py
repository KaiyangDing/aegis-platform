"""POST /v1/approvals/{approval_id}（M3.2；ADR-013 决策 3 / 6、ADR-014）：审批决定 + 同步续跑。

授权序（五步，顺序即语义）：401 身份 → 403 角色（只许坐席 / 管理员）→ 404 单不存在 → 403 他租的单（坐席与管理员都是租户级身份）
→ 惰性到期扫描（expire_due：过期单先翻 expired，命运表 B5）→ decide CAS（pending 且未过期才翻；False → 409 绝不覆盖赢家）
→ runtime.resume(approval_id)（SessionBusy = 并发续跑输家 → 409；ValueError = 同载荷仍有单 pending / 单不属挂起点 → 409）。
决定不经 resume 传入：先落审批表终态，resume 只从表翻译 decisions（形态守卫在恢复入口）。
同步吸干续跑并返回 JSON 摘要：done（reason = 终止原因、reply = 终答）/ awaiting_approval（续跑途中再撞闸门，next_approval_id）/ no_op（输给并发赢家，零事件）。
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.business import utterances as bu
from app.core.auth import Principal, Role, require_roles
from app.engine.runtime.events import AgentEvent
from app.engine.runtime.runtime import SessionBusy
from app.routers.common import services, spec_for

router = APIRouter(prefix="/v1", tags=["approvals"])


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]


def summarize(
    approval_id: str, decision: str, events: list[AgentEvent]
) -> dict[str, Any]:
    types = [e.type.value for e in events]
    reply = next(
        (
            e.payload.get("content")
            for e in reversed(events)
            if e.type.value == "assistant_message"
        ),
        None,
    )
    if not events:
        state, reason, next_id = "no_op", None, None
    elif types[-1] == "loop_terminated":
        state, reason, next_id = "done", events[-1].payload.get("reason"), None
    elif types[-1] == "approval_requested":
        state, reason, next_id = (
            "awaiting_approval",
            None,
            events[-1].payload.get("approval_id"),
        )
    else:
        state, reason, next_id = "ended", None, None
    return {
        "approval_id": approval_id,
        "decision": decision,
        "status": state,
        "reason": reason,
        "reply": reply,
        "next_approval_id": next_id,
        "events": types,
    }


@router.post("/approvals/{approval_id}")
async def decide(
    approval_id: str,
    body: DecisionRequest,
    request: Request,
    principal: Annotated[Principal, Depends(require_roles(Role.OPERATOR, Role.ADMIN))],
) -> dict[str, Any]:
    svc = services(request)
    ticket = await svc.approvals.get(approval_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=bu.APPROVAL_NOT_FOUND
        )
    if ticket["tenant_id"] != principal.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=bu.APPROVAL_FOREIGN_TENANT
        )
    await svc.approvals.expire_due()
    decided = await svc.approvals.decide(
        approval_id, approved=body.decision == "approve", operator_id=principal.user_id
    )
    if not decided:
        current = await svc.approvals.get(approval_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": bu.APPROVAL_NOT_PENDING,
                "status": current["status"] if current else None,
            },
        )
    spec = spec_for(svc, ticket["tenant_id"])
    try:
        events = [
            e
            async for e in svc.runtime.resume(
                tenant_id=ticket["tenant_id"],
                session_id=ticket["session_id"],
                spec=spec,
                approval_id=approval_id,
            )
        ]
    except SessionBusy as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=bu.SESSION_BUSY
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return summarize(approval_id, body.decision, events)
