"""POST /v1/chat（M3.2；ADR-014）：一次 run 的 HTTP 壳。

准入链（顺序即语义）：401 身份 → 429 租户限流（依赖列表里认证必须排在限流之前：身份段从 request.state 读）→ 403 租户未开通
→ 422 形态 → 会话首见即建 / 归属 404 → 等审批 409（带 pending 单号）→ **peek 首帧**再返回流：run() 是异步生成器，归属校验与
T1 CAS 在首次 anext 才执行，SessionBusy → 409、ValueError → 404 只有在首帧之前才能变成状态码；首帧之后的异常无法改状态码，
只能以 error 帧收流（只回显异常类型名）。
流的结束：见到 loop_terminated → done{reason}；末事件是 approval_requested（挂起，无 loop_terminated）→ done{reason: awaiting_approval, approvals}；
客户端断连不翻译为取消信号（cancel=None，命运表 B17）——事件已落盘、重订阅能续上。
"""

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.business import utterances as bu
from app.core.auth import Principal, Role, current_principal, tenant_identity
from app.core.limits import rate_limit
from app.core.logs import get_logger
from app.engine.runtime.events import AgentEvent
from app.engine.runtime.runtime import SessionBusy
from app.routers.common import (
    SESSION_ID_PATTERN,
    ensure_owned_session,
    services,
    spec_for,
)
from app.routers.sse import (
    MEDIA_TYPE,
    SSE_HEADERS,
    as_row,
    done_frame,
    error_frame,
    event_frame,
    visible,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])

CHAT_RATE_SCOPE, CHAT_RATE_TIMES, CHAT_RATE_SECONDS = "chat", 20, 60
"""租户级固定窗：每租户每分钟 20 次发消息（演示值）。"""


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


async def _chain(
    first: AgentEvent, rest: AsyncIterator[AgentEvent]
) -> AsyncIterator[AgentEvent]:
    yield first
    async for event in rest:
        yield event


async def _frames(
    first: AgentEvent, rest: AsyncIterator[AgentEvent], role: Role
) -> AsyncIterator[bytes]:
    reason: str | None = None
    pending: list[dict[str, Any]] = []
    try:
        async for event in _chain(first, rest):
            kind = event.type.value
            if kind == "approval_requested":
                pending.append(
                    {
                        "approval_id": event.payload.get("approval_id"),
                        "tool_name": event.payload.get("tool_name"),
                        "expires_at": event.payload.get("expires_at"),
                    }
                )
            elif kind == "loop_terminated":
                reason = str(event.payload.get("reason"))
            if visible(kind, role):
                yield event_frame(as_row(event))
    except Exception as e:
        logger.exception("chat_stream_failed", error=type(e).__name__)
        yield error_frame(bu.RUN_FAILED, type(e).__name__)
        return
    if reason is not None:
        yield done_frame(reason=reason)
    elif pending:
        yield done_frame(reason="awaiting_approval", approvals=pending)
    else:
        yield done_frame(reason="ended")  # 既无终止事件也无挂起：留痕而不装作完成


@router.post(
    "/chat",
    dependencies=[
        Depends(
            current_principal
        ),  # 先认证：限流的身份段从 request.state.principal 读（ADR-010 决策 6）
        Depends(
            rate_limit(
                CHAT_RATE_SCOPE,
                CHAT_RATE_TIMES,
                CHAT_RATE_SECONDS,
                identify=tenant_identity,
            )
        ),
    ],
)
async def chat(
    body: ChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
) -> StreamingResponse:
    svc = services(request)
    spec = spec_for(svc, principal.tenant_id)
    row = await ensure_owned_session(svc, principal, body.session_id)
    if row["run_state"] == "awaiting_approval":
        tickets = await svc.approvals.list_for_session(
            principal.tenant_id, body.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": bu.SESSION_AWAITING_APPROVAL,
                "pending_approvals": [
                    t["id"] for t in tickets if t["status"] == "pending"
                ],
            },
        )
    stream = svc.runtime.run(
        tenant_id=principal.tenant_id,
        session_id=body.session_id,
        user_input=body.message,
        spec=spec,
    )
    try:
        first = await anext(stream)
    except SessionBusy as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=bu.SESSION_BUSY
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=bu.SESSION_NOT_FOUND
        ) from e
    return StreamingResponse(
        _frames(first, stream, principal.role),
        media_type=MEDIA_TYPE,
        headers=SSE_HEADERS,
    )
