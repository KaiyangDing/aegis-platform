"""GET /v1/sessions/{session_id}/events?after_seq=&limit=（M3.2；ADR-011 决策 10、ADR-014）：事实源一次性回放。

游标 = max(after_seq, Last-Event-ID)：帧 id 就是 events.seq，断线后客户端带上最后一帧的 id 即零丢帧续读（命运表 A4：
活尾轮询与 LISTEN/NOTIFY 不做——续传正确性只依赖游标是 seq 这一条，唤醒是延迟问题）。
读面归属：本租户会话；终端用户只看本人，坐席 / 管理员看本租户任一会话（事件即审计）；他租一律 404。
帧序列以合成 done{reason: snapshot, run_state, next_seq, count} 收尾（无 id）——客户端据 run_state 决定是否继续轮询、据 next_seq 续读。
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.core.auth import Principal, current_principal
from app.routers.common import services, viewable_session
from app.routers.sse import (
    MEDIA_TYPE,
    SSE_HEADERS,
    as_row,
    done_frame,
    event_frame,
    visible,
)

router = APIRouter(prefix="/v1", tags=["sessions"])


def _last_event_id(raw: str | None) -> int:
    """Last-Event-ID 由客户端回传我们自己发过的帧 id；不可解析按 0（不因坏头拒绝续读）。"""
    try:
        return max(0, int(raw)) if raw is not None else 0
    except ValueError:
        return 0


@router.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
    after_seq: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> StreamingResponse:
    svc = services(request)
    row = await viewable_session(svc, principal, session_id)
    cursor = max(after_seq, _last_event_id(request.headers.get("Last-Event-ID")))
    rows = await svc.events.read(
        principal.tenant_id, session_id, after_seq=cursor, limit=limit
    )

    async def frames() -> AsyncIterator[bytes]:
        next_seq = cursor
        for r in rows:
            next_seq = r["seq"]
            if visible(r["type"], principal.role):
                yield event_frame(as_row(r))
        yield done_frame(
            reason="snapshot",
            run_state=row["run_state"],
            next_seq=next_seq,
            count=len(rows),
        )

    return StreamingResponse(frames(), media_type=MEDIA_TYPE, headers=SSE_HEADERS)
