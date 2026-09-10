"""HTTP 层共用件（M3.2；ADR-014）：从 app.state 取共享件、租户 spec 查找、会话归属两种口径、运行时异常 → 状态码。

路由只读 app.state，不 import app.main：组合根在 lifespan 里把 settings / runtime / runtime_parts / specs 挂上来，
测试用最小 FastAPI 应用 + 内存替身装同一批键即可零 PG 驱动路由。
状态码分工：401 身份无效（core/auth）、403 身份合法但无权（角色 / 租户未开通 / 他租审批单）、404 不泄露存在性
（他租 / 他人会话一律"不存在"）、409 状态冲突（会话正忙 / 等审批 / 审批单已非 pending / 并发续跑输家）、422 形态错误（FastAPI）。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy.exc import IntegrityError

from app.business import utterances as bu
from app.core.auth import Principal, Role
from app.engine.runtime.runtime import AgentRuntime
from app.engine.runtime.spec import AgentSpec

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
"""会话 id 由客户端给出（thread_id 同值）：字符集与租户 id 同一条规则——它要进 checkpoint 表与事件表的键。"""


@dataclass(frozen=True, slots=True)
class Services:
    runtime: AgentRuntime
    events: Any  # EventSource（domain.EventStore 或内存替身）
    sessions: Any  # SessionStateLike
    approvals: Any  # ApprovalStoreLike
    specs: Mapping[str, AgentSpec]


def services(request: Request) -> Services:
    state = request.app.state
    parts = state.runtime_parts
    return Services(
        runtime=state.runtime,
        events=parts.events,
        sessions=parts.sessions,
        approvals=parts.approvals,
        specs=state.specs,
    )


def spec_for(svc: Services, tenant_id: str) -> AgentSpec:
    """JWT 里的租户不在静态表：签名对、身份合法，只是未开通 → 403（不是 401）。"""
    try:
        return svc.specs[tenant_id]
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=bu.TENANT_NOT_ENABLED
        ) from e


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=bu.SESSION_NOT_FOUND
    )


async def ensure_owned_session(
    svc: Services, principal: Principal, session_id: str
) -> dict[str, Any]:
    """聊天口径：会话首见即建；既有会话必须属于本租户且由本人创建，否则一律 404（不泄露存在性）。
    并发首见：另一请求先建（IntegrityError / 内存替身覆盖）→ 回读再校归属。"""
    row = await svc.sessions.get(session_id)
    if row is None:
        try:
            await svc.sessions.create(
                session_id, tenant_id=principal.tenant_id, user_id=principal.user_id
            )
        except IntegrityError:
            pass
        row = await svc.sessions.get(session_id)
        if row is None:
            raise RuntimeError(f"会话 {session_id} 建行后回读为空——事实源不一致")
    if row["tenant_id"] != principal.tenant_id or row["user_id"] != principal.user_id:
        raise _not_found()
    return row


async def viewable_session(
    svc: Services, principal: Principal, session_id: str
) -> dict[str, Any]:
    """读面口径：本租户的会话；终端用户只能看本人的，坐席 / 管理员可看本租户任一会话；其余 404。"""
    row = await svc.sessions.get(session_id)
    if row is None or row["tenant_id"] != principal.tenant_id:
        raise _not_found()
    if principal.role is Role.USER and row["user_id"] != principal.user_id:
        raise _not_found()
    return row
