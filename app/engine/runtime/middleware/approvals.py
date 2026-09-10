"""Approvals：HITL 审批中间件——after_model 里用 interrupt() 原语自建（ADR-013 决策 1–3；契约 C10 与 C3 #6 的 HITL 半边）。

不用 HumanInTheLoopMiddleware：reject 继续循环而非 cancelled 终止、英文 ToolMessage 不可替换、无单 / TTL / CAS / 落账、
决定数错误令线程僵住、after_model 反序排不到闸门之后。借其载荷形态（action_requests / review_configs）与决定类型名（approve / reject）。
栈序：列表里在 AegisSummarization 与 Gates 之间——after_model 反序运行，Gates 先跑（#5 / #4），闸门终止的 jump end 绕过本钩子；
本钩子是列表首个 after_model = 循环出口节点，出边是模型→工具边（探针⒄）：返回通行证时未配对调用直送 tools，
返回配对 ToolMessage + jump end 时直达 after_agent、不再调模型。

首次进入：对最后一条 AIMessage 的每个未配对 tool_call 跑 risk_policy（谓词崩溃 = fail-closed：配对错误 ToolMessage、不进审批）；
命中 → 幂等开单（id 派生自稳定任务身份）→ approval_requested 事件 → T2 running→awaiting_approval → interrupt(载荷)：
run 干净返回，进程可下线（挂起点的 checkpoint 已落盘，durability=sync）。
恢复（resume 单入口发 Command(resume=…)）：同节点重放——开单命中既有单、事件去重、T2 跳过（单已终态）、interrupt() 返回决定；
节点以**审批表终态**为准（决定形态由恢复入口守卫，节点不解释它）：全部批准 → approval_decided + 通行证 approved_calls
写入通道（ToolExec ③ 放行、write-ahead 后回填 approval.event_id）；任一拒绝 / 撤回 / 超时 → 对应事件 + 全部未配对调用配对
ToolMessage + termination(cancelled) jump end，零 LLM 调用。
崩溃重放（ainvoke(None)，无 resume 值）：interrupt() 再次 raise → 原样再挂起。每个动作都幂等（单 / 事件 / 翻转）——这是与前作最大的实现差异。
"""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import ValidationError

from app.core.logs import get_logger
from app.engine.gateway.errors import sanitize_error_text
from app.engine.runtime import utterances as u
from app.engine.runtime.events import EventType, event_id
from app.engine.runtime.protocols import ApprovalStatus, SessionRunState
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import (
    RunContext,
    RunState,
    emit,
    task_coords,
    terminated,
)

logger = get_logger(__name__)

APPROVAL_HOOK = "approval"
"""审批单 id 的派生点名：event_id(session_id, 任务 id, "approval", 模型侧 call id)——与事件 id 同一机制、不同点名
（不与 approval_requested 事件撞 id）。节点重放派生同一单号 → 开单幂等。"""

ALLOWED_DECISIONS = ["approve", "reject"]
"""借 HITL 载荷形态的决定名：本仓只有批准 / 拒绝两型（edit / respond 不在前作语义里）。"""

_DETAIL_BY_STATUS = {
    ApprovalStatus.REJECTED: "审批被拒绝",
    ApprovalStatus.CANCELLED: "审批被撤回",
    ApprovalStatus.EXPIRED: "审批超时",
}
"""cancelled 终止的 detail 前缀（进 loop_terminated 事件，前作逐字）。"""

_LABEL_BY_STATUS = {
    ApprovalStatus.REJECTED: u.APPROVAL_LABEL_REJECTED,
    ApprovalStatus.CANCELLED: u.APPROVAL_LABEL_CANCELLED,
    ApprovalStatus.EXPIRED: u.APPROVAL_LABEL_EXPIRED,
}
"""配对回填 TOOL_APPROVAL_DENIED 的 {status} 标签（话术常量只放字符串，映射住这里）。"""

Ticket = tuple[Mapping[str, Any], Mapping[str, Any]]
"""(模型侧 tool_call, 审批单现状)。"""


def _pair(call: Mapping[str, Any], content: str) -> ToolMessage:
    return ToolMessage(
        content=content, tool_call_id=call["id"], name=call["name"], status="error"
    )


def last_ai_and_unpaired(
    messages: Sequence[Any],
) -> tuple[AIMessage | None, list[dict[str, Any]]]:
    """最后一条 AIMessage 与其后尚未配对 ToolMessage 的 tool_calls（Gates 可能已配对一部分：打断 / 幻觉名）。"""
    paired: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message, [c for c in message.tool_calls if c["id"] not in paired]
        if isinstance(message, ToolMessage):
            paired.add(message.tool_call_id)
    return None, []


class Approvals(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        if state.get("termination") is not None:
            return None  # 闸门已终止（其 jump end 通常已绕过本钩子；防御性）
        ctx = runtime.context
        last, unpaired = last_ai_and_unpaired(state["messages"])
        if last is None or not unpaired:
            return None
        approved: Mapping[str, str] = state.get("approved_calls") or {}
        blocked: list[ToolMessage] = []
        gated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for call in unpaired:
            if call["id"] in approved:
                continue
            tool = ctx.registry.get(call["name"])
            if tool is None or tool.risk_policy is None or tool.args_model is None:
                continue  # 幻觉名（Gates 已配对）/ 读工具 / 显式豁免的写工具：不过审批
            try:
                args = tool.args_model.model_validate(call["args"])
            except ValidationError:
                continue  # 坏参数没有副作用要保护：交给 ToolExec ① 回填话术
            try:
                needs_approval = tool.risk_policy(args, ctx.spec.tenant_config)
            except Exception as exc:  # noqa: BLE001  —— 谓词崩溃 = fail-closed：不进审批、不执行
                logger.warning(
                    u.LOG_RISK_POLICY_CRASHED,
                    tool=call["name"],
                    error=type(exc).__name__,
                )
                blocked.append(
                    _pair(
                        call,
                        u.TOOL_RISK_EVAL_FAILED.format(
                            detail=sanitize_error_text(str(exc))
                        ),
                    )
                )
                continue
            if needs_approval:
                gated.append((call, args.model_dump(mode="json")))
        if not gated:
            return {"messages": blocked} if blocked else None
        if ctx.approvals is None:
            raise RuntimeError(
                "风险闸门命中但 RunContext.approvals 缺席——审批需要审批单存取件"
            )
        task_id, _ = task_coords()
        tickets: list[Ticket] = []
        for call, args in gated:
            approval_id = event_id(ctx.session_id, task_id, APPROVAL_HOOK, call["id"])
            row = await ctx.approvals.create(
                approval_id=approval_id,
                tenant_id=ctx.tenant_id,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                tool_name=call["name"],
                args=args,
                ttl_s=ctx.spec.policy.approval_ttl_s,
            )
            await emit(
                runtime,
                EventType.APPROVAL_REQUESTED,
                {
                    "approval_id": approval_id,
                    "tool_name": call["name"],
                    "args": args,
                    "expires_at": row["expires_at"],
                },
                hook="approval_requested",
                ordinal=call["id"],
            )
            tickets.append((call, row))
        if all(row["status"] == ApprovalStatus.PENDING for _, row in tickets):
            # 首次进入（或决定前的崩溃重放）：开单 → 事件 → 翻转 → 挂起，四步之后进程可下线
            await self._suspend_state(ctx)
        # 首次 raise GraphInterrupt（run 干净返回）；恢复时返回 Command(resume=...) 的值——决定形态由恢复入口守卫，
        # 节点只认审批表终态（上面 create 返回的现状就是终态）
        interrupt(self._payload(tickets))
        return await self._settle(runtime, tickets, unpaired, blocked, approved)

    @staticmethod
    async def _suspend_state(ctx: RunContext) -> None:
        """T2 running→awaiting_approval。CAS 失败但会话已在 awaiting_approval = 翻转之后、checkpoint 之前崩过的重放窗口（正常）；
        其余失败是状态机被旁路，响亮留痕不掀挂起。"""
        if ctx.sessions is None:
            return
        flipped = await ctx.sessions.transition(
            ctx.session_id,
            expected=SessionRunState.RUNNING.value,
            to=SessionRunState.AWAITING_APPROVAL.value,
        )
        if flipped:
            return
        row = await ctx.sessions.get(ctx.session_id)
        if row is None or row["run_state"] != SessionRunState.AWAITING_APPROVAL.value:
            logger.warning(u.LOG_APPROVAL_FLIP_FAILED, session_id=ctx.session_id)

    @staticmethod
    def _payload(tickets: Sequence[Ticket]) -> dict[str, Any]:
        """interrupt 载荷（借 HITL 形态以利前端兼容）：每个 action_request 多带 approval_id / expires_at 供坐席回调定位单据。"""
        return {
            "action_requests": [
                {
                    "name": call["name"],
                    "args": dict(row["args"]),
                    "description": u.APPROVAL_DESCRIPTION.format(name=call["name"]),
                    "approval_id": row["id"],
                    "expires_at": row["expires_at"],
                }
                for call, row in tickets
            ],
            "review_configs": [
                {"action_name": call["name"], "allowed_decisions": ALLOWED_DECISIONS}
                for call, _ in tickets
            ],
        }

    @staticmethod
    async def _settle(
        runtime: Runtime[RunContext],
        tickets: Sequence[Ticket],
        unpaired: Sequence[Mapping[str, Any]],
        blocked: list[ToolMessage],
        approved: Mapping[str, str],
    ) -> dict[str, Any]:
        """决定处理：逐单写事件；全部批准 → 通行证；任一非批准 → cancelled 终止（闸门 #6 的 HITL 半边），全部未配对调用配对回填。"""
        for call, row in tickets:
            status = row["status"]
            if status == ApprovalStatus.PENDING:
                raise RuntimeError(
                    f"审批单 {row['id']} 仍是 pending 却收到恢复——恢复入口的终态守卫失效"
                )
            if status in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED):
                await emit(
                    runtime,
                    EventType.APPROVAL_DECIDED,
                    {
                        "approval_id": row["id"],
                        "approved": status == ApprovalStatus.APPROVED,
                        "operator_id": row["operator_id"],
                    },
                    hook="approval_decided",
                    ordinal=call["id"],
                )
            elif status == ApprovalStatus.CANCELLED:
                await emit(
                    runtime,
                    EventType.APPROVAL_CANCELLED,
                    {"approval_id": row["id"]},
                    hook="approval_cancelled",
                    ordinal=call["id"],
                )
            else:
                await emit(
                    runtime,
                    EventType.APPROVAL_EXPIRED,
                    {"approval_id": row["id"]},
                    hook="approval_expired",
                    ordinal=call["id"],
                )
        denied = [
            (call, row)
            for call, row in tickets
            if row["status"] != ApprovalStatus.APPROVED
        ]
        if not denied:
            passport = {**approved, **{call["id"]: row["id"] for call, row in tickets}}
            out: dict[str, Any] = {"approved_calls": passport}
            if blocked:
                out["messages"] = blocked
            return out
        denied_by_id = {call["id"]: row for call, row in denied}
        blocked_ids = {m.tool_call_id for m in blocked}
        paired: list[ToolMessage] = list(blocked)
        for call in unpaired:
            if call["id"] in blocked_ids:
                continue
            row = denied_by_id.get(call["id"])
            content = (
                u.TOOL_APPROVAL_DENIED.format(status=_LABEL_BY_STATUS[row["status"]])
                if row is not None
                else u.TOOL_NOT_EXECUTED
            )
            paired.append(_pair(call, content))
        detail = "；".join(
            f"{_DETAIL_BY_STATUS[row['status']]}：approval_id={row['id']}"
            for _, row in denied
        )
        return {
            "termination": terminated(TerminationReason.CANCELLED, detail=detail),
            "messages": paired,
            "jump_to": "end",
        }
