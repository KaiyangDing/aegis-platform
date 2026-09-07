"""图状态私有通道 + run 载体 + 钩子内事件发射（M2.3）。

RunState：在框架 AgentState（messages / jump_to）之上加运行时私有通道，全部 PrivateStateAttr——不进 ainvoke 输出与 schema，
但随 checkpoint 持久化且**跨 run 延续**（探针⑵）：run 级计数由 RunEvents.before_agent 显式归零（ADR-012 决策 8）。
RunContext：每 run 一份、不进 checkpoint 的载体（身份 / spec / 注册表 / 事实源 / 状态机 / token 种子 / 工具串行器）；
跨崩溃需要的状态只能放通道或表，不能放这里。
emit()：钩子内写事件的唯一入口——id 派生自 (session_id, 框架任务 id, 钩子名, 序号)，先落盘再经 stream_writer 外流；
去重命中（重放）不再外流。
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langgraph.config import get_config
from langgraph.runtime import Runtime

from app.engine.runtime import utterances as u
from app.engine.runtime.events import AgentEvent, EventType, event_id
from app.engine.runtime.protocols import EventSink, SessionStateLike
from app.engine.runtime.spec import AgentSpec, TerminationReason
from app.engine.runtime.tools import ToolRegistry


class RunState(AgentState):
    """私有通道（无 reducer：后写覆盖）。iteration = 已发起的 LLM 调用数；tokens_used = 会话级估算累计（D8 种子起）；
    violations / repeat 归闸门 #5 / #4（M2.4）；termination 是唯一终止信号（ADR-012 决策 3）；
    approved_calls / disabled_tools 归审批（M2.7）/ 连败禁用（M2.5）。"""

    iteration: NotRequired[Annotated[int, PrivateStateAttr]]
    tokens_used: NotRequired[Annotated[int, PrivateStateAttr]]
    violations: NotRequired[Annotated[int, PrivateStateAttr]]
    repeat: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]
    termination: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]
    approved_calls: NotRequired[Annotated[list[str], PrivateStateAttr]]
    disabled_tools: NotRequired[Annotated[list[str], PrivateStateAttr]]


def run_channels_reset(token_seed: int) -> dict[str, Any]:
    """run 起点的通道归零（tokens_used 从历史种子起）：通道跨 run 持久化，不归零第二轮一开场就撞轮数上限。"""
    return {
        "iteration": 0,
        "tokens_used": token_seed,
        "violations": 0,
        "repeat": None,
        "termination": None,
        "approved_calls": [],
        "disabled_tools": [],
    }


class ToolSerializer:
    """每 run 一把：按模型声明的顺序逐个放行工具调用（ADR-012 决策 7）。

    框架 ToolNode 把一轮的多个 tool_calls 作为并行任务派发，到达顺序随调度不定（探针 F）；单纯互斥锁只能保证
    不重叠、保证不了顺序。这里让每个调用等它在声明序里的前驱全部结束再执行——事件序与执行序都等于声明序。
    前驱无论成败都会 finish（wrap 的 finally），等待者不会悬死。
    """

    def __init__(self) -> None:
        self._done: set[str] = set()
        self._cond = asyncio.Condition()

    async def wait_turn(self, predecessors: Sequence[str]) -> None:
        async with self._cond:
            await self._cond.wait_for(
                lambda: all(p in self._done for p in predecessors)
            )

    async def finish(self, call_id: str) -> None:
        async with self._cond:
            self._done.add(call_id)
            self._cond.notify_all()


@dataclass(frozen=True, slots=True)
class RunContext:
    """每 run 一份的载体（create_agent 的 context_schema）。sessions=None 是纯单元测试形态（无状态机）。"""

    tenant_id: str
    user_id: str
    session_id: str
    run_id: str
    spec: AgentSpec
    registry: ToolRegistry
    events: EventSink
    sessions: SessionStateLike | None = None
    token_seed: int = 0
    tool_order: ToolSerializer = field(default_factory=ToolSerializer)


FALLBACK_BY_REASON: dict[TerminationReason, str | None] = {
    TerminationReason.COMPLETED: None,
    TerminationReason.MAX_ITERATIONS: u.FALLBACK_MAX_ITERATIONS,
    TerminationReason.STEP_TIMEOUT: u.FALLBACK_STEP_FAILED,
    TerminationReason.TOKEN_BUDGET_EXCEEDED: u.FALLBACK_BUDGET,
    TerminationReason.REPEATED_CALLS: u.FALLBACK_REPEATED,
    TerminationReason.PROTOCOL_VIOLATION: u.FALLBACK_PROTOCOL,
    TerminationReason.CANCELLED: None,  # 用户主动取消无需道歉文
    TerminationReason.GATEWAY_REJECTED: None,  # 确定性拒绝零话术（契约 C4）
}
"""兜底话术按原因单点选取。终止方把它写进 termination["fallback"] 并同时追加为 AIMessage；
after_agent 只据此写 assistant_message 事件，不再自己选话术。"""


def terminated(
    reason: TerminationReason,
    *,
    detail: str,
    cause: str | None = None,
) -> dict[str, Any]:
    """构造 termination 通道值：{reason, detail, cause?, fallback}（值一律 .value：断言与构造两侧不混用枚举成员）。"""
    value: dict[str, Any] = {
        "reason": reason.value,
        "detail": detail,
        "fallback": FALLBACK_BY_REASON[reason],
    }
    if cause is not None:
        value["cause"] = cause
    return value


def task_coords() -> tuple[str, str | None]:
    """框架任务坐标：__pregel_task_id 跨崩溃 / 中断重放稳定（探针 G2、⑺）；checkpoint_id 是调度自的 checkpoint（可为 None）。"""
    conf = get_config()["configurable"]
    return conf["__pregel_task_id"], conf.get("checkpoint_id")


async def emit(
    runtime: Runtime[RunContext],
    event_type: EventType,
    payload: Mapping[str, Any],
    *,
    hook: str,
    ordinal: str | int = 0,
) -> tuple[AgentEvent, bool]:
    """钩子内写一条事实：派生 id → append（返回 seq 与是否新建）→ 新建的经 stream_writer 外流。

    返回 (event, created)：created=False 即重放命中既有事件（ToolExec 据此进 reexecute 分支，M2.5）。
    事实源异常（不可用 / 围栏）裸穿：run 炸出去，不接。
    """
    ctx = runtime.context
    task_id, checkpoint_id = task_coords()
    eid = event_id(ctx.session_id, task_id, hook, ordinal)
    seq, created = await ctx.events.append(
        event_id=eid,
        tenant_id=ctx.tenant_id,
        session_id=ctx.session_id,
        run_id=ctx.run_id,
        event_type=event_type.value,
        payload=payload,
        task_id=task_id,
        checkpoint_id=checkpoint_id,
    )
    event = AgentEvent(
        id=eid,
        tenant_id=ctx.tenant_id,
        session_id=ctx.session_id,
        run_id=ctx.run_id,
        seq=seq,
        type=event_type,
        payload=dict(payload),
        task_id=task_id,
        checkpoint_id=checkpoint_id,
    )
    if created:
        runtime.stream_writer(event)
    return event, created
