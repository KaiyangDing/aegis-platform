"""AgentRuntime 门面（M2.3；M2.4 插入 Gates、接取消信号；M2.5 网关句柄进 RunContext；M2.6 插入 AegisSummarization、栈经 build_middleware
注入依赖；M2.7 插入 Approvals、resume() 审批续跑单入口、审批单存取件与前置校验挂点进 RunContext；M2.8 插入 Guards——七件栈定稿）：
按 (tenant_id, spec 指纹) 编译并缓存图；run() 单入口驱动一次循环、事件按 seq 序外流。

一次 run（ADR-011 / 012）：读会话行取身份并核对租户归属 → D8 种子（历史 llm_call / llm_result 的估算字段求和）→
T1 idle→running（CAS 失败 = 会话正忙）→ agent.astream(..., durability="sync", recursion_limit=推导, stream_mode=["custom"])
→ 钩子内落盘的事件经 stream_writer 到达这里逐条 yield。
图缓存：编译成本按租户 spec 摊销；指纹覆盖注入面全部会影响图形态的字段（工具 schema 含在内），改 spec 即换图。
resume（ADR-013 决策 3 / 6；契约 C12）：审批续跑与崩溃恢复同一入口 = 从最后一个 checkpoint 之后重放；approval_id 非 None = 计划内审批续跑，
None = 崩溃恢复（前作 resume(spec, session_id, approval_id=None) 同一分野；M2 无会话锁，"运行中的会话是不是死了"只能由调用方断言）。
审批续跑：会话必须在 awaiting_approval → 取挂起点（缺失即先重放到挂起点）→ approval_id 属于挂起载荷 → 载荷里的审批单全部终态才放行
（仍 pending → ValueError）→ T3 awaiting→running（CAS 输家 = 并发恢复 → SessionBusy）→ Command(resume={"decisions": […]}) 从挂起节点重放；
决定数恒等于挂起数、决定由审批表终态翻译而来——这就是 API 层形态守卫的落点。
崩溃恢复：前作四支分诊在 v2 坍缩为"重放 + 去重"——idle / failed 没有可恢复的 run；挂起点在且审批单仍 pending = 健康的挂起（不计次、零事件）；
recovery_count +1，超上限 → T5 →failed + recovery_abandoned（图外事件）；挂起点在且单已终态 → 与审批续跑同路径；next 为空 → 图已收尾只是
T4 没翻，只修状态；其余 → astream(None) 重放：tools 节点 ToolExec 去重（原键 reexecute）、model 节点 ModelCall 对半截 llm_call 补
llm_result(interrupted, cause=replay) 再作废重发、after_agent 末事件去重 + T4。
"""

import hashlib
import json
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from app.core.logs import get_logger
from app.engine.runtime import utterances as u
from app.engine.runtime.events import AgentEvent, EventType, event_id
from app.engine.runtime.middleware.approvals import Approvals
from app.engine.runtime.middleware.gates import Gates
from app.engine.runtime.middleware.guards import Guards
from app.engine.runtime.middleware.model_call import ModelCall
from app.engine.runtime.middleware.run_events import RunEvents
from app.engine.runtime.middleware.summarization import AegisSummarization
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.protocols import (
    RECOVERY_LIMIT,
    ApprovalStatus,
    ApprovalStoreLike,
    CancelSignal,
    EventStoreLike,
    SessionRunState,
    SessionStateLike,
)
from app.engine.runtime.spec import AgentSpec, LoopPolicy
from app.engine.runtime.state import RunContext
from app.engine.runtime.tools import PrecheckHook, ToolRegistry, to_structured_tools

logger = get_logger(__name__)

MIDDLEWARE_STACK: tuple[type[AgentMiddleware], ...] = (
    RunEvents,
    Guards,
    AegisSummarization,
    Approvals,
    Gates,
    ModelCall,
    ToolExec,
)
"""栈序即语义（ADR-012 决策 1），七件定稿：before_agent 列表序 = 首事件 → 入口守卫（HIGH 跳 end 直达 after_agent）；
before_model 列表序 = 压缩 → 闸门；after_model 反序 = 闸门（#5 / #4）先于审批，闸门终止的 jump end 绕过审批，
Approvals 是列表首个 after_model = 循环出口节点；wrap 各一件；after_agent 只有 RunEvents（末事件）。
build_middleware 与本元组一一对应（静态测试互钉）。"""

_HOOKS_PER_PHASE = {
    "before_agent": ("before_agent", "abefore_agent"),
    "before_model": ("before_model", "abefore_model"),
    "after_model": ("after_model", "aafter_model"),
    "after_agent": ("after_agent", "aafter_agent"),
}


class SessionBusy(RuntimeError):
    """CAS 失败：会话不在期望的状态（另一 run 在跑 / 挂起等审批 / 并发恢复已有赢家）。M3 翻译为 409。"""


def build_middleware(gateway: BaseChatModel, spec: AgentSpec) -> list[AgentMiddleware]:
    """按栈序实例化：需要依赖的中间件在这里注入——入口分类器 = 该租户网关（仅 spec.entry_classifier 开通时，fast 档在调用时指定）、
    摘要模型 = 该租户网关、预算 = spec.context_config（都在指纹里，改即换图）。"""
    return [
        RunEvents(),
        Guards(gateway if spec.entry_classifier else None),
        AegisSummarization(gateway, config=spec.context_config),
        Approvals(),
        Gates(),
        ModelCall(),
        ToolExec(),
    ]


def _hook_count(middleware: Sequence[type[AgentMiddleware]], phase: str) -> int:
    """栈里实现了某钩子（同步或异步）的中间件数——每个钩子是一个图节点。"""
    names = _HOOKS_PER_PHASE[phase]
    return sum(
        1
        for cls in middleware
        if any(getattr(cls, n) is not getattr(AgentMiddleware, n) for n in names)
    )


def recursion_limit_for(
    policy: LoopPolicy, middleware: Sequence[type[AgentMiddleware]] = MIDDLEWARE_STACK
) -> int:
    """从栈形态推导：最长路径的节点执行数 + 1（探针⑻ / M2.4-Q4：最小可行 recursion_limit = 节点数 + 1）。

    最长路径 = 工具循环撞 max_iterations：before_agent 节点 + max_iterations × (before_model 节点 + model + after_model 节点 + tools)
    + 终止那一遍的 before_model 节点（Gates 在此 jump end）+ after_agent 节点。对这条路径公式精确（余量 0）；
    文本完成等更短的路径自然有余量。栈里没有 before_model 钩子时公式对该路径少算一个节点（M2.3 形态的已知缺口，Gates 入栈后消失）。
    恢复（resume）从挂起点重新计数：框架的步数上限相对本次调用的起点。
    """
    per_iteration = (
        _hook_count(middleware, "before_model")
        + _hook_count(middleware, "after_model")
        + 2
    )
    outside = _hook_count(middleware, "before_agent") + _hook_count(
        middleware, "after_agent"
    )
    return (
        outside
        + policy.max_iterations * per_iteration
        + _hook_count(middleware, "before_model")
        + 1
    )


def spec_fingerprint(spec: AgentSpec) -> str:
    """注入面指纹：凡影响图形态或模型所见的字段都进去（工具 schema / 说明 / 读写标记 / 策略 / 预算 / 档位 / 租户配置）。"""
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "side_effect": t.side_effect.value,
            "parameters": dict(t.parameters_schema),
            "risk": "policy"
            if t.risk_policy is not None
            else ("exempt" if t.risk_exempt else "none"),
            "timeout_s": t.timeout_s,
            "retries": t.retries,
        }
        for t in spec.tools
    ]
    essence = {
        "system_prompt": spec.system_prompt,
        "tools": tools,
        "policy": asdict(spec.policy),
        "context_config": asdict(spec.context_config),
        "model_tier": spec.model_tier,
        "sub_agent_policy": spec.sub_agent_policy.value,
        "tenant_config": dict(spec.tenant_config),
        "owned_values": list(spec.owned_values),
        "entry_classifier": spec.entry_classifier,
    }
    blob = json.dumps(essence, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AgentRuntime:
    """对外门面：一次 run = 一条事件流；图按 (tenant_id, spec 指纹) LRU 缓存；网关按租户装配（组合根注入闭包）。

    approvals=None 是无审批的单元测试形态（风险闸门命中即 RuntimeError）；precheck=None = 批准后前置校验全通过（M3 注入真实校验）；
    recovery_limit = 同一会话连续崩溃恢复次数上限（超过即 failed + recovery_abandoned）。
    """

    def __init__(
        self,
        *,
        gateway_for: Callable[[str], BaseChatModel],
        events: EventStoreLike,
        sessions: SessionStateLike,
        checkpointer: BaseCheckpointSaver,
        approvals: ApprovalStoreLike | None = None,
        precheck: PrecheckHook | None = None,
        recovery_limit: int = RECOVERY_LIMIT,
        cache_size: int = 64,
    ) -> None:
        self._gateway_for = gateway_for
        self._events = events
        self._sessions = sessions
        self._checkpointer = checkpointer
        self._approvals = approvals
        self._precheck = precheck
        self._recovery_limit = recovery_limit
        self._cache_size = cache_size
        self._agents: OrderedDict[tuple[str, str], Any] = OrderedDict()

    def build_agent(self, tenant_id: str, spec: AgentSpec) -> Any:
        """编译（或取缓存）该租户该 spec 的图。工具只作 schema 载体进图（执行由 ToolExec 接管）。"""
        key = (tenant_id, spec_fingerprint(spec))
        agent = self._agents.get(key)
        if agent is not None:
            self._agents.move_to_end(key)
            return agent
        gateway = self._gateway_for(tenant_id)
        agent = create_agent(
            model=gateway,
            tools=to_structured_tools(spec.tools),
            system_prompt=spec.system_prompt,
            middleware=build_middleware(gateway, spec),
            context_schema=RunContext,
            checkpointer=self._checkpointer,
        )
        self._agents[key] = agent
        while len(self._agents) > self._cache_size:
            self._agents.popitem(last=False)
        return agent

    async def _token_seed(self, tenant_id: str, session_id: str) -> int:
        """D8：从历史 llm_call / llm_result 的估算字段重建会话级 token 计数（不加新列）。"""
        rows = await self._events.read(tenant_id, session_id)
        seed = 0
        for row in rows:
            if row["type"] in (EventType.LLM_CALL.value, EventType.LLM_RESULT.value):
                payload = row["payload"]
                seed += payload.get("input_tokens_est", 0) + payload.get(
                    "output_tokens_est", 0
                )
        return seed

    async def _owned_session(self, tenant_id: str, session_id: str) -> dict[str, Any]:
        """起跑 / 恢复前的归属校验：会话行不存在或不属于该租户 → ValueError。"""
        row = await self._sessions.get(session_id)
        if row is None:
            raise ValueError(f"会话 {session_id} 不存在——run 之前必须先建 sessions 行")
        if row["tenant_id"] != tenant_id:
            raise ValueError(f"会话 {session_id} 不属于租户 {tenant_id}")
        return row

    def _context(
        self,
        row: dict[str, Any],
        spec: AgentSpec,
        *,
        cancel: CancelSignal | None,
        token_seed: int,
    ) -> RunContext:
        """每次调用一份载体（新 run_id：恢复的事件带新 run_id、seq 接续旧流）。"""
        return RunContext(
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            session_id=row["id"],
            run_id=uuid.uuid4().hex,
            spec=spec,
            registry=ToolRegistry(spec.tools),
            events=self._events,
            sessions=self._sessions,
            token_seed=token_seed,
            cancel=cancel,
            gateway=self._gateway_for(
                row["tenant_id"]
            ),  # 增强层（工具结果摘要）的 fast 档入口，与图内模型同一租户绑定
            approvals=self._approvals,
            precheck=self._precheck,
        )

    @staticmethod
    def _config(session_id: str, spec: AgentSpec) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": recursion_limit_for(spec.policy),
        }

    @staticmethod
    async def _drive(
        agent: Any, payload: Any, config: dict[str, Any], ctx: RunContext
    ) -> AsyncIterator[AgentEvent]:
        """驱动图：只订阅 custom 帧（事件），durability=sync（ADR-011 决策 2）。挂起（interrupt）时流干净结束、不外流中断对象（探针⒅）。"""
        async for _mode, frame in agent.astream(
            payload, config, context=ctx, stream_mode=["custom"], durability="sync"
        ):
            if isinstance(frame, AgentEvent):
                yield frame

    async def run(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_input: str,
        spec: AgentSpec,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """驱动一次完整循环；产出事件 ≡ 本 run 落盘事件，yield 序 ≡ seq 序。

        会话行不存在 / 不属于该租户 → ValueError（起跑前的归属校验）；不在 idle（运行中 / 挂起等审批）→ SessionBusy——
        挂起态的新输入不许进图（探针⒆：新输入会作废挂起、留悬空 tool_calls），会话互斥在这里前置。
        cancel 是闸门 #6 的取消源（None = 无取消源）。
        run 内异常（ProviderError 泄漏、事实源不可用、system 超预算）裸穿，run_state 留在 running——由恢复路径（M2.9）分诊。
        审批挂起时本流在 approval_requested 之后干净结束（无 loop_terminated），run_state = awaiting_approval。
        """
        row = await self._owned_session(tenant_id, session_id)
        token_seed = await self._token_seed(tenant_id, session_id)
        if not await self._sessions.transition(
            session_id,
            expected=SessionRunState.IDLE.value,
            to=SessionRunState.RUNNING.value,
        ):
            raise SessionBusy(f"会话 {session_id} 不在 idle：当前 {row['run_state']}")
        ctx = self._context(row, spec, cancel=cancel, token_seed=token_seed)
        agent = self.build_agent(tenant_id, spec)
        async for event in self._drive(
            agent,
            {"messages": [HumanMessage(user_input)]},
            self._config(session_id, spec),
            ctx,
        ):
            yield event

    async def _tickets(self, interrupts: Sequence[Any]) -> list[dict[str, Any]]:
        """挂起载荷里的审批单现状（顺序 = 载荷顺序）；单不存在 → ValueError（挂起载荷与审批表不一致）。"""
        if self._approvals is None:
            raise RuntimeError(
                "恢复审批需要审批单存取件（AgentRuntime(approvals=...)）"
            )
        tickets: list[dict[str, Any]] = []
        for pending in interrupts:
            for request in pending.value.get("action_requests", ()):
                ticket = await self._approvals.get(request["approval_id"])
                if ticket is None:
                    raise ValueError(
                        f"审批单 {request['approval_id']} 不存在——挂起载荷与审批表不一致"
                    )
                tickets.append(ticket)
        return tickets

    @staticmethod
    def _decisions(tickets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """审批表终态 → 决定列表（借 HITL 决定形态：approve / reject）；决定数恒等于挂起数。调用方已保证全部终态。"""
        return [
            {
                "type": "approve"
                if ticket["status"] == ApprovalStatus.APPROVED
                else "reject",
                "approval_id": ticket["id"],
                "status": ticket["status"],
            }
            for ticket in tickets
        ]

    @staticmethod
    async def _pending_interrupts(agent: Any, config: dict[str, Any]) -> list[Any]:
        """checkpoint 里 pending 的中断（挂起点）；没有即空列表。"""
        snapshot = await agent.aget_state(config)
        return [i for task in snapshot.tasks for i in task.interrupts]

    async def resume(
        self,
        *,
        tenant_id: str,
        session_id: str,
        spec: AgentSpec,
        approval_id: str | None = None,
        cancel: CancelSignal | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """恢复单入口（审批续跑 / 崩溃恢复同路径，见模块 docstring）。approval_id 非 None = 计划内审批续跑；None = 崩溃恢复。

        产出 = 恢复段新落盘的事件（重放去重命中的旧事件不再外流）。
        """
        row = await self._owned_session(tenant_id, session_id)
        source = (
            self._resume_approval(row, spec, approval_id, cancel)
            if approval_id is not None
            else self._recover(row, spec, cancel)
        )
        async for event in source:
            yield event

    async def _resume_approval(
        self,
        row: dict[str, Any],
        spec: AgentSpec,
        approval_id: str,
        cancel: CancelSignal | None,
    ) -> AsyncIterator[AgentEvent]:
        """计划内审批续跑：awaiting_approval → 挂起点（缺失即先重放到挂起点）→ 单号属于挂起点 → 单全部终态 → T3 → Command(resume)。"""
        session_id = row["id"]
        if row["run_state"] != SessionRunState.AWAITING_APPROVAL.value:
            raise SessionBusy(
                f"会话 {session_id} 不在 awaiting_approval（当前 {row['run_state']}）：并发恢复已有赢家或会话在跑"
            )
        agent = self.build_agent(row["tenant_id"], spec)
        config = self._config(session_id, spec)
        ctx = self._context(row, spec, cancel=cancel, token_seed=0)
        interrupts = await self._pending_interrupts(agent, config)
        if not interrupts:
            async for event in self._drive(agent, None, config, ctx):
                yield event  # 重放到挂起点通常零新事件（全部去重命中）
            interrupts = await self._pending_interrupts(agent, config)
            if not interrupts:
                raise RuntimeError(
                    f"会话 {session_id} 在 awaiting_approval 但重放后仍无挂起点——事实源不一致"
                )
        tickets = await self._tickets(interrupts)
        if approval_id not in {ticket["id"] for ticket in tickets}:
            raise ValueError(f"审批单 {approval_id} 不属于会话 {session_id} 的挂起点")
        pending = [t["id"] for t in tickets if t["status"] == ApprovalStatus.PENDING]
        if pending:
            raise ValueError(
                f"审批单 {pending[0]} 仍是 pending——先 decide / cancel / expire_due 再 resume"
            )
        if not await self._sessions.transition(
            session_id,
            expected=SessionRunState.AWAITING_APPROVAL.value,
            to=SessionRunState.RUNNING.value,
        ):
            raise SessionBusy(f"会话 {session_id} 的恢复已有并发赢家")
        async for event in self._drive(
            agent, Command(resume={"decisions": self._decisions(tickets)}), config, ctx
        ):
            yield event

    async def _recover(
        self, row: dict[str, Any], spec: AgentSpec, cancel: CancelSignal | None
    ) -> AsyncIterator[AgentEvent]:
        """崩溃恢复分诊（调用方断言上一进程已死）：健康挂起 → 零事件不计次；计次超上限 → 放弃；决定已落 → 同审批续跑；
        图已收尾 → 只修状态；其余 → 从最后一个 checkpoint 之后重放（去重 / 作废重发 / 末事件去重）。"""
        session_id, state = row["id"], row["run_state"]
        if state in (SessionRunState.IDLE.value, SessionRunState.FAILED.value):
            raise ValueError(f"会话 {session_id} 处于 {state}：没有可恢复的 run")
        agent = self.build_agent(row["tenant_id"], spec)
        config = self._config(session_id, spec)
        ctx = self._context(row, spec, cancel=cancel, token_seed=0)
        snapshot = await agent.aget_state(config)
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        tickets = await self._tickets(interrupts) if interrupts else []
        if any(t["status"] == ApprovalStatus.PENDING for t in tickets):
            # 健康的挂起（等坐席），不是崩溃：不计恢复次数、零事件；run_state 停在 running = T2 之后被旁路，修回 awaiting
            if (
                state == SessionRunState.RUNNING.value
                and not await self._sessions.transition(
                    session_id,
                    expected=SessionRunState.RUNNING.value,
                    to=SessionRunState.AWAITING_APPROVAL.value,
                )
            ):
                logger.warning(u.LOG_RECOVERY_FLIP_FAILED, session_id=session_id)
            return
        count = await self._sessions.bump_recovery(session_id)
        if count is not None and count > self._recovery_limit:
            async for event in self._abandon(row, count):
                yield event
            return
        if tickets:
            # 决定已落、续跑没来得及（或续跑中途崩）：与计划内续跑同一条路径
            if (
                state == SessionRunState.AWAITING_APPROVAL.value
                and not await self._sessions.transition(
                    session_id,
                    expected=SessionRunState.AWAITING_APPROVAL.value,
                    to=SessionRunState.RUNNING.value,
                )
            ):
                raise SessionBusy(f"会话 {session_id} 的恢复已有并发赢家")
            payload: Any = Command(resume={"decisions": self._decisions(tickets)})
        elif not snapshot.next:
            # 图已收尾（末事件已在事实源）只是 T4 没翻：只修状态、零事件（框架无节点可重放）
            logger.info(u.LOG_RECOVERY_STATE_REPAIRED, session_id=session_id)
            if not await self._sessions.transition(
                session_id, expected=state, to=SessionRunState.IDLE.value
            ):
                logger.warning(u.LOG_RECOVERY_FLIP_FAILED, session_id=session_id)
            await self._sessions.reset_recovery(session_id)
            return
        else:
            payload = None  # 从最后一个 checkpoint 之后重放：ToolExec 去重 / ModelCall 作废重发 / after_agent 末事件去重
        async for event in self._drive(agent, payload, config, ctx):
            yield event

    async def _abandon(
        self, row: dict[str, Any], count: int
    ) -> AsyncIterator[AgentEvent]:
        """恢复次数超上限：T5 →failed（毒会话交人工）+ recovery_abandoned 图外事件（无框架任务身份：以本次恢复的 run_id 顶替派生输入，每次放弃各一条）。"""
        session_id = row["id"]
        logger.warning(
            u.LOG_RECOVERY_ABANDONED, session_id=session_id, recovery_count=count
        )
        if not await self._sessions.transition(
            session_id, expected=row["run_state"], to=SessionRunState.FAILED.value
        ):
            logger.warning(u.LOG_RECOVERY_FLIP_FAILED, session_id=session_id)
        run_id = uuid.uuid4().hex
        eid = event_id(session_id, run_id, "recovery_abandoned", 0)
        payload = {
            "recovery_count": count,
            "limit": self._recovery_limit,
            "run_state": row["run_state"],
        }
        seq, _ = await self._events.append(
            event_id=eid,
            tenant_id=row["tenant_id"],
            session_id=session_id,
            run_id=run_id,
            event_type=EventType.RECOVERY_ABANDONED.value,
            payload=payload,
        )
        yield AgentEvent(
            id=eid,
            tenant_id=row["tenant_id"],
            session_id=session_id,
            run_id=run_id,
            seq=seq,
            type=EventType.RECOVERY_ABANDONED,
            payload=payload,
        )
