"""AgentRuntime 门面（M2.3；M2.4 插入 Gates、接取消信号；M2.5 网关句柄进 RunContext；M2.6 插入 AegisSummarization、栈经 build_middleware 注入依赖）：
按 (tenant_id, spec 指纹) 编译并缓存图；run() 单入口驱动一次循环、事件按 seq 序外流。

一次 run（ADR-011 / 012）：读会话行取身份并核对租户归属 → D8 种子（历史 llm_call / llm_result 的估算字段求和）→
T1 idle→running（CAS 失败 = 会话正忙）→ agent.astream(..., durability="sync", recursion_limit=推导, stream_mode=["custom"])
→ 钩子内落盘的事件经 stream_writer 到达这里逐条 yield。
图缓存：编译成本按租户 spec 摊销；指纹覆盖注入面全部会影响图形态的字段（工具 schema 含在内），改 spec 即换图。
resume（审批续跑 / 崩溃恢复单入口）随 M2.7 / M2.9。
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

from app.engine.runtime.events import AgentEvent, EventType
from app.engine.runtime.middleware.gates import Gates
from app.engine.runtime.middleware.model_call import ModelCall
from app.engine.runtime.middleware.run_events import RunEvents
from app.engine.runtime.middleware.summarization import AegisSummarization
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.protocols import (
    CancelSignal,
    EventStoreLike,
    SessionRunState,
    SessionStateLike,
)
from app.engine.runtime.spec import AgentSpec, LoopPolicy
from app.engine.runtime.state import RunContext
from app.engine.runtime.tools import ToolRegistry, to_structured_tools

MIDDLEWARE_STACK: tuple[type[AgentMiddleware], ...] = (
    RunEvents,
    AegisSummarization,
    Gates,
    ModelCall,
    ToolExec,
)
"""栈序即语义（ADR-012 决策 1）。M2.6 形态：before_model 按列表序 = 压缩 → 闸门；after_model 反序 = 闸门先于（M2.7 插在
AegisSummarization 与 Gates 之间的）审批；Guards 随 M2.8 插到 RunEvents 之后。build_middleware 与本元组一一对应（静态测试互钉）。"""

_HOOKS_PER_PHASE = {
    "before_agent": ("before_agent", "abefore_agent"),
    "before_model": ("before_model", "abefore_model"),
    "after_model": ("after_model", "aafter_model"),
    "after_agent": ("after_agent", "aafter_agent"),
}


class SessionBusy(RuntimeError):
    """T1 CAS 失败：会话不在 idle（另一 run 在跑或挂起等审批）。M3 翻译为 409。"""


def build_middleware(gateway: BaseChatModel, spec: AgentSpec) -> list[AgentMiddleware]:
    """按栈序实例化：需要依赖的中间件在这里注入——摘要模型 = 该租户网关（fast 档在调用时指定）、预算 = spec.context_config
    （在指纹里，改预算即换图）。"""
    return [
        RunEvents(),
        AegisSummarization(gateway, config=spec.context_config),
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
    """对外门面：一次 run = 一条事件流；图按 (tenant_id, spec 指纹) LRU 缓存；网关按租户装配（组合根注入闭包）。"""

    def __init__(
        self,
        *,
        gateway_for: Callable[[str], BaseChatModel],
        events: EventStoreLike,
        sessions: SessionStateLike,
        checkpointer: BaseCheckpointSaver,
        cache_size: int = 64,
    ) -> None:
        self._gateway_for = gateway_for
        self._events = events
        self._sessions = sessions
        self._checkpointer = checkpointer
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

        会话行不存在 / 不属于该租户 → ValueError（起跑前的归属校验）；不在 idle → SessionBusy。
        cancel 是闸门 #6 的取消源（None = 无取消源）。
        run 内异常（ProviderError 泄漏、事实源不可用、system 超预算）裸穿，run_state 留在 running——由恢复路径（M2.9）分诊。
        """
        row = await self._sessions.get(session_id)
        if row is None:
            raise ValueError(f"会话 {session_id} 不存在——run 之前必须先建 sessions 行")
        if row["tenant_id"] != tenant_id:
            raise ValueError(f"会话 {session_id} 不属于租户 {tenant_id}")
        token_seed = await self._token_seed(tenant_id, session_id)
        if not await self._sessions.transition(
            session_id,
            expected=SessionRunState.IDLE.value,
            to=SessionRunState.RUNNING.value,
        ):
            raise SessionBusy(f"会话 {session_id} 不在 idle：当前 {row['run_state']}")
        ctx = RunContext(
            tenant_id=tenant_id,
            user_id=row["user_id"],
            session_id=session_id,
            run_id=uuid.uuid4().hex,
            spec=spec,
            registry=ToolRegistry(spec.tools),
            events=self._events,
            sessions=self._sessions,
            token_seed=token_seed,
            cancel=cancel,
            gateway=self._gateway_for(
                tenant_id
            ),  # 增强层（工具结果摘要）的 fast 档入口，与图内模型同一租户绑定
        )
        agent = self.build_agent(tenant_id, spec)
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": recursion_limit_for(spec.policy),
        }
        async for _mode, payload in agent.astream(
            {"messages": [HumanMessage(user_input)]},
            config,
            context=ctx,
            stream_mode=["custom"],
            durability="sync",
        ):
            if isinstance(payload, AgentEvent):
                yield payload

    async def resume(
        self, *, tenant_id: str, session_id: str, spec: AgentSpec
    ) -> AsyncIterator[AgentEvent]:
        """恢复单入口（审批续跑 / 崩溃恢复同路径）：M2.7 / M2.9 实装。"""
        raise NotImplementedError("resume 随 M2.7（审批）/ M2.9（崩溃恢复）实装")
        yield  # 保持 async generator 形态
