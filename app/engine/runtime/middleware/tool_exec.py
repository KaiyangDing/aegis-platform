"""ToolExec（M2.3 骨架）：wrap_tool_call 不调框架 handler，自己执行——框架 ToolNode 只做路由（ADR-012 决策 7）。

本步只立七步里的 ①严格校验 ②身份注入 ④write-ahead ⑦事件 + 每 run 按声明序串行；
③风险闸门（通行证）⑤超时 / 读重试 / 写超时 RESULT_UNKNOWN ⑥结果规范化、连败禁用、重放去重 → reexecute、
wrap_untrusted 包裹随 M2.5 / M2.8 补齐。
两种 id 严禁混用：模型侧 tool_call["id"] 只进对话（ToolMessage.tool_call_id 配对）与事件 payload 的 model_call_id；
write-ahead 事件 id 进 ToolContext.tool_call_id（幂等键透传下游）与 tool_result / tool_error 的 tool_call_id。
幻觉工具名在 M2.3 仍由 ToolNode 以英文回填（wrap 之外，探针 F）——M2.4 的 Gates.after_model 先于它拦截。
"""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from app.engine.gateway.errors import sanitize_error_text
from app.engine.runtime import utterances as u
from app.engine.runtime.events import EventType
from app.engine.runtime.state import RunContext, RunState, emit
from app.engine.runtime.tools import ToolContext

_monotonic = time.monotonic  # 测试接缝

ToolHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]


def _jsonable(value: Any) -> Any:
    """事件 payload 存原文：非 JSON 原生类型以 str 落盘（Decimal / datetime 等），与归一化的往返口径一致。"""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _content(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )


def _predecessors(request: ToolCallRequest) -> list[str]:
    """同一轮里声明在本调用之前的 tool_call id（框架并行派发，本仓按声明序串行）。"""
    messages = request.state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return []
    ids = [call["id"] for call in last_ai.tool_calls]
    mine = request.tool_call["id"]
    return ids[: ids.index(mine)] if mine in ids else []


class ToolExec(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: ToolHandler
    ) -> ToolMessage | Command[Any]:
        ctx: RunContext = request.runtime.context
        call_id = request.tool_call["id"]
        await ctx.tool_order.wait_turn(_predecessors(request))
        try:
            return await self._execute(request)
        finally:
            await ctx.tool_order.finish(call_id)

    async def _execute(self, request: ToolCallRequest) -> ToolMessage:
        ctx: RunContext = request.runtime.context
        call = request.tool_call
        name, call_id = call["name"], call["id"]
        tool = ctx.registry.get(name)
        if tool is None or tool.args_model is None:
            # 骨架期：注册表查不到（M2.4 起 after_model 先拦）——中文回填而不是框架英文
            available = "、".join(t.name for t in ctx.registry.specs())
            return ToolMessage(
                content=u.TOOL_UNKNOWN.format(name=name, available=available),
                tool_call_id=call_id,
                name=name,
                status="error",
            )
        # ① 严格校验：lax + extra=forbid；坏参数不进 write-ahead（没有副作用要保护）
        try:
            args = tool.args_model.model_validate(call["args"])
        except ValidationError as exc:
            return ToolMessage(
                content=u.TOOL_ARGS_INVALID.format(
                    detail=sanitize_error_text(str(exc))
                ),
                tool_call_id=call_id,
                name=name,
                status="error",
            )
        args_json = args.model_dump(mode="json")
        # ④ write-ahead：tool_call 事件先落盘，事件 id 即幂等键（重放命中 created=False → M2.5 reexecute）
        event, _created = await emit(
            request.runtime,
            EventType.TOOL_CALL,
            {"tool_name": name, "args": args_json, "model_call_id": call_id},
            hook="tool_call",
            ordinal=call_id,
        )
        # ② 身份注入：LLM 不可控的四个 id + 幂等键
        tool_ctx = ToolContext(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            tool_call_id=event.id,
        )
        started = _monotonic()
        try:
            result = await tool.handler(tool_ctx, **args.model_dump())
        except Exception as exc:  # noqa: BLE001  —— 工具边界：任何失败都回填模型
            detail = sanitize_error_text(str(exc))
            await emit(
                request.runtime,
                EventType.TOOL_ERROR,
                {
                    "tool_call_id": event.id,
                    "error": detail,
                    "latency_ms": int((_monotonic() - started) * 1000),
                    "retry_count": 0,
                },
                hook="tool_error",
                ordinal=call_id,
            )
            return ToolMessage(
                content=u.TOOL_FAILED.format(detail=detail),
                tool_call_id=call_id,
                name=name,
                status="error",
            )
        # ⑦ 结果原文进事件；回填给模型的是同一份内容（规范化随 M2.5）
        await emit(
            request.runtime,
            EventType.TOOL_RESULT,
            {
                "tool_call_id": event.id,
                "result": _jsonable(result),
                "latency_ms": int((_monotonic() - started) * 1000),
                "retry_count": 0,
            },
            hook="tool_result",
            ordinal=call_id,
        )
        return ToolMessage(content=_content(result), tool_call_id=call_id, name=name)
