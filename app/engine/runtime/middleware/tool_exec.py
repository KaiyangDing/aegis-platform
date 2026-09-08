"""ToolExec：wrap_tool_call 不调框架 handler，自己执行七步——框架 ToolNode 只做路由（ADR-012 决策 7）。M2.3 立骨架，M2.5 补齐。

顺序（契约 C6 + 闸门 #6 / #2 工具半边）：
  闸门 #6 取消检查点（每个调用前）→ 可用性（幻觉名兜底 / 连败禁用）→ ① 严格校验（lax + extra=forbid；坏参数不进 write-ahead）
  → ③ 风险闸门（fail-closed：谓词崩溃即阻断；approved_calls 通行证只由 Approvals 写入，M2.7）
  → ④ write-ahead（tool_call 事件先落盘，事件 id 即幂等键；重放命中既有事件 = reexecute：同一把钥匙、绝不产生第二把）
  → ② 身份注入 ToolContext（LLM 不可控）→ ⑤ asyncio.timeout 取更严，读可退避重试、写恒单次、写超时 = RESULT_UNKNOWN 封死重试话术
  → ⑥ 超预算收缩（fast 档摘要经网关，fail-open 硬截断；产物随事件留痕）→ ⑦ tool_result / tool_error 事件 + 连败两次本轮禁用。
五结局 ToolOutcome（tools.py）。两种 id 严禁混用：模型侧 tool_call["id"] 只进对话配对与事件 payload 的 model_call_id；
write-ahead 事件 id 进 ToolContext.tool_call_id 与 tool_result / tool_error 的 tool_call_id。
每 run 按声明序串行（ToolSerializer）；连败账 / 禁用集 / "本步已写 termination" 住 RunContext.tool_health——
tools 节点每调用一任务，通道同一步只许一个写者（探针 M2.4-Q3）。工具实现一律 async def（同步函数走线程池会丢上下文）。
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.agents.middleware.internal_call_transformer import (
    internal_call_metadata,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import ValidationError

from app.core.logs import get_logger
from app.core.tokens import estimate_tokens
from app.engine.gateway.errors import sanitize_error_text
from app.engine.runtime import utterances as u
from app.engine.runtime.events import EventType
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import (
    FAIL_STREAK_LIMIT,
    GATEWAY_FAILURES,
    INTERNAL_CALL_TAG,
    RunContext,
    RunState,
    ToolHealth,
    discard_note,
    emit,
    terminated,
)
from app.engine.runtime.tools import (
    OutcomeKind,
    SideEffect,
    ToolContext,
    ToolDef,
    ToolOutcome,
)

logger = get_logger(__name__)
_monotonic = time.monotonic  # 测试接缝
_sleep = asyncio.sleep  # 读重试退避的测试接缝（测试不真睡）

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


def _declared_ids(request: ToolCallRequest) -> list[str]:
    """本轮最后一条 AIMessage 声明的 tool_call id（声明序 = 串行序 = 事件序）。"""
    messages = request.state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return [] if last_ai is None else [call["id"] for call in last_ai.tool_calls]


def _truncate_to_budget(text: str, budget_tokens: int) -> str:
    """确定性硬截断（fail-open 兜底）：按估算尺 0.8 倍循环缩短进预算，尾部标注去向（原文在事件流）。"""
    while estimate_tokens(text) > budget_tokens and len(text) > 1:
        text = text[: max(1, int(len(text) * 0.8))]
    return text + u.CLIP_SUFFIX


class ToolExec(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: ToolHandler
    ) -> ToolMessage | Command[Any]:
        ctx: RunContext = request.runtime.context
        call = request.tool_call
        ids = _declared_ids(request)
        index = ids.index(call["id"]) if call["id"] in ids else 0
        await ctx.tool_order.wait_turn(ids[:index])
        try:
            # 闸门 #6 工具检查点：每个调用执行前查一次（与 LLM 调用前那次成对）
            if ctx.cancel is not None and ctx.cancel.is_set():
                return self._cancelled(ctx, call, index=index, total=len(ids))
            outcome = await self._execute(request)
        finally:
            await ctx.tool_order.finish(call["id"])
        return ToolMessage(
            content=outcome.content,
            tool_call_id=call["id"],
            name=call["name"],
            status="success" if outcome.kind is OutcomeKind.OK else "error",
        )

    @staticmethod
    def _cancelled(
        ctx: RunContext, call: Mapping[str, Any], *, index: int, total: int
    ) -> ToolMessage | Command[Any]:
        """取消：该调用不执行、配对回填；一轮里只有第一个被弃置的调用写 termination（通道单写者），其余只回 ToolMessage。
        Gates.before_model 看到 termination 即 jump end——不再调模型。"""
        message = ToolMessage(
            content=u.TOOL_CANCELLED,
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
        if ctx.tool_health.terminated:
            return message
        ctx.tool_health.terminated = True
        return Command(
            update={
                "messages": [message],
                "termination": terminated(
                    TerminationReason.CANCELLED,
                    detail=discard_note("收到取消信号（工具检查点）", total, index),
                ),
            }
        )

    async def _execute(self, request: ToolCallRequest) -> ToolOutcome:
        ctx: RunContext = request.runtime.context
        health = ctx.tool_health
        call = request.tool_call
        name, call_id = call["name"], call["id"]
        tool = ctx.registry.get(name)
        if tool is None or tool.args_model is None:
            # 幻觉工具名：Gates.after_model 已先拦（M2.4），这里是兜底——没有工具可禁用，不进连败账
            available = "、".join(t.name for t in ctx.registry.specs())
            return ToolOutcome(
                OutcomeKind.ERROR,
                name,
                u.TOOL_UNKNOWN.format(name=name, available=available),
            )
        if name in health.disabled:
            return ToolOutcome(
                OutcomeKind.DISABLED,
                name,
                u.TOOL_DISABLED.format(name=name, limit=FAIL_STREAK_LIMIT),
            )
        # ① 严格校验：lax + extra=forbid——说明书答应的（数字字符串）验货必须认，说明书没有的（幻觉参数）零容忍；
        #    坏参数不进 write-ahead：没有副作用要保护
        try:
            args = tool.args_model.model_validate(call["args"])
        except ValidationError as exc:
            return self._fail(
                health,
                name,
                u.TOOL_ARGS_INVALID.format(detail=sanitize_error_text(str(exc))),
            )
        # ③ 风险闸门：确定性安全闸门，fail-closed——评估不了绝不放行；通行证（approved_calls）只由 Approvals 写入（M2.7）
        if tool.risk_policy is not None and call_id not in request.state.get(
            "approved_calls", []
        ):
            try:
                needs_approval = tool.risk_policy(args, ctx.spec.tenant_config)
            except Exception as exc:  # noqa: BLE001  —— 谓词崩溃 = 阻断
                return self._fail(
                    health,
                    name,
                    u.TOOL_RISK_EVAL_FAILED.format(
                        detail=sanitize_error_text(str(exc))
                    ),
                )
            if needs_approval:
                # M2.7 起由 Approvals.after_model 先于 tools 开单挂起；走到这里 = 没拿到通行证，最后一道闸：不执行
                return ToolOutcome(
                    OutcomeKind.NEEDS_APPROVAL,
                    name,
                    u.TOOL_NEEDS_APPROVAL.format(name=name),
                )
        # ④ write-ahead：tool_call 事实先落盘，插入成功是执行副作用的前置；事件 id 即幂等键。
        #    重放命中既有事件（created=False）= reexecute：同一把钥匙透传下游去重，绝不产生第二把
        event, created = await emit(
            request.runtime,
            EventType.TOOL_CALL,
            {
                "tool_name": name,
                "args": args.model_dump(mode="json"),
                "model_call_id": call_id,
            },
            hook="tool_call",
            ordinal=call_id,
        )
        if not created:
            logger.info(u.LOG_TOOL_REEXECUTE, tool=name, tool_call_id=event.id)
        # ② 身份注入：LLM 不可控的四个 id + 幂等键
        tool_ctx = ToolContext(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            tool_call_id=event.id,
        )
        return await self._run(
            request.runtime, tool, tool_ctx, args.model_dump(), model_call_id=call_id
        )

    async def _run(
        self,
        runtime: Runtime[RunContext],
        tool: ToolDef,
        tool_ctx: ToolContext,
        kwargs: Mapping[str, Any],
        *,
        model_call_id: str,
    ) -> ToolOutcome:
        """⑤ 超时取更严；读可退避重试、写绝不（attempts 按 side_effect 分支是第一道保险，ToolDef"写 retries 恒 0"是第二道）；
        ⑥ 成功后超预算收缩；⑦ tool_result / tool_error 留痕。"""
        ctx = runtime.context
        health = ctx.tool_health
        policy = ctx.spec.policy
        name, event_id = tool.name, tool_ctx.tool_call_id
        timeout_s = (
            policy.tool_step_timeout_s
            if tool.timeout_s is None
            else min(tool.timeout_s, policy.tool_step_timeout_s)
        )
        attempts_allowed = 1 + (
            tool.retries if tool.side_effect is SideEffect.READ else 0
        )
        started = _monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(timeout_s):
                    result = await tool.handler(tool_ctx, **kwargs)
                break
            except TimeoutError:
                if tool.side_effect is SideEffect.WRITE:
                    # 写超时 = 结果不明：副作用可能已在下游生效；模型若自发重试会生成新幂等键，下游去重当场失效——
                    # 话术封死重试、引导查询确认；不进连败账（这不是"失败"）
                    await self._error(
                        runtime,
                        event_id,
                        u.TOOL_ERROR_TIMEOUT_UNKNOWN,
                        started,
                        attempt - 1,
                        ordinal=model_call_id,
                    )
                    return ToolOutcome(
                        OutcomeKind.RESULT_UNKNOWN,
                        name,
                        u.TOOL_RESULT_UNKNOWN.format(name=name),
                        event_id,
                    )
                if attempt >= attempts_allowed:
                    await self._error(
                        runtime,
                        event_id,
                        u.TOOL_ERROR_TIMEOUT.format(timeout_s=timeout_s),
                        started,
                        attempt - 1,
                        ordinal=model_call_id,
                    )
                    return self._fail(
                        health,
                        name,
                        u.TOOL_TIMEOUT.format(timeout_s=timeout_s),
                        event_id,
                    )
                await _sleep(0.2 * attempt)
            except Exception as exc:  # noqa: BLE001  —— 工具边界：任何失败都回填模型
                detail = sanitize_error_text(str(exc))
                if attempt >= attempts_allowed:
                    await self._error(
                        runtime,
                        event_id,
                        detail,
                        started,
                        attempt - 1,
                        ordinal=model_call_id,
                    )
                    return self._fail(
                        health, name, u.TOOL_FAILED.format(detail=detail), event_id
                    )
                await _sleep(0.2 * attempt)
        # 成功：连败账清零 → ⑥ 规范化 → ⑦ 事件留痕（原文永远全量在事件流，收缩产物随事件留痕）
        health.record_success(name)
        content = _content(result)
        payload: dict[str, Any] = {
            "tool_call_id": event_id,
            "result": _jsonable(result),
            "latency_ms": int((_monotonic() - started) * 1000),
            "retry_count": attempt - 1,
        }
        if estimate_tokens(content) > ctx.spec.context_config.tool_results_budget:
            content = await self._shrink(ctx, content, payload)
        await emit(
            runtime,
            EventType.TOOL_RESULT,
            payload,
            hook="tool_result",
            ordinal=model_call_id,
        )
        return ToolOutcome(OutcomeKind.OK, name, content, event_id)

    @staticmethod
    def _fail(
        health: ToolHealth, name: str, content: str, event_id: str | None = None
    ) -> ToolOutcome:
        """记连败账：达上限即禁用，并在当次回填里宣告——模型立刻知道该改道。"""
        streak = health.record_failure(name)
        if streak >= FAIL_STREAK_LIMIT:
            content += u.TOOL_STREAK_DISABLED.format(streak=streak)
        return ToolOutcome(OutcomeKind.ERROR, name, content, event_id)

    @staticmethod
    async def _error(
        runtime: Runtime[RunContext],
        event_id: str,
        error: str,
        started: float,
        retry_count: int,
        *,
        ordinal: str,
    ) -> None:
        await emit(
            runtime,
            EventType.TOOL_ERROR,
            {
                "tool_call_id": event_id,
                "error": error,
                "latency_ms": int((_monotonic() - started) * 1000),
                "retry_count": retry_count,
            },
            hook="tool_error",
            ordinal=ordinal,
        )

    @staticmethod
    async def _shrink(ctx: RunContext, raw: str, payload: dict[str, Any]) -> str:
        """⑥ 超预算收缩：首选 fast 档摘要（经网关，内部调用打标不进外流），网关缺席或六类公开异常一律 fail-open 硬截断
        （增强层坏了往活里放，与风险闸门的 fail-closed 相反方向）；收缩产物随事件留痕——摘要是 LLM 产物，不留痕回放重建不出模型视界。"""
        budget = ctx.spec.context_config.tool_results_budget
        if ctx.gateway is not None:
            try:
                response = await ctx.gateway.ainvoke(
                    [HumanMessage(f"{u.TOOL_DIGEST_PROMPT}\n\n{raw}")],
                    config={
                        "tags": [INTERNAL_CALL_TAG],
                        "metadata": {
                            "lc_source": "tool_digest",
                            **internal_call_metadata(),
                        },
                    },
                    tier="fast",
                    session_id=ctx.session_id,
                    deadline_s=ctx.spec.policy.llm_step_timeout_s,
                )
            except GATEWAY_FAILURES as exc:
                payload["summarize_error"] = sanitize_error_text(str(exc))
                logger.warning(u.LOG_TOOL_DIGEST_FALLBACK, error=type(exc).__name__)
            else:
                summary = u.TOOL_SUMMARY_PREFIX + response.text.strip()
                if estimate_tokens(summary) > budget:
                    summary = _truncate_to_budget(summary, budget)
                payload["injected"] = summary
                payload["normalization"] = "summary"
                return summary
        shrunk = _truncate_to_budget(raw, budget)
        payload["injected"] = shrunk
        payload["normalization"] = "truncated"
        return shrunk
