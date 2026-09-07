"""ModelCall：唯一的 wrap_model_call——网关载体、闸门 #1（wrap 内重发也计）/ #3 预检、llm_call / llm_result 事件、
L1 四组异常映射为终止事实（ADR-012 决策 2/3；契约 C4 / C16）。

流程：载体（tier / deadline_s / session_id / max_tokens 经 model_settings → bind_tools kwargs → 网关 _astream 具名参数，探针 E）
→ 估算 input（自家尺：system + messages + 工具 schema）→ 循环 { #1 轮数 → #3 预算 → iteration+1 → llm_call →
handler → 四组 except → llm_result }。流级中断作废重发（消耗迭代，探针 B：wrap 拿不到流、撤不回已外送块）。
终止 = 返回 ExtendedModelResponse：结果里放兜底 AIMessage（零话术时放空 AIMessage——保证"最后一条 AIMessage 无 tool_calls"
让模型→工具边走向 end，探针⑹ + factory 边规则），command 写 termination / iteration / tokens_used（同 superstep 对 after_model 可见）。
绝不 except GatewayError 基类：ProviderError 泄漏与事实源异常裸炸。
M2.6 在本文件接上下文编译（只改 request）；M2.8 接出口守卫终检（AIMessage 进 state 之前替换）。
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from app.core.tokens import estimate_messages_tokens, estimate_tokens, message_text
from app.engine.gateway.errors import (
    BudgetExceeded,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    TenantQuotaExceeded,
)
from app.engine.runtime.events import EventType
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import RunContext, RunState, emit, terminated

_monotonic = time.monotonic  # 测试接缝

Handler = Callable[[ModelRequest[RunContext]], Awaitable[ModelResponse]]


def _tool_calls_payload(message: AIMessage) -> list[dict[str, Any]]:
    return [
        {"id": call["id"], "name": call["name"], "args": call["args"]}
        for call in message.tool_calls
    ]


class ModelCall(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    async def awrap_model_call(
        self, request: ModelRequest[RunContext], handler: Handler
    ) -> ModelResponse | ExtendedModelResponse:
        runtime = request.runtime
        ctx = runtime.context
        policy = ctx.spec.policy
        iteration = request.state.get("iteration", 0)
        tokens_used = request.state.get("tokens_used", 0)
        request = request.override(
            model_settings={
                **request.model_settings,
                "tier": ctx.spec.model_tier,
                "deadline_s": policy.llm_step_timeout_s,
                "session_id": ctx.session_id,
                "max_tokens": ctx.spec.context_config.output_reserve,
            }
        )
        prompt: list[BaseMessage] = list(request.messages)
        if request.system_message is not None:
            prompt.insert(0, request.system_message)
        input_est = estimate_messages_tokens(
            prompt, tools=[convert_to_openai_tool(t) for t in request.tools]
        )
        attempt = 0
        while True:
            # 闸门 #1：按 LLM 调用计、调用前查（wrap 内的作废重发同样受限）
            if iteration >= policy.max_iterations:
                return self._terminate(
                    TerminationReason.MAX_ITERATIONS,
                    detail=f"已完成 {iteration} 次 LLM 调用，达 max_iterations 上限",
                    iteration=iteration,
                    tokens_used=tokens_used,
                )
            # 闸门 #3：会话预算预检——不打注定超预算的半截请求（L2 预检不带 cause）
            if tokens_used + input_est > policy.session_token_budget:
                return self._terminate(
                    TerminationReason.TOKEN_BUDGET_EXCEEDED,
                    detail=(
                        f"累计估算 {tokens_used} + 本次输入 {input_est} "
                        f"> session_token_budget={policy.session_token_budget}"
                    ),
                    iteration=iteration,
                    tokens_used=tokens_used,
                )
            iteration += 1
            await emit(
                runtime,
                EventType.LLM_CALL,
                {
                    "iteration": iteration,
                    "tier": ctx.spec.model_tier,
                    "input_tokens_est": input_est,
                },
                hook="llm_call",
                ordinal=attempt,
            )
            started = _monotonic()
            try:
                response = await handler(request)
            except (GatewayExhausted, GatewayOverloadedError) as exc:
                # 组一（闸门 #2 的 LLM 半边）：deadline 耗尽 / 本地过载 = 步作废
                cause = (
                    "gateway_exhausted"
                    if isinstance(exc, GatewayExhausted)
                    else "gateway_overloaded"
                )
                return await self._fail_step(
                    runtime,
                    TerminationReason.STEP_TIMEOUT,
                    cause=cause,
                    detail=str(exc),
                    iteration=iteration,
                    tokens_used=tokens_used,
                    ordinal=attempt,
                )
            except (BudgetExceeded, TenantQuotaExceeded) as exc:
                # 组二：三级预算共用终止原因，cause 区分层级
                cause = (
                    "l1_request_budget"
                    if isinstance(exc, BudgetExceeded)
                    else "l1_tenant_quota"
                )
                return await self._fail_step(
                    runtime,
                    TerminationReason.TOKEN_BUDGET_EXCEEDED,
                    cause=cause,
                    detail=str(exc),
                    iteration=iteration,
                    tokens_used=tokens_used,
                    ordinal=attempt,
                )
            except GatewayRejected as exc:
                # 组三：确定性拒绝 = 配置 / 协议 bug 信号——零话术，detail 带已打码的错误文本
                return await self._fail_step(
                    runtime,
                    TerminationReason.GATEWAY_REJECTED,
                    cause="gateway_rejected",
                    detail=str(exc),
                    iteration=iteration,
                    tokens_used=tokens_used,
                    ordinal=attempt,
                )
            except GatewayStreamInterrupted as exc:
                # 组四：作废重发——配对 llm_result(interrupted) 后再发一次，重发消耗迭代
                await emit(
                    runtime,
                    EventType.LLM_RESULT,
                    {
                        "iteration": iteration,
                        "status": "interrupted",
                        "detail": f"{exc}；死因：{exc.__cause__!r}",
                    },
                    hook="llm_result",
                    ordinal=attempt,
                )
                attempt += 1
                continue
            message = response.result[-1]
            assert isinstance(message, AIMessage)
            output_est = estimate_tokens(message_text(message))
            usage = message.usage_metadata
            await emit(
                runtime,
                EventType.LLM_RESULT,
                {
                    "iteration": iteration,
                    "status": "ok",
                    "text": message.content
                    if isinstance(message.content, str)
                    else message_text(message),
                    "tool_calls": _tool_calls_payload(message),
                    "stop_reason": message.response_metadata.get("finish_reason"),
                    "model": message.response_metadata.get("model_name"),
                    "usage": (
                        {
                            "prompt_tokens": usage["input_tokens"],
                            "completion_tokens": usage["output_tokens"],
                        }
                        if usage
                        else None
                    ),
                    "output_tokens_est": output_est,
                    "latency_ms": int((_monotonic() - started) * 1000),
                },
                hook="llm_result",
                ordinal=attempt,
            )
            return ExtendedModelResponse(
                model_response=response,
                command=Command(
                    update={
                        "iteration": iteration,
                        "tokens_used": tokens_used + input_est + output_est,
                    }
                ),
            )

    async def _fail_step(
        self,
        runtime: Any,
        reason: TerminationReason,
        *,
        cause: str,
        detail: str,
        iteration: int,
        tokens_used: int,
        ordinal: int,
    ) -> ExtendedModelResponse:
        """终止型异常的统一收尾：先配对 llm_result(failed)（进程内不留孤儿 llm_call），再终止。"""
        await emit(
            runtime,
            EventType.LLM_RESULT,
            {
                "iteration": iteration,
                "status": "failed",
                "cause": cause,
                "detail": detail,
            },
            hook="llm_result",
            ordinal=ordinal,
        )
        return self._terminate(
            reason,
            detail=detail,
            cause=cause,
            iteration=iteration,
            tokens_used=tokens_used,
        )

    @staticmethod
    def _terminate(
        reason: TerminationReason,
        *,
        detail: str,
        iteration: int,
        tokens_used: int,
        cause: str | None = None,
    ) -> ExtendedModelResponse:
        """wrap 内唯一合法的终止出口：兜底 AIMessage 进结果（零话术时为空 AIMessage），事实进 termination 通道。"""
        term = terminated(reason, detail=detail, cause=cause)
        return ExtendedModelResponse(
            model_response=ModelResponse(
                result=[AIMessage(content=term["fallback"] or "")]
            ),
            command=Command(
                update={
                    "iteration": iteration,
                    "tokens_used": tokens_used,
                    "termination": term,
                }
            ),
        )
