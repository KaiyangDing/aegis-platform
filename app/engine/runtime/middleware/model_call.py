"""ModelCall：唯一的 wrap_model_call——上下文编译（M2.6）、网关载体、闸门 #1（wrap 内重发也计）/ #3 预检、llm_call / llm_result 事件、
L1 四组异常映射为终止事实（ADR-012 决策 2/3；契约 C4 / C8 / C16）。

流程：上下文编译（compile_prompt 只改 request 不改 state——system 层 + 摘要 / 旧轮 + 当前 user + 折叠后的工具结果层）
→ 载体（tier / deadline_s / session_id / max_tokens 经 model_settings → bind_tools kwargs → 网关 _astream 具名参数，探针 E）
→ 估算 input（自家尺：编译后 prompt + 工具 schema）→ 循环 { #1 轮数 → #3 预算 → iteration+1 → llm_call →
handler → 四组 except → llm_result }。流级中断作废重发（消耗迭代，探针 B：wrap 拿不到流、撤不回已外送块）。
终止 = 返回 ExtendedModelResponse：结果里放兜底 AIMessage（零话术时放空 AIMessage——保证"最后一条 AIMessage 无 tool_calls"
让模型→工具边走向 end，探针⑹ + factory 边规则），command 写 termination / iteration / tokens_used（同 superstep 对 after_model 可见）。
绝不 except GatewayError 基类：ProviderError 泄漏与事实源异常裸炸。
出口守卫终检（M2.8 挂点③，ADR-012 判据④）：llm_result 写原文之后、AIMessage 进 state 之前——OutputGuard 整段 feed + flush
（逐字符 ≡ 整段的不变量让 M3 真流式共享同一行为）+ final_check 兜底；命中 → guardrail_triggered(stream|final) 事件 + 替换后的
AIMessage 进 state（流中命中 = 已放行前缀 + SAFE_REPLY，终局命中 = 整条 SAFE_REPLY；工具轮只留放行前缀不补话术）并打
GUARDRAIL_TRUNCATED 标记，checkpoint 不留泄漏原文。入口打标（entry_notice 通道）随编译进 system 层。
半截 LLM（M2.9，契约 C12）：重放命中既有 llm_call（进程死在该次调用完成之前）→ 补配对 llm_result(interrupted, cause=replay)
后以下一序号重发，重发消耗迭代；显式接受重生成文本不同（保事实不保字节）。
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
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from app.core.logs import get_logger
from app.core.tokens import estimate_messages_tokens, estimate_tokens, message_text
from app.engine.gateway.errors import (
    BudgetExceeded,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    TenantQuotaExceeded,
)
from app.engine.runtime import utterances as u
from app.engine.runtime.context import compile_prompt
from app.engine.runtime.events import EventType
from app.engine.runtime.guards import Guardrails, output_audit_payload
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import (
    GUARDRAIL_TRUNCATED,
    RunContext,
    RunState,
    emit,
    terminated,
)

logger = get_logger(__name__)
_monotonic = time.monotonic  # 测试接缝

Handler = Callable[[ModelRequest[RunContext]], Awaitable[ModelResponse]]


def _tool_calls_payload(message: AIMessage) -> list[dict[str, Any]]:
    return [
        {"id": call["id"], "name": call["name"], "args": call["args"]}
        for call in message.tool_calls
    ]


class ModelCall(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    def __init__(self, guards: Guardrails | None = None) -> None:
        """guards：出口守卫工厂（规则库缺省 v1；每次模型调用按 spec 新建一个 OutputGuard 实例）。"""
        super().__init__()
        self._guards = guards or Guardrails()

    async def awrap_model_call(
        self, request: ModelRequest[RunContext], handler: Handler
    ) -> ModelResponse | ExtendedModelResponse:
        runtime = request.runtime
        ctx = runtime.context
        policy = ctx.spec.policy
        iteration = request.state.get("iteration", 0)
        tokens_used = request.state.get("tokens_used", 0)
        # 上下文编译：prompt 是 state 的有损投影（system 超预算在此 fail-loud，ValueError 裸穿）
        compiled = compile_prompt(
            request.messages, ctx.spec, notice=request.state.get("entry_notice")
        )
        request = request.override(
            system_message=compiled.system,
            messages=compiled.messages,
            model_settings={
                **request.model_settings,
                "tier": ctx.spec.model_tier,
                "deadline_s": policy.llm_step_timeout_s,
                "session_id": ctx.session_id,
                "max_tokens": ctx.spec.context_config.output_reserve,
            },
        )
        input_est = estimate_messages_tokens(
            compiled.all_messages,
            tools=[convert_to_openai_tool(t) for t in request.tools],
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
            _, created = await emit(
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
            if not created:
                # 重放命中既有 llm_call = 上一进程死在这次调用完成之前（有 llm_call 无 llm_result；若旧结果其实已落盘，
                # 这条 interrupted 会被同 id 去重吸收）：作废重发——配对 llm_result(interrupted, cause=replay)，下一序号重发，消耗迭代
                logger.info(u.LOG_LLM_REPLAY, iteration=iteration)
                await emit(
                    runtime,
                    EventType.LLM_RESULT,
                    {
                        "iteration": iteration,
                        "status": "interrupted",
                        "cause": "replay",
                        "detail": u.LLM_REPLAY_DETAIL,
                    },
                    hook="llm_result",
                    ordinal=attempt,
                )
                attempt += 1
                continue
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
            screened, audit = self._screen(message, ctx)
            if audit is not None:
                # 出口守卫命中：审计事件先于替换后的 AIMessage 进 state（checkpoint 不留泄漏原文；llm_result 保留原文供审计）
                await emit(
                    runtime,
                    EventType.GUARDRAIL_TRIGGERED,
                    audit,
                    hook="guardrail_output",
                    ordinal=attempt,
                )
                response = ModelResponse(result=[*response.result[:-1], screened])
            return ExtendedModelResponse(
                model_response=response,
                command=Command(
                    update={
                        "iteration": iteration,
                        "tokens_used": tokens_used + input_est + output_est,
                    }
                ),
            )

    def _screen(
        self, message: AIMessage, ctx: RunContext
    ) -> tuple[AIMessage, dict[str, Any] | None]:
        """出口终检：无命中原样返回 (message, None)；命中返回 (替换后的 AIMessage, 审计 payload)。

        文本回复：流中命中 = 已放行前缀 + SAFE_REPLY，终局命中 = 整条 SAFE_REPLY；工具轮的前置文本命中只留放行前缀、不补话术
        （本轮没有对用户的回复位，链路继续走工具）——审计是义务，改写回复不是。
        """
        text = message.text
        if not text:
            return message, None
        spec = ctx.spec
        guard = self._guards.output_guard(
            system_prompt=spec.system_prompt,  # 片段集用 spec 原文：不可信声明是公开机制说明，模型复述无害不设防
            tool_names=[t.name for t in spec.tools],
            owned_values=spec.owned_values,
        )
        visible = guard.feed(text) + guard.flush()
        if guard.hit is not None:
            audit = output_audit_payload(guard.hit, stage="stream")
            replaced = visible if message.tool_calls else visible + u.SAFE_REPLY
        else:
            hits = guard.final_check(visible)
            if not hits:
                return message, None
            audit = output_audit_payload(hits[0], stage="final")
            replaced = "" if message.tool_calls else u.SAFE_REPLY
        screened = message.model_copy(
            update={
                "content": replaced,
                "additional_kwargs": {
                    **message.additional_kwargs,
                    GUARDRAIL_TRUNCATED: True,
                },
            }
        )
        return screened, audit

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
