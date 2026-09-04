"""AegisGateway：档位路由 + 候选环 + 终局三段——网关的总装车间。

网关就是一个 BaseChatModel（ADR-005）：`_astream` 是主实现，`_agenerate` 聚合流，同步路径不支持，
`bind_tools` 返回绑定视图（原实例不变），类级 `cache=False`。agent 层对网关内部零知识；
档位与 deadline 以 kwargs 表达（`gateway.bind(tier="strong", deadline_s=30)`），载体终裁归 ADR-009。

九步（v1 顺序即设计）：①路由防御 ②deadline 换算 ③缓存 ④月度预算 ⑤单请求预算 ⑥租户出站闸
⑦候选环 ⑧簿记在流尾 ⑨终局三段。本步立 ①②⑦⑨与⑧的熔断上报；③④⑤⑥及⑧的缓存/记账留钩子位（M1.5）。

候选环每站：deadline 预检 → 熔断入口判定（allow/probe/deny）→ 受控重试（出站闸按尝试计，在
complete_with_retry 的 acquire 缝内）→ 三待遇分流。
两条红线（v1）：半截不换路；租户配额在环外（M1.5c）。
三待遇：5xx/超时进熔断账再换路；429 换路不进账；出站闸满换路不进账且不作终局死因；Auth/BadRequest 换路不进账、计确定性拒绝；
Overloaded 不进账不换路裸穿；弃流/取消/未知异常不进账不清账，只归还试探锁。
"""

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.engine.gateway import utterances as u
from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    GatewayError,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    OutboundGateTimeout,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
)
from app.engine.gateway.protocols import BreakerLike, LimiterLike
from app.engine.gateway.resilience import AcquireFn, RetryPolicy, complete_with_retry
from app.engine.gateway.routing import Candidate

# 只有这两类进熔断账：429 是限流的领地（上游活着），Auth/BadRequest 是本家的问题，供应商无辜
_BREAKER_COUNTED = (ProviderServerError, ProviderTimeoutError)

_monotonic = time.monotonic  # 测试接缝


class AegisGateway(BaseChatModel):
    cache: BaseCache | bool | None = False  # 框架缓存显式关闭：ainvoke 内部流式会查缓存
    routes: dict[str, list[Candidate]]
    models: dict[Candidate, BaseChatModel]
    breaker: BreakerLike
    limiter: LimiterLike
    retry_policy: RetryPolicy = RetryPolicy()
    default_tier: str = "standard"

    @property
    def _llm_type(self) -> str:
        return "aegis-gateway"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(u.SYNC_NOT_SUPPORTED)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """返回绑定视图，原实例不变；tools 统一转成 OpenAI 格式经 kwargs 转发给候选。
        tool_choice=None 不上线：ChatOpenAI 会把 None 原样放进请求体（探针）。"""
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """整段回 = 流式回的聚合：同一条候选环，不存在第二条路径。"""
        merged: AIMessageChunk | None = None
        async for chunk in self._astream(messages, stop=stop, **kwargs):
            piece = chunk.message
            merged = piece if merged is None else merged + piece
        if merged is None:  # complete_with_retry 对零块流抛错，此处只是类型收口
            merged = AIMessageChunk(content="")
        return ChatResult(
            generations=[ChatGeneration(message=message_chunk_to_message(merged))]
        )

    def _gate(self, provider: str) -> AcquireFn:
        async def acquire(max_wait: float | None) -> bool:
            return await self.limiter.acquire(provider, max_wait)

        return acquire

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tier: str | None = None,
        deadline_s: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # ① 路由防御：parse_routes 已保证齐档，但手工构造的路由表可能缺档——在消耗任何配额之前干净地失败
        tier = tier or self.default_tier
        candidates = self.routes.get(tier)
        if not candidates:
            raise GatewayExhausted(u.ROUTE_MISSING.format(tier=tier))
        # ② deadline 只约束"首块前"的空转：换算成绝对单调钟沿候选链与重试层传播；
        #    首块流出后不再看它——整流不设上限，块间空闲由传输层双闸守护
        deadline = None if deadline_s is None else _monotonic() + deadline_s
        # ③ 缓存 ④ 月度预算 ⑤ 单请求预算 ⑥ 租户出站闸：钩子位（M1.5a/M1.5c）
        policy = self.retry_policy
        last_error: GatewayError | None = None
        budget_out = False
        rejections = 0  # 确定性拒绝（Auth/BadRequest）计数
        transients = 0  # 暂时性因素计数：熔断拒/限流拒/5xx/超时/429
        for cand in candidates:  # ⑦ 候选环
            if (
                deadline is not None
                and deadline - _monotonic() < policy.min_attempt_budget
            ):
                budget_out = True  # 剩余预算连一次像样的尝试都开不起：停止换路
                break
            decision = await self.breaker.allow(cand.key)
            if decision == "deny":
                transients += 1
                continue  # 秒拒：一次都不打扰，也不排出站闸的队
            probe = decision == "probe"
            adjudicated = (
                False  # 上报过成功/失败即为裁决；无裁决的结局在 finally 归还试探锁
            )
            yielded = False
            try:
                stream = complete_with_retry(
                    self.models[cand],
                    messages,
                    provider=cand.provider,
                    policy=replace(policy, max_attempts=1) if probe else policy,
                    deadline=deadline,
                    acquire=self._gate(cand.provider),
                    stop=stop,
                    **kwargs,
                )
                async with aclosing(stream) as chunks:
                    async for chunk in chunks:
                        yielded = True
                        yield ChatGenerationChunk(message=chunk)
                # ⑧ 簿记在流尾：只有流耗尽才算成功；弃流/取消走不到这里（入缓存/记账钩子位同此）
                await self.breaker.report_success(cand.key, probe=probe)
                adjudicated = True
                return
            except _BREAKER_COUNTED as e:
                await self.breaker.report_failure(cand.key, probe=probe)
                adjudicated = True
                if yielded:
                    raise self._interrupted(cand) from e  # 红线一：半截不换路，账照记
                transients += 1
                last_error = e
            except OutboundGateTimeout as e:
                if yielded:
                    raise self._interrupted(cand) from e
                transients += 1  # 本地闸排不上：换路、不进账、不作终局死因（v1 wait_take False 语义）
            except RateLimitedError as e:
                if yielded:
                    raise self._interrupted(cand) from e
                transients += 1
                last_error = e  # 429 不记熔断账：上游活着，只是挤
            except (AuthError, BadRequestError) as e:
                if yielded:
                    raise self._interrupted(cand) from e
                rejections += 1
                last_error = e  # 本家的配置/转换问题，别家未必过不去
            except GatewayOverloadedError as e:
                if yielded:
                    raise self._interrupted(cand) from e
                raise  # 本地连接池排队超时：不进账、不换路（所有候选共用一个池）
            finally:
                if probe and not adjudicated:
                    await self.breaker.release_probe(cand.key)
        # ⑨ 终局三段
        if budget_out:
            raise GatewayExhausted(
                u.FIRST_CHUNK_BUDGET_EXHAUSTED.format(tier=tier, deadline_s=deadline_s)
            ) from last_error
        if rejections > 0 and transients == 0:
            raise GatewayRejected(u.ALL_REJECTED.format(tier=tier)) from last_error
        raise GatewayExhausted(u.ALL_UNAVAILABLE.format(tier=tier)) from last_error

    @staticmethod
    def _interrupted(cand: Candidate) -> GatewayStreamInterrupted:
        return GatewayStreamInterrupted(
            u.STREAM_INTERRUPTED.format(provider=cand.provider, model=cand.model)
        )
