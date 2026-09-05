"""AegisGateway：档位路由 + 候选环 + 终局三段——网关的总装车间。

网关就是一个 BaseChatModel（ADR-005）：`_astream` 是主实现，`_agenerate` 聚合流，同步路径不支持，
`bind_tools` 返回绑定视图（原实例不变），类级 `cache=False`。agent 层对网关内部零知识；
档位 / deadline / session_id 以 kwargs 表达（`gateway.bind(tier="strong", deadline_s=30, session_id=sid)`），
租户身份在构造时绑定（`tenant_id`，组合根每请求装配一个网关实例），request_id 由网关每次调用自生成——四项载体见 ADR-009。

九步（v1 顺序即设计）：①路由防御 ②deadline 换算 ③缓存 ④月度预算 ⑤单请求预算 ⑥租户出站闸
⑦候选环 ⑧簿记在流尾 ⑨终局三段。

候选环每站：deadline 预检 → 熔断入口判定（allow/probe/deny）→ 受控重试（出站闸按尝试计，在
complete_with_retry 的 acquire 缝内）→ 三待遇分流。
两条红线（v1）：半截不换路；租户配额在候选环外（换供应商换不掉租户身份，配额尽则立刻明确失败）。
三待遇：5xx/超时进熔断账再换路；429 换路不进账；出站闸满换路不进账且不作终局死因；Auth/BadRequest 换路不进账、计确定性拒绝；
Overloaded 不进账不换路裸穿；弃流/取消/未知异常不进账不清账，只归还试探锁。
缓存：命中 = 零上游成本，在一切闸门之前短路，仍记一行零成本账（命中率的分母在账本里）；只有流耗尽的完整回复才入库。
预算：月度（resolver 三态：值 / 静态配置 / None=读挂 fail-open；budget≤0 不查账）与单请求估算（自家尺，0=关）
都在出站闸之前——被拒的请求不该消耗配额；成本护栏不是安全边界，读挂放行并告警。
簿记在流尾：熔断上报 / 入缓存 / 记账全在 `async for` 耗尽之后；记账失败只告警（为了发票烧掉货物是荒唐的）。
"""

import time
import uuid
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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    UsageMetadata,
)
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import field_validator

from app.core.logs import get_logger
from app.core.tokens import estimate_messages_tokens
from app.engine.gateway import utterances as u
from app.engine.gateway.cache import CachedReply, CacheLike, request_digest
from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    BudgetExceeded,
    GatewayError,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    OutboundGateTimeout,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
    TenantQuotaExceeded,
)
from app.engine.gateway.protocols import (
    BreakerLike,
    BudgetResolver,
    LimiterLike,
    MeterLike,
)
from app.engine.gateway.resilience import AcquireFn, RetryPolicy, complete_with_retry
from app.engine.gateway.routing import Candidate
from app.engine.gateway.tenancy import validate_tenant_id

# 只有这两类进熔断账：429 是限流的领地（上游活着），Auth/BadRequest 是本家的问题，供应商无辜
_BREAKER_COUNTED = (ProviderServerError, ProviderTimeoutError)
CACHE_PROVIDER = "cache"  # 命中回放行的 provider 值（与 domain.usage.CACHE_PROVIDER 同串，分层不许互引）

_monotonic = time.monotonic  # 测试接缝

logger = get_logger(__name__)


def _new_request_id() -> str:
    return uuid.uuid4().hex


class AegisGateway(BaseChatModel):
    cache: BaseCache | bool | None = False  # 框架缓存显式关闭：ainvoke 内部流式会查缓存
    tenant_id: str  # 租户身份：缓存前缀/账本列/配额桶键共用；必填，构造即校验
    routes: dict[str, list[Candidate]]
    models: dict[Candidate, BaseChatModel]
    breaker: BreakerLike
    limiter: LimiterLike
    reply_cache: CacheLike | None = None  # 租户前缀精确缓存（M1.5a）；None = 关缓存
    meter: MeterLike | None = None  # 记账员（M1.5c）；None = 不记账、月度闸关闭
    tenant_limiter: LimiterLike | None = (
        None  # 租户出站桶（ADR-008 A′）；None = 无租户节流
    )
    budget_resolver: BudgetResolver | None = None  # 月度预算事实源；None = 用静态配置
    monthly_token_budget: int = 0  # 静态月度预算；0 = 关闭
    request_token_budget: int = 0  # 单请求估算预算；0 = 关闭
    retry_policy: RetryPolicy = RetryPolicy()
    default_tier: str = "standard"

    @field_validator("tenant_id")
    @classmethod
    def _check_tenant_id(cls, value: str) -> str:
        return validate_tenant_id(value)

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

    # ---------------------------------------------------------------- 预算与记账

    async def _resolve_monthly_budget(self) -> int | None:
        """预算三态：resolver 值（租户表，M3）/ 静态配置（resolver=None）/ None=读挂 fail-open 跳闸门。"""
        if self.budget_resolver is None:
            return self.monthly_token_budget
        try:
            return await self.budget_resolver(self.tenant_id)
        except Exception:
            logger.warning(
                u.LOG_BUDGET_READ_FAILED, tenant_id=self.tenant_id, exc_info=True
            )
            return None

    async def _check_monthly_budget(self) -> None:
        """④ 租户月度预算闸门（软预算 fail-open：账本/租户表读挂了放行并告警——为一次读抖动拒绝所有用户是代价倒挂）。"""
        if self.meter is None or (
            self.budget_resolver is None and self.monthly_token_budget <= 0
        ):
            return
        budget = await self._resolve_monthly_budget()
        if budget is None or budget <= 0:
            return  # None=读挂 fail-open；≤0=该租户闸门关闭（不白查账本）
        try:
            spent = await self.meter.month_spend(self.tenant_id)
        except Exception:
            logger.warning(
                u.LOG_BUDGET_READ_FAILED, tenant_id=self.tenant_id, exc_info=True
            )
            return
        if spent >= budget:
            raise BudgetExceeded(
                u.BUDGET_MONTHLY.format(
                    tenant_id=self.tenant_id, spent=spent, budget=budget
                )
            )

    def _check_request_budget(
        self, messages: Sequence[BaseMessage], kwargs: dict[str, Any]
    ) -> None:
        """⑤ 单请求预算闸门（三级预算的 L1 级）：挡超长上下文炸弹；只估 prompt 侧（输出上界由 max_tokens 约束）。"""
        if self.request_token_budget <= 0:
            return
        estimated = estimate_messages_tokens(messages, kwargs.get("tools", ()))
        if estimated > self.request_token_budget:
            raise BudgetExceeded(
                u.BUDGET_REQUEST.format(
                    estimated=estimated, budget=self.request_token_budget
                )
            )

    async def _check_tenant_gate(
        self, deadline: float | None, policy: RetryPolicy
    ) -> None:
        """⑥ 租户出站闸：候选环外把关；等待预算由 deadline 剩余反推（与供应商闸同一算法）。"""
        if self.tenant_limiter is None:
            return
        max_wait = None
        if deadline is not None:
            max_wait = max(0.0, deadline - _monotonic() - policy.min_attempt_budget)
        if not await self.tenant_limiter.acquire(self.tenant_id, max_wait):
            raise TenantQuotaExceeded(u.TENANT_QUOTA.format(tenant_id=self.tenant_id))

    async def _safe_record(
        self,
        *,
        request_id: str,
        session_id: str | None,
        tier: str,
        provider: str,
        model: str,
        usage: UsageMetadata | None,
        cached: bool,
    ) -> None:
        """记账失败绝不拖垮请求——为了发票烧掉货物是荒唐的；缺口留给对账脚本暴露。"""
        if self.meter is None:
            return
        try:
            await self.meter.record(
                tenant_id=self.tenant_id,
                request_id=request_id,
                session_id=session_id,
                tier=tier,
                provider=provider,
                model=model,
                prompt_tokens=usage["input_tokens"] if usage else 0,
                completion_tokens=usage["output_tokens"] if usage else 0,
                cached=cached,
                usage_missing=usage is None,
            )
        except Exception:
            logger.warning(
                u.LOG_METER_WRITE_FAILED,
                tenant_id=self.tenant_id,
                request_id=request_id,
                exc_info=True,
            )

    # ---------------------------------------------------------------- 主路径

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tier: str | None = None,
        deadline_s: float | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # ① 路由防御：parse_routes 已保证齐档，但手工构造的路由表可能缺档——在消耗任何配额之前干净地失败
        tier = tier or self.default_tier
        candidates = self.routes.get(tier)
        if not candidates:
            raise GatewayExhausted(u.ROUTE_MISSING.format(tier=tier))
        request_id = (
            _new_request_id()
        )  # 一次调用一行账：agent 一轮多次调用各有各的 request_id
        # ② deadline 只约束"首块前"的空转：换算成绝对单调钟沿候选链与重试层传播；
        #    首块流出后不再看它——整流不设上限，块间空闲由传输层双闸守护
        deadline = None if deadline_s is None else _monotonic() + deadline_s
        # ③ 缓存：最外圈——命中 = 零上游成本，不消耗任何配额、不问任何闸门；
        #    key 不含 deadline/session_id（传输参数不是语义），故障在缓存内部降级为 miss
        digest: str | None = None
        if self.reply_cache is not None:
            digest = request_digest(tier, messages, stop=stop, **kwargs)
            hit = await self.reply_cache.get(digest)
            if hit is not None:
                usage: UsageMetadata | None = None
                for chunk in hit.chunks:
                    if chunk.usage_metadata is not None:
                        usage = chunk.usage_metadata
                    yield ChatGenerationChunk(message=chunk)
                # 命中也记账（provider="cache"，cached=True → 成本 0）：命中率统计的分母在这
                await self._safe_record(
                    request_id=request_id,
                    session_id=session_id,
                    tier=tier,
                    provider=CACHE_PROVIDER,
                    model=hit.model,
                    usage=usage,
                    cached=True,
                )
                return
        policy = self.retry_policy
        await self._check_monthly_budget()  # ④
        self._check_request_budget(messages, kwargs)  # ⑤
        await self._check_tenant_gate(deadline, policy)  # ⑥ 红线二：租户配额在候选环外
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
            buffer: list[AIMessageChunk] = []  # 只在缓存开启时收集：整流耗尽才入库
            usage_seen: UsageMetadata | None = (
                None  # 上游通常只在末块给 usage；缺失 → usage_missing
            )
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
                        if chunk.usage_metadata is not None:
                            usage_seen = chunk.usage_metadata
                        if digest is not None:
                            buffer.append(chunk)
                        yield ChatGenerationChunk(message=chunk)
                # ⑧ 簿记在流尾：只有流耗尽才算成功；弃流/取消走不到这里——熔断上报、入缓存、记账一个都不发生
                await self.breaker.report_success(cand.key, probe=probe)
                adjudicated = True
                if digest is not None and self.reply_cache is not None:
                    await self.reply_cache.put(
                        digest, CachedReply(cand.provider, cand.model, buffer)
                    )
                await self._safe_record(
                    request_id=request_id,
                    session_id=session_id,
                    tier=tier,
                    provider=cand.provider,
                    model=cand.model,
                    usage=usage_seen,
                    cached=False,
                )
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
