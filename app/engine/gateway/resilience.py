"""单候选韧性：翻译补全 + 受控重试 + 三段超时 + deadline 传播。

三条铁律（v1 逐字沿用）：
1. 只重试无业务副作用的操作——LLM 补全可重复执行，唯一代价是重复计费，
   所以尝试次数与总时限都有硬预算；
2. 只在"首块之前"重试——一旦有 chunk 流向下游，重试会造成重复输出；
   中途失败属于"半截输出"问题，归上层（候选环包成 GatewayStreamInterrupted，L2 恢复语义）处置；
3. 退避 = 指数 + 满抖动，429 优先服从服务端 Retry-After——
   无抖动的同步重试会让所有客户端一起冲撞刚恢复的上游（惊群）。

首块 = 首个可见块（content / tool_call_chunks / finish_reason / usage 任一在场）：思考型模型的 reasoning delta
被框架吐成 content='' 的空块（探针），不喂首块计时器、不向下游透传。
三段超时（C5）：connect 归候选工厂的 httpx2.Timeout；首块归本模块的 asyncio.timeout（与 deadline 剩余取小）；
块间归 httpx2 read（字节级）与 stream_chunk_timeout（已解析块级，取值 ≥ 首块窗，ADR-006）。整流不设上限。
出站闸（C8）：每次尝试先过 acquire 缝，等待发生在首块计时器之外，等待上限由 deadline 与 total_timeout 剩余反推——
本地排队不冤枉供应商、不进熔断账、不吃掉首块预算。
翻译（C3）：候选抛出的 SDK/传输异常在这里过 classify；截断流（无 finish_reason）与零块流不当成功。
"""

import asyncio
import math
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage

from app.core.logs import get_logger
from app.engine.gateway import utterances as u
from app.engine.gateway.errors import (
    AuthError,
    GatewayError,
    OutboundGateTimeout,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
    classify,
)

RETRYABLE_ERRORS = (RateLimitedError, ProviderTimeoutError, ProviderServerError)

# 测试接缝：单测替换这三个名字来记录/加速，而不是打全局 asyncio/random/time 的补丁
_sleep = asyncio.sleep
_uniform = random.uniform
_monotonic = time.monotonic

# 出站闸缝：入参 = 本次允许的最长等待秒数（<= 0 表示只试一次不排队；None 表示无上限约束，闸自定），
# 返回是否取得令牌。闸自身抛出的异常是编程错误：不翻译、不重试、原样穿出。
# M1.3 默认无闸；M1.4b 由候选环注入"限时取令牌"的真件。
AcquireFn = Callable[[float | None], Awaitable[bool]]

logger = get_logger(__name__)


async def _no_gate(max_wait: float | None) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3  # 总尝试次数（含第一次）；半开试探用 replace(max_attempts=1)
    base_backoff: float = 0.5  # 首次退避基数（秒）
    max_backoff: float = 8.0  # 单次退避上限（Retry-After 同样封顶）
    total_timeout: float = 60.0  # 单候选墙钟预算（含闸内排队与退避）：耗尽不再开新尝试
    first_chunk_timeout: float = 25.0  # 首块超时：切断"连上了但不吐字"的挂起形态
    min_attempt_budget: float = 8.0  # 剩余预算低于此值不再开新尝试

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(u.RETRY_POLICY_INVALID.format(field="max_attempts"))
        if self.first_chunk_timeout <= 0:
            raise ValueError(u.RETRY_POLICY_INVALID.format(field="first_chunk_timeout"))
        for name in (
            "base_backoff",
            "max_backoff",
            "total_timeout",
            "min_attempt_budget",
        ):
            if getattr(self, name) < 0:
                raise ValueError(u.RETRY_POLICY_INVALID.format(field=name))


def compute_backoff(
    attempt: int, policy: RetryPolicy, retry_after: float | None
) -> float:
    """服务端指令优先（封顶、不套抖动）；否则指数增长 + 满抖动。非有限/负的 Retry-After 视为没给。"""
    if retry_after is not None and math.isfinite(retry_after):
        return min(max(0.0, retry_after), policy.max_backoff)
    cap = min(policy.base_backoff * (2 ** (attempt - 1)), policy.max_backoff)
    return _uniform(0.0, cap)


def _first_wait(policy: RetryPolicy, deadline: float | None) -> float:
    """首块窗口上双闸：参数闸（first_chunk_timeout）与全局闸（deadline 剩余）取小。"""
    wait = policy.first_chunk_timeout
    if deadline is not None:
        wait = min(wait, max(0.0, deadline - _monotonic()))
    return wait


def _gate_budget(policy: RetryPolicy, deadline: float | None, start: float) -> float:
    """出站闸最多可等多久：total_timeout 剩余与 deadline 剩余取小，再扣掉一次像样尝试的下限。"""
    now = _monotonic()
    remaining = policy.total_timeout - (now - start)
    if deadline is not None:
        remaining = min(remaining, deadline - now)
    return max(0.0, remaining - policy.min_attempt_budget)


def _translate(provider: str, exc: Exception) -> GatewayError | None:
    """翻译并挂 __cause__；未知异常返回 None 让调用方裸抛（把编程错误伪装成上游故障会藏起 bug）。

    AuthError 例外：401 响应体回显密钥原文，SDK 异常不进 __cause__ 链——否则 traceback 渲染
    （日志 exc_info）会绕过 sanitize_error_text 的打码。调用方在 except 块之外 raise，__context__ 也不会挂上。
    """
    err = classify(provider, exc)
    if err is None or err is exc:
        return err
    err.__cause__ = None if isinstance(err, AuthError) else exc
    err.__suppress_context__ = True
    return err


def _finished(chunk: AIMessageChunk) -> bool:
    """截断检测判据（探针⑨）：v1 的 [DONE] 哨兵在 SDK 层不可见，改看 finish_reason 是否到场。"""
    return chunk.response_metadata.get("finish_reason") is not None


def _visible(chunk: AIMessageChunk) -> bool:
    """首块判据：可见 token / 工具块 / 终止信号 / usage 任一在场；只含 reasoning delta 的空块不算。"""
    return (
        bool(chunk.content)
        or bool(chunk.tool_call_chunks)
        or _finished(chunk)
        or chunk.usage_metadata is not None
    )


async def _first_chunk(
    stream: AsyncGenerator[AIMessageChunk], provider: str, first_wait: float
) -> AIMessageChunk:
    """首块窗口：首块前无任何输出流向下游，取消 anext 是安全的。

    deadline 已过（first_wait=0）时立即超时——开首次尝试前的预检归调用方（候选环）。
    """
    try:
        async with asyncio.timeout(first_wait) as cut:
            while True:
                chunk = await anext(stream)
                if _visible(chunk):
                    return chunk
                # 空块不满足首块窗、也不透传：思考期的挂起照样会被切断
    except TimeoutError as e:
        if not cut.expired():
            # 框架的 StreamChunkTimeoutError（只在其阈值 < 首块窗时发生，探针⑥）：交给翻译表，不冒充首块超时
            raise
        # 挂起统一翻译成 ProviderTimeoutError：与 5xx 走同一条重试/记熔断账/换路流水线。
        # 探针⑧：切断时底层连接已随取消展开同步释放，这里的 aclose() 是保险且与 v1 同形
        await stream.aclose()
        raise ProviderTimeoutError(
            provider, u.FIRST_CHUNK_TIMEOUT.format(wait=first_wait)
        ) from e
    except StopAsyncIteration:
        # 框架候选一块都没吐时抛 ValueError（探针⑨，classify 已翻译）；这里兜住非框架候选的空流
        raise ProviderServerError(provider, u.STREAM_EMPTY) from None


async def complete_with_retry(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    provider: str,
    policy: RetryPolicy | None = None,
    deadline: float | None = None,  # 绝对单调钟时刻；只约束首块前
    acquire: AcquireFn = _no_gate,
    **kwargs: Any,
) -> AsyncGenerator[AIMessageChunk]:
    policy = policy or RetryPolicy()
    start = _monotonic()  # 单调钟：测时长不用壁钟（壁钟会被校时跳变）
    attempt = 0
    last_error: GatewayError | None = None
    while True:
        attempt += 1
        # 出站闸：按尝试计、在首块计时器之外；排不上时首次尝试抛 OutboundGateTimeout（候选环按 429 待遇：
        # 换路、不进账、不作终局死因），重试尝试则裸抛上一次的真实死因——闸满不是新故障
        gate_started = _monotonic()
        if not await acquire(_gate_budget(policy, deadline, start)):
            waited = _monotonic() - gate_started
            if last_error is not None:
                logger.info(
                    "gateway.gate_refused",
                    provider=provider,
                    attempt=attempt,
                    waited_s=round(waited, 3),
                    error=type(last_error).__name__,
                )
                raise last_error
            raise OutboundGateTimeout(
                provider, u.OUTBOUND_GATE_TIMEOUT.format(wait=waited)
            )
        stream = model.astream(messages, **kwargs)
        err: GatewayError | None = None
        try:
            first = await _first_chunk(stream, provider, _first_wait(policy, deadline))
        except Exception as e:
            # 翻译漏斗：未知类型立刻裸抛，实际捕获面 = 翻译表认识的类型
            err = _translate(provider, e)
            if err is None:
                raise
        if err is not None:
            # 离开 except 块再 raise：SDK 异常不会被解释器挂到 __context__（AuthError 断链才彻底）
            if not isinstance(err, RETRYABLE_ERRORS):
                raise err
            last_error = err
            if attempt >= policy.max_attempts:
                raise err
            delay = compute_backoff(attempt, policy, getattr(err, "retry_after", None))
            now = _monotonic()
            if now - start + delay > policy.total_timeout:
                raise err
            if (
                deadline is not None
                and now + delay + policy.min_attempt_budget > deadline
            ):
                raise err  # 全局首块预算不够再开一次像样的尝试：真实死因原样上抛，不造新异常
            logger.info(
                "gateway.retry",
                provider=provider,
                attempt=attempt,
                delay_s=round(delay, 3),
                error=type(err).__name__,
            )
            await _sleep(delay)
            continue
        # 首块已到手：从此进入"不可重试区"，任何错误翻译后原样上抛（候选环包成 StreamInterrupted）。
        # aclosing：消费者提前挂断时 GeneratorExit 同步传进候选流；底层 httpx2 连接由 asyncio 事件循环的
        # 异步生成器终结钩子在下一个循环周期释放（内层生成器引用计数归零即调度 aclose，探针⑧），不依赖循环回收器
        saw_finish = _finished(first)
        try:
            async with aclosing(stream) as inner:
                yield first
                async for chunk in inner:
                    saw_finish = saw_finish or _finished(chunk)
                    yield chunk
        except Exception as e:
            err = _translate(provider, e)
            if err is None:
                raise
        if err is not None:
            raise err
        if not saw_finish:
            # 干净断连/正文半途结束都走到这：没见到 finish_reason 的流是截断，不是成功（C3）
            raise ProviderServerError(provider, u.STREAM_TRUNCATED)
        return
