"""故障注入器：包装候选模型，按概率注入三种故障形态——演示/实验专用，prod 启动即炸（config 校验器）。

自己就是 BaseChatModel，重试/熔断/候选环对它一视同仁，不知道故障是演的：
  error     首块前抛 5xx（模拟连接阶段失败）；
  hang      首块前挂起不吐字——由网关首块超时真实切断，不是模拟切断；
  midstream 吐出首块后死掉——触发半截语义 GatewayStreamInterrupted。
包装发生在候选表层（inject_faults，组合根调用），网关本体对注入零知识。
"""

import asyncio
import random
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from contextlib import aclosing
from typing import Any, Literal

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

from app.engine.gateway import utterances as u
from app.engine.gateway.errors import ProviderServerError, ProviderTimeoutError
from app.engine.gateway.routing import Candidate

FaultMode = Literal["error", "hang", "midstream"]

# 测试接缝：注入是否命中、hang 的睡眠
_random = random.random
_hang_sleep = asyncio.sleep


class FaultInjector(BaseChatModel):
    inner: BaseChatModel
    provider: str
    rate: float
    mode: FaultMode = "error"
    hang_s: float = 30.0
    cache: BaseCache | bool | None = False

    @property
    def _llm_type(self) -> str:
        return "aegis-fault-injector"

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
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=list(tools), **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """与网关同形：整段回 = 流式回的聚合，注入形态对两条路径一致。"""
        merged: AIMessageChunk | None = None
        async for chunk in self._astream(messages, stop=stop, **kwargs):
            piece = chunk.message
            merged = piece if merged is None else merged + piece
        if merged is None:
            merged = AIMessageChunk(content="")
        return ChatResult(
            generations=[ChatGeneration(message=message_chunk_to_message(merged))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        inject = _random() < self.rate
        if inject and self.mode == "error":
            raise ProviderServerError(self.provider, u.FAULT_ERROR)
        if inject and self.mode == "hang":
            await _hang_sleep(self.hang_s)  # 等着被首块超时取消——考验真实机制
            raise ProviderServerError(self.provider, u.FAULT_HANG)
        async with aclosing(self.inner.astream(messages, stop=stop, **kwargs)) as inner:
            if inject and self.mode == "midstream":
                first = await anext(inner, None)
                if first is not None:
                    yield ChatGenerationChunk(
                        message=first
                    )  # 首块已流出：下游进入半截境地
                raise ProviderTimeoutError(self.provider, u.FAULT_MIDSTREAM)
            async for chunk in inner:
                yield ChatGenerationChunk(message=chunk)


def inject_faults(
    models: Mapping[Candidate, BaseChatModel],
    *,
    rate: float,
    targets: Collection[str],
    mode: FaultMode,
    hang_s: float,
) -> dict[Candidate, BaseChatModel]:
    """候选表 → 被点名的候选换成注入器包装（`provider:model` 点名）；rate=0 原样返回。"""
    if rate <= 0:
        return dict(models)
    targets = {t.strip() for t in targets}  # 环境变量里的空白不该让点名静默失效
    return {
        cand: (
            FaultInjector(
                inner=model, provider=cand.provider, rate=rate, mode=mode, hang_s=hang_s
            )
            if cand.key in targets
            else model
        )
        for cand, model in models.items()
    }
