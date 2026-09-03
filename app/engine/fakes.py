"""测试与压测线 B 共用的 fake 层：零真实 LLM 调用的根基。

两种 fake：
  FakeToolChatModel  剧本回放（测试用）：按预写顺序吐 AIMessage，内容与 tool_calls 完全由测试掌控；
                     流式路径吐带 tool_call_chunks 的块——GenericFakeChatModel._stream 忽略 tool_calls，
                     工具剧本走流式路径会产出 0 块而崩（探针 probe_m12_b 复现）。
  FakeReplyChatModel 默认回复（AEGIS_FAKE_LLM=1 / 线 B）：固定文案、固定非零 usage、首块前延迟。
开关落点在候选工厂（engine/gateway/candidates.py），本文件不读配置。
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    UsageMetadata,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

CHUNK_CHARS = 4

FAKE_REPLY = (
    "感谢您的耐心等待。根据当前记录，您的问题已登记并转交相应人员处理，"
    "预计 1–3 个工作日内回复；如需进一步帮助，请随时告诉我。"
)
# 非零 usage 是刻意的：记账落行、月度 SUM 这些写路径也要被压到（数值任意）
FAKE_USAGE = UsageMetadata(input_tokens=120, output_tokens=48, total_tokens=168)

_sleep = asyncio.sleep  # 测试接缝：延迟可替换/可观测，测试不真睡


def script_chunks(message: AIMessage) -> list[AIMessageChunk]:
    """一条 AIMessage → 流式块序列，形状对齐 ChatOpenAI 的真流：
    文本按定长切块；每个 tool_call 一块 tool_call_chunks（args 为 JSON 串，合并后由框架解析回 dict）；
    末块带 usage_metadata 与 finish_reason，并标 chunk_position="last"（否则框架再补一个空块）。
    """
    chunks: list[AIMessageChunk] = []
    content = message.content if isinstance(message.content, str) else ""
    for i in range(0, len(content), CHUNK_CHARS):
        chunks.append(
            AIMessageChunk(content=content[i : i + CHUNK_CHARS], id=message.id)
        )
    for index, call in enumerate(message.tool_calls):
        chunks.append(
            AIMessageChunk(
                content="",
                id=message.id,
                tool_call_chunks=[
                    tool_call_chunk(
                        name=call["name"],
                        args=json.dumps(call["args"], ensure_ascii=False),
                        id=call["id"],
                        index=index,
                    )
                ],
            )
        )
    finish = "tool_calls" if message.tool_calls else "stop"
    chunks.append(
        AIMessageChunk(
            content="",
            id=message.id,
            usage_metadata=message.usage_metadata,
            response_metadata={"finish_reason": finish},
            chunk_position="last",
        )
    )
    return chunks


class FakeToolChatModel(GenericFakeChatModel):
    """支持 bind_tools 与流式工具块的剧本回放模型。

    GenericFakeChatModel.bind_tools 抛 NotImplementedError（core 1.6.1 探针实证，ADR-002），
    而 create_agent 必须调用它——覆写为返回自身：工具 schema 被忽略，模型行为只由剧本决定。
    """

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeToolChatModel:
        return self

    def _next_message(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        result = self._generate(messages, run_manager=None, **kwargs)
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        return message

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for piece in script_chunks(self._next_message(messages, stop=stop, **kwargs)):
            chunk = ChatGenerationChunk(message=piece)
            if run_manager:
                run_manager.on_llm_new_token(str(piece.content), chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for piece in script_chunks(self._next_message(messages, stop=stop, **kwargs)):
            chunk = ChatGenerationChunk(message=piece)
            if run_manager:
                await run_manager.on_llm_new_token(str(piece.content), chunk=chunk)
            yield chunk


def scripted_model(*messages: AIMessage) -> FakeToolChatModel:
    """从剧本构造 fake 模型：scripted_model(AIMessage(...), AIMessage(...))。"""
    return FakeToolChatModel(messages=iter(messages))


class FakeReplyChatModel(BaseChatModel):
    """确定性默认回复替身：整段回与流式回同一文案、同一 usage；延迟只在首块前一次性发生
    （真模型的时延大头在首 token 之前）。"""

    reply: str = FAKE_REPLY
    delay_s: float = 0.0
    usage: UsageMetadata = FAKE_USAGE

    @property
    def _llm_type(self) -> str:
        return "aegis-fake-reply"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeReplyChatModel:
        return self

    def _message(self) -> AIMessage:
        return AIMessage(content=self.reply, usage_metadata=self.usage)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        return ChatResult(generations=[ChatGeneration(message=self._message())])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay_s > 0:
            await _sleep(self.delay_s)
        return ChatResult(generations=[ChatGeneration(message=self._message())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        for piece in script_chunks(self._message()):
            yield ChatGenerationChunk(message=piece)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.delay_s > 0:
            await _sleep(self.delay_s)
        for piece in script_chunks(self._message()):
            yield ChatGenerationChunk(message=piece)
