"""网关测试共用替身：剧本候选、熔断/出站闸桩、SDK 异常构造。不是测试文件（不以 test_ 开头）。"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field

REQ = httpx2.Request("POST", "http://test/v1/chat/completions")
HANG = object()


def text(s: str) -> AIMessageChunk:
    return AIMessageChunk(content=s)


def finish(reason: str = "stop", *, usage: bool = True) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        response_metadata={"finish_reason": reason},
        usage_metadata=(
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            if usage
            else None
        ),
        chunk_position="last",
    )


def ok() -> list[Any]:
    return [text("好"), finish()]


class ScriptedCandidate(BaseChatModel):
    """按剧本演出的假候选：每次 astream 消费一幕（幕用完停在最后一幕）。
    幕内元素：块→yield，异常→raise，HANG→挂起，可调用→执行（假时钟）。走真实的 BaseChatModel.astream 外壳。"""

    acts: list[list[Any]]
    calls: int = 0
    closed: int = 0
    seen_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.calls += 1
        self.seen_kwargs.append({**kwargs, "stop": stop})
        act = self.acts[min(self.calls, len(self.acts)) - 1]
        try:
            for item in act:
                if item is HANG:
                    await asyncio.sleep(999)
                elif isinstance(item, BaseException):
                    raise item
                elif callable(item):
                    item()
                else:
                    yield ChatGenerationChunk(message=item)
        finally:
            self.closed += 1


def scripted(*acts: list[Any]) -> ScriptedCandidate:
    return ScriptedCandidate(acts=list(acts))


def status_error(
    status: int, body: dict | None = None, headers: dict | None = None
) -> openai.APIStatusError:
    """按 SDK 自己的映射造异常：与真实响应路径同一张表。"""
    resp = httpx2.Response(
        status,
        request=REQ,
        headers=headers or {},
        json={"error": body or {"message": "m"}},
    )
    client = openai.OpenAI(api_key="fake", base_url="http://test/v1", max_retries=0)
    return client._make_status_error_from_response(resp)


class StubBreaker:
    """决策表按候选 key（provider:model）；记录三种上报与入口判定。"""

    def __init__(self, decisions: dict[str, str] | None = None) -> None:
        self.decisions = decisions or {}
        self.allowed: list[str] = []
        self.successes: list[str] = []
        self.failures: list[str] = []
        self.releases: list[str] = []

    async def allow(self, key: str) -> str:
        self.allowed.append(key)
        return self.decisions.get(key, "allow")

    async def report_success(self, key: str, *, probe: bool) -> None:
        self.successes.append(key)

    async def report_failure(self, key: str, *, probe: bool) -> None:
        self.failures.append(key)

    async def release_probe(self, key: str) -> None:
        self.releases.append(key)


class StubLimiter:
    """按 provider 拒绝；记录每次取令牌的 (provider, max_wait)。"""

    def __init__(self, deny: set[str] | None = None) -> None:
        self.deny = deny or set()
        self.asked: list[tuple[str, float | None]] = []

    async def acquire(self, key: str, max_wait: float | None) -> bool:
        self.asked.append((key, max_wait))
        return key not in self.deny
