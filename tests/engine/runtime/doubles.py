"""runtime 测试共用替身（非测试文件）：内存事实源 / 内存会话状态机 / 剧本网关 / 一键装配的 AgentRuntime。

网关是真 AegisGateway（租户绑定、候选环、异常翻译全在），候选是走真实框架外壳的 ScriptedCandidate；
事实源与状态机用内存版（结构匹配 engine 协议），checkpointer 用 InMemorySaver——真 PG 只在 test_runtime_pg 用。
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from app.engine.fakes import script_chunks
from app.engine.gateway.errors import ProviderServerError
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate
from app.engine.runtime.events import AgentEvent
from app.engine.runtime.runtime import AgentRuntime
from tests.engine.gateway.doubles import ScriptedCandidate, StubBreaker, StubLimiter

USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def turn(message: AIMessage) -> list[AIMessageChunk]:
    """一幕 = 一条 AIMessage 的流式块序列（含 usage 与 finish_reason 末块）。"""
    if message.usage_metadata is None:
        message = AIMessage(
            content=message.content,
            tool_calls=message.tool_calls,
            usage_metadata=USAGE,
        )
    return script_chunks(message)


def text_turn(text: str) -> list[AIMessageChunk]:
    return turn(AIMessage(text))


def tool_turn(*calls: tuple[str, dict[str, Any], str]) -> list[AIMessageChunk]:
    return turn(
        AIMessage(
            content="",
            tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in calls],
        )
    )


class MemoryEventStore:
    """EventSink + EventSource 的内存实现：按 id 去重、按会话播 seq；可预置历史行（D8 种子测试）。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}

    async def append(
        self,
        *,
        event_id: str,
        tenant_id: str,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: Any,
        task_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> tuple[int, bool]:
        if event_id in self.by_id:
            return self.by_id[event_id]["seq"], False
        seq = 1 + max(
            (r["seq"] for r in self.rows if r["session_id"] == session_id), default=0
        )
        row = {
            "id": event_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "run_id": run_id,
            "seq": seq,
            "type": event_type,
            "payload": dict(payload),
            "schema_version": 1,
            "task_id": task_id,
            "checkpoint_id": checkpoint_id,
        }
        self.rows.append(row)
        self.by_id[event_id] = row
        return seq, True

    async def read(
        self,
        tenant_id: str,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = sorted(
            (
                r
                for r in self.rows
                if r["tenant_id"] == tenant_id
                and r["session_id"] == session_id
                and r["seq"] > after_seq
            ),
            key=lambda r: r["seq"],
        )
        if limit is not None:
            rows = rows[:limit]
        return [dict(r) for r in rows]

    def types(self, session_id: str) -> list[str]:
        return [r["type"] for r in self.rows if r["session_id"] == session_id]


class MemorySessionStore:
    """SessionStateLike 的内存实现（CAS 语义与 domain 版一致）。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def create(self, session_id: str, *, tenant_id: str, user_id: str) -> None:
        self.rows[session_id] = {
            "id": session_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "run_state": "idle",
            "recovery_count": 0,
        }

    async def get(self, session_id: str) -> dict[str, Any] | None:
        row = self.rows.get(session_id)
        return dict(row) if row is not None else None

    async def transition(self, session_id: str, *, expected: str, to: str) -> bool:
        row = self.rows.get(session_id)
        if row is None or row["run_state"] != expected:
            return False
        row["run_state"] = to
        return True

    async def bump_recovery(self, session_id: str) -> int | None:
        row = self.rows.get(session_id)
        if row is None:
            return None
        row["recovery_count"] += 1
        return row["recovery_count"]

    async def reset_recovery(self, session_id: str) -> None:
        row = self.rows.get(session_id)
        if row is not None:
            row["recovery_count"] = 0


class RaisingModel(BaseChatModel):
    """把网关内部家族异常直接抛出的"网关"替身——ProviderError 泄漏必须裸炸的证人。"""

    @property
    def _llm_type(self) -> str:
        return "raising"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RaisingModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ProviderServerError("p1", "泄漏的内部异常")

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise ProviderServerError("p1", "泄漏的内部异常")
        yield  # 保持 async generator 形态


def scripted_gateway_factory(
    candidate: ScriptedCandidate, *, tier: str = "fast", **gateway_kwargs: Any
) -> Callable[[str], AegisGateway]:
    """按租户装配真网关（单候选 p1、桩熔断 / 桩出站闸），候选实例跨租户共享以便断言调用。"""
    cand = Candidate("p1", "model-p1")

    def factory(tenant_id: str) -> AegisGateway:
        return AegisGateway(
            tenant_id=tenant_id,
            routes={tier: [cand]},
            models={cand: candidate},
            breaker=StubBreaker(),
            limiter=StubLimiter(),
            default_tier=tier,
            **gateway_kwargs,
        )

    return factory


def make_runtime(
    *acts: list[Any],
    events: MemoryEventStore | None = None,
    sessions: MemorySessionStore | None = None,
    checkpointer: Any = None,
    gateway_for: Callable[[str], BaseChatModel] | None = None,
    **gateway_kwargs: Any,
) -> tuple[AgentRuntime, ScriptedCandidate, MemoryEventStore, MemorySessionStore]:
    """一键装配：剧本幕 → 候选 → 按租户网关闭包 → AgentRuntime（内存事实源 / 状态机 / InMemorySaver）。"""
    candidate = ScriptedCandidate(acts=list(acts) or [text_turn("好")])
    events = events or MemoryEventStore()
    sessions = sessions or MemorySessionStore()
    runtime = AgentRuntime(
        gateway_for=gateway_for
        or scripted_gateway_factory(candidate, **gateway_kwargs),
        events=events,
        sessions=sessions,
        checkpointer=checkpointer or InMemorySaver(),
    )
    return runtime, candidate, events, sessions


async def collect(runtime: AgentRuntime, **kwargs: Any) -> list[AgentEvent]:
    return [event async for event in runtime.run(**kwargs)]
