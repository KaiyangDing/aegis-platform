"""runtime 测试共用替身（非测试文件）：内存事实源 / 内存会话状态机 / 内存审批单（可注入时钟）/ 剧本网关 / 一键装配的 AgentRuntime。

网关是真 AegisGateway（租户绑定、候选环、异常翻译全在），候选是走真实框架外壳的 ScriptedCandidate；
事实源与状态机用内存版（结构匹配 engine 协议），checkpointer 用 InMemorySaver——真 PG 只在 test_runtime_pg 用。
"""

from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime, timedelta
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
from app.engine.runtime import utterances as u
from app.engine.runtime.events import AgentEvent
from app.engine.runtime.runtime import AgentRuntime
from tests.engine.gateway.doubles import (
    ScriptedCandidate,
    StubBreaker,
    StubLimiter,
    finish,
)

USAGE = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def unwrap_untrusted(text: str) -> str:
    """取 ToolMessage 回填正文：去掉 M2.8 挂点②的不可信包裹（首尾各一行标记）；未包裹（M2.8 之前的树）原样返回。"""
    lines = text.split("\n")
    if (
        len(lines) >= 2
        and lines[0].startswith(u.UNTRUSTED_OPEN)
        and lines[-1].startswith(u.UNTRUSTED_CLOSE)
    ):
        return "\n".join(lines[1:-1])
    return text


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


def empty_turn(stop: str = "stop") -> list[AIMessageChunk]:
    """空输出轮（闸门 #5 的两种违规形态）：只有收尾块——stop="tool_calls" 即"宣告工具停却没给调用"。"""
    return [finish(stop)]


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


class MemoryApprovalStore:
    """ApprovalStoreLike 的内存实现（CAS 语义与 domain 版一致）：rows 按单号保存原始行（expires_at 为 datetime），
    create / get 返回 ISO 视图；now 可替换（拨时钟测到期 fail-closed 与 expire_due）。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.now: Callable[[], datetime] = lambda: datetime.now(UTC)

    @staticmethod
    def _view(row: dict[str, Any]) -> dict[str, Any]:
        view = dict(row)
        for key in ("expires_at", "decided_at", "created_at"):
            value = view[key]
            view[key] = None if value is None else value.isoformat()
        return view

    async def create(
        self,
        *,
        approval_id: str,
        tenant_id: str,
        session_id: str,
        run_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        ttl_s: float,
    ) -> dict[str, Any]:
        row = self.rows.get(approval_id)
        if row is not None:
            return {**self._view(row), "created": False}
        now = self.now()
        row = {
            "id": approval_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "run_id": run_id,
            "tool_name": tool_name,
            "args": dict(args),
            "status": "pending",
            "operator_id": None,
            "event_id": None,
            "expires_at": now + timedelta(seconds=ttl_s),
            "decided_at": None,
            "created_at": now,
        }
        self.rows[approval_id] = row
        return {**self._view(row), "created": True}

    async def get(self, approval_id: str) -> dict[str, Any] | None:
        row = self.rows.get(approval_id)
        return None if row is None else self._view(row)

    async def decide(
        self, approval_id: str, *, approved: bool, operator_id: str
    ) -> bool:
        row = self.rows.get(approval_id)
        if row is None or row["status"] != "pending" or row["expires_at"] <= self.now():
            return False
        row.update(
            status="approved" if approved else "rejected",
            operator_id=operator_id,
            decided_at=self.now(),
        )
        return True

    async def cancel(self, approval_id: str) -> bool:
        row = self.rows.get(approval_id)
        if row is None or row["status"] != "pending":
            return False
        row.update(status="cancelled", decided_at=self.now())
        return True

    async def expire_due(self, *, now: datetime | None = None) -> list[str]:
        cutoff = self.now() if now is None else now
        flipped: list[str] = []
        for row in self.rows.values():
            if row["status"] == "pending" and row["expires_at"] <= cutoff:
                row.update(status="expired", decided_at=self.now())
                flipped.append(row["id"])
        return flipped

    async def attach_event(self, approval_id: str, *, event_id: str) -> bool:
        row = self.rows.get(approval_id)
        if row is None or row["event_id"] is not None:
            return False
        row["event_id"] = event_id
        return True

    async def list_for_session(
        self, tenant_id: str, session_id: str
    ) -> list[dict[str, Any]]:
        return [
            self._view(r)
            for r in self.rows.values()
            if r["tenant_id"] == tenant_id and r["session_id"] == session_id
        ]


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
    approvals: Any = None,
    precheck: Any = None,
    **gateway_kwargs: Any,
) -> tuple[AgentRuntime, ScriptedCandidate, MemoryEventStore, MemorySessionStore]:
    """一键装配：剧本幕 → 候选 → 按租户网关闭包 → AgentRuntime（内存事实源 / 状态机 / InMemorySaver；审批单存取件与
    批准后前置校验挂点按需注入，缺席 = 无审批 / 全通过）。"""
    candidate = ScriptedCandidate(acts=list(acts) or [text_turn("好")])
    events = events or MemoryEventStore()
    sessions = sessions or MemorySessionStore()
    extras: dict[
        str, Any
    ] = {}  # M2.7 之前的 AgentRuntime 没有这两个形参：只在给了值时才传
    if approvals is not None:
        extras["approvals"] = approvals
    if precheck is not None:
        extras["precheck"] = precheck
    runtime = AgentRuntime(
        gateway_for=gateway_for
        or scripted_gateway_factory(candidate, **gateway_kwargs),
        events=events,
        sessions=sessions,
        checkpointer=checkpointer or InMemorySaver(),
        **extras,
    )
    return runtime, candidate, events, sessions


async def collect(runtime: AgentRuntime, **kwargs: Any) -> list[AgentEvent]:
    return [event async for event in runtime.run(**kwargs)]
