"""HITL 审批（M2.7 Approvals + resume 单入口；ADR-013）：挂起链路（开单 → 事件 → T2 → interrupt，进程可下线）、批准续跑（通行证、
回填审计链、v1 形态 C 十一事件）、拒绝 / 撤回 / 超时 → cancelled 终止零 LLM、pending 不许恢复、挂起态新输入被拒（会话互斥前置）、
重放不重复开单、并发恢复恰一赢家、多单批处理与混合决定、谓词崩溃 fail-closed、非闸门调用等审批后按声明序执行、
前置校验否决、闸门终止绕过审批、取消信号在恢复段的工具检查点生效、栈位与跳转声明。内存事实源 + InMemorySaver，零真实调用。"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.approvals",
    reason="M2.7 未敲：app/engine/runtime/middleware/approvals.py 不存在",
)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.engine.runtime import utterances as u
from app.engine.runtime.events import normalize_events
from app.engine.runtime.middleware.approvals import Approvals, last_ai_and_unpaired
from app.engine.runtime.middleware.gates import Gates
from app.engine.runtime.middleware.summarization import AegisSummarization
from app.engine.runtime.runtime import MIDDLEWARE_STACK, AgentRuntime, SessionBusy
from app.engine.runtime.spec import AgentSpec, LoopPolicy
from app.engine.runtime.tools import (
    PrecheckVeto,
    SideEffect,
    ToolContext,
    ToolRegistry,
    tool,
)
from tests.engine.gateway.doubles import ScriptedCandidate
from tests.engine.runtime.demo_tools import build_registry, demo_order_query
from tests.engine.runtime.doubles import (
    MemoryApprovalStore,
    MemoryEventStore,
    MemorySessionStore,
    collect,
    make_runtime,
    scripted_gateway_factory,
    text_turn,
    tool_turn,
    unwrap_untrusted,
)


class SimulatedCrash(BaseException):
    """模拟 kill -9：不是 Exception，框架与钩子的错误处理都接不住。"""


SUSPENDED = ["user_message", "llm_call", "llm_result", "approval_requested"]
RESUMED_OK = [
    "approval_decided",
    "tool_call",
    "tool_result",
    "llm_call",
    "llm_result",
    "assistant_message",
    "loop_terminated",
]


def _spec(registry: ToolRegistry | None = None, **policy: Any) -> AgentSpec:
    return AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=(registry or build_registry()).specs(),
        policy=LoopPolicy(**policy),
        tenant_config={"approval_threshold": 200},
    )


def _refund(amount: int, cid: str = "c1") -> list[Any]:
    return tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": amount}, cid))


def _types(got) -> list[str]:
    return [e.type.value for e in got]


async def _suspend(
    *acts: list[Any],
    approvals: MemoryApprovalStore | None = None,
    precheck: Any = None,
    spec: AgentSpec | None = None,
    sid: str = "s-1",
    user_input: str = "退款350",
):
    approvals = approvals or MemoryApprovalStore()
    rt, cand, events, sessions = make_runtime(
        *(acts or [_refund(350), text_turn("已退款")]),
        approvals=approvals,
        precheck=precheck,
    )
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input=user_input, spec=spec or _spec()
    )
    return rt, cand, events, sessions, approvals, got


async def _resume(
    rt,
    *,
    approval_id: str | None = None,
    spec: AgentSpec | None = None,
    sid: str = "s-1",
    **kw: Any,
):
    """计划内审批续跑：approval_id 缺省取运行时审批单存取件里的第一张（坐席刚决定的那张）。"""
    aid = approval_id or next(iter(rt._approvals.rows))
    return [
        e
        async for e in rt.resume(
            tenant_id="t-a",
            session_id=sid,
            spec=spec or _spec(),
            approval_id=aid,
            **kw,
        )
    ]


async def _snapshot(rt, sid: str = "s-1", spec: AgentSpec | None = None):
    return await rt.build_agent("t-a", spec or _spec()).aget_state(
        {"configurable": {"thread_id": sid}}
    )


def _ticket(approvals: MemoryApprovalStore) -> dict[str, Any]:
    (row,) = approvals.rows.values()
    return row


def _tool_messages(snap) -> dict[str, ToolMessage]:
    return {
        m.tool_call_id: m for m in snap.values["messages"] if isinstance(m, ToolMessage)
    }


def _assert_no_dangling_tool_calls(messages) -> None:
    paired = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                assert call["id"] in paired, f"悬空 tool_call {call['id']}"


def _assert_event_invariants(events: MemoryEventStore, sid: str) -> None:
    rows = [r for r in events.rows if r["session_id"] == sid]
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    assert len({r["id"] for r in rows}) == len(rows)


# ---------------------------------------------------------------- 挂起链路


async def test_gated_call_suspends_cleanly_and_process_can_go_away():
    """开单 → approval_requested → T2 → interrupt：无 loop_terminated、run_state=awaiting_approval、checkpoint next 指向审批节点。"""
    rt, cand, events, sessions, approvals, got = await _suspend()
    assert _types(got) == SUSPENDED and events.types("s-1") == SUSPENDED
    req = got[3].payload
    ticket = _ticket(approvals)
    assert req == {
        "approval_id": ticket["id"],
        "tool_name": "demo_refund_apply",
        "args": {"order_id": "1024", "amount": 350},
        "expires_at": ticket["expires_at"].isoformat(),
    }
    assert ticket["status"] == "pending" and ticket["tenant_id"] == "t-a"
    assert ticket["session_id"] == "s-1" and ticket["run_id"] == got[0].run_id
    assert (await sessions.get("s-1"))["run_state"] == "awaiting_approval"
    snap = await _snapshot(rt)
    assert snap.next == ("Approvals.after_model",)
    (task,) = snap.tasks
    (pending,) = task.interrupts
    assert pending.value["action_requests"] == [
        {
            "name": "demo_refund_apply",
            "args": {"order_id": "1024", "amount": 350},
            "description": u.APPROVAL_DESCRIPTION.format(name="demo_refund_apply"),
            "approval_id": ticket["id"],
            "expires_at": ticket["expires_at"].isoformat(),
        }
    ]
    assert pending.value["review_configs"] == [
        {"action_name": "demo_refund_apply", "allowed_decisions": ["approve", "reject"]}
    ]
    assert cand.calls == 1
    assert not _tool_messages(snap)  # 挂起期间调用未配对：等决定


async def test_approve_then_resume_executes_with_passport_and_completes():
    """v1 形态 C 十一事件：4（挂起）+ 7（恢复）；通行证放行、write-ahead 后回填审批单 event_id、恢复用新 run_id、seq 接续。"""
    rt, cand, events, sessions, approvals, got = await _suspend()
    aid = _ticket(approvals)["id"]
    assert await approvals.decide(aid, approved=True, operator_id="op-1") is True
    resumed = await _resume(rt)
    assert _types(resumed) == RESUMED_OK
    assert events.types("s-1") == SUSPENDED + RESUMED_OK
    _assert_event_invariants(events, "s-1")
    assert [e.seq for e in resumed] == list(range(5, 12))
    assert len({e.run_id for e in got}) == 1 and len({e.run_id for e in resumed}) == 1
    assert got[0].run_id != resumed[0].run_id
    assert resumed[0].payload == {
        "approval_id": aid,
        "approved": True,
        "operator_id": "op-1",
    }
    call, result = resumed[1], resumed[2]
    assert call.payload == {
        "tool_name": "demo_refund_apply",
        "args": {"order_id": "1024", "amount": 350},
        "model_call_id": "c1",
    }
    assert result.payload["result"] == {"refunded": 350, "idempotency_key": call.id}
    assert _ticket(approvals)["event_id"] == call.id  # 批准已兑现的凭证
    assert resumed[-1].payload["reason"] == "completed"
    assert cand.calls == 2
    assert (await sessions.get("s-1"))["run_state"] == "idle"
    snap = await _snapshot(rt)
    assert snap.next == () and snap.values["approved_calls"] == {"c1": aid}
    _assert_no_dangling_tool_calls(snap.values["messages"])


async def test_reject_terminates_cancelled_with_zero_llm_calls():
    rt, cand, events, sessions, approvals, _ = await _suspend()
    aid = _ticket(approvals)["id"]
    assert await approvals.decide(aid, approved=False, operator_id="op-1") is True
    resumed = await _resume(rt)
    assert _types(resumed) == ["approval_decided", "loop_terminated"]
    assert resumed[0].payload == {
        "approval_id": aid,
        "approved": False,
        "operator_id": "op-1",
    }
    assert resumed[1].payload == {
        "reason": "cancelled",
        "iteration": 1,
        "detail": f"审批被拒绝：approval_id={aid}",
    }
    assert cand.calls == 1 and "tool_call" not in events.types("s-1")
    snap = await _snapshot(rt)
    messages = snap.values["messages"]
    assert _tool_messages(snap)["c1"].content == u.TOOL_APPROVAL_DENIED.format(
        status="被拒绝"
    )
    assert isinstance(messages[-1], ToolMessage)  # 取消零话术：不追加兜底 AIMessage
    _assert_no_dangling_tool_calls(messages)
    assert snap.values["termination"]["reason"] == "cancelled"
    assert (await sessions.get("s-1"))["run_state"] == "idle"


async def test_cancel_terminates_cancelled():
    rt, cand, _, _, approvals, _ = await _suspend()
    aid = _ticket(approvals)["id"]
    assert await approvals.cancel(aid) is True
    resumed = await _resume(rt)
    assert _types(resumed) == ["approval_cancelled", "loop_terminated"]
    assert resumed[0].payload == {"approval_id": aid}
    assert resumed[1].payload["detail"] == f"审批被撤回：approval_id={aid}"
    assert _tool_messages(await _snapshot(rt))[
        "c1"
    ].content == u.TOOL_APPROVAL_DENIED.format(status="被撤回")
    assert cand.calls == 1


async def test_expired_via_injected_clock_terminates_and_late_approval_is_refused():
    rt, cand, _, _, approvals, _ = await _suspend()
    aid = _ticket(approvals)["id"]
    approvals.now = lambda: datetime.now(UTC) + timedelta(hours=2)  # 时钟拨过 TTL
    assert (
        await approvals.decide(aid, approved=True, operator_id="op-1") is False
    )  # 到期 fail-closed
    assert await approvals.expire_due() == [aid]
    resumed = await _resume(rt)
    assert _types(resumed) == ["approval_expired", "loop_terminated"]
    assert resumed[1].payload["detail"] == f"审批超时：approval_id={aid}"
    assert _tool_messages(await _snapshot(rt))[
        "c1"
    ].content == u.TOOL_APPROVAL_DENIED.format(status="已超时")
    assert cand.calls == 1


async def test_resume_while_pending_is_refused_and_state_untouched():
    rt, _, events, sessions, _, _ = await _suspend()
    with pytest.raises(ValueError, match="pending"):
        await _resume(rt)
    assert (await sessions.get("s-1"))["run_state"] == "awaiting_approval"
    assert events.types("s-1") == SUSPENDED


async def test_new_input_while_awaiting_is_rejected_by_session_mutex():
    """探针⒆：挂起态直接喂新输入会作废中断并留悬空 tool_calls——会话互斥在 run() 的 T1 前置挡住。"""
    rt, _, events, sessions, _, _ = await _suspend()
    with pytest.raises(SessionBusy):
        await collect(
            rt, tenant_id="t-a", session_id="s-1", user_input="再来", spec=_spec()
        )
    assert (await sessions.get("s-1"))["run_state"] == "awaiting_approval"
    assert events.types("s-1") == SUSPENDED
    assert (await _snapshot(rt)).next == ("Approvals.after_model",)


async def test_resume_replays_node_without_reopening_ticket_or_duplicating_events():
    rt, _, events, _, approvals, _ = await _suspend()
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    await _resume(rt)
    assert len(approvals.rows) == 1
    assert events.types("s-1").count("approval_requested") == 1
    _assert_event_invariants(events, "s-1")


async def test_resume_with_foreign_approval_id_is_refused():
    rt, _, _, sessions, approvals, _ = await _suspend()
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    with pytest.raises(ValueError, match="不属于"):
        await _resume(rt, approval_id=str(uuid.uuid4()))
    assert (await sessions.get("s-1"))["run_state"] == "awaiting_approval"


class _CrashAfterSuspendFlip(MemorySessionStore):
    """T2 翻转之后、挂起 checkpoint 之前进程死亡（一次）。"""

    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    async def transition(self, session_id: str, *, expected: str, to: str) -> bool:
        flipped = await super().transition(session_id, expected=expected, to=to)
        if flipped and to == "awaiting_approval" and self.armed:
            self.armed = False
            raise SimulatedCrash("T2 之后、挂起 checkpoint 之前进程死亡")
        return flipped


async def test_resume_repairs_missing_suspension_checkpoint_before_continuing():
    """崩溃窗口（探针 21c）：单已开、事件已写、T2 已翻，挂起点没进 checkpoint——resume 先重放到挂起点（全部幂等、零新事件），再按决定续跑。"""
    sessions = _CrashAfterSuspendFlip()
    approvals = MemoryApprovalStore()
    rt, cand, events, _ = make_runtime(
        _refund(350), text_turn("已退款"), sessions=sessions, approvals=approvals
    )
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    with pytest.raises(SimulatedCrash):
        await collect(
            rt, tenant_id="t-a", session_id="s-1", user_input="退款350", spec=_spec()
        )
    assert events.types("s-1") == SUSPENDED
    assert (await sessions.get("s-1"))["run_state"] == "awaiting_approval"
    snap = await _snapshot(rt)
    assert snap.next == ("Approvals.after_model",) and not snap.tasks[0].interrupts
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    resumed = await _resume(rt)
    assert _types(resumed) == RESUMED_OK
    assert events.types("s-1") == SUSPENDED + RESUMED_OK
    _assert_event_invariants(events, "s-1")
    assert len(approvals.rows) == 1 and cand.calls == 2
    assert (await sessions.get("s-1"))["run_state"] == "idle"


async def test_concurrent_resume_has_exactly_one_winner():
    rt, _, events, sessions, approvals, _ = await _suspend()
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    results = await asyncio.gather(_resume(rt), _resume(rt), return_exceptions=True)
    winners = [r for r in results if isinstance(r, list)]
    losers = [r for r in results if isinstance(r, SessionBusy)]
    assert len(winners) == 1 and len(losers) == 1
    assert _types(winners[0]) == RESUMED_OK
    assert events.types("s-1") == SUSPENDED + RESUMED_OK
    assert (await sessions.get("s-1"))["run_state"] == "idle"


# ---------------------------------------------------------------- 多单、混合决定、非闸门调用


async def test_two_gated_calls_share_one_interrupt_and_execute_in_declared_order():
    rt, _, _events, _, approvals, got = await _suspend(
        tool_turn(
            ("demo_refund_apply", {"order_id": "1", "amount": 300}, "c1"),
            ("demo_refund_apply", {"order_id": "2", "amount": 400}, "c2"),
        ),
        text_turn("两笔已退"),
    )
    assert _types(got) == SUSPENDED + ["approval_requested"]
    snap = await _snapshot(rt)
    (task,) = snap.tasks
    (pending,) = (
        task.interrupts
    )  # 一个节点一个 interrupt：多单进同一载荷（探针⒄：同节点第二个 interrupt 只会再挂起）
    ids = [r["approval_id"] for r in pending.value["action_requests"]]
    assert ids == [got[3].payload["approval_id"], got[4].payload["approval_id"]]
    for aid in ids:
        assert await approvals.decide(aid, approved=True, operator_id="op-1")
    resumed = await _resume(rt)
    assert _types(resumed) == [
        "approval_decided",
        "approval_decided",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert [
        e.payload["model_call_id"] for e in resumed if e.type.value == "tool_call"
    ] == ["c1", "c2"]
    assert (await _snapshot(rt)).values["approved_calls"] == {
        "c1": ids[0],
        "c2": ids[1],
    }
    assert [r["event_id"] for r in approvals.rows.values()] == [
        resumed[2].id,
        resumed[4].id,
    ]


async def test_mixed_decisions_cancel_everything():
    rt, cand, events, _, approvals, got = await _suspend(
        tool_turn(
            ("demo_refund_apply", {"order_id": "1", "amount": 300}, "c1"),
            ("demo_refund_apply", {"order_id": "2", "amount": 400}, "c2"),
        ),
        text_turn("不该发出"),
    )
    a1, a2 = got[3].payload["approval_id"], got[4].payload["approval_id"]
    await approvals.decide(a1, approved=True, operator_id="op-1")
    await approvals.decide(a2, approved=False, operator_id="op-2")
    resumed = await _resume(rt)
    assert _types(resumed) == [
        "approval_decided",
        "approval_decided",
        "loop_terminated",
    ]
    assert [e.payload["approved"] for e in resumed[:2]] == [True, False]
    assert resumed[2].payload["detail"] == f"审批被拒绝：approval_id={a2}"
    msgs = _tool_messages(await _snapshot(rt))
    assert msgs["c1"].content == u.TOOL_NOT_EXECUTED
    assert msgs["c2"].content == u.TOOL_APPROVAL_DENIED.format(status="被拒绝")
    assert "tool_call" not in events.types("s-1") and cand.calls == 1


async def test_non_gated_calls_in_same_turn_wait_and_run_after_approval():
    """v2 差异：审批节点先于 tools——同一轮里闸门之外的调用也等决定，批准后按声明序全部执行（v1 挂起时弃置其后调用）。"""
    rt, _, _events, _, approvals, got = await _suspend(
        tool_turn(
            ("demo_order_query", {"order_id": "1024"}, "c1"),
            ("demo_refund_apply", {"order_id": "1024", "amount": 350}, "c2"),
        ),
        text_turn("查到并已退"),
    )
    assert _types(got) == SUSPENDED  # 查询也没执行
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    resumed = await _resume(rt)
    assert _types(resumed) == [
        "approval_decided",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert [e.payload["tool_name"] for e in resumed if e.type.value == "tool_call"] == [
        "demo_order_query",
        "demo_refund_apply",
    ]


def _gate_boom(args: Any, cfg: Any) -> bool:
    raise RuntimeError("闸门自己炸了 sk-abcdefghijklmnop")


@tool(side_effect=SideEffect.WRITE, risk_policy=_gate_boom)
async def demo_gate_crash(ctx: ToolContext, amount: int) -> dict:
    """风险闸门会崩溃的演示写工具（审批测试专用）。"""
    return {}


async def test_risk_policy_crash_fails_closed_without_ticket_and_run_continues():
    registry = ToolRegistry([demo_gate_crash, demo_order_query])
    rt, _, _events, sessions, approvals, got = await _suspend(
        tool_turn(
            ("demo_gate_crash", {"amount": 1}, "c1"),
            ("demo_order_query", {"order_id": "1024"}, "c2"),
        ),
        text_turn("查到了"),
        spec=_spec(registry),
    )
    assert _types(got) == [
        "user_message",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert got[3].payload["tool_name"] == "demo_order_query"
    assert approvals.rows == {}
    blocked = _tool_messages(await _snapshot(rt, spec=_spec(registry)))["c1"]
    assert blocked.status == "error" and "fail-closed" in blocked.content
    assert "sk-***" in blocked.content and "abcdefghijklmnop" not in blocked.content
    assert (await sessions.get("s-1"))["run_state"] == "idle"


async def test_precheck_veto_feeds_back_without_execution():
    """TOCTOU 挂点：批准后前置校验否决 → precheck_vetoed 事件、工具不执行（无 write-ahead）、observation 回填、run 照常完成。"""
    seen: list[tuple[str, dict[str, Any]]] = []

    async def precheck(name: str, args: Any) -> PrecheckVeto | None:
        seen.append((name, dict(args)))
        return PrecheckVeto(observation="订单已关闭", detail="status=closed")

    rt, _, _, _, approvals, _ = await _suspend(precheck=precheck)
    aid = _ticket(approvals)["id"]
    await approvals.decide(aid, approved=True, operator_id="op-1")
    resumed = await _resume(rt)
    assert _types(resumed) == [
        "approval_decided",
        "precheck_vetoed",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert resumed[1].payload == {
        "approval_id": aid,
        "tool_name": "demo_refund_apply",
        "observation": "订单已关闭",
        "detail": "status=closed",
    }
    assert seen == [("demo_refund_apply", {"order_id": "1024", "amount": 350})]
    veto = _tool_messages(await _snapshot(rt))["c1"]
    assert veto.status == "error"
    assert unwrap_untrusted(veto.content) == u.PRECHECK_VETO_TEMPLATE.format(
        reason="订单已关闭"
    )  # M2.8 起 ToolExec 的回填经不可信包裹
    assert "status=closed" not in veto.content  # detail 只进事件，不进模型上下文
    assert _ticket(approvals)["event_id"] is None  # 未兑现


async def test_gate_termination_bypasses_approvals():
    """栈序判据②：Gates.after_model 先跑并终止 → jump end 绕过审批钩子——不开单、不翻转。"""
    _rt, _, _, sessions, approvals, got = await _suspend(
        tool_turn(
            ("ghost_tool", {}, "c1"),
            ("demo_refund_apply", {"order_id": "1024", "amount": 350}, "c2"),
        ),
        text_turn("不该发出"),
        spec=_spec(protocol_retry_limit=0),
    )
    assert _types(got) == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert got[-1].payload["reason"] == "protocol_violation"
    assert approvals.rows == {}
    assert (await sessions.get("s-1"))["run_state"] == "idle"


async def test_cancel_signal_during_suspension_applies_at_tool_checkpoint():
    rt, cand, _, sessions, approvals, _ = await _suspend()
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    cancel = asyncio.Event()
    cancel.set()
    resumed = await _resume(rt, cancel=cancel)
    assert _types(resumed) == ["approval_decided", "loop_terminated"]
    assert resumed[1].payload["reason"] == "cancelled"
    assert resumed[1].payload["detail"].startswith("收到取消信号（工具检查点）")
    assert cand.calls == 1
    assert (await sessions.get("s-1"))["run_state"] == "idle"


# ---------------------------------------------------------------- 归属、缺件、形态


async def test_resume_rejects_missing_session_and_wrong_tenant():
    rt, _, _, _, _, _ = await _suspend()
    with pytest.raises(ValueError, match="不存在"):
        await _resume(rt, sid="ghost")
    with pytest.raises(ValueError, match="不属于租户"):
        [e async for e in rt.resume(tenant_id="t-b", session_id="s-1", spec=_spec())]


async def test_missing_approval_store_is_loud():
    """无审批单存取件的运行时撞上闸门命中：RuntimeError 裸穿（run_state 留 running，crash-only）。"""
    cand = ScriptedCandidate(acts=[_refund(350)])
    events, sessions = MemoryEventStore(), MemorySessionStore()
    rt = AgentRuntime(
        gateway_for=scripted_gateway_factory(cand),
        events=events,
        sessions=sessions,
        checkpointer=InMemorySaver(),
    )
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    with pytest.raises(RuntimeError, match="approvals"):
        await collect(
            rt, tenant_id="t-a", session_id="s-1", user_input="退", spec=_spec()
        )
    assert (await sessions.get("s-1"))["run_state"] == "running"


async def test_approval_flow_event_shape_snapshot():
    """v1 形态 C 的 v2 版：归一化后的类型序列与审批 payload（approval_id 别名 a1、tool_call_id 别名 e6（tool_call 是流内第 6 个事件）、expires_at 豁免）。"""
    rt, _, _events, _, approvals, got = await _suspend()
    await approvals.decide(_ticket(approvals)["id"], approved=True, operator_id="op-1")
    resumed = await _resume(rt)
    normalized = normalize_events([*got, *resumed])
    assert [n["type"] for n in normalized] == SUSPENDED + RESUMED_OK
    assert normalized[3]["payload"] == {
        "approval_id": "a1",
        "tool_name": "demo_refund_apply",
        "args": {"order_id": "1024", "amount": 350},
    }
    assert normalized[4]["payload"] == {
        "approval_id": "a1",
        "approved": True,
        "operator_id": "op-1",
    }
    assert normalized[6]["payload"]["tool_call_id"] == "e6"
    assert normalized[6]["payload"]["result"]["idempotency_key"] == resumed[1].id
    assert normalized[-1]["payload"] == {
        "reason": "completed",
        "iteration": 2,
        "detail": "stop_reason=stop",
    }


def test_stack_position_and_jump_declaration():
    stack = list(MIDDLEWARE_STACK)
    assert stack.index(AegisSummarization) < stack.index(Approvals) < stack.index(Gates)
    assert Approvals.aafter_model.__can_jump_to__ == ["end"]


async def test_loop_exit_node_edges_and_graph_nodes():
    rt, _, _, _ = make_runtime()
    graph = rt.build_agent("t-a", _spec()).get_graph()
    assert "Approvals.after_model" in graph.nodes
    targets = {e.target for e in graph.edges if e.source == "Approvals.after_model"}
    assert {
        "tools",
        "RunEvents.after_agent",
        "AegisSummarization.before_model",
    } <= targets


def test_last_ai_and_unpaired_helper():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "a", "args": {}, "id": "c1"},
            {"name": "b", "args": {}, "id": "c2"},
        ],
    )
    paired = ToolMessage(content="x", tool_call_id="c1", name="a")
    last, unpaired = last_ai_and_unpaired([HumanMessage("hi"), ai, paired])
    assert last is ai and [c["id"] for c in unpaired] == ["c2"]
    assert last_ai_and_unpaired([HumanMessage("hi")]) == (None, [])
