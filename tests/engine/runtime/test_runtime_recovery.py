"""崩溃恢复（M2.9 resume 崩溃分诊；契约 C12 / C11 T5）：工具副作用后崩溃 → 恢复恰一把幂等键；模型调用中崩溃 → 半截 llm_call 补
interrupted(cause=replay) 再作废重发并消耗迭代；after_agent 末事件前崩溃 → 补末事件；连续崩溃计数与成功清零；超上限 → failed +
recovery_abandoned 图外事件；图已收尾只是 T4 没翻 → 只修状态零事件；idle / failed 无可恢复；健康挂起不是崩溃；决定已落的挂起经崩溃路径续跑；
挂起点缺失（T2 之后崩）经崩溃路径重放回挂起；上限常量两层互钉；真 PG 的 kill -9 演示（工具 / 模型两处）。零真实调用。"""

import uuid
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.guards",
    reason="M2.8 未敲：middleware/guards.py 不存在（M2.9 稿以 M2.8 栈为基）",
)

from app.engine.runtime import protocols as protocols_mod

if not hasattr(protocols_mod, "RECOVERY_LIMIT"):
    pytest.skip("M2.9 未敲：protocols.py 尚无 RECOVERY_LIMIT", allow_module_level=True)

from app.engine.runtime import utterances as u
from app.engine.runtime.protocols import RECOVERY_LIMIT
from app.engine.runtime.spec import AgentSpec
from app.engine.runtime.tools import SideEffect, ToolContext, tool
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import (
    MemoryApprovalStore,
    MemoryEventStore,
    MemorySessionStore,
    collect,
    make_runtime,
    text_turn,
    tool_turn,
)


class SimulatedCrash(BaseException):
    """模拟 kill -9：不是 Exception，框架与钩子的错误处理都接不住。"""


NINE = [
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


def _types(got) -> list[str]:
    return [e.type.value for e in got]


def _crashing_ship(times: int):
    """副作用之后崩溃 times 次的写工具；keys 记录每次执行拿到的幂等键。"""
    keys: list[str] = []
    armed = {"left": times}

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def ship(ctx: ToolContext, order_id: str) -> dict:
        """发货（副作用之后可能崩溃）。"""
        keys.append(ctx.tool_call_id)
        if armed["left"] > 0:
            armed["left"] -= 1
            raise SimulatedCrash("kill -9")
        return {"shipped": order_id}

    return ship, keys


def _spec(tools) -> AgentSpec:
    return AgentSpec(
        system_prompt="你是演示客服。", model_tier="fast", tools=tuple(tools)
    )


async def _session(sessions, sid: str = "s-1") -> str:
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    return sid


async def _recover(rt, spec: AgentSpec, sid: str = "s-1", **kw: Any):
    return [
        e async for e in rt.resume(tenant_id="t-a", session_id=sid, spec=spec, **kw)
    ]


def _assert_event_invariants(events: MemoryEventStore, sid: str) -> None:
    rows = [r for r in events.rows if r["session_id"] == sid]
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))
    assert len({r["id"] for r in rows}) == len(rows)


# ---------------------------------------------------------------- 半截工具 / 半截 LLM / 末事件


async def test_tool_crash_recovery_keeps_single_key_and_resets_count():
    ship, keys = _crashing_ship(1)
    spec = _spec([ship])
    rt, _, events, sessions = make_runtime(
        tool_turn(("ship", {"order_id": "A"}, "c1")), text_turn("已发货")
    )
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="发货", spec=spec)
    assert (
        events.types(sid) == NINE[:4]
        and (await sessions.get(sid))["run_state"] == "running"
    )
    resumed = await _recover(rt, spec)
    assert _types(resumed) == NINE[4:]  # 重放的 tool_call 去重命中，不再外流
    assert events.types(sid) == NINE
    _assert_event_invariants(events, sid)
    assert len(keys) == 2 and len(set(keys)) == 1 and keys[0] == events.rows[3]["id"]
    assert events.rows[4]["payload"]["tool_call_id"] == events.rows[3]["id"]
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0  # 计到 1、成功清零
    assert (
        len({e.run_id for e in resumed}) == 1
        and resumed[0].run_id != events.rows[0]["run_id"]
    )


class _CrashBeforeResult(MemoryEventStore):
    """模型已返回、llm_result 落盘之前进程死亡（一次）：有 llm_call 无 llm_result，checkpoint 停在 model 之前。
    （候选里直接抛 BaseException 不可用：langchain-core agenerate 把非 Exception 的 BaseException 当成功结果取 .generations → AttributeError，探针登记。）"""

    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    async def append(self, **kw: Any) -> tuple[int, bool]:
        if kw["event_type"] == "llm_result" and self.armed:
            self.armed = False
            raise SimulatedCrash("llm_result 落盘之前进程死亡")
        return await super().append(**kw)


async def test_model_crash_recovery_marks_orphan_call_interrupted_and_reissues():
    """半截 LLM：llm_call 已落、进程死在模型调用完成之前 → 恢复重放 model 节点：同 id 命中 → 补 llm_result(interrupted, cause=replay)，
    下一序号重发（消耗迭代：终止时 iteration=2），显式接受重生成文本不同（第一次的回答永远丢了）。"""
    events = _CrashBeforeResult()
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, cand, _, sessions = make_runtime(
        text_turn("第一次的回答"), text_turn("重发的回答"), events=events
    )
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec)
    assert events.types(sid) == ["user_message", "llm_call"]
    assert (
        await rt.build_agent("t-a", spec).aget_state(
            {"configurable": {"thread_id": sid}}
        )
    ).next == ("model",)
    resumed = await _recover(rt, spec)
    assert _types(resumed) == [
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert resumed[0].payload == {
        "iteration": 1,
        "status": "interrupted",
        "cause": "replay",
        "detail": u.LLM_REPLAY_DETAIL,
    }
    assert resumed[1].payload["iteration"] == 2
    assert resumed[2].payload["text"] == "重发的回答"
    assert resumed[-1].payload == {
        "reason": "completed",
        "iteration": 2,
        "detail": "stop_reason=stop",
    }
    assert events.types(sid) == [
        "user_message",
        "llm_call",
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    _assert_event_invariants(events, sid)
    assert cand.calls == 2 and (await sessions.get(sid))["run_state"] == "idle"


class _CrashBeforeTail(MemoryEventStore):
    """after_agent 里 assistant_message 已落、loop_terminated 落盘之前进程死亡（一次）。"""

    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    async def append(self, **kw: Any) -> tuple[int, bool]:
        if kw["event_type"] == "loop_terminated" and self.armed:
            self.armed = False
            raise SimulatedCrash("末事件之前进程死亡")
        return await super().append(**kw)


async def test_after_agent_crash_recovery_writes_missing_tail_and_flips_idle():
    events = _CrashBeforeTail()
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, cand, _, sessions = make_runtime(text_turn("好"), events=events)
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec)
    assert events.types(sid) == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
    ]
    resumed = await _recover(rt, spec)
    assert _types(resumed) == [
        "loop_terminated"
    ]  # assistant_message 重放去重命中、不外流；末事件补上
    assert events.types(sid) == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    _assert_event_invariants(events, sid)
    assert cand.calls == 1
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0


# ---------------------------------------------------------------- 计数、上限、只修状态、拒绝


async def test_repeated_crashes_accumulate_and_success_resets():
    ship, keys = _crashing_ship(2)
    spec = _spec([ship])
    rt, _, events, sessions = make_runtime(
        tool_turn(("ship", {"order_id": "A"}, "c1")), text_turn("已发货")
    )
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="发货", spec=spec)
    with pytest.raises(SimulatedCrash):
        await _recover(rt, spec)  # 恢复中再次崩溃
    assert (await sessions.get(sid))["recovery_count"] == 1
    resumed = await _recover(rt, spec)
    assert _types(resumed) == NINE[4:] and events.types(sid) == NINE
    _assert_event_invariants(events, sid)
    assert len(keys) == 3 and len(set(keys)) == 1
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0


async def test_recovery_limit_marks_session_failed_with_audit_event():
    ship, keys = _crashing_ship(1)
    spec = _spec([ship])
    rt, _, events, sessions = make_runtime(
        tool_turn(("ship", {"order_id": "A"}, "c1")), text_turn("不该发出")
    )
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="发货", spec=spec)
    for _ in range(RECOVERY_LIMIT):
        await sessions.bump_recovery(sid)  # 假装已连续恢复 3 次
    abandoned = await _recover(rt, spec)
    assert _types(abandoned) == ["recovery_abandoned"]
    event = abandoned[0]
    assert event.payload == {
        "recovery_count": RECOVERY_LIMIT + 1,
        "limit": RECOVERY_LIMIT,
        "run_state": "running",
    }
    assert event.task_id is None and event.checkpoint_id is None and event.seq == 5
    assert events.types(sid) == NINE[:4] + ["recovery_abandoned"]
    _assert_event_invariants(events, sid)
    assert (await sessions.get(sid))["run_state"] == "failed"
    assert len(keys) == 1  # 没有再重放
    with pytest.raises(ValueError, match="没有可恢复"):
        await _recover(rt, spec)


async def test_finished_graph_with_stale_running_state_is_repaired_without_events():
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, _, events, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec
    )
    assert got[-1].type.value == "loop_terminated"
    sessions.rows[sid]["run_state"] = "running"  # 后门：T4 翻转丢了
    await sessions.bump_recovery(sid)
    assert await _recover(rt, spec) == []
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0
    assert len(events.rows) == 5


async def test_idle_and_failed_sessions_have_nothing_to_recover():
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, _, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    with pytest.raises(ValueError, match="没有可恢复"):
        await _recover(rt, spec)
    sessions.rows[sid]["run_state"] = "failed"
    with pytest.raises(ValueError, match="没有可恢复"):
        await _recover(rt, spec)


# ---------------------------------------------------------------- 与审批的交界


def _gated_runtime(sessions=None):
    approvals = MemoryApprovalStore()
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=build_registry().specs(),
        tenant_config={"approval_threshold": 200},
    )
    rt, cand, events, sessions = make_runtime(
        tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": 350}, "c1")),
        text_turn("已退款"),
        sessions=sessions,
        approvals=approvals,
    )
    return rt, cand, events, sessions, approvals, spec


async def test_healthy_suspension_is_not_a_crash():
    rt, cand, events, sessions, approvals, spec = _gated_runtime()
    sid = await _session(sessions)
    await collect(rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=spec)
    assert (await sessions.get(sid))["run_state"] == "awaiting_approval"
    assert await _recover(rt, spec) == []  # 等坐席：零事件、不计次
    row = await sessions.get(sid)
    assert row["run_state"] == "awaiting_approval" and row["recovery_count"] == 0
    assert len(approvals.rows) == 1 and cand.calls == 1 and len(events.rows) == 4


async def test_decided_but_not_resumed_suspension_is_continued_by_crash_path():
    """决定已落、坐席的 resume 没来得及（进程死了）：崩溃路径与计划内续跑同一条路——T3、通行证、完成。"""
    rt, cand, events, sessions, approvals, spec = _gated_runtime()
    sid = await _session(sessions)
    await collect(rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=spec)
    (aid,) = approvals.rows
    await approvals.decide(aid, approved=True, operator_id="op-1")
    resumed = await _recover(rt, spec)
    assert _types(resumed) == [
        "approval_decided",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert approvals.rows[aid]["event_id"] == resumed[1].id and cand.calls == 2
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0
    _assert_event_invariants(events, sid)


class _CrashAfterSuspendFlip(MemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.armed = True

    async def transition(self, session_id: str, *, expected: str, to: str) -> bool:
        flipped = await super().transition(session_id, expected=expected, to=to)
        if flipped and to == "awaiting_approval" and self.armed:
            self.armed = False
            raise SimulatedCrash("T2 之后、挂起 checkpoint 之前进程死亡")
        return flipped


async def test_missing_suspension_checkpoint_is_replayed_back_to_suspension_by_crash_path():
    rt, cand, events, sessions, approvals, spec = _gated_runtime(
        sessions=_CrashAfterSuspendFlip()
    )
    sid = await _session(sessions)
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=spec)
    agent = rt.build_agent("t-a", spec)
    cfg = {"configurable": {"thread_id": sid}}
    assert not (await agent.aget_state(cfg)).tasks[0].interrupts  # 挂起点缺失
    assert (
        await _recover(rt, spec) == []
    )  # 重放到挂起点：单命中、事件去重、T2 CAS 失败被容忍
    snap = await agent.aget_state(cfg)
    assert (
        snap.next == ("Approvals.after_model",) and len(snap.tasks[0].interrupts) == 1
    )
    row = await sessions.get(sid)
    assert row["run_state"] == "awaiting_approval" and row["recovery_count"] == 1
    assert len(approvals.rows) == 1 and len(events.rows) == 4 and cand.calls == 1
    (aid,) = approvals.rows
    await approvals.decide(aid, approved=True, operator_id="op-1")
    resumed = await _recover(rt, spec, approval_id=aid)  # 计划内续跑照常
    assert resumed[-1].type.value == "loop_terminated"
    assert (await sessions.get(sid))["recovery_count"] == 0


def test_recovery_limit_matches_domain_constant():
    sessions_mod = pytest.importorskip("app.domain.sessions", reason="M2.2 未敲")
    assert RECOVERY_LIMIT == sessions_mod.RECOVERY_LIMIT == 3


# ---------------------------------------------------------------- 真 PG：kill -9 演示（reports/ 记录一次运行）


async def _pg_runtime(db_session_factory, pg_checkpointer, *acts, events=None):
    from app.domain.approvals import ApprovalStore
    from app.domain.events import EventStore
    from app.domain.sessions import SessionStateStore

    events = events or EventStore(db_session_factory)
    sessions = SessionStateStore(db_session_factory)
    rt, cand, _, _ = make_runtime(
        *acts,
        events=events,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        checkpointer=pg_checkpointer,
        approvals=ApprovalStore(db_session_factory),
    )
    sid = f"s-{uuid.uuid4().hex[:8]}"
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    return rt, cand, events, sessions, sid


async def test_kill9_demo_tool_crash_on_real_pg(db_session_factory, pg_checkpointer):
    """真 PG + durability=sync：工具副作用之后中断 → 恢复 → 下游只见一把幂等键、tool_call 恰一条、tool_result 以原 id 闭合、seq 连续。"""
    pytest.importorskip("app.domain.approvals", reason="M2.2 未敲")
    ship, keys = _crashing_ship(1)
    spec = _spec([ship])
    rt, _, events, sessions, sid = await _pg_runtime(
        db_session_factory,
        pg_checkpointer,
        tool_turn(("ship", {"order_id": "A"}, "c1")),
        text_turn("已发货"),
    )
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="发货", spec=spec)
    agent = rt.build_agent("t-a", spec)
    cfg = {"configurable": {"thread_id": sid}}
    assert (await agent.aget_state(cfg)).next == ("tools",)
    assert (await sessions.get(sid))["run_state"] == "running"
    resumed = await _recover(rt, spec, sid=sid)
    assert _types(resumed) == NINE[4:]
    rows = await events.read("t-a", sid)
    assert [r["type"] for r in rows] == NINE
    assert [r["seq"] for r in rows] == list(range(1, 10)) and len(
        {r["id"] for r in rows}
    ) == 9
    assert all(r["task_id"] for r in rows)
    assert rows[4]["payload"]["tool_call_id"] == rows[3]["id"]
    assert len(keys) == 2 and set(keys) == {rows[3]["id"]}
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0
    assert (await agent.aget_state(cfg)).next == ()
    print(
        f"\n[kill -9 demo/tool] session={sid} events={[r['type'] for r in rows]} keys={sorted(set(keys))} executions={len(keys)}"
    )


async def test_kill9_demo_model_crash_on_real_pg(db_session_factory, pg_checkpointer):
    """真 PG：模型返回后、llm_result 落盘之前中断 → 恢复 → 旧 llm_call 补 llm_result(interrupted, cause=replay) + 新 llm_call；
    事件无重复、seq 连续。"""
    pytest.importorskip("app.domain.approvals", reason="M2.2 未敲")
    from app.domain.events import EventStore

    class CrashBeforeResult(EventStore):
        armed = True

        async def append(self, **kw: Any) -> tuple[int, bool]:
            if kw["event_type"] == "llm_result" and self.armed:
                self.armed = False
                raise SimulatedCrash("llm_result 落盘之前进程死亡")
            return await super().append(**kw)

    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, cand, events, sessions, sid = await _pg_runtime(
        db_session_factory,
        pg_checkpointer,
        text_turn("第一次的回答"),
        text_turn("重发的回答"),
        events=CrashBeforeResult(db_session_factory),
    )
    with pytest.raises(SimulatedCrash):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec)
    agent = rt.build_agent("t-a", spec)
    cfg = {"configurable": {"thread_id": sid}}
    assert (await agent.aget_state(cfg)).next == ("model",)
    resumed = await _recover(rt, spec, sid=sid)
    assert _types(resumed) == [
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert resumed[0].payload["status"] == "interrupted"
    assert resumed[0].payload["cause"] == "replay"
    rows = await events.read("t-a", sid)
    assert [r["type"] for r in rows] == [
        "user_message",
        "llm_call",
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert [r["seq"] for r in rows] == list(range(1, 8))
    assert len({r["id"] for r in rows}) == 7
    assert rows[1]["payload"]["iteration"] == 1 and rows[3]["payload"]["iteration"] == 2
    assert rows[4]["payload"]["text"] == "重发的回答"
    assert cand.calls == 2 and (await sessions.get(sid))["run_state"] == "idle"
    print(
        f"\n[kill -9 demo/model] session={sid} events={[r['type'] for r in rows]} calls={cand.calls}"
    )
