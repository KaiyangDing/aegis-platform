"""AgentRuntime 首条端到端（M2.3）：文本 run 五事件序列与 payload 形态、yield 序 ≡ seq 序、状态机 T1/T4 与归属校验、
私有通道在 run 起点归零（发现 F1）、第二轮接续 seq、model_settings 载体到达候选、D8 种子。零真实调用。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)

from app.core.tokens import estimate_messages_tokens
from app.engine.runtime import utterances as u
from app.engine.runtime.events import AgentEvent, EventType
from app.engine.runtime.runtime import SessionBusy
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy
from tests.engine.runtime.doubles import (
    MemoryEventStore,
    collect,
    make_runtime,
    text_turn,
)

SPEC = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
FIVE = [
    "user_message",
    "llm_call",
    "llm_result",
    "assistant_message",
    "loop_terminated",
]


async def _session(sessions, sid: str = "s-1", tenant: str = "t-a") -> str:
    await sessions.create(sid, tenant_id=tenant, user_id="u-1")
    return sid


async def test_text_run_emits_five_events_in_seq_order():
    rt, cand, events, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=SPEC
    )
    assert [e.type.value for e in got] == FIVE
    assert [e.seq for e in got] == [1, 2, 3, 4, 5]
    assert all(isinstance(e, AgentEvent) for e in got)
    assert len({e.run_id for e in got}) == 1
    assert all(e.tenant_id == "t-a" and e.session_id == sid for e in got)
    assert all(e.task_id for e in got)  # 框架任务身份在场
    assert events.types(sid) == FIVE  # yield 序 ≡ 落盘序
    assert cand.calls == 1


async def test_payload_shapes_match_v1_contract():
    rt, _, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=SPEC
    )
    user, call, result, answer, done = got
    assert user.payload == {"content": "你好"}
    assert call.payload["iteration"] == 1 and call.payload["tier"] == "fast"
    assert call.payload["input_tokens_est"] > 0
    assert result.payload["status"] == "ok" and result.payload["iteration"] == 1
    assert result.payload["text"] == "好" and result.payload["tool_calls"] == []
    assert result.payload["stop_reason"] == "stop"
    assert result.payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert result.payload["output_tokens_est"] == 1
    assert isinstance(result.payload["latency_ms"], int)
    assert answer.payload == {"content": "好", "token_usage": 2}
    assert done.payload == {
        "reason": "completed",
        "iteration": 1,
        "detail": "stop_reason=stop",
    }
    assert "cause" not in done.payload


async def test_state_channels_after_run():
    rt, _, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=SPEC
    )
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    assert snap.next == ()
    assert snap.values["iteration"] == 1 and snap.values["termination"] is None
    est = got[1].payload["input_tokens_est"] + got[2].payload["output_tokens_est"]
    assert snap.values["tokens_used"] == est
    assert [type(m).__name__ for m in snap.values["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]


async def test_session_transitions_t1_t4_and_recovery_reset():
    rt, _, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    await sessions.bump_recovery(sid)
    await sessions.bump_recovery(sid)
    await collect(rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=SPEC)
    row = await sessions.get(sid)
    assert row["run_state"] == "idle" and row["recovery_count"] == 0


async def test_missing_session_wrong_tenant_and_busy_are_rejected():
    rt, _, _, sessions = make_runtime(text_turn("好"))
    with pytest.raises(ValueError, match="不存在"):
        await collect(
            rt, tenant_id="t-a", session_id="ghost", user_input="x", spec=SPEC
        )
    sid = await _session(sessions)
    with pytest.raises(ValueError, match="不属于租户"):
        await collect(rt, tenant_id="t-b", session_id=sid, user_input="x", spec=SPEC)
    await sessions.transition(sid, expected="idle", to="running")
    with pytest.raises(SessionBusy):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    assert (await sessions.get(sid))["run_state"] == "running"  # 拒绝不改状态


async def test_second_turn_continues_seq_and_resets_run_channels():
    """发现 F1：私有通道跨 run 持久化——第二轮 iteration 从 0 重新计，seq 接着上一轮。"""
    rt, cand, _events, sessions = make_runtime(text_turn("一"), text_turn("二"))
    sid = await _session(sessions)
    first = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="第一问", spec=SPEC
    )
    second = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="第二问", spec=SPEC
    )
    assert [e.seq for e in second] == [6, 7, 8, 9, 10]
    assert second[0].type is EventType.USER_MESSAGE
    assert second[0].payload == {"content": "第二问"}
    assert second[-1].payload["iteration"] == 1  # 不是 2：run 起点归零
    assert first[0].run_id != second[0].run_id
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    assert [m.content for m in snap.values["messages"]] == [
        "第一问",
        "一",
        "第二问",
        "二",
    ]
    assert cand.calls == 2


async def test_model_settings_carrier_reaches_gateway_and_candidate():
    """tier 走网关路由（只有 fast 档有候选：档位没到网关就是 ROUTE_MISSING）；max_tokens 透传到候选 kwargs。"""
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        context_config=ContextConfig(output_reserve=321),
    )
    rt, cand, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    await collect(rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec)
    seen = cand.seen_kwargs[0]
    assert seen["max_tokens"] == 321
    assert "tier" not in seen and "deadline_s" not in seen and "session_id" not in seen


async def test_input_estimate_covers_system_prompt_and_user_message():
    rt, _, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="退款申请", spec=SPEC
    )
    expected = estimate_messages_tokens(
        [AIMessage(SPEC.system_prompt), HumanMessage("退款申请")]
    )
    assert got[1].payload["input_tokens_est"] == expected


async def test_token_seed_rebuilt_from_history_events():
    """D8：会话级预算从历史 llm_call / llm_result 的估算字段重建，不加新列。"""
    store = MemoryEventStore()
    rt, _, _, sessions = make_runtime(text_turn("好"), events=store)
    sid = await _session(sessions)
    await store.append(
        event_id="00000000-0000-5000-8000-000000000001",
        tenant_id="t-a",
        session_id=sid,
        run_id="r-old",
        event_type="llm_call",
        payload={"input_tokens_est": 40},
    )
    await store.append(
        event_id="00000000-0000-5000-8000-000000000002",
        tenant_id="t-a",
        session_id=sid,
        run_id="r-old",
        event_type="llm_result",
        payload={"output_tokens_est": 20, "status": "ok"},
    )
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=SPEC
    )
    assert got[0].seq == 3  # 接着历史 seq
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    est = got[1].payload["input_tokens_est"] + got[2].payload["output_tokens_est"]
    assert snap.values["tokens_used"] == 60 + est


async def test_session_budget_precheck_stops_before_llm_call():
    """闸门 #3（L2 预检，无 cause）：注定超预算就不发 llm_call；兜底话术进事件与 state。"""
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        policy=LoopPolicy(session_token_budget=1),
    )
    rt, cand, _, sessions = make_runtime(text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec
    )
    assert [e.type.value for e in got] == [
        "user_message",
        "assistant_message",
        "loop_terminated",
    ]
    assert got[1].payload == {"content": u.FALLBACK_BUDGET}
    assert got[2].payload["reason"] == "token_budget_exceeded"
    assert "cause" not in got[2].payload and got[2].payload["iteration"] == 0
    assert cand.calls == 0
    snap = await rt.build_agent("t-a", spec).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    assert snap.values["messages"][-1].content == u.FALLBACK_BUDGET
    assert (await sessions.get(sid))["run_state"] == "idle"
