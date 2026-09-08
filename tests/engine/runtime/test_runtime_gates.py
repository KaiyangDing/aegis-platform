"""六道闸门（M2.4 Gates）：#1 轮数 / #4 重复 / #5 协议（含幻觉工具名）/ #6 取消各正例 + 边界；终止时不留悬空 tool_call；
栈序与出边静态快照；未声明 jump_to 静默无效回归；推导式 recursion_limit 在最长路径上紧致；八值终止原因全部可达。零真实调用。"""

import asyncio
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.gates",
    reason="M2.4 未敲：app/engine/runtime/middleware/gates.py 不存在",
)

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from app.engine.fakes import FakeToolChatModel
from app.engine.gateway.errors import AuthError, ProviderServerError
from app.engine.runtime import utterances as u
from app.engine.runtime.middleware.gates import Gates, canonical_key, classify
from app.engine.runtime.middleware.model_call import ModelCall
from app.engine.runtime.middleware.run_events import RunEvents
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.runtime import MIDDLEWARE_STACK, recursion_limit_for
from app.engine.runtime.spec import AgentSpec, LoopPolicy, TerminationReason
from app.engine.runtime.state import AEGIS_SOURCE, RunContext
from app.engine.runtime.tools import SideEffect, ToolContext, ToolRegistry, tool
from tests.engine.gateway.doubles import finish, text
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import (
    MemoryEventStore,
    collect,
    empty_turn,
    make_runtime,
    text_turn,
    tool_turn,
)


def _spec(registry: ToolRegistry | None = None, **policy: Any) -> AgentSpec:
    return AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=(registry or build_registry()).specs(),
        policy=LoopPolicy(**policy),
    )


async def _session(sessions, sid: str = "s-1") -> str:
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    return sid


def _query(order_id: str, cid: str) -> list[Any]:
    return tool_turn(("demo_order_query", {"order_id": order_id}, cid))


def _types(got) -> list[str]:
    return [e.type.value for e in got]


def _count(got, kind: str) -> int:
    return sum(1 for e in got if e.type.value == kind)


async def _snapshot(rt, spec: AgentSpec, sid: str):
    return await rt.build_agent("t-a", spec).aget_state(
        {"configurable": {"thread_id": sid}}
    )


def _assert_no_dangling_tool_calls(messages) -> None:
    """ADR-012 决策 5：每条 AIMessage 的每个 tool_call 都有配对 ToolMessage。"""
    paired = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                assert call["id"] in paired, f"悬空 tool_call {call['id']}"


# ---------------------------------------------------------------- 闸门 #1 轮数


async def test_max_iterations_terminates_before_next_call_with_handoff():
    """三轮参数各异的工具调用（绕开 #4）后再欲发起第四次即终止：Gates.before_model 挡在 llm_call 之前。"""
    spec = _spec(max_iterations=3)
    rt, cand, _, sessions = make_runtime(
        _query("A-1", "c1"),
        _query("A-2", "c2"),
        _query("A-3", "c3"),
        text_turn("不该发出"),
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    done = got[-1].payload
    assert done == {
        "reason": "max_iterations",
        "iteration": 3,
        "detail": "已完成 3 次 LLM 调用，达 max_iterations 上限",
    }
    assert _count(got, "llm_call") == 3 and _count(got, "tool_call") == 3
    assert cand.calls == 3
    assert _types(got)[-3:] == ["tool_result", "assistant_message", "loop_terminated"]
    assert got[-2].payload == {"content": u.FALLBACK_MAX_ITERATIONS}
    assert "转人工" in got[-2].payload["content"]
    snap = await _snapshot(rt, spec, sid)
    assert snap.values["messages"][-1].content == u.FALLBACK_MAX_ITERATIONS
    assert snap.values["termination"]["reason"] == "max_iterations"
    _assert_no_dangling_tool_calls(snap.values["messages"])
    assert (await sessions.get(sid))["run_state"] == "idle"


async def test_last_allowed_call_can_still_complete():
    """边界：max_iterations=2，第 2 次调用给出文本 → completed（上限是"再欲发起"才拦）。"""
    spec = _spec(max_iterations=2)
    rt, _, _, sessions = make_runtime(_query("A-1", "c1"), text_turn("查到了"))
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert got[-1].payload["reason"] == "completed"
    assert got[-1].payload["iteration"] == 2


# ---------------------------------------------------------------- 闸门 #4 重复调用


async def test_repeat_break_pairs_prompt_without_tool_call_event():
    """连续 3 次同 (工具, 参数规范形)：第 3 次不执行（无 write-ahead 即无 tool_call 事件），打断话术以配对 ToolMessage 回填；
    第二次故意换键序——规范形不变、计数不重置。"""
    spec = _spec()
    rt, cand, _, sessions = make_runtime(
        tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": 80}, "c1")),
        tool_turn(("demo_refund_apply", {"amount": 80, "order_id": "1024"}, "c2")),
        tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": 80}, "c3")),
        text_turn("我们换个方式核实。"),
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=spec
    )
    assert _count(got, "tool_call") == 2 and _count(got, "llm_call") == 4
    assert got[-1].payload["reason"] == "completed"
    assert cand.calls == 4
    snap = await _snapshot(rt, spec, sid)
    broken = next(
        m
        for m in snap.values["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "c3"
    )
    assert broken.content == u.PROMPT_REPEAT_BREAK.format(limit=3)
    assert broken.status == "error"
    assert snap.values["repeat"] == {
        "key": canonical_key("demo_refund_apply", {"order_id": "1024", "amount": 80}),
        "streak": 3,
    }
    _assert_no_dangling_tool_calls(snap.values["messages"])


async def test_repeat_after_break_terminates():
    """打断不清零：原样再犯 → repeated_calls；恰 limit+1=4 次 LLM 调用、2 次真执行；第 4 次配对"未执行"。"""
    spec = _spec()
    rt, _, _, sessions = make_runtime(*[_query("A-8", f"c{i}") for i in range(1, 5)])
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    done = got[-1].payload
    assert done["reason"] == "repeated_calls" and done["iteration"] == 4
    assert done["detail"] == "打断后仍第 4 次重复调用 demo_order_query"
    assert _count(got, "llm_call") == 4 and _count(got, "tool_call") == 2
    fallbacks = [e for e in got if e.type.value == "assistant_message"]
    assert [e.payload for e in fallbacks] == [{"content": u.FALLBACK_REPEATED}]
    assert (
        len(got) == 15
    )  # 1 user + 4×(call+result) + 2×(tool_call+tool_result) + 兜底 + 终止
    snap = await _snapshot(rt, spec, sid)
    messages = snap.values["messages"]
    last_tool = [m for m in messages if isinstance(m, ToolMessage)][-1]
    assert last_tool.tool_call_id == "c4" and last_tool.content == u.TOOL_NOT_EXECUTED
    assert messages[-1].content == u.FALLBACK_REPEATED
    _assert_no_dangling_tool_calls(messages)


async def test_varied_args_reset_repeat_streak():
    spec = _spec()
    rt, _, _, sessions = make_runtime(
        _query("A-9", "c1"),
        _query("A-9", "c2"),
        _query("B-1", "c3"),
        text_turn("三笔都查到了。"),
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert _count(got, "tool_call") == 3
    assert got[-1].payload["reason"] == "completed"
    snap = await _snapshot(rt, spec, sid)
    assert snap.values["repeat"]["streak"] == 1


async def test_repeat_streak_counts_across_calls_within_one_turn():
    """一轮里三个相同调用：前两个执行，第三个达阈值被打断——计数按声明序跨调用连续。"""
    spec = _spec()
    rt, _, _, sessions = make_runtime(
        tool_turn(
            *[("demo_order_query", {"order_id": "A-1"}, f"c{i}") for i in (1, 2, 3)]
        ),
        text_turn("好"),
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert _count(got, "tool_call") == 2 and got[-1].payload["reason"] == "completed"
    snap = await _snapshot(rt, spec, sid)
    contents = {
        m.tool_call_id: m.content
        for m in snap.values["messages"]
        if isinstance(m, ToolMessage)
    }
    assert contents["c3"] == u.PROMPT_REPEAT_BREAK.format(limit=3)
    assert "已发货" in contents["c1"] and "已发货" in contents["c2"]


# ---------------------------------------------------------------- 闸门 #5 协议违规


async def test_protocol_violation_retry_then_terminate():
    """空输出 / 宣告工具停却没给调用 / 空输出 → 纠错两次仍违规即终止；纠错以 user 消息注入并带运行时标记。"""
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, cand, _, sessions = make_runtime(
        empty_turn(), empty_turn("tool_calls"), empty_turn()
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="在吗", spec=spec
    )
    done = got[-1].payload
    assert done["reason"] == "protocol_violation" and done["iteration"] == 3
    assert done["detail"] == "连续 3 次协议违规，超过纠错上限"
    assert _count(got, "llm_call") == 3 and cand.calls == 3
    assert [e.payload for e in got if e.type.value == "assistant_message"] == [
        {"content": u.FALLBACK_PROTOCOL}
    ]
    snap = await _snapshot(rt, spec, sid)
    retries = [
        m
        for m in snap.values["messages"]
        if isinstance(m, HumanMessage) and m.additional_kwargs.get(AEGIS_SOURCE)
    ]
    assert len(retries) == 2 and all(
        m.content == u.PROMPT_PROTOCOL_RETRY for m in retries
    )
    assert snap.values["violations"] == 3


async def test_valid_output_resets_violation_count():
    """两次违规 → 一轮合法工具调用清零 → 再两次违规仍可纠错 → 文本完成（不清零则第 3 次违规即触杀）。"""
    spec = _spec()
    rt, _, _, sessions = make_runtime(
        empty_turn(),
        empty_turn(),
        _query("A-1", "c1"),
        empty_turn(),
        empty_turn(),
        text_turn("查到了。"),
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert got[-1].payload["reason"] == "completed"
    assert _count(got, "llm_call") == 6
    snap = await _snapshot(rt, spec, sid)
    assert snap.values["violations"] == 0


async def test_hallucinated_tool_names_count_as_violations_and_are_fed_back_in_chinese():
    """一轮三个不同幻觉名 → 连续违规 3 > 2 终止；零 tool_call 事件；三个调用各配对中文回填，框架英文串不进 state。"""
    spec = _spec()
    rt, cand, _, sessions = make_runtime(
        tool_turn(("ghost_a", {}, "g1"), ("ghost_b", {}, "g2"), ("ghost_c", {}, "g3")),
        text_turn("不该发出"),
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="随便查查", spec=spec
    )
    done = got[-1].payload
    assert done["reason"] == "protocol_violation" and "ghost_c" in done["detail"]
    assert _count(got, "tool_call") == 0 and cand.calls == 1
    snap = await _snapshot(rt, spec, sid)
    messages = snap.values["messages"]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["g1", "g2", "g3"]
    assert all(
        m.content.startswith("工具 ghost_") and "不存在" in m.content for m in tool_msgs
    )
    assert all("is not a valid tool" not in str(m.content) for m in messages)
    assert messages[-1].content == u.FALLBACK_PROTOCOL
    _assert_no_dangling_tool_calls(messages)


async def test_single_hallucination_is_fed_back_and_loop_continues():
    spec = _spec()
    rt, _, _, sessions = make_runtime(
        tool_turn(("ghost", {}, "g1")), text_turn("换个方式")
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert got[-1].payload["reason"] == "completed" and _count(got, "llm_call") == 2
    snap = await _snapshot(rt, spec, sid)
    ghost = next(m for m in snap.values["messages"] if isinstance(m, ToolMessage))
    assert ghost.content == u.TOOL_UNKNOWN.format(
        name="ghost",
        available="demo_order_query、demo_refund_apply、demo_ticket_create",
    )
    assert snap.values["violations"] == 0  # 文本轮清零


async def test_truncated_text_counts_as_completed():
    """D18：max_tokens 截断但文本非空 → 正常完成，stop_reason 留痕。"""
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, _, _, sessions = make_runtime([text("回答到一半被截"), finish("length")])
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="讲讲政策", spec=spec
    )
    assert got[2].payload["stop_reason"] == "length"
    assert got[-1].payload == {
        "reason": "completed",
        "iteration": 1,
        "detail": "stop_reason=length",
    }


# ---------------------------------------------------------------- 闸门 #6 取消


async def test_cancel_before_first_llm_call_terminates_quietly():
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    cancel = asyncio.Event()
    cancel.set()
    rt, cand, _, sessions = make_runtime(text_turn("不该发出"))
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="你好", spec=spec, cancel=cancel
    )
    assert _types(got) == ["user_message", "loop_terminated"]
    assert got[-1].payload == {
        "reason": "cancelled",
        "iteration": 0,
        "detail": "收到取消信号",
    }
    assert cand.calls == 0
    snap = await _snapshot(rt, spec, sid)
    assert [type(m).__name__ for m in snap.values["messages"]] == ["HumanMessage"]
    assert (await sessions.get(sid))["run_state"] == "idle"


async def test_cancel_raised_during_tool_is_honored_at_next_llm_checkpoint():
    cancel = asyncio.Event()

    @tool(side_effect=SideEffect.READ)
    async def pull_plug(ctx: ToolContext) -> str:
        """执行中触发取消。"""
        cancel.set()
        return "ok"

    spec = _spec(ToolRegistry([pull_plug]))
    rt, cand, _, sessions = make_runtime(
        tool_turn(("pull_plug", {}, "c1")), text_turn("不该发出")
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="x", spec=spec, cancel=cancel
    )
    assert _types(got) == [
        "user_message",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "loop_terminated",
    ]
    assert (
        got[-1].payload["reason"] == "cancelled" and got[-1].payload["iteration"] == 1
    )
    assert cand.calls == 1


# ---------------------------------------------------------------- 静态：栈序、出边、跳转纪律、推导式


def test_stack_order_and_jump_declarations():
    stack = list(MIDDLEWARE_STACK)
    assert stack[0] is RunEvents and stack[-1] is ToolExec
    assert stack.index(Gates) == stack.index(ModelCall) - 1  # Gates 紧贴 ModelCall 之前
    assert Gates.abefore_model.__can_jump_to__ == ["end"]
    assert Gates.aafter_model.__can_jump_to__ == ["end", "model"]


def test_gate_nodes_have_conditional_edges_to_end_and_loop_entry():
    rt, _, _, _ = make_runtime()
    agent = rt.build_agent("t-a", _spec())
    graph = agent.get_graph()
    targets: dict[str, set[str]] = {}
    for edge in graph.edges:
        targets.setdefault(edge.source, set()).add(edge.target)
    loop_entry = next(
        f"{cls.__name__}.before_model"
        for cls in MIDDLEWARE_STACK
        if cls.abefore_model is not AgentMiddleware.abefore_model
        or cls.before_model is not AgentMiddleware.before_model
    )
    assert {"model", "RunEvents.after_agent"} <= targets["Gates.before_model"]
    assert {"tools", "RunEvents.after_agent", loop_entry} <= targets[
        "Gates.after_model"
    ]


async def test_undeclared_jump_is_silently_ignored_regression():
    """框架事实（ADR-012 决策 4 的反面）：未声明 can_jump_to 的钩子返回 jump_to 无任何效果——模型照跑。
    这条回归防止真中间件漂移成"看起来会跳、其实不跳"。"""

    class Undeclared(AgentMiddleware):
        async def abefore_model(self, state, runtime):
            return {"jump_to": "end"}

    model = FakeToolChatModel(messages=iter([AIMessage("照跑")]))
    agent = create_agent(model, middleware=[Undeclared()])
    out = await agent.ainvoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "照跑"


async def test_recursion_limit_is_exact_on_longest_path():
    """探针 M2.4-Q4：工具循环撞 max_iterations 是最长路径，推导值恰为最小可行值——减 1 即 GraphRecursionError。"""
    spec = _spec(max_iterations=2)
    acts = [_query("A-1", "c1"), _query("A-2", "c2"), text_turn("不该发出")]
    rt, _, _, sessions = make_runtime(*acts)
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="查", spec=spec)
    assert got[-1].payload["reason"] == "max_iterations"

    rt2, _, _, _ = make_runtime(*acts)
    agent = rt2.build_agent("t-a", spec)
    ctx = RunContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-2",
        run_id="r-2",
        spec=spec,
        registry=ToolRegistry(spec.tools),
        events=MemoryEventStore(),
    )
    with pytest.raises(GraphRecursionError):
        await agent.ainvoke(
            {"messages": [HumanMessage("查")]},
            {
                "configurable": {"thread_id": "s-2"},
                "recursion_limit": recursion_limit_for(spec.policy) - 1,
            },
            context=ctx,
        )


def test_canonical_key_and_classify_matrix():
    assert canonical_key("t", {"a": 1, "b": "x"}) == canonical_key(
        "t", {"b": "x", "a": 1}
    )
    assert canonical_key("t", {"a": 1}) != canonical_key("t", {"a": 2})
    assert canonical_key("t", {"a": 1}) != canonical_key("u", {"a": 1})
    assert (
        classify(AIMessage("", response_metadata={"finish_reason": "tool_calls"}))
        == "violation"
    )
    assert (
        classify(AIMessage("", response_metadata={"finish_reason": "stop"}))
        == "violation"
    )
    assert classify(AIMessage("  ")) == "violation"
    assert (
        classify(AIMessage("半截", response_metadata={"finish_reason": "length"}))
        == "text"
    )
    assert (
        classify(AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]))
        == "tools"
    )


# ---------------------------------------------------------------- DoD：八值全部可达


async def test_all_eight_termination_reasons_are_reachable():
    cancel = asyncio.Event()
    cancel.set()
    scenarios: list[tuple[AgentSpec, list[Any], dict[str, Any]]] = [
        (_spec(), [text_turn("好")], {}),
        (_spec(max_iterations=1), [_query("A-1", "c1"), text_turn("x")], {}),
        (_spec(), [[ProviderServerError("p1", "boom")]], {}),
        (_spec(session_token_budget=1), [text_turn("x")], {}),
        (_spec(), [_query("A-1", f"c{i}") for i in range(4)], {}),
        (_spec(), [empty_turn(), empty_turn(), empty_turn()], {}),
        (_spec(), [text_turn("x")], {"cancel": cancel}),
        (_spec(), [[AuthError("p1", "401")]], {}),
    ]
    seen: set[str] = set()
    for index, (spec, acts, extra) in enumerate(scenarios):
        rt, _, _, sessions = make_runtime(*acts)
        sid = await _session(sessions, f"s-{index}")
        got = await collect(
            rt, tenant_id="t-a", session_id=sid, user_input="x", spec=spec, **extra
        )
        seen.add(got[-1].payload["reason"])
    assert seen == {r.value for r in TerminationReason}
