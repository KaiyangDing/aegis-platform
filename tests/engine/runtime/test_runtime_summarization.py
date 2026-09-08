"""滚动摘要（M2.6 AegisSummarization）：触发 / 不触发、自家尺、summary_updated 事件（同钩子、先于 llm_call）、中文包装、
state 被 REMOVE_ALL 替换而 events 原文不动、网关失败 fail-open 提取式降级且不经框架 with_retry、注定终止的 run 不压缩、关层。
零真实调用：摘要调用消耗剧本候选的一幕（fast 档）。"""

import asyncio

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.summarization",
    reason="M2.6 未敲：middleware/summarization.py 不存在",
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.retry import RunnableRetry

from app.engine.gateway.errors import ProviderServerError
from app.engine.gateway.resilience import RetryPolicy
from app.engine.runtime import utterances as u
from app.engine.runtime.context import SUMMARY_SOURCE
from app.engine.runtime.middleware.summarization import (
    AegisSummarization,
    count_message_tokens,
    extractive_fallback,
    render_transcript,
)
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy
from tests.engine.gateway.doubles import ScriptedCandidate
from tests.engine.runtime.doubles import (
    collect,
    make_runtime,
    scripted_gateway_factory,
    text_turn,
)

FIRST_Q = "第一问" + "长" * 20  # 23 token
FIRST_A = "第一答" + "长" * 10  # 13 token
SECOND_Q = "第二问"  # 3 token：第二轮 before_model 时 state 共 39 ≥ 0.8×40=32 → 触发；keep=20 → 只压第一问


def _spec(history_budget: int = 40, **policy) -> AgentSpec:
    return AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        context_config=ContextConfig(history_budget=history_budget),
        policy=LoopPolicy(**policy),
    )


async def _two_runs(rt, sessions, spec: AgentSpec):
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    first = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input=FIRST_Q, spec=spec
    )
    second = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input=SECOND_Q, spec=spec
    )
    return first, second


def test_construction_uses_own_ruler_and_disables_framework_retry():
    gateway = scripted_gateway_factory(ScriptedCandidate(acts=[]))("t-a")
    mw = AegisSummarization(gateway, config=ContextConfig(history_budget=40))
    assert mw.trigger == ("tokens", 32) and mw.keep == ("tokens", 20)
    assert mw.token_counter is count_message_tokens
    assert count_message_tokens([HumanMessage("长" * 10), AIMessage("ab")]) == 11
    assert mw._summary_model is gateway and not isinstance(
        mw._summary_model, RunnableRetry
    )
    closed = AegisSummarization(gateway, config=ContextConfig(history_budget=0))
    assert closed.trigger is None


async def test_under_threshold_no_summary_and_no_extra_call():
    rt, cand, _, sessions = make_runtime(text_turn("好"))
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input="你好", spec=_spec()
    )
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert cand.calls == 1


async def test_trigger_writes_event_before_llm_call_and_replaces_state_with_chinese_wrapper():
    rt, cand, events, sessions = make_runtime(
        text_turn(FIRST_A), text_turn("摘要产物"), text_turn("第二答")
    )
    spec = _spec()
    _first, second = await _two_runs(rt, sessions, spec)
    assert [e.type.value for e in second] == [
        "user_message",
        "summary_updated",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert second[1].payload == {
        "summary": "摘要产物",
        "covered": 1,
        "kept": 2,
        "fallback": False,
    }
    assert cand.calls == 3  # 第一答 + 摘要 + 第二答
    snap = await rt.build_agent("t-a", spec).aget_state(
        {"configurable": {"thread_id": "s-1"}}
    )
    messages = snap.values["messages"]
    assert messages[0].content == u.SUMMARY_WRAPPER.format(summary="摘要产物")
    assert messages[0].additional_kwargs["lc_source"] == SUMMARY_SOURCE
    assert [m.content for m in messages[1:]] == [FIRST_A, SECOND_Q, "第二答"]
    assert all("Here is a summary" not in str(m.content) for m in messages)
    # events 原文不动：第一轮的用户原话与回答仍在事实源里，seq 连续
    assert events.rows[0]["payload"] == {"content": FIRST_Q}
    assert [r["seq"] for r in events.rows] == list(range(1, 12))
    assert second[2].payload["input_tokens_est"] > 0  # 编译后的 prompt 含摘要


async def test_gateway_failure_falls_open_to_extractive_summary_without_retry():
    """摘要那一幕由网关自己的受控重试耗尽（每次重试消耗候选的下一幕，故失败幕铺满 max_attempts）→ GatewayExhausted →
    提取式降级；框架的 with_retry（×3）已关：候选调用数恰 = 第一答 + 网关尝试数 + 第二答。"""
    attempts = RetryPolicy().max_attempts
    rt, cand, _, sessions = make_runtime(
        text_turn(FIRST_A),
        *[[ProviderServerError("p1", "fast 档挂了")] for _ in range(attempts)],
        text_turn("第二答"),
    )
    spec = _spec()
    _, second = await _two_runs(rt, sessions, spec)
    payload = second[1].payload
    assert payload["fallback"] is True
    assert payload["summary"].startswith(u.SUMMARY_FALLBACK_PREFIX)
    assert "用户：第一问" in payload["summary"]
    assert second[-1].payload["reason"] == "completed"
    assert second[-2].payload["content"] == "第二答"
    assert cand.calls == 1 + attempts + 1


async def test_doomed_run_is_not_summarized():
    """取消信号已置位：Gates 紧接着终止，压缩只会白花一次 LLM 调用——零候选调用。"""
    cancel = asyncio.Event()
    cancel.set()
    rt, cand, _, sessions = make_runtime(text_turn(FIRST_A), text_turn("不该发出"))
    spec = _spec()
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    await collect(rt, tenant_id="t-a", session_id="s-1", user_input=FIRST_Q, spec=spec)
    second = await collect(
        rt,
        tenant_id="t-a",
        session_id="s-1",
        user_input=SECOND_Q,
        spec=spec,
        cancel=cancel,
    )
    assert [e.type.value for e in second] == ["user_message", "loop_terminated"]
    assert cand.calls == 1


async def test_closed_history_layer_never_summarizes():
    rt, cand, _, sessions = make_runtime(text_turn(FIRST_A), text_turn("第二答"))
    _, second = await _two_runs(rt, sessions, _spec(history_budget=0))
    assert "summary_updated" not in [e.type.value for e in second]
    assert cand.calls == 2


def test_transcript_and_fallback_are_deterministic_and_chinese():
    messages = [
        HumanMessage("问", additional_kwargs={"lc_source": SUMMARY_SOURCE}),
        HumanMessage("用户问题"),
        AIMessage("", tool_calls=[{"name": "q", "args": {"id": 1}, "id": "c1"}]),
        AIMessage("助手回答"),
    ]
    assert render_transcript(messages) == render_transcript(messages)
    assert render_transcript(messages).splitlines() == [
        "此前摘要：问",
        "用户：用户问题",
        '助手：（调用 q({"id": 1})）',
        "助手：助手回答",
    ]
    fallback = extractive_fallback(messages, 200)
    assert fallback.splitlines() == [
        u.SUMMARY_FALLBACK_PREFIX,
        "此前摘要：问",
        "用户：用户问题",
        "助手：助手回答",
    ]
    assert extractive_fallback(messages, 5).endswith(u.CLIP_SUFFIX)  # 裁进预算
