"""L1 异常四组映射（契约 C4）：(Exhausted, Overloaded)→step_timeout、(Budget, TenantQuota)→token_budget_exceeded、
Rejected→gateway_rejected 零话术、StreamInterrupted→配对 interrupted 后作废重发（消耗迭代）；
ProviderError 泄漏裸炸（绝不 except 基类）；wrap 内重发受 #1 限制。零真实调用，网关是真候选环 + 剧本候选。"""

import pytest

pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)

from app.engine.gateway.errors import (
    AuthError,
    GatewayOverloadedError,
    ProviderServerError,
)
from app.engine.runtime import utterances as u
from app.engine.runtime.spec import AgentSpec, LoopPolicy
from tests.engine.gateway.doubles import StubLimiter, text
from tests.engine.runtime.doubles import (
    RaisingModel,
    collect,
    make_runtime,
    text_turn,
)

SPEC = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
FAILED_CHAIN = [
    "user_message",
    "llm_call",
    "llm_result",
    "assistant_message",
    "loop_terminated",
]


async def _session(sessions, sid: str = "s-1") -> str:
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    return sid


def _assert_failed(got, *, reason: str, cause: str, fallback: str) -> None:
    assert [e.type.value for e in got] == FAILED_CHAIN
    result = got[2].payload
    assert (
        result["status"] == "failed"
        and result["cause"] == cause
        and result["iteration"] == 1
    )
    assert got[3].payload == {"content": fallback}
    done = got[4].payload
    assert (
        done["reason"] == reason and done["cause"] == cause and done["iteration"] == 1
    )
    assert done["detail"] == result["detail"]


async def test_gateway_exhausted_maps_to_step_timeout():
    rt, cand, _, sessions = make_runtime([ProviderServerError("p1", "boom")])
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    _assert_failed(
        got,
        reason="step_timeout",
        cause="gateway_exhausted",
        fallback=u.FALLBACK_STEP_FAILED,
    )
    assert cand.calls >= 1
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    assert snap.values["messages"][-1].content == u.FALLBACK_STEP_FAILED
    assert snap.values["termination"]["reason"] == "step_timeout"
    assert (await sessions.get(sid))["run_state"] == "idle"


async def test_overloaded_maps_to_step_timeout():
    rt, _, _, sessions = make_runtime(
        [GatewayOverloadedError("[p1] 本地连接池排队超时")]
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    _assert_failed(
        got,
        reason="step_timeout",
        cause="gateway_overloaded",
        fallback=u.FALLBACK_STEP_FAILED,
    )


async def test_request_budget_maps_to_token_budget_with_l1_cause():
    rt, cand, _, sessions = make_runtime(text_turn("好"), request_token_budget=1)
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    _assert_failed(
        got,
        reason="token_budget_exceeded",
        cause="l1_request_budget",
        fallback=u.FALLBACK_BUDGET,
    )
    assert cand.calls == 0


async def test_tenant_quota_maps_to_token_budget_with_l1_cause():
    rt, _, _, sessions = make_runtime(
        text_turn("好"), tenant_limiter=StubLimiter(deny={"t-a"})
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    _assert_failed(
        got,
        reason="token_budget_exceeded",
        cause="l1_tenant_quota",
        fallback=u.FALLBACK_BUDGET,
    )


async def test_gateway_rejected_terminates_without_any_utterance():
    rt, _, _, sessions = make_runtime([AuthError("p1", "401 bad key sk-abcdefghij")])
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "loop_terminated",
    ]
    assert got[2].payload["status"] == "failed"
    assert got[2].payload["cause"] == "gateway_rejected"
    done = got[3].payload
    assert done["reason"] == "gateway_rejected" and done["cause"] == "gateway_rejected"
    assert "全部候选均被确定性拒绝" in done["detail"]
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    last = snap.values["messages"][-1]
    assert (
        last.content == "" and last.tool_calls == []
    )  # 空 AIMessage：无话术但让出边走向 end
    assert (await sessions.get(sid))["run_state"] == "idle"


async def test_stream_interrupted_is_voided_and_resent_consuming_an_iteration():
    rt, cand, _, sessions = make_runtime(
        [text("半截"), ProviderServerError("p1", "reset")],
        text_turn("重发后的完整回答。"),
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert (
        got[2].payload["status"] == "interrupted" and got[2].payload["iteration"] == 1
    )
    assert "死因" in got[2].payload["detail"]
    assert got[3].payload["iteration"] == 2 and got[4].payload["status"] == "ok"
    assert got[5].payload["content"] == "重发后的完整回答。"
    assert got[6].payload == {
        "reason": "completed",
        "iteration": 2,
        "detail": "stop_reason=stop",
    }
    assert cand.calls == 2
    snap = await rt.build_agent("t-a", SPEC).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    assert [m.content for m in snap.values["messages"]] == [
        "x",
        "重发后的完整回答。",
    ]  # 半截未入 state


async def test_resend_is_bounded_by_max_iterations():
    """wrap 内的作废重发同样受闸门 #1：max_iterations=1 时第二次不再发，走 max_iterations 终止。"""
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        policy=LoopPolicy(max_iterations=1),
    )
    rt, cand, _, sessions = make_runtime(
        [text("半截"), ProviderServerError("p1", "reset")], text_turn("不该发出")
    )
    sid = await _session(sessions)
    got = await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=spec)
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    assert got[2].payload["status"] == "interrupted"
    assert got[3].payload == {"content": u.FALLBACK_MAX_ITERATIONS}
    assert (
        got[4].payload["reason"] == "max_iterations"
        and got[4].payload["iteration"] == 1
    )
    assert "cause" not in got[4].payload
    assert cand.calls == 1


async def test_provider_error_leak_propagates_out_of_run():
    """绝不 except GatewayError 基类：内部家族泄漏是 bug 信号，run 裸炸、会话留在 running（交给恢复路径）。"""
    rt, _, events, sessions = make_runtime(gateway_for=lambda tenant_id: RaisingModel())
    sid = await _session(sessions)
    with pytest.raises(ProviderServerError):
        await collect(rt, tenant_id="t-a", session_id=sid, user_input="x", spec=SPEC)
    assert events.types(sid) == [
        "user_message",
        "llm_call",
    ]  # 孤儿 llm_call 留给恢复分诊
    assert (await sessions.get(sid))["run_state"] == "running"


async def test_all_eight_reasons_reachable_snapshot():
    """M2.3 已能产生的终止原因：completed / step_timeout / token_budget_exceeded / gateway_rejected / max_iterations；
    repeated_calls / protocol_violation / cancelled 随 M2.4 / M2.7。"""
    from app.engine.runtime.spec import TerminationReason

    reachable = {
        TerminationReason.COMPLETED,
        TerminationReason.STEP_TIMEOUT,
        TerminationReason.TOKEN_BUDGET_EXCEEDED,
        TerminationReason.GATEWAY_REJECTED,
        TerminationReason.MAX_ITERATIONS,
    }
    assert reachable < set(TerminationReason)
