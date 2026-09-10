"""工具执行七步（M2.5 ToolExec）：v1 test_executor 三件平移（前厅校验 / 闸门 fail-closed / 连败禁用；write-ahead / 超时取更严 /
读重试写不重试 / 写超时 RESULT_UNKNOWN；结果规范化 fail-open 留痕）+ v2 新增：重放去重（同任务身份二次进入 → 无第二把幂等键）、
取消检查点弃置剩余调用、通行证放行、增强层只接六类公开异常；M2.7：通行证是 {call id: approval_id}、持通行证者过批准后前置校验
（否决 → precheck_vetoed 事件、无 write-ahead）、write-ahead 后回填审批单 event_id 恰一次。直连 ToolExec 的用例用最小替身 Runtime
（task_coords 打桩），链路用例走真图。零真实调用。"""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.gates",
    reason="M2.4 未敲：app/engine/runtime/middleware/gates.py 不存在",
)

from app.engine.runtime import tools as tools_mod

if not hasattr(tools_mod, "OutcomeKind"):
    pytest.skip(
        "M2.5 未敲：tools.py 尚无 OutcomeKind / ToolOutcome", allow_module_level=True
    )

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.core.tokens import estimate_tokens
from app.engine.gateway.errors import ProviderServerError
from app.engine.runtime import utterances as u
from app.engine.runtime.middleware import tool_exec as tool_exec_mod
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.runtime import recursion_limit_for
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy
from app.engine.runtime.state import RunContext
from app.engine.runtime.tools import (
    OutcomeKind,
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
    RaisingModel,
    collect,
    make_runtime,
    scripted_gateway_factory,
    text_turn,
    tool_turn,
    unwrap_untrusted,
)

TENANT_CFG = {"approval_threshold": 200}


class SimulatedCrash(BaseException):
    """模拟 kill -9：不是 Exception，框架与 wrap 的错误处理都接不住。"""


# ---------------------------------------------------------------- 直连替身（不进图）


@pytest.fixture(autouse=True)
def stable_task_coords(monkeypatch):
    """直连 ToolExec 时不在图里：任务身份打桩成常量——同一测试内两次进入即"重放"。"""
    monkeypatch.setattr(
        "app.engine.runtime.state.task_coords", lambda: ("task-fixed", None)
    )


@pytest.fixture
def no_retry_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(tool_exec_mod, "_sleep", fake_sleep)
    return slept


def _harness(
    registry: ToolRegistry,
    *,
    gateway: Any = None,
    cancel: Any = None,
    policy: LoopPolicy | None = None,
    context_config: ContextConfig | None = None,
    approvals: Any = None,
    precheck: Any = None,
) -> tuple[RunContext, MemoryEventStore, Any]:
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        tools=registry.specs(),
        policy=policy or LoopPolicy(),
        context_config=context_config or ContextConfig(),
        tenant_config=TENANT_CFG,
    )
    events = MemoryEventStore()
    extras: dict[
        str, Any
    ] = {}  # M2.7 之前的 RunContext 没有这两个字段：只在给了值时才传
    if approvals is not None:
        extras["approvals"] = approvals
    if precheck is not None:
        extras["precheck"] = precheck
    ctx = RunContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-1",
        run_id="r-1",
        spec=spec,
        registry=registry,
        events=events,
        gateway=gateway,
        cancel=cancel,
        **extras,
    )
    return ctx, events, SimpleNamespace(context=ctx, stream_writer=lambda e: None)


def _request(
    runtime: Any,
    name: str,
    args: dict[str, Any],
    cid: str = "c1",
    *,
    approved: Any = (),
    calls: list[tuple[str, dict[str, Any], str]] | None = None,
) -> ToolCallRequest:
    """approved：call id 元组（通行证单号取 ap-{id}）或 {call id: approval_id} 映射（M2.7 通行证形态）。"""
    declared = calls or [(name, args, cid)]
    passport = (
        dict(approved)
        if isinstance(approved, dict)
        else {call_id: f"ap-{call_id}" for call_id in approved}
    )
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in declared],
            )
        ],
        "approved_calls": passport,
    }
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": cid, "type": "tool_call"},
        tool=None,
        state=state,
        runtime=runtime,
    )


async def _call(runtime: Any, request: ToolCallRequest) -> ToolMessage:
    out = await ToolExec().awrap_tool_call(request, handler=None)  # type: ignore[arg-type]
    assert isinstance(out, ToolMessage)
    return out


def _body(message: ToolMessage) -> str:
    """回填正文：M2.8 起经不可信包裹（首尾各一行标记），之前的树原样。"""
    return unwrap_untrusted(message.content)


def _m27() -> None:
    """M2.7 才有的符号（PrecheckVeto / 通行证回填 / 串行器修正）：未敲时跳过本用例。"""
    if not hasattr(tools_mod, "PrecheckVeto"):
        pytest.skip("M2.7 未敲：tools.py 尚无 PrecheckVeto")


def test_outcome_kind_values_are_stable():
    assert {k.value for k in OutcomeKind} == {
        "ok",
        "error",
        "result_unknown",
        "needs_approval",
        "disabled",
    }


# ---------------------------------------------------------------- ① ③ 前厅：校验、闸门、通行证


async def test_hallucinated_param_named_and_no_write_ahead():
    """extra=forbid 的报错点名多余字段——模型看到才能自我修正；坏参数不进 write-ahead。"""
    _, events, rt = _harness(build_registry())
    out = await _call(
        rt,
        _request(
            rt, "demo_refund_apply", {"order_id": "1024", "amount": 80, "coupon": "X"}
        ),
    )
    assert out.status == "error" and _body(out).startswith("参数校验失败：")
    assert "coupon" in _body(out) and events.rows == []


async def test_lax_numeric_string_passes_then_gate_blocks_without_passport():
    """校验宽容度与导出 schema 一致（lax）：数字字符串放行；随后闸门按值命中——没有通行证就不执行、不写事件。"""
    _, events, rt = _harness(build_registry())
    out = await _call(
        rt, _request(rt, "demo_refund_apply", {"order_id": "1", "amount": "350"})
    )
    assert out.status == "error"
    assert _body(out) == u.TOOL_NEEDS_APPROVAL.format(name="demo_refund_apply")
    assert events.rows == []


async def test_passport_lets_gated_call_execute():
    """approved_calls 通行证（M2.7 由 Approvals 写入）：同一调用带通行证即放行执行，幂等键进结果。"""
    _, events, rt = _harness(build_registry())
    out = await _call(
        rt,
        _request(
            rt, "demo_refund_apply", {"order_id": "1", "amount": 350}, approved=("c1",)
        ),
    )
    assert out.status == "success" and "refunded" in _body(out)
    assert events.types("s-1") == ["tool_call", "tool_result"]
    assert events.rows[0]["id"] in _body(out)  # 幂等键透传进了工具


def _gate_boom(args: Any, cfg: Any) -> bool:
    raise RuntimeError("闸门自己炸了 sk-abcdefghijklmnop")


@tool(side_effect=SideEffect.WRITE, risk_policy=_gate_boom)
async def demo_gate_bug(ctx: ToolContext, amount: int) -> dict:
    """风险闸门会崩溃的演示工具。"""
    return {}


async def test_gate_crash_fails_closed_and_is_sanitized():
    ctx, events, rt = _harness(ToolRegistry([demo_gate_bug]))
    out = await _call(rt, _request(rt, "demo_gate_bug", {"amount": 1}))
    assert out.status == "error"
    assert "fail-closed" in _body(out) and "未执行" in _body(out)
    assert "sk-***" in _body(out) and "abcdefghijklmnop" not in _body(out)
    assert events.rows == [] and ctx.tool_health.fail_streaks == {"demo_gate_bug": 1}


async def test_read_tool_never_consults_gate_and_unknown_tool_has_no_streak():
    ctx, _, rt = _harness(build_registry())
    out = await _call(rt, _request(rt, "demo_order_query", {"order_id": "A"}))
    assert out.status == "success"
    ghost = await _call(rt, _request(rt, "ghost", {}, "c9"))
    assert (
        _body(ghost).startswith("工具 ghost 不存在")
        and ctx.tool_health.fail_streaks == {}
    )


# ---------------------------------------------------------------- ④ write-ahead 与重放


@tool(side_effect=SideEffect.READ)
async def echo_ctx(ctx: ToolContext, ping: str) -> dict:
    """回显注入的身份与幂等键。"""
    return {"ping": ping, "tool_call_id": ctx.tool_call_id, "user": ctx.user_id}


async def test_write_ahead_lands_before_side_effect_and_key_reaches_handler():
    seen: list[tuple[int, str, str]] = []

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def observe(ctx: ToolContext) -> str:
        """执行时观察事实源。"""
        rows = events.rows
        seen.append((len(rows), rows[-1]["type"], rows[-1]["id"] == ctx.tool_call_id))
        return "done"

    _, events, rt = _harness(ToolRegistry([observe]))
    out = await _call(rt, _request(rt, "observe", {}))
    assert out.status == "success"
    assert seen == [
        (1, "tool_call", True)
    ]  # 副作用发生时 tool_call 已在盘上，钥匙就是它的 id
    assert events.types("s-1") == ["tool_call", "tool_result"]
    assert events.rows[1]["payload"]["tool_call_id"] == events.rows[0]["id"]


async def test_replay_reuses_write_ahead_key_and_dedupes_events():
    """同任务身份二次进入（崩溃重放）：无第二条 tool_call、tool_result 去重；副作用两次但下游只见一把钥匙。"""
    downstream: set[str] = set()
    calls = {"n": 0}

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def ship(ctx: ToolContext, order_id: str) -> dict:
        """发货：以幂等键去重。"""
        calls["n"] += 1
        downstream.add(ctx.tool_call_id)
        return {"shipped": order_id}

    _, events, rt = _harness(ToolRegistry([ship]))
    first = await _call(rt, _request(rt, "ship", {"order_id": "A"}))
    second = await _call(rt, _request(rt, "ship", {"order_id": "A"}))
    assert first.content == second.content
    assert calls["n"] == 2 and len(downstream) == 1
    assert events.types("s-1") == ["tool_call", "tool_result"]


async def test_crash_after_side_effect_then_resume_keeps_single_key(monkeypatch):
    """真图：工具副作用之后 BaseException 中断（checkpoint 停在 tools 之前）→ 恢复重放 → 恰一条 tool_call 事件、一把幂等键、
    tool_result 以原 id 闭合；重放去重命中的事件不再外流。"""
    monkeypatch.undo()  # 真图里用真实任务身份
    keys: list[str] = []
    armed = {"crash": True}

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def ship(ctx: ToolContext, order_id: str) -> dict:
        """发货（副作用之后可能崩溃）。"""
        keys.append(ctx.tool_call_id)
        if armed["crash"]:
            armed["crash"] = False
            raise SimulatedCrash("kill -9")
        return {"shipped": order_id}

    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast", tools=(ship,))
    rt, _, events, sessions = make_runtime(
        tool_turn(("ship", {"order_id": "A"}, "c1")), text_turn("已发货")
    )
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    with pytest.raises(SimulatedCrash):
        await collect(
            rt, tenant_id="t-a", session_id="s-1", user_input="发货", spec=spec
        )
    assert events.types("s-1") == [
        "user_message",
        "llm_call",
        "llm_result",
        "tool_call",
    ]
    assert (await sessions.get("s-1"))["run_state"] == "running"
    agent = rt.build_agent("t-a", spec)
    cfg = {
        "configurable": {"thread_id": "s-1"},
        "recursion_limit": recursion_limit_for(spec.policy),
    }
    assert (await agent.aget_state(cfg)).next == ("tools",)
    ctx = RunContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-1",
        run_id="r-resume",
        spec=spec,
        registry=ToolRegistry(spec.tools),
        events=events,
        sessions=sessions,
    )
    resumed = [
        payload
        async for _mode, payload in agent.astream(
            None, cfg, context=ctx, stream_mode=["custom"], durability="sync"
        )
    ]
    assert events.types("s-1") == [
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
    assert [e.type.value for e in resumed] == [
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]  # 重放的 tool_call 去重命中，不再外流
    assert len(keys) == 2 and len(set(keys)) == 1
    assert events.rows[4]["payload"]["tool_call_id"] == events.rows[3]["id"]
    assert (await sessions.get("s-1"))["run_state"] == "idle"


# ---------------------------------------------------------------- ⑤ 超时与重试


async def test_write_timeout_is_result_unknown_and_not_a_failure(no_retry_sleep):
    @tool(side_effect=SideEffect.WRITE, risk_exempt=True, timeout_s=0.05)
    async def slow_write(ctx: ToolContext) -> dict:
        """睡不醒的写工具。"""
        await asyncio.sleep(5)
        return {}

    ctx, events, rt = _harness(ToolRegistry([slow_write]))
    out = await _call(rt, _request(rt, "slow_write", {}))
    assert out.status == "error"
    assert _body(out) == u.TOOL_RESULT_UNKNOWN.format(name="slow_write")
    assert "禁止重试" in _body(out) and "查询" in _body(out)
    assert events.types("s-1") == ["tool_call", "tool_error"]
    assert events.rows[1]["payload"]["error"] == u.TOOL_ERROR_TIMEOUT_UNKNOWN
    assert ctx.tool_health.fail_streaks == {} and no_retry_sleep == []


async def test_read_timeout_final_error_counts_streak(no_retry_sleep):
    @tool(side_effect=SideEffect.READ, timeout_s=0.05)
    async def slow_read(ctx: ToolContext) -> dict:
        """睡不醒的读工具。"""
        await asyncio.sleep(5)
        return {}

    ctx, events, rt = _harness(ToolRegistry([slow_read]))
    out = await _call(rt, _request(rt, "slow_read", {}))
    assert out.status == "error" and _body(out) == u.TOOL_TIMEOUT.format(timeout_s=0.05)
    assert events.rows[1]["payload"]["error"] == u.TOOL_ERROR_TIMEOUT.format(
        timeout_s=0.05
    )
    assert ctx.tool_health.fail_streaks == {"slow_read": 1}


async def test_read_retries_with_backoff_then_succeeds(no_retry_sleep):
    calls = {"n": 0}

    @tool(side_effect=SideEffect.READ, retries=2)
    async def flaky_read(ctx: ToolContext) -> dict:
        """时好时坏的读工具。"""
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("抖一下")
        return {"ok": calls["n"]}

    _, events, rt = _harness(ToolRegistry([flaky_read]))
    out = await _call(rt, _request(rt, "flaky_read", {}))
    assert out.status == "success" and calls["n"] == 3
    assert no_retry_sleep == [pytest.approx(0.2), pytest.approx(0.4)]
    assert events.types("s-1") == ["tool_call", "tool_result"]
    assert events.rows[1]["payload"]["retry_count"] == 2


async def test_write_exception_is_single_attempt(no_retry_sleep):
    calls = {"n": 0}

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def boom_write(ctx: ToolContext) -> dict:
        """一碰就炸的写工具。"""
        calls["n"] += 1
        raise RuntimeError("下游拒绝")

    _, events, rt = _harness(ToolRegistry([boom_write]))
    out = await _call(rt, _request(rt, "boom_write", {}))
    assert out.status == "error" and _body(out) == u.TOOL_FAILED.format(
        detail="下游拒绝"
    )
    assert calls["n"] == 1 and no_retry_sleep == []
    assert events.types("s-1") == ["tool_call", "tool_error"]


async def test_stricter_timeout_wins():
    @tool(side_effect=SideEffect.READ, timeout_s=99.0)
    async def optimistic(ctx: ToolContext) -> dict:
        """自以为有 99 秒的工具。"""
        await asyncio.sleep(5)
        return {}

    _, _, rt = _harness(
        ToolRegistry([optimistic]), policy=LoopPolicy(tool_step_timeout_s=0.05)
    )
    out = await _call(rt, _request(rt, "optimistic", {}))
    assert _body(out) == u.TOOL_TIMEOUT.format(timeout_s=0.05)


# ---------------------------------------------------------------- ⑦ 连败禁用


async def test_two_failures_disable_tool_for_run():
    ctx, events, rt = _harness(build_registry())
    bad = {"order_id": "1", "extra": 1}
    first = await _call(rt, _request(rt, "demo_order_query", bad, "c1"))
    second = await _call(rt, _request(rt, "demo_order_query", bad, "c2"))
    assert not _body(first).endswith(u.TOOL_STREAK_DISABLED.format(streak=1))
    assert _body(second).endswith(u.TOOL_STREAK_DISABLED.format(streak=2))
    third = await _call(rt, _request(rt, "demo_order_query", {"order_id": "1"}, "c3"))
    assert _body(third) == u.TOOL_DISABLED.format(name="demo_order_query", limit=2)
    assert events.rows == [] and ctx.tool_health.disabled == {"demo_order_query"}


async def test_success_resets_fail_streak_and_streak_is_per_tool_and_per_run():
    ctx, _, rt = _harness(build_registry())
    await _call(rt, _request(rt, "demo_order_query", {"x": 1}, "c1"))
    ok = await _call(rt, _request(rt, "demo_order_query", {"order_id": "1"}, "c2"))
    assert ok.status == "success" and ctx.tool_health.fail_streaks == {}
    again = await _call(rt, _request(rt, "demo_order_query", {"x": 1}, "c3"))
    assert "本轮已禁用" not in again.content
    other = await _call(rt, _request(rt, "demo_ticket_create", {"title": 1}, "c4"))
    assert (
        other.status == "error" and "本轮已禁用" not in other.content
    )  # 首败，不受连累
    fresh_ctx, _, fresh_rt = _harness(build_registry())  # 新 run：新账本
    assert fresh_ctx.tool_health.fail_streaks == {}
    out = await _call(fresh_rt, _request(fresh_rt, "demo_order_query", {"x": 1}, "c5"))
    assert "本轮已禁用" not in _body(out)


# ---------------------------------------------------------------- ⑥ 结果规范化


@tool(side_effect=SideEffect.READ)
async def big_read(ctx: ToolContext) -> dict:
    """返回超大结果的读工具。"""
    return {"rows": ["订单数据条目内容" * 5 for _ in range(200)]}


@tool(side_effect=SideEffect.READ)
async def small_read(ctx: ToolContext) -> dict:
    """返回小结果的读工具。"""
    return {"status": "已发货", "eta": "明天"}


def _digest_gateway(*acts: list[Any]) -> Any:
    return scripted_gateway_factory(ScriptedCandidate(acts=list(acts)))("t-a")


async def test_small_result_passes_through_without_injected():
    _, events, rt = _harness(ToolRegistry([small_read]), gateway=_digest_gateway())
    out = await _call(rt, _request(rt, "small_read", {}))
    assert _body(out) == '{"status": "已发货", "eta": "明天"}'
    payload = events.rows[1]["payload"]
    assert "injected" not in payload and "normalization" not in payload


async def test_over_budget_uses_fast_tier_digest_and_keeps_raw():
    gateway = _digest_gateway(text_turn("共 200 条订单数据，全部已发货"))
    _, events, rt = _harness(
        ToolRegistry([big_read]),
        gateway=gateway,
        context_config=ContextConfig(tool_results_budget=100),
    )
    out = await _call(rt, _request(rt, "big_read", {}))
    assert out.status == "success"
    assert _body(out) == u.TOOL_SUMMARY_PREFIX + "共 200 条订单数据，全部已发货"
    payload = events.rows[1]["payload"]
    assert len(payload["result"]["rows"]) == 200  # 原文一条不少
    assert payload["injected"] == _body(out) and payload["normalization"] == "summary"


async def test_digest_gateway_failure_fails_open_to_truncation():
    gateway = _digest_gateway([ProviderServerError("p1", "fast 档挂了")])
    _, events, rt = _harness(
        ToolRegistry([big_read]),
        gateway=gateway,
        context_config=ContextConfig(tool_results_budget=100),
    )
    out = await _call(rt, _request(rt, "big_read", {}))
    assert out.status == "success" and _body(out).endswith(u.CLIP_SUFFIX)
    assert estimate_tokens(_body(out)) < 200
    payload = events.rows[1]["payload"]
    assert payload["normalization"] == "truncated" and "summarize_error" in payload


async def test_no_gateway_truncates_deterministically():
    _, events, rt = _harness(
        ToolRegistry([big_read]), context_config=ContextConfig(tool_results_budget=100)
    )
    out = await _call(rt, _request(rt, "big_read", {}))
    assert _body(out).endswith(u.CLIP_SUFFIX)
    payload = events.rows[1]["payload"]
    assert payload["normalization"] == "truncated" and "summarize_error" not in payload


async def test_oversized_digest_gets_truncated_too():
    gateway = _digest_gateway(text_turn("长" * 5000))
    _, events, rt = _harness(
        ToolRegistry([big_read]),
        gateway=gateway,
        context_config=ContextConfig(tool_results_budget=100),
    )
    out = await _call(rt, _request(rt, "big_read", {}))
    assert _body(out).startswith(u.TOOL_SUMMARY_PREFIX) and _body(out).endswith(
        u.CLIP_SUFFIX
    )
    assert estimate_tokens(_body(out)) < 200
    assert events.rows[1]["payload"]["normalization"] == "summary"


async def test_provider_error_leak_in_digest_propagates():
    """增强层只接六类公开异常：网关内部家族泄漏仍是 bug 信号，裸炸。"""
    _, _, rt = _harness(
        ToolRegistry([big_read]),
        gateway=RaisingModel(),
        context_config=ContextConfig(tool_results_budget=100),
    )
    with pytest.raises(ProviderServerError):
        await _call(rt, _request(rt, "big_read", {}))


# ---------------------------------------------------------------- 闸门 #6 工具检查点


async def test_cancel_at_tool_checkpoint_discards_remaining_calls_with_single_writer():
    """一轮三个调用：第一个执行中触发取消 → 第二个不执行且写 termination（含弃置计数）→ 第三个只配对回填（不再写通道）；
    Gates.before_model 看到 termination 直接收尾，零后续 LLM 调用。"""
    cancel = asyncio.Event()

    @tool(side_effect=SideEffect.READ)
    async def pull_plug(ctx: ToolContext) -> str:
        """执行中触发取消。"""
        cancel.set()
        return "ok"

    registry = ToolRegistry([pull_plug, demo_order_query])
    spec = AgentSpec(
        system_prompt="你是演示客服。", model_tier="fast", tools=registry.specs()
    )
    rt, cand, _, sessions = make_runtime(
        tool_turn(
            ("pull_plug", {}, "c1"),
            ("demo_order_query", {"order_id": "1"}, "c2"),
            ("demo_order_query", {"order_id": "2"}, "c3"),
        ),
        text_turn("不该发出"),
    )
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input="x", spec=spec, cancel=cancel
    )
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "loop_terminated",
    ]
    assert got[-1].payload == {
        "reason": "cancelled",
        "iteration": 1,
        "detail": "收到取消信号（工具检查点）；弃置本轮剩余 1 个调用",
    }
    assert cand.calls == 1
    snap = await rt.build_agent("t-a", spec).aget_state(
        {"configurable": {"thread_id": "s-1"}}
    )
    tool_msgs = {
        m.tool_call_id: m for m in snap.values["messages"] if isinstance(m, ToolMessage)
    }
    assert _body(tool_msgs["c1"]) == "ok"
    assert (
        tool_msgs["c2"].content == u.TOOL_CANCELLED
        and tool_msgs["c3"].content == u.TOOL_CANCELLED
    )
    assert not isinstance(
        snap.values["messages"][-1], AIMessage
    )  # 取消零话术：不追加兜底


async def test_serializer_does_not_wait_for_calls_paired_by_after_model_hooks(
    monkeypatch,
):
    """发现 F7（M2.7 修）：同一轮 [幻觉名 c1, 真调用 c2]——c1 被 Gates 配对、永不进 tools；c2 的串行器不能等 c1，否则死锁。"""
    _m27()
    monkeypatch.undo()  # 真图里用真实任务身份（钉死的任务 id 会让第二轮事件被当作重放去重）
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=build_registry().specs(),
    )
    rt, cand, _events, sessions = make_runtime(
        tool_turn(
            ("ghost_tool", {}, "c1"),
            ("demo_order_query", {"order_id": "1024"}, "c2"),
        ),
        text_turn("查到了"),
    )
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    got = await asyncio.wait_for(
        collect(rt, tenant_id="t-a", session_id="s-1", user_input="查", spec=spec),
        timeout=10,
    )
    assert [e.type.value for e in got] == [
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
    assert got[3].payload["model_call_id"] == "c2" and cand.calls == 2
    snap = await rt.build_agent("t-a", spec).aget_state(
        {"configurable": {"thread_id": "s-1"}}
    )
    tool_msgs = {
        m.tool_call_id: m for m in snap.values["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_msgs["c1"].content.startswith("工具 ghost_tool 不存在")
    assert "已发货" in tool_msgs["c2"].content


# ---------------------------------------------------------------- ③′ 通行证：前置校验挂点与审批单回填（M2.7）


async def test_precheck_veto_writes_event_and_skips_write_ahead():
    """TOCTOU 挂点：持通行证者执行前重跑业务校验——否决即 precheck_vetoed 事件（detail 只进事件）、不进 write-ahead、不记连败账。"""
    _m27()

    async def precheck(name: str, args: Any) -> Any:
        from app.engine.runtime.tools import PrecheckVeto

        return PrecheckVeto(observation="订单已关闭", detail="status=closed")

    ctx, events, rt = _harness(build_registry(), precheck=precheck)
    out = await _call(
        rt,
        _request(
            rt, "demo_refund_apply", {"order_id": "1", "amount": 350}, approved=("c1",)
        ),
    )
    assert out.status == "error"
    assert _body(out) == u.PRECHECK_VETO_TEMPLATE.format(reason="订单已关闭")
    assert "status=closed" not in _body(out)
    assert events.types("s-1") == ["precheck_vetoed"]
    assert events.rows[0]["payload"] == {
        "approval_id": "ap-c1",
        "tool_name": "demo_refund_apply",
        "observation": "订单已关闭",
        "detail": "status=closed",
    }
    assert ctx.tool_health.fail_streaks == {}


async def test_precheck_only_consulted_with_passport_and_pass_executes():
    _m27()
    consulted: list[tuple[str, dict[str, Any]]] = []

    async def precheck(name: str, args: Any) -> Any:
        consulted.append((name, dict(args)))
        return None

    _, events, rt = _harness(build_registry(), precheck=precheck)
    await _call(
        rt, _request(rt, "demo_order_query", {"order_id": "1"})
    )  # 无通行证的读工具：不问
    out = await _call(
        rt,
        _request(
            rt,
            "demo_refund_apply",
            {"order_id": "1", "amount": 350},
            "c2",
            approved=("c2",),
        ),
    )
    assert out.status == "success"
    assert consulted == [("demo_refund_apply", {"order_id": "1", "amount": 350})]
    assert events.types("s-1") == ["tool_call", "tool_result"] * 2


async def test_write_ahead_backfills_approval_event_id_exactly_once():
    """批准已兑现的凭证：write-ahead 之后把 tool_call 事件 id 回填审批单；重放二次进入不覆盖。"""
    _m27()
    approvals = MemoryApprovalStore()
    aid = str(uuid.uuid4())
    await approvals.create(
        approval_id=aid,
        tenant_id="t-a",
        session_id="s-1",
        run_id="r-1",
        tool_name="demo_refund_apply",
        args={"order_id": "1", "amount": 350},
        ttl_s=60,
    )
    _, events, rt = _harness(build_registry(), approvals=approvals)
    request = _request(
        rt, "demo_refund_apply", {"order_id": "1", "amount": 350}, approved={"c1": aid}
    )
    first = await _call(rt, request)
    assert first.status == "success"
    assert approvals.rows[aid]["event_id"] == events.rows[0]["id"]
    second = await _call(rt, request)  # 重放：同一把钥匙、回填不变
    assert second.content == first.content
    assert approvals.rows[aid]["event_id"] == events.rows[0]["id"]
    assert events.types("s-1") == ["tool_call", "tool_result"]


async def test_cancelled_call_returns_command_once_then_plain_messages():
    """直连：同一轮里取消检查点只有第一次返回 Command（写 termination），之后只回 ToolMessage——通道单写者。"""
    cancel = asyncio.Event()
    cancel.set()
    ctx, _, rt = _harness(build_registry(), cancel=cancel)
    calls = [("demo_order_query", {"order_id": str(i)}, f"c{i}") for i in (1, 2)]
    first = await ToolExec().awrap_tool_call(
        _request(rt, *calls[0], calls=calls),
        handler=None,  # type: ignore[arg-type]
    )
    second = await ToolExec().awrap_tool_call(
        _request(rt, *calls[1], calls=calls),
        handler=None,  # type: ignore[arg-type]
    )
    assert (
        isinstance(first, Command)
        and first.update["termination"]["reason"] == "cancelled"
    )
    assert (
        first.update["termination"]["detail"]
        == "收到取消信号（工具检查点）；弃置本轮剩余 1 个调用"
    )
    assert isinstance(second, ToolMessage) and second.content == u.TOOL_CANCELLED
    assert ctx.tool_health.terminated is True
