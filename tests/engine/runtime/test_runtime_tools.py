"""工具链路（M2.3 骨架 ToolExec）：九事件序列、write-ahead 事件 id 即 ToolContext.tool_call_id、两种 id 各进哪里、
多调用串行、工具异常 → tool_error + 中文回填且循环继续、坏参数不进 write-ahead、异常文本消毒。零真实调用。"""

import asyncio
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)

from langchain_core.messages import ToolMessage

from app.engine.runtime import utterances as u
from app.engine.runtime.spec import AgentSpec
from app.engine.runtime.tools import SideEffect, ToolContext, ToolRegistry, tool
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import collect, make_runtime, text_turn, tool_turn

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


def _spec(registry: ToolRegistry | None = None) -> AgentSpec:
    return AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=(registry or build_registry()).specs(),
    )


async def _session(sessions, sid: str = "s-1") -> str:
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    return sid


async def test_tool_round_then_completion_event_chain():
    rt, _, _, sessions = make_runtime(
        tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": 80}, "c1")),
        text_turn("已退款"),
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=_spec()
    )
    assert [e.type.value for e in got] == NINE
    call, result = got[3], got[4]
    assert call.payload == {
        "tool_name": "demo_refund_apply",
        "args": {"order_id": "1024", "amount": 80},
        "model_call_id": "c1",
    }
    # write-ahead 事件 id 即幂等键：透传进 ToolContext，被工具返回；tool_result 以它闭合
    assert result.payload["tool_call_id"] == call.id
    assert result.payload["result"] == {"refunded": 80, "idempotency_key": call.id}
    assert got[2].payload["tool_calls"] == [
        {
            "id": "c1",
            "name": "demo_refund_apply",
            "args": {"order_id": "1024", "amount": 80},
        }
    ]
    assert got[-1].payload["iteration"] == 2
    snap = await rt.build_agent("t-a", _spec()).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    tool_msgs = [m for m in snap.values["messages"] if isinstance(m, ToolMessage)]
    assert (
        len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "c1"
    )  # 模型侧 id 进对话配对
    assert "idempotency_key" in tool_msgs[0].content


async def test_multiple_calls_in_one_turn_run_serially_in_order():
    """框架默认并行；每 run 一把锁串行（ADR-012 决策 7）：事件序确定、执行不重叠。"""
    running: list[str] = []
    overlaps: list[bool] = []

    @tool(side_effect=SideEffect.READ)
    async def slow(ctx: ToolContext, tag: str) -> str:
        """慢工具。"""
        overlaps.append(bool(running))
        running.append(tag)
        await asyncio.sleep(0.01)
        running.remove(tag)
        return tag

    registry = ToolRegistry([slow])
    rt, _, _, sessions = make_runtime(
        tool_turn(("slow", {"tag": "a"}, "c1"), ("slow", {"tag": "b"}, "c2")),
        text_turn("完成"),
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="x", spec=_spec(registry)
    )
    tool_events = [e for e in got if e.type.value in ("tool_call", "tool_result")]
    assert [e.type.value for e in tool_events] == ["tool_call", "tool_result"] * 2
    assert [e.payload["args"]["tag"] for e in tool_events[::2]] == ["a", "b"]
    assert overlaps == [False, False]


async def test_tool_exception_becomes_tool_error_and_loop_continues():
    @tool(side_effect=SideEffect.READ)
    async def boom(ctx: ToolContext, order_id: str) -> str:
        """会炸的工具。"""
        raise RuntimeError("下游炸了 sk-abcdefghijklmnop")

    rt, _, _, sessions = make_runtime(
        tool_turn(("boom", {"order_id": "1"}, "c1")), text_turn("抱歉")
    )
    sid = await _session(sessions)
    got = await collect(
        rt,
        tenant_id="t-a",
        session_id=sid,
        user_input="x",
        spec=_spec(ToolRegistry([boom])),
    )
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_error",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    err = got[4].payload
    assert err["tool_call_id"] == got[3].id and err["retry_count"] == 0
    assert "sk-***" in err["error"] and "abcdefghijklmnop" not in err["error"]  # 消毒
    snap = await rt.build_agent("t-a", _spec(ToolRegistry([boom]))).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    tool_msg = next(m for m in snap.values["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert tool_msg.content.startswith(u.TOOL_FAILED.format(detail="")[:6])
    assert "sk-***" in tool_msg.content
    assert got[-1].payload["reason"] == "completed"


async def test_invalid_args_do_not_reach_write_ahead():
    """坏参数在 ① 就被拒：没有副作用要保护，所以没有 tool_call 事件；中文回填而不是框架英文。"""
    rt, _, _, sessions = make_runtime(
        tool_turn(("demo_refund_apply", {"order_id": "1024", "amount": "很多"}, "c1")),
        text_turn("请提供金额"),
    )
    sid = await _session(sessions)
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="退款", spec=_spec()
    )
    assert [e.type.value for e in got] == [
        "user_message",
        "llm_call",
        "llm_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    snap = await rt.build_agent("t-a", _spec()).aget_state(
        {"configurable": {"thread_id": sid}}
    )
    tool_msg = next(m for m in snap.values["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status == "error" and tool_msg.content.startswith("参数校验失败：")


async def test_tool_context_carries_identity_not_model_input():
    seen: list[dict[str, Any]] = []

    @tool(side_effect=SideEffect.READ)
    async def whoami(ctx: ToolContext) -> str:
        """回显身份。"""
        seen.append(
            {
                "tenant_id": ctx.tenant_id,
                "user_id": ctx.user_id,
                "session_id": ctx.session_id,
                "run_id": ctx.run_id,
                "tool_call_id": ctx.tool_call_id,
            }
        )
        return "ok"

    rt, _, _, sessions = make_runtime(tool_turn(("whoami", {}, "c1")), text_turn("好"))
    sid = await _session(sessions)
    got = await collect(
        rt,
        tenant_id="t-a",
        session_id=sid,
        user_input="x",
        spec=_spec(ToolRegistry([whoami])),
    )
    assert seen == [
        {
            "tenant_id": "t-a",
            "user_id": "u-1",
            "session_id": sid,
            "run_id": got[0].run_id,
            "tool_call_id": got[3].id,
        }
    ]
