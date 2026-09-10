"""业务片接真运行时（S2）：build_spec 的 spec + 真三工具 + 内存后端 + 剧本网关，零真实调用。
读工具端到端事件序列；越权在事件流里是同一话术；退款超阈值挂起等审批（零副作用）；退款在阈值内直接执行且后端去重键
== write-ahead 的 tool_call 事件 id；工单豁免直通不开单。"""

import json

import pytest

from app.business import utterances as bu
from app.business.backend import demo_backend, set_backend
from app.business.spec import build_spec
from tests.engine.runtime.doubles import (
    MemoryApprovalStore,
    collect,
    make_runtime,
    text_turn,
    tool_turn,
)

FULL_TOOL_ROUND = [
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


@pytest.fixture
def backend():
    instance = demo_backend()
    set_backend(instance)
    try:
        yield instance
    finally:
        set_backend(None)


def _types(events) -> list[str]:
    return [e.type.value for e in events]


def _payload_text(events, event_type: str) -> str:
    return json.dumps(
        [e.payload for e in events if e.type.value == event_type],
        ensure_ascii=False,
        default=str,
    )


async def _run(acts, *, user_id: str = "u-a1", text: str = "你好"):
    runtime, _candidate, _events, sessions = make_runtime(
        *acts, approvals=MemoryApprovalStore(), tier="standard"
    )
    await sessions.create("s1", tenant_id="tenant-a", user_id=user_id)
    out = await collect(
        runtime,
        tenant_id="tenant-a",
        session_id="s1",
        user_input=text,
        spec=build_spec("tenant-a"),
    )
    return out, sessions


async def test_order_query_end_to_end(backend):
    events, _ = await _run(
        [
            tool_turn(("order_query", {"order_id": "AZ-1001"}, "c1")),
            text_turn("您的订单 AZ-1001 已发货。"),
        ],
        text="查一下 AZ-1001",
    )
    assert _types(events) == FULL_TOOL_ROUND
    assert "shipped" in _payload_text(events, "tool_result")
    assert events[-1].payload["reason"] == "completed"


async def test_other_users_order_yields_denied_text_in_event_stream(backend):
    events, _ = await _run(
        [
            tool_turn(("order_query", {"order_id": "AZ-1001"}, "c1")),
            text_turn("抱歉，没有找到该订单。"),
        ],
        user_id="u-a2",
    )
    text = _payload_text(events, "tool_result")
    assert bu.DENIED_TEXT in text and "shipped" not in text


async def test_refund_over_threshold_suspends_for_approval(backend):
    events, sessions = await _run(
        [tool_turn(("refund_apply", {"order_id": "AZ-1001", "amount": 300}, "c1"))],
        text="退 300",
    )
    assert _types(events) == [
        "user_message",
        "llm_call",
        "llm_result",
        "approval_requested",
    ]
    assert events[-1].payload["tool_name"] == "refund_apply"
    assert (await sessions.get("s1"))["run_state"] == "awaiting_approval"
    assert backend.write_calls == 0


async def test_refund_within_threshold_uses_write_ahead_id_as_downstream_key(backend):
    events, _ = await _run(
        [
            tool_turn(("refund_apply", {"order_id": "AZ-1001", "amount": 100}, "c1")),
            text_turn("已为您退款 100 元。"),
        ],
        text="退 100",
    )
    assert _types(events) == FULL_TOOL_ROUND
    tool_call = next(e for e in events if e.type.value == "tool_call")
    assert backend.write_calls == 1
    assert backend.replay(tool_call.id) is not None  # 下游去重键 == write-ahead 事件 id
    assert "refunded" in _payload_text(events, "tool_result")


async def test_ticket_create_is_exempt_from_approval(backend):
    events, _ = await _run(
        [
            tool_turn(("ticket_create", {"title": "投诉"}, "c1")),
            text_turn("已建工单。"),
        ],
        text="投诉",
    )
    assert _types(events) == FULL_TOOL_ROUND
    assert "T-0001" in _payload_text(events, "tool_result")
