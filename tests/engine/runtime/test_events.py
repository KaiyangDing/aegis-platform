"""事件类型快照、AgentEvent 防呆、event_id 派生（确定性 / 逐段敏感 / 空段拒绝 / 命名空间与一枚输出钉死）。"""

import uuid
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

pytest.importorskip(
    "app.engine.runtime.events", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.runtime.events import (
    EVENT_ID_NAMESPACE,
    SCHEMA_VERSION,
    AgentEvent,
    EventType,
    event_id,
)


def test_event_type_values_are_stable():
    """17 类值快照。再加成员先让这里红、过口径再改。"""
    assert {e.value for e in EventType} == {
        "user_message",
        "assistant_message",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "tool_error",
        "approval_requested",
        "approval_decided",
        "approval_cancelled",
        "approval_expired",
        "loop_terminated",
        "handoff",
        "summary_updated",
        "guardrail_triggered",
        "recovery_abandoned",
        "precheck_vetoed",
    }


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "e",
        "tenant_id": "t",
        "session_id": "s",
        "run_id": "r",
        "seq": 1,
        "type": EventType.HANDOFF,
        "payload": {},
    }
    return {**base, **overrides}


def test_agent_event_fields_and_frozen():
    e = AgentEvent(
        id="e-1",
        tenant_id="t-a",
        session_id="s-1",
        run_id="r-1",
        seq=1,
        type=EventType.USER_MESSAGE,
        payload={"content": "你好"},
    )
    assert e.schema_version == SCHEMA_VERSION
    assert e.type is EventType.USER_MESSAGE
    assert e.task_id is None and e.checkpoint_id is None
    with pytest.raises(FrozenInstanceError):
        e.seq = 2  # type: ignore[misc]


def test_agent_event_carries_framework_coordinates():
    e = AgentEvent(**_kwargs(task_id="task-1", checkpoint_id="ck-1"))
    assert (e.task_id, e.checkpoint_id) == ("task-1", "ck-1")


@pytest.mark.parametrize("blank", ["id", "tenant_id", "session_id", "run_id"])
def test_agent_event_rejects_blank_ids(blank: str):
    with pytest.raises(ValueError, match=blank):
        AgentEvent(**_kwargs(**{blank: ""}))


@pytest.mark.parametrize("bad_seq", [0, -1])
def test_agent_event_rejects_bad_seq(bad_seq: int):
    with pytest.raises(ValueError, match="seq"):
        AgentEvent(**_kwargs(seq=bad_seq))


def test_agent_event_rejects_bad_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        AgentEvent(**_kwargs(schema_version=0))


# ---------------------------------------------------------------- event_id


def test_event_id_is_uuid5_and_deterministic():
    a = event_id("s-1", "task-1", "tool_call", "call_1")
    assert a == event_id("s-1", "task-1", "tool_call", "call_1")
    assert uuid.UUID(a).version == 5


def test_event_id_namespace_and_one_output_pinned():
    """命名空间与一枚派生值钉死：改任何一处 = 历史事件 id 全部换算不出来、去重失效。"""
    assert str(EVENT_ID_NAMESPACE) == "9aa11ec8-940c-5bf7-b377-cbf8faafe45d"
    assert (
        event_id("s-1", "task-1", "tool_call", "call_1")
        == "b9a47431-15fd-591a-b911-1c3e5062f7ea"
    )


def test_event_id_changes_with_every_part():
    base = event_id("s-1", "task-1", "tool_call", "call_1")
    variants = {
        event_id("s-2", "task-1", "tool_call", "call_1"),
        event_id("s-1", "task-2", "tool_call", "call_1"),
        event_id("s-1", "task-1", "tool_result", "call_1"),
        event_id("s-1", "task-1", "tool_call", "call_2"),
    }
    assert base not in variants and len(variants) == 4


def test_event_id_ordinal_int_and_str_equivalent():
    assert event_id("s", "t", "llm_call", 1) == event_id("s", "t", "llm_call", "1")


def test_event_id_parts_cannot_bleed_into_each_other():
    """段之间不是字符串拼接：段内出现分隔符也不会撞键。"""
    assert event_id("a", "b:c", "h", "1") != event_id("a:b", "c", "h", "1")
    assert event_id("ab", "c", "h", "1") != event_id("a", "bc", "h", "1")


@pytest.mark.parametrize("blank", ["thread_id", "task_id", "hook", "ordinal"])
def test_event_id_rejects_blank_parts(blank: str):
    parts = {"thread_id": "s", "task_id": "t", "hook": "h", "ordinal": "1"}
    parts[blank] = ""
    with pytest.raises(ValueError, match=blank):
        event_id(**parts)
