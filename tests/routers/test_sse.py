"""SSE 编码器：帧形态（id = seq、单行 JSON、中文不转义、Decimal 走 default=str）、AgentEvent 与行字典归一视图一致、
角色可见性矩阵、合成帧无 id。"""

from decimal import Decimal

from app.core.auth import Role
from app.engine.runtime.events import AgentEvent, EventType
from app.routers.sse import (
    INTERNAL_EVENT_TYPES,
    as_row,
    done_frame,
    encode,
    error_frame,
    event_frame,
    visible,
)

EVENT = AgentEvent(
    id="e1",
    tenant_id="t",
    session_id="s",
    run_id="r",
    seq=3,
    type=EventType.USER_MESSAGE,
    payload={"content": "hi"},
)
ROW = {
    "id": "e1",
    "tenant_id": "t",
    "session_id": "s",
    "run_id": "r",
    "seq": 3,
    "type": "user_message",
    "payload": {"content": "hi"},
    "schema_version": 1,
    "task_id": None,
    "checkpoint_id": None,
}


def test_encode_with_and_without_seq():
    frame = encode("x", {"a": "中", "d": Decimal("1.5")}, seq=7).decode()
    assert frame == 'id: 7\nevent: x\ndata: {"a":"中","d":"1.5"}\n\n'
    assert (
        encode("done", {"reason": "completed"}).decode()
        == 'event: done\ndata: {"reason":"completed"}\n\n'
    )


def test_as_row_from_event_and_dict_agree_and_drop_tenant():
    expected = {
        "id": "e1",
        "seq": 3,
        "run_id": "r",
        "type": "user_message",
        "payload": {"content": "hi"},
    }
    assert as_row(EVENT) == as_row(ROW) == expected
    assert "tenant_id" not in as_row(EVENT)


def test_event_frame_uses_seq_as_id():
    assert (
        event_frame(as_row(EVENT)).decode().startswith("id: 3\nevent: user_message\n")
    )


def test_visibility_matrix():
    for kind in INTERNAL_EVENT_TYPES:
        assert not visible(kind, Role.USER)
        assert visible(kind, Role.OPERATOR) and visible(kind, Role.ADMIN)
    for kind in (
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
        "tool_error",
        "approval_requested",
        "approval_decided",
        "loop_terminated",
    ):
        assert visible(kind, Role.USER)


def test_synthetic_frames_have_no_id():
    assert not done_frame(reason="x").startswith(b"id:")
    err = error_frame("d", "E")
    assert err.startswith(b"event: error\n") and b'"error":"E"' in err
