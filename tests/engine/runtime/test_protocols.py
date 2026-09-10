"""跨包协议（M2.1 / M2.2 / M2.7）：domain 的存取件靠结构匹配实现 engine 协议；两层的状态值 / 审批五态快照互钉（不 import 对方）。"""

import pytest

pytest.importorskip(
    "app.engine.runtime.protocols", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.runtime import protocols as protocols_mod

if not hasattr(protocols_mod, "EventSource"):
    pytest.skip(
        "M2.2 未敲：protocols.py 尚无 EventSource / SessionRunState / SessionStateLike",
        allow_module_level=True,
    )

from app.engine.runtime.protocols import (
    EventSink,
    EventSource,
    SessionRunState,
    SessionStateLike,
)


def test_session_run_state_values_are_stable():
    assert [s.value for s in SessionRunState] == [
        "idle",
        "running",
        "awaiting_approval",
        "failed",
    ]


def test_event_store_matches_sink_and_source_protocols():
    events = pytest.importorskip(
        "app.domain.events", reason="M2.2 未敲：app/domain/events.py 不存在"
    )
    store = events.EventStore(None)  # type: ignore[arg-type]
    assert isinstance(store, EventSink) and isinstance(store, EventSource)


def test_session_store_matches_protocol_and_shares_state_values():
    sessions = pytest.importorskip(
        "app.domain.sessions", reason="M2.2 未敲：app/domain/sessions.py 不存在"
    )
    assert isinstance(sessions.SessionStateStore(None), SessionStateLike)  # type: ignore[arg-type]
    assert tuple(s.value for s in SessionRunState) == sessions.RUN_STATES


def test_approval_store_matches_protocol_and_shares_status_values():
    approvals = pytest.importorskip(
        "app.domain.approvals", reason="M2.2 未敲：app/domain/approvals.py 不存在"
    )
    if not hasattr(protocols_mod, "ApprovalStoreLike") or not hasattr(
        approvals, "ApprovalStore"
    ):
        pytest.skip("M2.7 未敲：ApprovalStoreLike / ApprovalStore 不存在")
    from app.engine.runtime.protocols import ApprovalStatus, ApprovalStoreLike

    assert isinstance(approvals.ApprovalStore(None), ApprovalStoreLike)  # type: ignore[arg-type]
    assert tuple(s.value for s in ApprovalStatus) == approvals.APPROVAL_STATUSES
