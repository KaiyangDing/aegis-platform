"""真 PG 的 HTTP 端到端（M3.3）：三路由 + 真 EventStore / SessionStateStore / ApprovalStore + AsyncPostgresSaver + 剧本网关（零真实调用）。
审批闭环 approve / reject 两路经 HTTP 落真库；越权在 HTTP 帧里是同一话术；挂起中第二条消息 409 带真库单号；
GET 回放 after_seq 接续、帧 id = seq 连续、事件 id 全不同。需 docker compose up -d postgres（checkpoint 表不在回滚夹具内，会话 id 逐测唯一）。"""

import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.business import utterances as bu
from app.business.backend import demo_backend, set_backend
from app.business.spec import preheat_specs
from app.core.auth import Role
from app.core.config import Settings
from app.domain.approvals import ApprovalStore
from app.domain.events import EventStore
from app.domain.sessions import SessionStateStore
from app.routers import approvals, chat, sessions
from tests.engine.runtime.doubles import make_runtime, text_turn, tool_turn
from tests.routers.sse_client import SECRET, auth, parse_sse

REFUND = tool_turn(("refund_apply", {"order_id": "AZ-1001", "amount": 300}, "c1"))
QUERY = tool_turn(("order_query", {"order_id": "AZ-1001"}, "c1"))
STAFF = auth(Role.OPERATOR, uid="op-a1")


@pytest.fixture(autouse=True)
def _backend():
    backend = demo_backend()
    set_backend(backend)
    try:
        yield backend
    finally:
        set_backend(None)


def _real_app(acts, db_session_factory, pg_checkpointer):
    events = EventStore(db_session_factory)
    sessions_store = SessionStateStore(db_session_factory)
    approvals_store = ApprovalStore(db_session_factory)
    runtime, candidate, _, _ = make_runtime(
        *acts,
        events=events,  # type: ignore[arg-type]
        sessions=sessions_store,  # type: ignore[arg-type]
        checkpointer=pg_checkpointer,
        approvals=approvals_store,
        tier="standard",
    )
    app = FastAPI()
    for router in (chat.router, approvals.router, sessions.router):
        app.include_router(router)
    app.state.settings = Settings(
        _env_file=None, jwt_secret=SecretStr(SECRET), redis_url=""
    )
    app.state.runtime = runtime
    app.state.runtime_parts = SimpleNamespace(
        events=events, sessions=sessions_store, approvals=approvals_store
    )
    app.state.specs = preheat_specs()
    return app, candidate, approvals_store, sessions_store


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _sid() -> str:
    return f"e2e-{uuid.uuid4().hex[:10]}"


async def test_approve_loop_over_http_on_real_pg(
    db_session_factory, pg_checkpointer, _backend
):
    app, _, approvals_store, sessions_store = _real_app(
        [REFUND, text_turn("已为您退款 300 元。")], db_session_factory, pg_checkpointer
    )
    sid = _sid()
    async with _client(app) as c:
        first = await c.post(
            "/v1/chat", json={"session_id": sid, "message": "退 300"}, headers=auth()
        )
        assert first.status_code == 200
        done = parse_sse(first.text)[-1]["data"]
        assert done["reason"] == "awaiting_approval"
        approval_id = done["approvals"][0]["approval_id"]
        (ticket,) = await approvals_store.list_for_session("tenant-a", sid)
        assert ticket["id"] == approval_id and ticket["status"] == "pending"
        assert (await sessions_store.get(sid))["run_state"] == "awaiting_approval"

        blocked = await c.post(
            "/v1/chat", json={"session_id": sid, "message": "在吗"}, headers=auth()
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["pending_approvals"] == [approval_id]

        decided = await c.post(
            f"/v1/approvals/{approval_id}", json={"decision": "approve"}, headers=STAFF
        )
        assert decided.status_code == 200, decided.text
        body = decided.json()
        assert body["status"] == "done" and body["reason"] == "completed"
        assert body["reply"] == "已为您退款 300 元。"

        replay = await c.get(f"/v1/sessions/{sid}/events", headers=STAFF)
        tail = await c.get(f"/v1/sessions/{sid}/events?after_seq=4", headers=STAFF)
    frames = parse_sse(replay.text)
    kinds = [f["event"] for f in frames]
    assert kinds == [
        "user_message",
        "llm_call",
        "llm_result",
        "approval_requested",
        "approval_decided",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
        "done",
    ]
    assert [f["id"] for f in frames[:-1]] == list(range(1, 12))  # seq 连续
    assert len({f["data"]["id"] for f in frames[:-1]}) == 11  # 事件 id 全不同
    assert frames[-1]["data"]["run_state"] == "idle"
    assert [f["id"] for f in parse_sse(tail.text)[:-1]] == list(
        range(5, 12)
    )  # after_seq 接续
    assert _backend.write_calls == 1
    assert (await approvals_store.get(approval_id))["operator_id"] == "op-a1"


async def test_reject_loop_over_http_on_real_pg(
    db_session_factory, pg_checkpointer, _backend
):
    app, candidate, _, sessions_store = _real_app(
        [REFUND, text_turn("不该被调用")], db_session_factory, pg_checkpointer
    )
    sid = _sid()
    async with _client(app) as c:
        first = await c.post(
            "/v1/chat", json={"session_id": sid, "message": "退 300"}, headers=auth()
        )
        approval_id = parse_sse(first.text)[-1]["data"]["approvals"][0]["approval_id"]
        calls_before = candidate.calls
        decided = await c.post(
            f"/v1/approvals/{approval_id}", json={"decision": "reject"}, headers=STAFF
        )
        assert decided.status_code == 200, decided.text
        body = decided.json()
        assert body["status"] == "done" and body["reason"] == "cancelled"
        assert "tool_call" not in body["events"]
        again = await c.post(
            f"/v1/approvals/{approval_id}", json={"decision": "approve"}, headers=STAFF
        )
    assert again.status_code == 409 and again.json()["detail"]["status"] == "rejected"
    assert candidate.calls == calls_before and _backend.write_calls == 0
    assert (await sessions_store.get(sid))["run_state"] == "idle"


async def test_cross_user_denial_is_same_utterance_over_http(
    db_session_factory, pg_checkpointer
):
    app, _, _, _ = _real_app(
        [QUERY, text_turn("没有找到该订单。")], db_session_factory, pg_checkpointer
    )
    sid = _sid()
    async with _client(app) as c:
        r = await c.post(
            "/v1/chat",
            json={"session_id": sid, "message": "查 AZ-1001"},
            headers=auth(uid="u-a2"),
        )
    frames = parse_sse(r.text)
    tool_result = next(f for f in frames if f["event"] == "tool_result")
    assert bu.DENIED_TEXT in str(tool_result["data"]["payload"])
    assert "shipped" not in str(tool_result["data"]["payload"])
