"""POST /v1/approvals/{id}：401 / USER 403 / 404 / 他租坐席 403 不触碰单 / 批准续跑摘要与副作用恰一次 / 拒绝零 LLM 且 cancelled /
二次决定 409 / 过期单惰性翻 expired 后 409 / 决定形态 422。"""

from datetime import UTC, datetime, timedelta

from app.core.auth import Role
from tests.engine.runtime.doubles import text_turn, tool_turn
from tests.routers.conftest import Harness, build_harness
from tests.routers.sse_client import auth, parse_sse

REFUND = tool_turn(("refund_apply", {"order_id": "AZ-1001", "amount": 300}, "c1"))
STAFF = auth(Role.OPERATOR, uid="op-a1")
APPROVE = {"decision": "approve"}


async def _suspend(h: Harness, client) -> str:
    r = await client.post(
        "/v1/chat", json={"session_id": "s-1", "message": "退 300"}, headers=auth()
    )
    assert r.status_code == 200
    done = parse_sse(r.text)[-1]["data"]
    assert done["reason"] == "awaiting_approval"
    return done["approvals"][0]["approval_id"]


async def test_401_and_user_role_403():
    h = build_harness(REFUND, text_turn("已退款"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        anonymous = await c.post(f"/v1/approvals/{aid}", json=APPROVE)
        user = await c.post(f"/v1/approvals/{aid}", json=APPROVE, headers=auth())
    assert anonymous.status_code == 401 and user.status_code == 403


async def test_unknown_ticket_404_and_foreign_tenant_staff_403():
    h = build_harness(REFUND, text_turn("已退款"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        missing = await c.post("/v1/approvals/nope", json=APPROVE, headers=STAFF)
        foreign = await c.post(
            f"/v1/approvals/{aid}",
            json=APPROVE,
            headers=auth(Role.ADMIN, uid="admin-b", tid="tenant-b"),
        )
    assert missing.status_code == 404 and foreign.status_code == 403
    assert (await h.approvals.get(aid))["status"] == "pending"  # 越权尝试不触碰单


async def test_approve_resumes_and_summarizes():
    h = build_harness(REFUND, text_turn("已为您退款 300 元。"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        r = await c.post(f"/v1/approvals/{aid}", json=APPROVE, headers=STAFF)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done" and body["reason"] == "completed"
    assert body["reply"] == "已为您退款 300 元。" and body["next_approval_id"] is None
    assert body["events"] == [
        "approval_decided",
        "tool_call",
        "tool_result",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
    ]
    ticket = await h.approvals.get(aid)
    assert ticket["status"] == "approved" and ticket["operator_id"] == "op-a1"
    assert h.backend.write_calls == 1
    assert (await h.sessions.get("s-1"))["run_state"] == "idle"


async def test_reject_terminates_cancelled_with_zero_llm_calls():
    h = build_harness(REFUND, text_turn("不该被调用"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        calls_before = h.candidate.calls
        r = await c.post(
            f"/v1/approvals/{aid}", json={"decision": "reject"}, headers=STAFF
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done" and body["reason"] == "cancelled"
    assert (
        body["events"][0] == "approval_decided"
        and body["events"][-1] == "loop_terminated"
    )
    assert "tool_call" not in body["events"]
    assert h.candidate.calls == calls_before and h.backend.write_calls == 0
    assert (await h.approvals.get(aid))["status"] == "rejected"


async def test_second_decision_is_409():
    h = build_harness(REFUND, text_turn("已退款"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        assert (
            await c.post(f"/v1/approvals/{aid}", json=APPROVE, headers=STAFF)
        ).status_code == 200
        again = await c.post(
            f"/v1/approvals/{aid}", json={"decision": "reject"}, headers=STAFF
        )
    assert again.status_code == 409
    assert (
        again.json()["detail"]["status"] == "approved"
    )  # 输家看到赢家的终态，绝不覆盖


async def test_expired_ticket_is_409_after_lazy_expire():
    h = build_harness(REFUND, text_turn("已退款"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        h.approvals.now = lambda: (
            datetime.now(UTC) + timedelta(days=1)
        )  # 拨钟到期：端点入口惰性 expire_due
        r = await c.post(f"/v1/approvals/{aid}", json=APPROVE, headers=STAFF)
    assert r.status_code == 409
    assert r.json()["detail"]["status"] == "expired"
    assert h.backend.write_calls == 0


async def test_invalid_decision_is_422():
    h = build_harness(REFUND, text_turn("已退款"))
    async with h.client() as c:
        aid = await _suspend(h, c)
        r = await c.post(
            f"/v1/approvals/{aid}", json={"decision": "maybe"}, headers=STAFF
        )
    assert r.status_code == 422
