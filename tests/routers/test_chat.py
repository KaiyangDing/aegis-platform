"""POST /v1/chat：401 / 422 / 403 租户未开通 / 首见即建与用户可见帧（id = seq、done 无 id）/ 坐席见内部事件 / 他租与他人会话 404 /
超阈值挂起的 done 帧与等审批 409（带单号）/ 运行中 409 / 首帧后失败只能 error 帧 / 依赖序（认证先于限流）/ 真 Redis 下租户级 429。"""

import httpx
from fastapi import Depends, FastAPI
from pydantic import SecretStr

from app.core.auth import Role, current_principal, tenant_identity
from app.core.config import Settings
from app.core.limits import InboundLimiter, rate_limit
from app.routers.chat import router as chat_router
from tests.engine.runtime.doubles import RaisingModel, text_turn, tool_turn
from tests.routers.conftest import build_harness
from tests.routers.sse_client import SECRET, auth, parse_sse

BODY = {"session_id": "s-1", "message": "你好"}
REFUND = tool_turn(("refund_apply", {"order_id": "AZ-1001", "amount": 300}, "c1"))


async def test_requires_token():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        assert (await c.post("/v1/chat", json=BODY)).status_code == 401


async def test_validates_body():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        empty = await c.post(
            "/v1/chat", json={"session_id": "s-1", "message": ""}, headers=auth()
        )
        bad_id = await c.post(
            "/v1/chat", json={"session_id": "a:b", "message": "x"}, headers=auth()
        )
    assert empty.status_code == 422 and bad_id.status_code == 422


async def test_unknown_tenant_is_403():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        r = await c.post("/v1/chat", json=BODY, headers=auth(tid="tenant-z"))
    assert r.status_code == 403


async def test_first_message_creates_session_and_streams_user_visible_frames():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        r = await c.post("/v1/chat", json=BODY, headers=auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    frames = parse_sse(r.text)
    assert [f["event"] for f in frames] == [
        "user_message",
        "assistant_message",
        "loop_terminated",
        "done",
    ]
    assert [f["id"] for f in frames] == [
        1,
        4,
        5,
        None,
    ]  # llm_call(2) / llm_result(3) 对终端用户不可见；done 无 id
    assert frames[1]["data"]["payload"]["content"] == "你好！"
    assert frames[-1]["data"] == {"reason": "completed"}
    row = await h.sessions.get("s-1")
    assert row["tenant_id"] == "tenant-a" and row["user_id"] == "u-a1"
    assert row["run_state"] == "idle"


async def test_staff_sees_internal_events():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        r = await c.post(
            "/v1/chat", json=BODY, headers=auth(Role.OPERATOR, uid="op-a1")
        )
    assert [f["event"] for f in parse_sse(r.text)] == [
        "user_message",
        "llm_call",
        "llm_result",
        "assistant_message",
        "loop_terminated",
        "done",
    ]


async def test_foreign_tenant_or_other_user_session_is_404():
    h = build_harness(text_turn("你好！"))
    await h.sessions.create("s-b", tenant_id="tenant-b", user_id="u-b1")
    await h.sessions.create("s-2", tenant_id="tenant-a", user_id="u-a2")
    async with h.client() as c:
        foreign = await c.post(
            "/v1/chat", json={**BODY, "session_id": "s-b"}, headers=auth()
        )
        other = await c.post(
            "/v1/chat", json={**BODY, "session_id": "s-2"}, headers=auth()
        )
    assert foreign.status_code == 404 and other.status_code == 404


async def test_refund_over_threshold_ends_with_awaiting_done_and_next_message_is_409():
    h = build_harness(REFUND)
    async with h.client() as c:
        r = await c.post("/v1/chat", json={**BODY, "message": "退 300"}, headers=auth())
        frames = parse_sse(r.text)
        assert [f["event"] for f in frames] == [
            "user_message",
            "approval_requested",
            "done",
        ]
        done = frames[-1]["data"]
        assert done["reason"] == "awaiting_approval"
        assert done["approvals"][0]["tool_name"] == "refund_apply"
        approval_id = done["approvals"][0]["approval_id"]
        assert (await h.sessions.get("s-1"))["run_state"] == "awaiting_approval"
        assert h.backend.write_calls == 0
        again = await c.post("/v1/chat", json=BODY, headers=auth())
    assert again.status_code == 409
    assert again.json()["detail"]["pending_approvals"] == [approval_id]


async def test_running_session_is_409():
    h = build_harness(text_turn("你好！"))
    await h.sessions.create("s-1", tenant_id="tenant-a", user_id="u-a1")
    assert await h.sessions.transition("s-1", expected="idle", to="running")
    async with h.client() as c:
        r = await c.post("/v1/chat", json=BODY, headers=auth())
    assert r.status_code == 409


async def test_midstream_failure_yields_error_frame_not_status_code():
    h = build_harness(gateway_for=lambda tenant_id: RaisingModel())
    async with h.client() as c:
        r = await c.post("/v1/chat", json=BODY, headers=auth())
    assert r.status_code == 200
    frames = parse_sse(r.text)
    assert [f["event"] for f in frames] == ["user_message", "error"]
    assert frames[-1]["data"]["error"] == "ProviderServerError"
    assert (await h.sessions.get("s-1"))[
        "run_state"
    ] == "running"  # crash-only：由恢复入口分诊


def test_auth_dependency_precedes_rate_limit():
    """依赖按声明序解析：认证在前，限流的身份段才读得到主体（ADR-010 决策 6）。"""
    route = next(r for r in chat_router.routes if r.path == "/v1/chat")
    names = [d.call.__qualname__ for d in route.dependant.dependencies]
    auth_index = names.index("current_principal")
    limit_index = next(i for i, n in enumerate(names) if n.startswith("rate_limit"))
    assert auth_index < limit_index


async def test_tenant_rate_limit_429_with_real_redis(redis_async):
    """同款挂载形态在真 Redis 上：租户 A 第二次 429 带 Retry-After，租户 B 不受影响（键身份段 = 租户）。"""
    app = FastAPI()
    app.state.settings = Settings(_env_file=None, jwt_secret=SecretStr(SECRET))
    limiter = InboundLimiter(redis_async)
    await limiter.start()
    app.state.inbound_limiter = limiter

    @app.get(
        "/ping",
        dependencies=[
            Depends(current_principal),
            Depends(rate_limit("ping-test", 1, 60, identify=tenant_identity)),
        ],
    )
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as c:
        first = await c.get("/ping", headers=auth())
        second = await c.get("/ping", headers=auth())
        other = await c.get("/ping", headers=auth(tid="tenant-b"))
    assert first.status_code == 200
    assert second.status_code == 429 and "Retry-After" in second.headers
    assert other.status_code == 200
