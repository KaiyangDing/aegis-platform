"""薄 HTTP 面接真 lifespan（M3.2）：specs 启动期预热、三路由挂载、fake 模式下 POST /v1/chat → SSE 往返落真 PG + Redis 限流器在场、
GET 回放快照 run_state 回 idle；零真实 LLM（AEGIS_FAKE_LLM=1）。"""

import uuid

import httpx
import structlog

from app.core.auth import Role, issue_token
from app.core.config import get_settings
from app.engine.fakes import FAKE_REPLY
from app.main import app
from tests.conftest import PG_TEST_DSN
from tests.routers.sse_client import parse_sse

SECRET = "main-api-test-secret-0123456789abcdef"


async def test_chat_round_trip_through_real_lifespan(
    monkeypatch, redis_db1, pg_test_db
):
    monkeypatch.setenv("REDIS_URL", redis_db1)
    monkeypatch.setenv("DATABASE_URL", pg_test_db)
    monkeypatch.setenv("CHECKPOINT_DATABASE_URL", PG_TEST_DSN)
    monkeypatch.setenv("AEGIS_FAKE_LLM", "1")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    get_settings.cache_clear()
    session_id = f"s-{uuid.uuid4().hex[:12]}"
    token = issue_token(
        user_id="u-a1", tenant_id="tenant-a", role=Role.USER, ttl_s=60, secret=SECRET
    )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with app.router.lifespan_context(app):
            assert set(app.state.specs) == {"tenant-a", "tenant-b"}
            # FastAPI 0.141 把 include_router 的结果包成 _IncludedRouter 放进 app.routes（无 path 属性）：看 OpenAPI 这层公开契约
            paths = set(app.openapi()["paths"])
            assert {
                "/v1/chat",
                "/v1/approvals/{approval_id}",
                "/v1/sessions/{session_id}/events",
            } <= paths
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat",
                    json={"session_id": session_id, "message": "你好"},
                    headers=headers,
                )
                assert resp.status_code == 200, resp.text
                frames = parse_sse(resp.text)
                assert [f["event"] for f in frames] == [
                    "user_message",
                    "assistant_message",
                    "loop_terminated",
                    "done",
                ]
                assert frames[1]["data"]["payload"]["content"] == FAKE_REPLY
                snap = await client.get(
                    f"/v1/sessions/{session_id}/events", headers=headers
                )
                assert snap.status_code == 200
                assert parse_sse(snap.text)[-1]["data"]["run_state"] == "idle"
        assert app.state.specs is None  # 关停摘下
    finally:
        get_settings.cache_clear()
        structlog.reset_defaults()
