"""GET /v1/sessions/{id}/events：401 / 404（不存在、他租、他人）/ 坐席看本租户任一会话 / 帧 id = seq 与 done 快照 /
游标 = max(after_seq, Last-Event-ID) 且坏头按 0 / 终端用户过滤内部事件而坐席全量 / limit 与 next_seq。"""

from app.core.auth import Role
from tests.engine.runtime.doubles import text_turn
from tests.routers.conftest import Harness, build_harness
from tests.routers.sse_client import auth, parse_sse

STAFF = auth(Role.OPERATOR, uid="op-a1")


async def _chat(h: Harness, client, session_id: str = "s-1") -> None:
    r = await client.post(
        "/v1/chat", json={"session_id": session_id, "message": "你好"}, headers=auth()
    )
    assert r.status_code == 200


async def test_401_and_404_cases_and_staff_visibility():
    h = build_harness(text_turn("你好！"))
    await h.sessions.create("s-b", tenant_id="tenant-b", user_id="u-b1")
    await h.sessions.create("s-2", tenant_id="tenant-a", user_id="u-a2")
    async with h.client() as c:
        await _chat(h, c)
        assert (await c.get("/v1/sessions/s-1/events")).status_code == 401
        assert (
            await c.get("/v1/sessions/nope/events", headers=auth())
        ).status_code == 404
        assert (
            await c.get("/v1/sessions/s-b/events", headers=auth())
        ).status_code == 404
        assert (
            await c.get("/v1/sessions/s-2/events", headers=auth())
        ).status_code == 404
        staff = await c.get("/v1/sessions/s-2/events", headers=STAFF)
    assert staff.status_code == 200  # 坐席看本租户任一会话（事件即审计）


async def test_snapshot_frames_and_done():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        await _chat(h, c)
        r = await c.get("/v1/sessions/s-1/events", headers=auth())
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(r.text)
    assert [f["event"] for f in frames] == [
        "user_message",
        "assistant_message",
        "loop_terminated",
        "done",
    ]
    assert [f["id"] for f in frames[:-1]] == [1, 4, 5]
    assert frames[-1]["data"] == {
        "reason": "snapshot",
        "run_state": "idle",
        "next_seq": 5,
        "count": 5,
    }


async def test_cursor_is_max_of_after_seq_and_last_event_id():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        await _chat(h, c)
        by_header = await c.get(
            "/v1/sessions/s-1/events?after_seq=1",
            headers={**STAFF, "Last-Event-ID": "3"},
        )
        by_query = await c.get(
            "/v1/sessions/s-1/events?after_seq=4",
            headers={**STAFF, "Last-Event-ID": "junk"},
        )
    assert [f["id"] for f in parse_sse(by_header.text)[:-1]] == [4, 5]
    assert [f["id"] for f in parse_sse(by_query.text)[:-1]] == [5]


async def test_staff_sees_internal_events_and_limit_applies():
    h = build_harness(text_turn("你好！"))
    async with h.client() as c:
        await _chat(h, c)
        r = await c.get("/v1/sessions/s-1/events?limit=2", headers=STAFF)
    frames = parse_sse(r.text)
    assert [f["event"] for f in frames] == ["user_message", "llm_call", "done"]
    assert frames[-1]["data"]["next_seq"] == 2 and frames[-1]["data"]["count"] == 2
