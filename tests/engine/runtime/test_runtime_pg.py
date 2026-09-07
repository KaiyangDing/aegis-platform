"""真 PG 端到端（M2.3）：EventStore + SessionStateStore + AsyncPostgresSaver 三件真件 + 剧本网关——
事件带框架坐标落库、seq 连续、会话状态机翻转、checkpoint 链完整。需 docker compose up -d postgres。"""

import uuid

import pytest

pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)
pytest.importorskip(
    "app.domain.events", reason="M2.2 未敲：app/domain/events.py 不存在"
)

from app.domain.events import EventStore
from app.domain.sessions import SessionStateStore
from app.engine.runtime.spec import AgentSpec
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import collect, make_runtime, text_turn, tool_turn


async def test_end_to_end_with_real_stores_and_checkpointer(
    db_session_factory, pg_checkpointer
):
    events = EventStore(db_session_factory)
    sessions = SessionStateStore(db_session_factory)
    spec = AgentSpec(
        system_prompt="你是演示客服。",
        model_tier="fast",
        tools=build_registry().specs(),
    )
    rt, _, _, _ = make_runtime(
        tool_turn(("demo_order_query", {"order_id": "1024"}, "c1")),
        text_turn("已发货"),
        events=events,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        checkpointer=pg_checkpointer,
    )
    sid = f"s-{uuid.uuid4().hex[:8]}"
    await sessions.create(sid, tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id=sid, user_input="查订单", spec=spec
    )
    assert [e.type.value for e in got] == [
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
    rows = await events.read("t-a", sid)
    assert [r["seq"] for r in rows] == list(range(1, 10))
    assert [r["id"] for r in rows] == [e.id for e in got]
    assert all(r["task_id"] for r in rows)
    assert all(r["tenant_id"] == "t-a" for r in rows)
    assert (await sessions.get(sid))["run_state"] == "idle"
    agent = rt.build_agent("t-a", spec)
    snap = await agent.aget_state({"configurable": {"thread_id": sid}})
    assert snap.next == () and snap.config["configurable"]["checkpoint_id"]
    history = [
        s async for s in agent.aget_state_history({"configurable": {"thread_id": sid}})
    ]
    assert (
        len(history) >= 6
    )  # 输入 + 起点 + before_agent + model + tools + model + after_agent
