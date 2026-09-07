"""checkpointer 工厂（M2.2；真 PG，Selector 循环）：开池 + 框架迁移幂等、两 thread 各自 checkpoint、每节点一次 checkpoint、
同 thread 续跑、关池后报 PoolClosed、非池形态拒绝、Proactor 下启动期即报明白话。"""

import asyncio
import sys
import uuid

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from psycopg_pool import PoolClosed

pytest.importorskip(
    "app.core.checkpoint", reason="M2.2 未敲：app/core/checkpoint.py 不存在"
)

from app.core.checkpoint import (
    close_checkpointer,
    make_checkpointer,
    open_checkpointer,
)
from app.engine.fakes import scripted_model
from tests.conftest import FRAMEWORK_TABLES, PG_TEST_DSN


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"t-{uuid.uuid4().hex}"}}


async def test_open_is_idempotent_and_creates_framework_tables(pg_test_db):
    saver = make_checkpointer(PG_TEST_DSN, pool_size=2)
    await open_checkpointer(saver)
    try:
        await saver.setup()  # 再跑一次：幂等（探针⑷）
        async with saver.conn.connection() as conn:
            cur = await conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            names = {row["table_name"] for row in await cur.fetchall()}
    finally:
        await close_checkpointer(saver)
    assert set(FRAMEWORK_TABLES) <= names


async def test_two_threads_are_isolated_and_each_node_checkpoints(pg_checkpointer):
    agent = create_agent(
        scripted_model(AIMessage("甲"), AIMessage("乙")), checkpointer=pg_checkpointer
    )
    c1, c2 = _cfg(), _cfg()
    await agent.ainvoke({"messages": [HumanMessage("一")]}, c1, durability="sync")
    await agent.ainvoke({"messages": [HumanMessage("二")]}, c2, durability="sync")
    s1 = await agent.aget_state(c1)
    s2 = await agent.aget_state(c2)
    assert [m.content for m in s1.values["messages"]] == ["一", "甲"]
    assert [m.content for m in s2.values["messages"]] == ["二", "乙"]
    history = [s async for s in agent.aget_state_history(c1)]
    # 输入 + 起点 + model 节点 = 3：每个节点一次 checkpoint（探针⑷；框架升级后此数变化即需重跑探针）
    assert len(history) == 3
    assert s1.config["configurable"]["checkpoint_id"]


async def test_same_thread_continues_from_checkpoint(pg_checkpointer):
    agent = create_agent(
        scripted_model(AIMessage("甲"), AIMessage("乙")), checkpointer=pg_checkpointer
    )
    cfg = _cfg()
    await agent.ainvoke({"messages": [HumanMessage("一")]}, cfg, durability="sync")
    out = await agent.ainvoke(
        {"messages": [HumanMessage("二")]}, cfg, durability="sync"
    )
    assert [m.content for m in out["messages"]] == ["一", "甲", "二", "乙"]


async def test_closed_pool_fails_loud(pg_test_db):
    saver = make_checkpointer(PG_TEST_DSN, pool_size=2)
    await open_checkpointer(saver)
    await close_checkpointer(saver)
    with pytest.raises(PoolClosed):
        await saver.aget_tuple(_cfg())


def test_non_pool_saver_rejected():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    class NotAPool: ...

    async def probe() -> None:
        await open_checkpointer(AsyncPostgresSaver(NotAPool()))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="连接池"):
        asyncio.run(probe(), loop_factory=asyncio.SelectorEventLoop)


@pytest.mark.skipif(sys.platform != "win32", reason="Proactor 循环只有 Windows 有")
def test_proactor_loop_rejected_before_touching_network():
    """探针⑷：Proactor 下池会沉默重连到 30s 超时；工厂在开池前就把它变成一句明白话。"""

    async def probe() -> None:
        saver = make_checkpointer(PG_TEST_DSN, pool_size=1)
        await open_checkpointer(saver)

    with pytest.raises(RuntimeError, match="selector_loop_factory"):
        asyncio.run(probe(), loop_factory=asyncio.ProactorEventLoop)
