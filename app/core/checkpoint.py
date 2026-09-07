"""LangGraph checkpointer 工厂：AsyncPostgresSaver over psycopg_pool.AsyncConnectionPool（ADR-011 决策 1/8）。

与 db.py 同一纪律：不在模块级建池（池绑定创建时的事件循环）；lifespan 建（make_checkpointer）→
open_checkpointer（开池 + 框架自带迁移 setup，幂等）→ 关停 close_checkpointer。
框架四表（checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations）由 setup() 管、不进 alembic；
无 tenant 列，thread_id = session_id（全局唯一）。
池参数照抄框架自己的 from_conn_string（autocommit / prepare_threshold=0 / dict_row）——saver 的 SQL 假定这三样。
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.loops import require_selector_loop

POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 5  # 每节点一次 checkpoint 写入，串行居多；并发 run 数才是上限的依据
# 取连接 / 开池的等待上限：默认 30s 会把"连不上"拖成半分钟的沉默
CONNECT_TIMEOUT_S = 10.0


def make_checkpointer(
    dsn: str, *, pool_size: int = POOL_MAX_SIZE
) -> AsyncPostgresSaver:
    """只造对象不碰网络：池 open=False，连接在 open_checkpointer 里建。"""
    pool = AsyncConnectionPool(
        dsn,
        min_size=POOL_MIN_SIZE,
        max_size=pool_size,
        open=False,
        timeout=CONNECT_TIMEOUT_S,
        # 每条连接的参数：照抄框架 from_conn_string——saver 的 SQL 假定 autocommit 与 dict_row
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    return AsyncPostgresSaver(pool)


def _pool(saver: AsyncPostgresSaver) -> AsyncConnectionPool:
    conn = saver.conn
    if not isinstance(conn, AsyncConnectionPool):
        raise TypeError("checkpointer 须由 make_checkpointer 构造（连接池形态）")
    return conn


async def open_checkpointer(saver: AsyncPostgresSaver) -> None:
    """开池（wait=True：min_size 条连接就绪才返回，连不上立刻报错而不是拖到首个请求）+ 框架迁移。

    先查循环类型：Proactor 下池会沉默重连直到超时，这里直接给出一句明白话。
    """
    require_selector_loop()
    pool = _pool(saver)
    await pool.open(wait=True, timeout=CONNECT_TIMEOUT_S)
    await saver.setup()


async def close_checkpointer(saver: AsyncPostgresSaver) -> None:
    """关池；之后任何 checkpoint 读写抛 PoolClosed（探针⑷），不再沉默。"""
    await _pool(saver).close()
