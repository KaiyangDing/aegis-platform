"""全仓 fixture：真 Redis db1（熔断/缓存/限流测试）与真 Postgres 测试库 aegis_test（账本/迁移/事件/checkpointer 测试），
不可达则整组跳过并给出启动命令。

flushdb 是全仓唯一的破坏性触点，且只许清 db1（fixture 里断言）。
测试库由 alembic downgrade base → upgrade head 建表（迁移=被测物，不用 create_all）；
LangGraph checkpointer 的四张框架表不归 alembic：每个测试会话开局整体删掉，由 setup() 重建（M2.2）。
每测一个引擎（asyncpg 连接绑定事件循环，探针⑰）+ 外层连接事务 + create_savepoint 会话工厂：
被测组件"自己开会话自己 commit"，外层 rollback 一笔勾销，测试库零污染。
事件循环：全仓统一 Selector（pytest_asyncio_loop_factories 钩子）——psycopg 拒绝 Windows 的 Proactor（ADR-011 决策 9）。
"""

import asyncio
import socket
import uuid
from pathlib import Path

import asyncpg
import pytest
import redis
import redis.asyncio as aioredis
from alembic import command
from alembic.config import Config
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

ROOT = Path(__file__).resolve().parents[1]
REDIS_TEST_URL = "redis://127.0.0.1:6379/1"
DEAD_REDIS_URL = "redis://127.0.0.1:6390/1"
PG_TEST_DB = "aegis_test"
PG_TEST_URL = f"postgresql+asyncpg://aegis:aegis_dev_pw@127.0.0.1:5432/{PG_TEST_DB}"
PG_TEST_DSN = f"postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/{PG_TEST_DB}"  # psycopg 形态（checkpointer）
PG_ADMIN_DSN = "postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
FRAMEWORK_TABLES: tuple[str, ...] = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)


def pytest_asyncio_loop_factories(config, item):
    """全仓统一 Selector 循环（Linux 默认即此；Windows 由此绕开 Proactor）。单一工厂时 pytest 不改测试 id。"""
    return {"selector": asyncio.SelectorEventLoop}


@pytest.fixture(scope="session")
def redis_db1() -> str:
    """会话级：探活 + 开局 flushdb（同步客户端，只用一次）；返回测试库 URL。"""
    client = redis.Redis.from_url(
        REDIS_TEST_URL, socket_connect_timeout=0.5, socket_timeout=0.5
    )
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("Redis db1 不可达：docker compose up -d redis")
    assert client.connection_pool.connection_kwargs.get("db") == 1
    client.flushdb()
    client.close()
    return REDIS_TEST_URL


@pytest.fixture
async def redis_async(redis_db1: str) -> aioredis.Redis:
    # 延迟导入：M1.4b 落地前整个测试集合不能因 conftest 炸掉
    from app.core.redis import make_async_redis

    client = make_async_redis(redis_db1)
    yield client
    await client.aclose()


@pytest.fixture
async def dead_redis_async() -> aioredis.Redis:
    """无人监听的端口：每个触点只付一次极短超时（生产快速失败常量另由 test_redis 钉住）。"""
    probe = socket.socket()
    probe.settimeout(0.2)  # Windows 回环闭端口不回 RST：不设超时的 connect_ex 会等约 2s
    occupied = probe.connect_ex(("127.0.0.1", 6390)) == 0
    probe.close()
    if occupied:
        pytest.skip("127.0.0.1:6390 有服务在听，dead_redis 无法模拟断连")
    client = aioredis.Redis.from_url(
        DEAD_REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=0.05,
        socket_timeout=0.05,
        retry=Retry(NoBackoff(), 0),
        retry_on_timeout=False,
    )
    yield client
    await client.aclose()


@pytest.fixture
def namespace() -> str:
    return f"t{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- Postgres 测试库


async def _ensure_test_database() -> None:
    conn = await asyncpg.connect(PG_ADMIN_DSN, timeout=2.0)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", PG_TEST_DB
        )
        if not exists:
            await conn.execute(f"CREATE DATABASE {PG_TEST_DB}")
    finally:
        await conn.close()


async def _reset_schema() -> None:
    """测试库从零开始：整个 public schema 重建。

    两个理由：checkpointer 四表不归 alembic（由 setup() 重建，thread 间靠唯一 id 隔离）；
    库里的 alembic_version 可能指向当前代码树还没有的迁移（稿件先于手敲落库），downgrade 会因找不到修订而炸。
    """
    conn = await asyncpg.connect(PG_TEST_DSN, timeout=2.0)
    try:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def pg_test_db() -> str:
    """会话级：探活 + 建库 + 清 schema + 迁移到 head，再 downgrade base / upgrade head 走一遍降级脚本
    （同步夹具里 asyncio.run，与各测试的事件循环无关）。"""
    pytest.importorskip(
        "app.domain.usage", reason="M1.5b 未敲：app/domain/usage.py 不存在"
    )
    try:
        asyncio.run(_ensure_test_database())
        asyncio.run(_reset_schema())
    except OSError, asyncpg.PostgresError, TimeoutError:
        pytest.skip("Postgres 不可达：docker compose up -d postgres")
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", PG_TEST_URL)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    return PG_TEST_URL


@pytest.fixture
async def db_conn(pg_test_db: str) -> AsyncConnection:
    """一条带外层事务的连接：测试里发生的一切在结束时整体回滚。引擎每测新建（跨 loop 不可复用）。"""
    engine = create_async_engine(pg_test_db)
    async with engine.connect() as conn:
        trans = await conn.begin()
        yield conn
        await trans.rollback()  # 一笔勾销
    await engine.dispose()


@pytest.fixture
def db_session_factory(db_conn: AsyncConnection) -> async_sessionmaker[AsyncSession]:
    """绑在测试连接上的会话工厂：给"自己开会话自己 commit"的组件（记账员 / 事件存取）注入。

    join_transaction_mode="create_savepoint"：这些会话的 commit 只提交 SAVEPOINT，
    外层 rollback 照样把一切吞掉——被测组件真实提交，测试库零污染。
    """
    return async_sessionmaker(
        bind=db_conn, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
async def db_session(db_session_factory) -> AsyncSession:
    async with db_session_factory() as session:
        yield session


@pytest.fixture
async def pg_checkpointer(pg_test_db: str):
    """每测一个 checkpointer（psycopg 池绑定事件循环）：开池 + 框架迁移；关停关池。

    checkpoint 写入走框架自己的连接与 autocommit，不在回滚夹具内——测试用唯一 thread_id 互不干扰，
    表在下个会话开局整体删除。
    """
    pytest.importorskip(
        "app.core.checkpoint", reason="M2.2 未敲：app/core/checkpoint.py 不存在"
    )
    from app.core.checkpoint import (
        close_checkpointer,
        make_checkpointer,
        open_checkpointer,
    )

    saver = make_checkpointer(PG_TEST_DSN, pool_size=2)
    await open_checkpointer(saver)
    yield saver
    await close_checkpointer(saver)
