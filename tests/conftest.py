"""全仓 fixture：真 Redis db1（熔断/缓存/限流测试），不可达则整组跳过并给出启动命令。

flushdb 是全仓唯一的破坏性触点，且只许清 db1（fixture 里断言）。
"""

import socket
import uuid

import pytest
import redis
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

REDIS_TEST_URL = "redis://127.0.0.1:6379/1"
DEAD_REDIS_URL = "redis://127.0.0.1:6390/1"


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
