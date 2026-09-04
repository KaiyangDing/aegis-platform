"""Redis 客户端工厂：快速失败常量被钉住——Redis 故障期每个触点只付一次短超时，不重试，且并发触点不排队。"""

import asyncio
import time

import pytest
import redis
import redis.asyncio as aioredis

from app.core.redis import (
    CONNECT_TIMEOUT_S,
    POOL_MAX_CONNECTIONS,
    POOL_WAIT_S,
    SOCKET_TIMEOUT_S,
    make_async_redis,
)
from tests.conftest import DEAD_REDIS_URL


def test_client_is_configured_to_fail_fast():
    client = make_async_redis("redis://127.0.0.1:6379/1")
    pool = client.connection_pool
    kw = pool.connection_kwargs
    assert isinstance(pool, aioredis.BlockingConnectionPool)  # 建连在池锁之外
    assert pool.max_connections == POOL_MAX_CONNECTIONS == 64
    assert pool.timeout == POOL_WAIT_S == 0.5
    assert kw["decode_responses"] is True
    assert kw["socket_connect_timeout"] == CONNECT_TIMEOUT_S == 0.5
    assert kw["socket_timeout"] == SOCKET_TIMEOUT_S == 0.5
    assert kw["retry"]._retries == 0
    assert kw["retry_on_timeout"] is False


async def test_dead_port_raises_within_one_timeout(dead_redis_async):
    """生产参数对死端口的代价上界：一次连接超时（0.5s），不是重试叠加。"""
    client = make_async_redis(DEAD_REDIS_URL)
    t0 = time.perf_counter()
    with pytest.raises(redis.RedisError):
        await client.exists("x")
    assert time.perf_counter() - t0 < CONNECT_TIMEOUT_S + 0.5
    await client.aclose()


async def test_concurrent_touches_on_dead_port_do_not_queue_up(dead_redis_async):
    """4 个并发触点各付各的 0.5s（默认 ConnectionPool 在锁内 connect 会排成 4 × 0.5s）。"""
    client = make_async_redis(DEAD_REDIS_URL)

    async def touch() -> None:
        with pytest.raises(redis.RedisError):
            await client.exists("x")

    t0 = time.perf_counter()
    await asyncio.gather(*(touch() for _ in range(4)))
    assert time.perf_counter() - t0 < 2 * CONNECT_TIMEOUT_S
    await client.aclose()
