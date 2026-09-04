"""Redis 异步客户端工厂：熔断（RedisBreaker）、限流、缓存共用一种客户端。

快速失败：连接/读写超时短且不重试——Redis 故障期每个触点只付一次短超时，
上层各自 fail-open（熔断放行、缓存直通），护栏挂了不连坐核心功能。
连接池用 BlockingConnectionPool：建连在池锁之外，故障期并发触点各付各的一次超时，
而不是像默认 ConnectionPool 那样在锁内 connect、N 个并发触点排队付 N 次；
池满等 POOL_WAIT_S 后抛 ConnectionError，同样落入上层的 fail-open。
"""

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

CONNECT_TIMEOUT_S = 0.5
SOCKET_TIMEOUT_S = 0.5
POOL_MAX_CONNECTIONS = 64
POOL_WAIT_S = 0.5


def make_async_redis(url: str) -> aioredis.Redis:
    pool = aioredis.BlockingConnectionPool.from_url(
        url,
        max_connections=POOL_MAX_CONNECTIONS,
        timeout=POOL_WAIT_S,
        decode_responses=True,
        socket_connect_timeout=CONNECT_TIMEOUT_S,
        socket_timeout=SOCKET_TIMEOUT_S,
        retry=Retry(NoBackoff(), 0),
        retry_on_timeout=False,
    )
    return aioredis.Redis.from_pool(pool)  # aclose() 连池一起关
