"""API 进程入口：lifespan 建共享单例（Redis 客户端 / 上游 httpx2 客户端 / 账本引擎 / 网关共享件 / 入站限流器 /
LangGraph checkpointer / AgentRuntime），关停时逐个收尾。

这些对象都绑定创建时的事件循环（asyncpg 连接、httpx2 连接池、psycopg 连接池），所以在 lifespan 里建而不在模块级建；
worker 进程（app/worker.py，M3）在自己的 loop 里用同一批 build_* 再建一份。
入站限流器只在这里建（InboundLimiter，共用同一个 Redis 客户端）并挂到 app.state；无 Redis 配置则挂 None = 永远 fail-open；
端点挂载随 M3。
checkpointer（ADR-011）：psycopg 连接池 + 框架自带迁移，挂 app.state.checkpointer；Windows 开发进程须以
`--loop app.core.loops:selector_loop_factory` 启动，否则 open_checkpointer 在启动期就报明白话。
AgentRuntime（M2.3）：进程级单件，挂 app.state.runtime；图按租户 spec 在它里面缓存。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.checkpoint import (
    close_checkpointer,
    make_checkpointer,
    open_checkpointer,
)
from app.core.config import get_settings
from app.core.db import make_engine, make_session_factory
from app.core.limits import InboundLimiter
from app.core.logs import configure_logging
from app.core.redis import make_async_redis
from app.deps import build_gateway_parts, build_runtime, build_runtime_parts
from app.engine.gateway.candidates import make_http_client


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json=settings.app_env != "dev")
    redis = make_async_redis(settings.redis_url) if settings.redis_url else None
    limiter = None
    if redis is not None:
        limiter = InboundLimiter(
            redis, probe_interval=settings.inbound_probe_interval_s
        )
        await limiter.start()
    app_.state.inbound_limiter = limiter
    http_client = make_http_client(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=settings.upstream_max_keepalive,
    )
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    gateway_parts = build_gateway_parts(
        settings, http_client=http_client, redis=redis, session_factory=session_factory
    )
    app_.state.gateway_parts = gateway_parts
    # 开池 + 框架迁移；连不上 / 事件循环不对在这里就炸，不拖到首个请求
    checkpointer = make_checkpointer(settings.checkpoint_database_url)
    await open_checkpointer(checkpointer)
    app_.state.checkpointer = checkpointer
    app_.state.runtime = build_runtime(
        gateway_parts, build_runtime_parts(session_factory, checkpointer)
    )
    try:
        yield
    finally:
        app_.state.inbound_limiter = None  # 先摘闸（之后 fail-open）再关客户端
        app_.state.runtime = None
        app_.state.checkpointer = None
        await close_checkpointer(checkpointer)
        await http_client.aclose()
        if redis is not None:
            await redis.aclose()
        await engine.dispose()


app = FastAPI(title="Aegis v2", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
