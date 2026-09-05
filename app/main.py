"""API 进程入口：lifespan 建共享单例（Redis 客户端 / 上游 httpx2 客户端 / 账本引擎 / 网关共享件），关停时逐个收尾。

这些对象都绑定创建时的事件循环（asyncpg 连接、httpx2 连接池），所以在 lifespan 里建而不在模块级建；
worker 进程（app/worker.py，M3）在自己的 loop 里用同一个 build_gateway_parts 再建一份。
入站限流（FastAPILimiter.init）随 M1.6 接入；路由随 M3 挂载。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.db import make_engine, make_session_factory
from app.core.logs import configure_logging
from app.core.redis import make_async_redis
from app.deps import build_gateway_parts
from app.engine.gateway.candidates import make_http_client


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json=settings.app_env != "dev")
    redis = make_async_redis(settings.redis_url) if settings.redis_url else None
    http_client = make_http_client(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=settings.upstream_max_keepalive,
    )
    engine = make_engine(settings.database_url)
    app_.state.gateway_parts = build_gateway_parts(
        settings,
        http_client=http_client,
        redis=redis,
        session_factory=make_session_factory(engine),
    )
    try:
        yield
    finally:
        await http_client.aclose()
        if redis is not None:
            await redis.aclose()
        await engine.dispose()


app = FastAPI(title="Aegis v2", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
