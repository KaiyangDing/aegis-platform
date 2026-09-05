"""入口冒烟：healthz 契约；lifespan 在 fake 模式下建起共享件，按租户装配的网关能流出回复并落账。"""

import httpx
import pytest
import structlog
from langchain_core.messages import HumanMessage

from app.main import app


async def test_healthz_ok():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_lifespan_builds_gateway_parts_in_fake_mode(
    monkeypatch, redis_db1, pg_test_db, namespace
):
    """真 lifespan：Redis 指到 db1、账本指到 aegis_test（都由 conftest 保证可达），fake 上游零网络。"""
    from app import deps as deps_mod
    from app.core.config import get_settings
    from app.engine.fakes import FAKE_REPLY

    if not hasattr(deps_mod, "gateway_for"):
        pytest.skip("M1.5c 未敲：deps.py 尚无组合根")
    gateway_for = deps_mod.gateway_for

    monkeypatch.setenv("REDIS_URL", redis_db1)
    monkeypatch.setenv("DATABASE_URL", pg_test_db)
    monkeypatch.setenv("AEGIS_FAKE_LLM", "1")
    get_settings.cache_clear()
    try:
        async with app.router.lifespan_context(app):
            parts = app.state.gateway_parts
            gw = gateway_for(parts, "tA")
            chunks = [
                str(c.content) async for c in gw.astream([HumanMessage(namespace)])
            ]
            assert "".join(chunks) == FAKE_REPLY
            assert gw.meter is not None and gw.reply_cache is not None
    finally:
        get_settings.cache_clear()
        structlog.reset_defaults()  # lifespan 配置了进程级日志，别泄漏到别的测试


def test_lifespan_is_registered():
    assert app.router.lifespan_context is not None
