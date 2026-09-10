"""HTTP 面测试夹具：最小 FastAPI 应用挂三路由 + app.state 装内存替身（事件 / 会话 / 审批 / 静态 specs）+ 剧本网关运行时；
零 PG 零 Redis 零真实调用（入站限流器未挂载 = fail-open）。每测一个干净模拟后端经 set_backend 注入，测后归还。
路由只读 app.state，所以这里装的键与 app/main.py 的 lifespan 一一对应（settings / runtime / runtime_parts / specs）。"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.business.backend import MockBackend, demo_backend, set_backend
from app.business.spec import preheat_specs
from app.core.config import Settings
from app.routers import approvals, chat, sessions
from tests.engine.gateway.doubles import ScriptedCandidate
from tests.engine.runtime.doubles import (
    MemoryApprovalStore,
    MemoryEventStore,
    MemorySessionStore,
    make_runtime,
)
from tests.routers.sse_client import SECRET


@dataclass
class Harness:
    app: FastAPI
    candidate: ScriptedCandidate
    events: MemoryEventStore
    sessions: MemorySessionStore
    approvals: MemoryApprovalStore
    backend: MockBackend

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )


def build_harness(
    *acts: list[Any], tier: str = "standard", gateway_for: Any = None
) -> Harness:
    approvals_store = MemoryApprovalStore()
    runtime, candidate, events, sessions_store = make_runtime(
        *acts, approvals=approvals_store, tier=tier, gateway_for=gateway_for
    )
    app = FastAPI()
    for router in (chat.router, approvals.router, sessions.router):
        app.include_router(router)
    app.state.settings = Settings(
        _env_file=None, jwt_secret=SecretStr(SECRET), redis_url=""
    )
    app.state.runtime = runtime
    app.state.runtime_parts = SimpleNamespace(
        events=events, sessions=sessions_store, approvals=approvals_store
    )
    app.state.specs = preheat_specs()
    backend = demo_backend()
    set_backend(backend)
    return Harness(app, candidate, events, sessions_store, approvals_store, backend)


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    set_backend(None)
