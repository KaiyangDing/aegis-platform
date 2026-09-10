"""组合根的运行时装配（M2.3；M2.7 加审批单存取件）：RuntimeParts → AgentRuntime，网关按租户由闭包装配、图按租户 spec 编译。"""

import httpx2
import pytest
from langgraph.checkpoint.memory import InMemorySaver

pytest.importorskip(
    "app.domain.events", reason="M2.2 未敲：app/domain/events.py 不存在"
)
pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)

from app import deps as deps_mod

if not hasattr(deps_mod, "build_runtime"):
    pytest.skip("M2.3 未敲：deps.py 尚无 build_runtime", allow_module_level=True)
if "approvals" not in deps_mod.RuntimeParts.__dataclass_fields__:
    pytest.skip("M2.7 未敲：RuntimeParts 尚无 approvals", allow_module_level=True)

from app.core.config import Settings
from app.deps import (
    RuntimeParts,
    build_gateway_parts,
    build_runtime,
    build_runtime_parts,
)
from app.domain.approvals import ApprovalStore
from app.domain.events import EventStore
from app.domain.sessions import SessionStateStore
from app.engine.runtime.runtime import AgentRuntime
from app.engine.runtime.spec import AgentSpec


@pytest.fixture
async def http_client():
    client = httpx2.AsyncClient()
    yield client
    await client.aclose()


def test_build_runtime_parts_wraps_stores(db_session_factory):
    parts = build_runtime_parts(db_session_factory, InMemorySaver())
    assert isinstance(parts, RuntimeParts)
    assert isinstance(parts.events, EventStore)
    assert isinstance(parts.sessions, SessionStateStore)
    assert isinstance(parts.approvals, ApprovalStore)


def test_build_runtime_compiles_tenant_bound_graph(http_client):
    gateway_parts = build_gateway_parts(
        Settings(_env_file=None, aegis_fake_llm=True),
        http_client=http_client,
        redis=None,
        session_factory=None,
    )
    runtime_parts = RuntimeParts(
        events=EventStore(None),  # type: ignore[arg-type]
        sessions=SessionStateStore(None),  # type: ignore[arg-type]
        approvals=ApprovalStore(None),  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )
    runtime = build_runtime(gateway_parts, runtime_parts)
    assert isinstance(runtime, AgentRuntime)
    assert runtime._approvals is runtime_parts.approvals  # type: ignore[attr-defined]
    agent = runtime.build_agent("tA", AgentSpec(system_prompt="演示"))
    assert "model" in agent.get_graph().nodes
    with pytest.raises(ValueError):
        runtime.build_agent("bad tenant!", AgentSpec(system_prompt="演示"))  # 入口守卫
