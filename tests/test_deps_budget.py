"""组合根接线（M3.2）：gateway_for 透传月度预算 resolver（L3 租户配置注入网关的缝）；缺省 None = 静态配置兜底。"""

import httpx2
import pytest

from app.core.config import Settings
from app.deps import build_gateway_parts, gateway_for


@pytest.fixture
async def http_client():
    client = httpx2.AsyncClient()
    yield client
    await client.aclose()


async def test_gateway_for_passes_budget_resolver(http_client):
    parts = build_gateway_parts(
        Settings(_env_file=None, aegis_fake_llm=True),
        http_client=http_client,
        redis=None,
        session_factory=None,
    )

    async def resolver(tenant_id: str) -> int | None:
        return 42

    assert (
        gateway_for(parts, "tA", budget_resolver=resolver).budget_resolver is resolver
    )
    assert gateway_for(parts, "tA").budget_resolver is None
