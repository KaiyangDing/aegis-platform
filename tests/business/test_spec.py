"""AgentSpec 装配（S2）：白名单点名与声明序、租户配置透传、两层预算关闭、未知租户 / 未知工具 / 非法档位在装配期炸、
启动期预热覆盖全部租户且 fail-loud、system 模板在预算内、月度预算 resolver 三态、指纹按租户不同且可复现。"""

import pytest

from app.business import config as bc
from app.business.config import UnknownTenant, monthly_budget_for, tenant_ids
from app.business.spec import build_spec, preheat_specs
from app.core.tokens import estimate_tokens
from app.engine.runtime.runtime import spec_fingerprint
from app.engine.runtime.spec import ContextConfig


def test_tenant_a_spec():
    spec = build_spec("tenant-a")
    assert [t.name for t in spec.tools] == [
        "order_query",
        "refund_apply",
        "ticket_create",
    ]
    assert spec.tenant_config["approval_threshold"] == 200
    assert spec.model_tier == "standard" and spec.entry_classifier is False
    assert spec.owned_values == ()
    assert spec.context_config.memory_budget == 0
    assert spec.context_config.retrieval_budget == 0
    assert spec.policy.session_token_budget == 50_000
    assert spec.policy.approval_ttl_s == 3600.0
    assert "云杉商城" in spec.system_prompt


def test_tenant_b_has_no_refund_tool_and_classifier_on():
    spec = build_spec("tenant-b")
    assert [t.name for t in spec.tools] == ["order_query", "ticket_create"]
    assert spec.entry_classifier is True and spec.model_tier == "fast"
    assert spec.policy.session_token_budget == 20_000
    assert spec.policy.approval_ttl_s == 1800.0


def test_unknown_tenant_is_lookup_error():
    with pytest.raises(UnknownTenant):
        build_spec("tenant-z")
    assert issubclass(UnknownTenant, LookupError)


def test_unknown_tool_or_bad_tier_fails_at_assembly(monkeypatch):
    monkeypatch.setitem(
        bc._TENANTS, "tenant-x", {"name": "X", "tools": ("order_query", "coupon_grant")}
    )
    with pytest.raises(ValueError, match="coupon_grant"):
        build_spec("tenant-x")
    monkeypatch.setitem(
        bc._TENANTS, "tenant-y", {"name": "Y", "tools": (), "model_tier": "turbo"}
    )
    with pytest.raises(ValueError, match="model_tier"):
        build_spec("tenant-y")


def test_preheat_covers_every_tenant_and_fails_loud(monkeypatch):
    specs = preheat_specs()
    assert set(specs) == set(tenant_ids()) == {"tenant-a", "tenant-b"}
    monkeypatch.setitem(bc._TENANTS, "tenant-x", {"name": "X", "tools": ("nope",)})
    with pytest.raises(ValueError, match="nope"):
        preheat_specs()


def test_system_prompt_fits_system_budget():
    spec = build_spec("tenant-a")
    assert estimate_tokens(spec.system_prompt) < ContextConfig().system_budget // 2


async def test_monthly_budget_resolver_three_states():
    assert await monthly_budget_for("tenant-a") is None  # 0 = 关闭
    assert await monthly_budget_for("tenant-b") == 2_000_000
    assert await monthly_budget_for("tenant-z") is None


def test_fingerprint_differs_by_tenant_and_is_stable():
    a1 = spec_fingerprint(build_spec("tenant-a"))
    a2 = spec_fingerprint(build_spec("tenant-a"))
    b = spec_fingerprint(build_spec("tenant-b"))
    assert a1 == a2 and a1 != b
