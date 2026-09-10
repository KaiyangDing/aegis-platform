"""租户静态配置（S2；命运表 B7 降级：无 tenants 表，变更入口 = 代码提交，可 diff 可回滚）。

运行时对这些键零知识：tenant_config 整体原样透传给 AgentSpec，risk_policy 只认 approval_threshold；
其余键在 build_spec（tools / model_tier / entry_classifier / 两个预算）与组合根（monthly_token_budget → 网关 resolver）消费。
键缺席的语义各处自定：approval_threshold 缺席 = 0 = 任意正金额都要人批（fail-closed）；monthly_token_budget 缺席或 0 = 关闭。
两个演示租户覆盖三条分岔：工具白名单不同（B 没有退款）、入口分类器开关不同、档位不同。
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class UnknownTenant(LookupError):
    """JWT 里的租户不在静态表：签名是对的、身份合法，只是租户未开通——API 层翻译为 403 而不是 401。"""


_TENANTS: dict[str, Mapping[str, Any]] = {
    "tenant-a": MappingProxyType(
        {
            "name": "云杉商城",
            "tools": ("order_query", "refund_apply", "ticket_create"),
            "approval_threshold": 200,
            "model_tier": "standard",
            "entry_classifier": False,
            "session_token_budget": 50_000,
            "approval_ttl_s": 3600.0,
            "monthly_token_budget": 0,
        }
    ),
    "tenant-b": MappingProxyType(
        {
            "name": "北辰家电",
            "tools": ("order_query", "ticket_create"),
            "approval_threshold": 0,
            "model_tier": "fast",
            "entry_classifier": True,
            "session_token_budget": 20_000,
            "approval_ttl_s": 1800.0,
            "monthly_token_budget": 2_000_000,
        }
    ),
}
TENANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(_TENANTS)
"""只读视图：测试想加坏租户用 monkeypatch.setitem(config._TENANTS, …)，视图随之可见。"""


def tenant_config(tenant_id: str) -> Mapping[str, Any]:
    try:
        return TENANTS[tenant_id]
    except KeyError as e:
        raise UnknownTenant(tenant_id) from e


def tenant_ids() -> tuple[str, ...]:
    return tuple(TENANTS)


async def monthly_budget_for(tenant_id: str) -> int | None:
    """网关 BudgetResolver 形态（组合根接进 gateway_for）：0 / 缺席 = 关闭 → None；未知租户 → None
    （网关月度闸 fail-open 语义不变，租户是否开通由 API 层身份守卫先拦）。"""
    cfg = TENANTS.get(tenant_id)
    if cfg is None:
        return None
    budget = int(cfg.get("monthly_token_budget", 0))
    return budget if budget > 0 else None
