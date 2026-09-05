"""tenant_id 字符集守卫（C9 入口守卫）：合法四例过、非法四例拒；网关构造与组合根共用同一条规则。"""

import pytest

pytest.importorskip(
    "app.engine.gateway.tenancy",
    reason="M1.5a 未敲：app/engine/gateway/tenancy.py 不存在",
)

from app.engine.gateway.tenancy import TENANT_ID_RE, validate_tenant_id


@pytest.mark.parametrize("tenant", ["tA", "tenant-1", "T_x-9", "a" * 64])
def test_valid_tenant_ids_pass(tenant):
    assert validate_tenant_id(tenant) == tenant


@pytest.mark.parametrize(
    "tenant",
    ["tA:evil", "", "租户甲", "a" * 65, "t a", "t*", "tA\n", 123],
)
def test_invalid_tenant_ids_rejected(tenant):
    with pytest.raises(ValueError, match="tenant_id"):
        validate_tenant_id(tenant)


def test_pattern_matches_ledger_column_width():
    assert TENANT_ID_RE.pattern == r"^[A-Za-z0-9_-]{1,64}$"
