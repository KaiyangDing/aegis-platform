"""M1.5 新增配置字段：缓存探针间隔 / 价目表形态 / 租户配额与预算 / 上游连接池——非法值启动即炸，默认值可用。"""

import math

import pytest
from pydantic import ValidationError

from app.core.config import Settings

if "tenant_burst" not in Settings.model_fields:
    pytest.skip("M1.5c 未敲：Settings 尚无租户配额字段", allow_module_level=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"cache_probe_interval_s": 0},
        {"cache_probe_interval_s": math.inf},
        {"model_prices": {"m": [0.1]}},  # 缺输出价
        {"model_prices": {"m": [0.1, -0.2]}},  # 负价
        {"tenant_rate_per_s": 0},
        {"tenant_burst": 0.5},  # 库要求整枚令牌
        {"tenant_limiter_max_keys": 0},
        {"tenant_monthly_token_budget": -1},
        {"request_token_budget": -1},
        {"upstream_max_connections": 0},
        {"upstream_max_keepalive": -1},
    ],
)
def test_m15_fields_fail_loud(bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **bad)


def test_m15_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.cache_probe_interval_s == 5.0
    assert (
        s.tenant_monthly_token_budget == 0 and s.request_token_budget == 0
    )  # 0 = 关闭
    assert s.tenant_burst >= 1 and s.tenant_rate_per_s > 0
    assert set(s.model_prices) >= {
        "qwen-flash",
        "qwen-turbo",
        "qwen-plus",
        "qwen3.7-max",
    }
    assert all(len(pair) == 2 for pair in s.model_prices.values())


def test_prices_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", '{"x": [1.0, 2.0]}')
    assert Settings(_env_file=None).model_prices == {"x": [1.0, 2.0]}
