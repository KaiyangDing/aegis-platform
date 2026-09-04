"""配置安全底线：密钥不出 repr，非法环境名启动即炸，生产环境禁止故障注入。"""

import math

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_secret_never_leaks_in_repr():
    settings = Settings(_env_file=None, dashscope_api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in str(settings)
    assert settings.dashscope_api_key.get_secret_value() == "sk-super-secret"


def test_default_key_is_empty_so_tests_need_no_secret(monkeypatch):
    # 不依赖本机环境干净：显式摘掉环境变量，只看默认值
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert Settings(_env_file=None).dashscope_api_key.get_secret_value() == ""


def test_app_env_is_validated():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production")


def test_fault_injection_is_forbidden_in_prod():
    with pytest.raises(ValidationError, match="生产环境禁止故障注入"):
        Settings(_env_file=None, app_env="prod", fault_injection_rate=0.5)


def test_fault_injection_zero_in_prod_and_positive_in_dev_are_fine():
    assert Settings(_env_file=None, app_env="prod").fault_injection_rate == 0.0
    assert (
        Settings(_env_file=None, fault_injection_rate=0.5).fault_injection_rate == 0.5
    )


def test_fault_injection_hang_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fault_injection_hang_s=0)


@pytest.mark.parametrize(
    "bad",
    [
        {"breaker_fail_max": 0},
        {"breaker_reset_timeout_s": 0},
        {"breaker_probe_ttl_s": 0},
        {"breaker_fail_window_s": 0},
        {"breaker_fail_window_s": math.inf},
        {"breaker_probe_interval_s": 0},
        {"outbound_rate_per_s": 0},
        {"outbound_burst": 0.5},
        {"outbound_max_wait_s": -0.1},
    ],
)
def test_breaker_and_outbound_fields_fail_loud(bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **bad)


def test_outbound_max_wait_zero_means_try_only():
    assert Settings(_env_file=None, outbound_max_wait_s=0).outbound_max_wait_s == 0
