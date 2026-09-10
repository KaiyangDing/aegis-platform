"""S2 配置：JWT 三字段默认值、密钥不出 repr、生产环境空密钥启动即炸、TTL 必须为正。"""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_jwt_defaults_are_empty_secret_and_two_ttl_tiers(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    settings = Settings(_env_file=None)
    assert settings.jwt_secret.get_secret_value() == ""
    assert (settings.jwt_user_ttl_s, settings.jwt_staff_ttl_s) == (7200, 28800)


def test_jwt_secret_never_leaks_in_repr():
    settings = Settings(
        _env_file=None, jwt_secret=SecretStr("top-secret-0123456789abcdefghijklmn")
    )
    assert "top-secret" not in repr(settings) and "top-secret" not in str(settings)


def test_prod_requires_jwt_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, app_env="prod")
    ok = Settings(_env_file=None, app_env="prod", jwt_secret=SecretStr("x" * 32))
    assert ok.app_env == "prod"


def test_ttl_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, jwt_user_ttl_s=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, jwt_staff_ttl_s=-1)
