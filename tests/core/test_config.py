"""配置安全底线：密钥不出 repr，非法环境名启动即炸。"""

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
