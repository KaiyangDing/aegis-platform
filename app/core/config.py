"""全局配置：唯一的 .env 消费方。字段名即环境变量名（大小写不敏感）。

密钥只从环境变量读，类型用 SecretStr：repr/日志自动打码，取真值必须显式 .get_secret_value()。
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    database_url: str = "postgresql+asyncpg://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    checkpoint_database_url: str = (
        "postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    dashscope_api_key: SecretStr = SecretStr("")
    aegis_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    aegis_fake_llm: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
