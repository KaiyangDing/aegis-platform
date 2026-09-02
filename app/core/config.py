"""全局配置：唯一的 .env 消费方。字段名即环境变量名（大小写不敏感）。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    checkpoint_database_url: str = (
        "postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    dashscope_api_key: str = ""
    aegis_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    aegis_fake_llm: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
