"""全局配置：唯一的 .env 消费方。字段名即环境变量名（大小写不敏感）。

密钥只从环境变量读，类型用 SecretStr：repr/日志自动打码，取真值必须显式 .get_secret_value()
（全仓只允许候选工厂一处取）。只存原始类型：档位/候选/异常类型全在 engine，core 不认识它们。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    database_url: str = "postgresql+asyncpg://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    checkpoint_database_url: str = (
        "postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"

    # --- 上游供应商与档位路由（M1.2） ---
    dashscope_api_key: SecretStr = SecretStr("")
    # 供应商名 → OpenAI 兼容端点。M1 单供应商多模型；第二个兼容端点只需加一行（密钥沿用同一把，
    # 换供应商时再拆成按供应商的密钥表）。
    providers: dict[str, str] = {
        "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    # 档位 → 候选链（"provider:model"，按序 fallback）。启动时 parse_routes 四项校验；
    # 能力断崖（strong 退到 fast 级模型）是配置纪律，不做硬校验。环境变量可用 JSON 整体覆盖。
    model_routes: dict[str, list[str]] = {
        "fast": ["bailian:qwen-flash", "bailian:qwen-turbo"],
        "standard": ["bailian:qwen-plus", "bailian:qwen-turbo"],
        "strong": ["bailian:qwen3.7-max", "bailian:qwen-plus"],
    }
    # fake 开关只在候选工厂生效：组合根/网关/测试的注入点零改动统一换成 FakeReplyChatModel。
    aegis_fake_llm: bool = True
    # 线 B：fake 首块前延迟，让任务时长与队列形态接近真实（0 = 瞬间完成，什么都压不出来）
    aegis_fake_llm_delay_s: float = Field(default=0.0, ge=0)

    # --- 缓存与故障注入（字段先立；prod 禁注入校验器随 M1.4a） ---
    cache_ttl_seconds: int = Field(default=300, ge=0)  # 0 = 关缓存（组合根不装）
    fault_injection_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # 0 = 关闭
    fault_injection_targets: list[str] = []  # 点名 "provider:model"
    fault_injection_mode: Literal["error", "hang", "midstream"] = "error"


@lru_cache
def get_settings() -> Settings:
    return Settings()
