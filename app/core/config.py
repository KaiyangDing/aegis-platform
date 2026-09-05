"""全局配置：唯一的 .env 消费方。字段名即环境变量名（大小写不敏感）。

密钥只从环境变量读，类型用 SecretStr：repr/日志自动打码，取真值必须显式 .get_secret_value()
（全仓只允许候选工厂一处取）。只存原始类型：档位/候选/异常类型全在 engine，core 不认识它们。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    database_url: str = "postgresql+asyncpg://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    checkpoint_database_url: str = (
        "postgresql://aegis:aegis_dev_pw@127.0.0.1:5432/aegis"
    )
    redis_url: str = (
        "redis://127.0.0.1:6379/0"  # 空串 = 无 Redis（熔断退化进程内、缓存关闭）
    )

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
    # 共享上游 httpx2 客户端的连接池：池满等 pool 超时即 GatewayOverloadedError（本地过载，不换路不进账）
    upstream_max_connections: int = Field(default=100, gt=0)
    upstream_max_keepalive: int = Field(default=20, ge=0)

    # --- 熔断与出站闸（M1.4b；v1 口径起点，参数重推见 ADR-007/008） ---
    breaker_fail_max: int = Field(default=5, gt=0)  # 连续入账失败达到即开路
    # 三个时长 allow_inf_nan=False：inf 会过 gt=0，却让 Redis 版永不跳闸（TTL 换算溢出被 fail-open 吞）
    breaker_reset_timeout_s: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    # 试探令牌租约：试探方崩溃后到期自愈
    breaker_probe_ttl_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    # 失败计数遗忘窗（须 > reset + probe_ttl，BreakerPolicy 校验）
    breaker_fail_window_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    # 降级期顺路探针间隔：Redis 断后每隔这么久才再碰一次
    breaker_probe_interval_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    outbound_rate_per_s: float = Field(default=8.0, gt=0)  # 每 provider 令牌补给速率
    outbound_burst: float = Field(
        default=16.0, ge=1
    )  # 桶容量（冷启动满桶；<1 库永远发不出令牌）
    outbound_max_wait_s: float = Field(
        default=10.0, ge=0
    )  # 取令牌最长排队；0 = 只试不排

    # --- 租户配额与预算（M1.5c；ADR-008 决策 3/4） ---
    tenant_rate_per_s: float = Field(default=5.0, gt=0)  # 每租户出站速率（进程内近似）
    tenant_burst: float = Field(default=10.0, ge=1)
    tenant_limiter_max_keys: int = Field(default=10_000, ge=1)  # 租户桶 LRU 上限
    tenant_monthly_token_budget: int = Field(
        default=0, ge=0
    )  # 0 = 关闭；超额抛 BudgetExceeded
    request_token_budget: int = Field(
        default=0, ge=0
    )  # 单请求估算预算（自家尺）；0 = 关闭

    # --- 计量（M1.5b） ---
    # 模型单价（元/千 token，[输入, 输出]）——演示值，以百炼价目页为准；调价改这里不改代码。
    # 组合根经 domain.usage.price_table 转 Decimal(str(p))；不在价目表的模型记 0 并告警。
    model_prices: dict[str, list[float]] = {
        "qwen-flash": [0.00015, 0.0015],
        "qwen-turbo": [0.0003, 0.0006],
        "qwen-plus": [0.0008, 0.002],
        "qwen3.7-max": [0.012, 0.036],
        "text-embedding-v4": [0.0005, 0.0],
    }

    # --- 缓存与故障注入 ---
    cache_ttl_seconds: int = Field(default=300, ge=0)  # 0 = 关缓存（组合根不装）
    # 缓存降级期顺路探针间隔：Redis 断后每隔这么久才再碰一次（与熔断同款粘滞）
    cache_probe_interval_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    fault_injection_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # 0 = 关闭
    fault_injection_targets: list[str] = []  # 点名 "provider:model"
    fault_injection_mode: Literal["error", "hang", "midstream"] = "error"
    # hang 模式的挂起时长：由网关首块超时真实切断，取值只需大于首块窗口
    fault_injection_hang_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _forbid_fault_injection_in_prod(self) -> Settings:
        # 注入器是演示/实验专用：生产环境配置了注入率，进程启动即炸，不许带病上线
        if self.app_env == "prod" and self.fault_injection_rate > 0:
            raise ValueError("生产环境禁止故障注入：FAULT_INJECTION_RATE 必须为 0")
        return self

    @model_validator(mode="after")
    def _prices_are_pairs(self) -> Settings:
        # 价目表形态在启动时钉死：[输入, 输出] 两项非负——环境变量 JSON 写错不许拖到第一笔账
        for model, pair in self.model_prices.items():
            if len(pair) != 2 or any(p < 0 for p in pair):
                raise ValueError(
                    f"MODEL_PRICES 非法：{model} 须为 [输入价, 输出价] 且非负"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
