"""L1 网关：档位路由 / 异常契约 / 受控重试 / 熔断 / 缓存 / 计量。不 import runtime。"""

from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    BudgetExceeded,
    GatewayError,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    ProviderError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
    TenantQuotaExceeded,
    classify,
    sanitize_error_text,
)

__all__ = [
    "AuthError",
    "BadRequestError",
    "BudgetExceeded",
    "GatewayError",
    "GatewayExhausted",
    "GatewayOverloadedError",
    "GatewayRejected",
    "GatewayStreamInterrupted",
    "ProviderError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "TenantQuotaExceeded",
    "classify",
    "sanitize_error_text",
]
