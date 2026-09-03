"""L1 网关：档位路由 / 异常契约 / 受控重试 / 熔断 / 缓存 / 计量。不 import runtime。"""

from app.engine.gateway.candidates import build_candidates, make_candidate
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
from app.engine.gateway.routing import (
    TIERS,
    Candidate,
    Tier,
    parse_routes,
    unique_candidates,
)

__all__ = [
    "TIERS",
    "AuthError",
    "BadRequestError",
    "BudgetExceeded",
    "Candidate",
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
    "Tier",
    "build_candidates",
    "classify",
    "make_candidate",
    "parse_routes",
    "sanitize_error_text",
    "unique_candidates",
]
