"""L1 网关：档位路由 / 异常契约 / 受控重试 / 熔断 / 缓存 / 计量。不 import runtime。"""

from app.engine.gateway.breakers import BreakerPolicy, MemoryBreaker, RedisBreaker
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
    OutboundGateTimeout,
    ProviderError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
    TenantQuotaExceeded,
    classify,
    sanitize_error_text,
)
from app.engine.gateway.faults import FaultInjector, inject_faults
from app.engine.gateway.outbound import ProviderLimiter
from app.engine.gateway.protocols import BreakerLike, Decision, LimiterLike
from app.engine.gateway.resilience import (
    RetryPolicy,
    complete_with_retry,
    compute_backoff,
)
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import (
    TIERS,
    Candidate,
    Tier,
    parse_routes,
    unique_candidates,
)

__all__ = [
    "TIERS",
    "AegisGateway",
    "AuthError",
    "BadRequestError",
    "BreakerLike",
    "BreakerPolicy",
    "BudgetExceeded",
    "Candidate",
    "Decision",
    "FaultInjector",
    "GatewayError",
    "GatewayExhausted",
    "GatewayOverloadedError",
    "GatewayRejected",
    "GatewayStreamInterrupted",
    "LimiterLike",
    "MemoryBreaker",
    "OutboundGateTimeout",
    "ProviderError",
    "ProviderLimiter",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "RedisBreaker",
    "RetryPolicy",
    "TenantQuotaExceeded",
    "Tier",
    "build_candidates",
    "classify",
    "complete_with_retry",
    "compute_backoff",
    "inject_faults",
    "make_candidate",
    "parse_routes",
    "sanitize_error_text",
    "unique_candidates",
]
