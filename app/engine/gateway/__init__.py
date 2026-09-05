"""L1 网关：档位路由 / 异常契约 / 受控重试 / 熔断 / 缓存 / 计量。不 import runtime。"""

from app.engine.gateway.breakers import BreakerPolicy, MemoryBreaker, RedisBreaker
from app.engine.gateway.cache import (
    CachedReply,
    CacheLike,
    CacheStore,
    TenantCache,
    request_digest,
)
from app.engine.gateway.candidates import (
    build_candidates,
    make_candidate,
    make_http_client,
)
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
from app.engine.gateway.protocols import (
    BreakerLike,
    BudgetResolver,
    Decision,
    LimiterLike,
    MeterLike,
)
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
from app.engine.gateway.tenancy import validate_tenant_id

__all__ = [
    "TIERS",
    "AegisGateway",
    "AuthError",
    "BadRequestError",
    "BreakerLike",
    "BreakerPolicy",
    "BudgetExceeded",
    "BudgetResolver",
    "CacheLike",
    "CacheStore",
    "CachedReply",
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
    "MeterLike",
    "OutboundGateTimeout",
    "ProviderError",
    "ProviderLimiter",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "RedisBreaker",
    "RetryPolicy",
    "TenantCache",
    "TenantQuotaExceeded",
    "Tier",
    "build_candidates",
    "classify",
    "complete_with_retry",
    "compute_backoff",
    "inject_faults",
    "make_candidate",
    "make_http_client",
    "parse_routes",
    "request_digest",
    "sanitize_error_text",
    "unique_candidates",
    "validate_tenant_id",
]
