"""组合根：真实依赖只在这里聚合成完整网关与运行时，其余代码一律靠注入（C12；ADR-009 / ADR-011）。

两级装配：
- 进程级共享件（lifespan 建一次）：路由表、候选实例（含注入器包装）、熔断器、供应商出站闸、租户桶、
  缓存存取（CacheStore）、记账员——都是无租户状态或按 key 分片的对象，跨请求复用；
  运行时共享件（RuntimeParts）：事件事实源、会话状态机、审批单存取件、checkpointer——同样进程级；
- 租户绑定件（每请求 `gateway_for`）：AegisGateway 实例 + TenantCache 视图——网关是轻量 pydantic 对象，
  租户身份在构造时绑定，缓存前缀 / 账本列 / 配额桶键由此而来。
AgentRuntime 是进程级单件：网关按租户由闭包装配、图按 (tenant_id, spec 指纹) 缓存在它里面。
入口守卫：tenant_id 先过字符集校验（网关字段校验器是第二道，规则同源），非法 → ValueError。
fake 开关在候选工厂生效（M1.2），这里不感知。本模块是全仓唯一同时 import engine 与 domain 的地方。
"""

from dataclasses import dataclass

import httpx2
import redis.asyncio as aioredis
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.domain.approvals import ApprovalStore
from app.domain.events import EventStore
from app.domain.sessions import SessionStateStore
from app.domain.usage import MeteringRecorder, price_table
from app.engine.gateway.breakers import BreakerPolicy, MemoryBreaker, RedisBreaker
from app.engine.gateway.cache import CacheStore, TenantCache
from app.engine.gateway.candidates import build_candidates
from app.engine.gateway.faults import inject_faults
from app.engine.gateway.outbound import ProviderLimiter
from app.engine.gateway.protocols import BreakerLike, LimiterLike, MeterLike
from app.engine.gateway.resilience import RetryPolicy
from app.engine.gateway.router import AegisGateway
from app.engine.gateway.routing import Candidate, parse_routes, unique_candidates
from app.engine.gateway.tenancy import validate_tenant_id
from app.engine.runtime.runtime import AgentRuntime


@dataclass(frozen=True, slots=True)
class GatewayParts:
    """进程级共享件：lifespan 建一次，挂在 app.state；worker 进程同样用它。"""

    settings: Settings
    routes: dict[str, list[Candidate]]
    models: dict[Candidate, BaseChatModel]
    breaker: BreakerLike
    limiter: LimiterLike
    tenant_limiter: LimiterLike
    cache_store: CacheStore | None
    meter: MeterLike | None
    retry_policy: RetryPolicy


@dataclass(frozen=True, slots=True)
class RuntimeParts:
    """运行时的进程级共享件（M2.3；M2.7 加审批单）：事实源 / 状态机 / 审批单各一份，checkpointer 由 lifespan 开池后交来。"""

    events: EventStore
    sessions: SessionStateStore
    approvals: ApprovalStore
    checkpointer: BaseCheckpointSaver


def build_gateway_parts(
    settings: Settings,
    *,
    http_client: httpx2.AsyncClient,
    redis: aioredis.Redis | None,
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> GatewayParts:
    """配置 → 共享件。redis=None 时熔断退化为进程内状态机、缓存关闭；session_factory=None 时不记账。"""
    routes = parse_routes(settings.model_routes, set(settings.providers))
    models = inject_faults(
        build_candidates(
            settings, unique_candidates(routes), http_async_client=http_client
        ),
        rate=settings.fault_injection_rate,
        targets=settings.fault_injection_targets,
        mode=settings.fault_injection_mode,
        hang_s=settings.fault_injection_hang_s,
    )
    policy = BreakerPolicy(
        fail_max=settings.breaker_fail_max,
        reset_timeout=settings.breaker_reset_timeout_s,
        probe_ttl=settings.breaker_probe_ttl_s,
        fail_window=settings.breaker_fail_window_s,
        probe_interval=settings.breaker_probe_interval_s,
    )
    breaker: BreakerLike = (
        RedisBreaker(redis, policy) if redis is not None else MemoryBreaker(policy)
    )
    cache_store = (
        CacheStore(
            redis,
            ttl_seconds=settings.cache_ttl_seconds,
            probe_interval=settings.cache_probe_interval_s,
        )
        if redis is not None and settings.cache_ttl_seconds > 0
        else None
    )
    meter = (
        MeteringRecorder(session_factory, price_table(settings.model_prices))
        if session_factory is not None
        else None
    )
    # 供应商闸：key = provider，无界（供应商就那几个）
    limiter = ProviderLimiter(
        rate=settings.outbound_rate_per_s,
        burst=settings.outbound_burst,
        max_wait=settings.outbound_max_wait_s,
    )
    # 租户桶（ADR-008 A′）：key = tenant_id，有界 LRU；等待上限沿用出站闸的
    tenant_limiter = ProviderLimiter(
        rate=settings.tenant_rate_per_s,
        burst=settings.tenant_burst,
        max_wait=settings.outbound_max_wait_s,
        max_keys=settings.tenant_limiter_max_keys,
    )
    return GatewayParts(
        settings=settings,
        routes=routes,
        models=models,
        breaker=breaker,
        limiter=limiter,
        tenant_limiter=tenant_limiter,
        cache_store=cache_store,
        meter=meter,
        retry_policy=RetryPolicy(),
    )


def gateway_for(parts: GatewayParts, tenant_id: str) -> AegisGateway:
    """按租户装配一个网关实例（每请求一个，构造开销只是 pydantic 校验）。"""
    tenant_id = validate_tenant_id(tenant_id)  # 入口守卫：非法身份不许碰任何共享件
    settings = parts.settings
    return AegisGateway(
        tenant_id=tenant_id,
        routes=parts.routes,
        models=parts.models,
        breaker=parts.breaker,
        limiter=parts.limiter,
        tenant_limiter=parts.tenant_limiter,
        reply_cache=(
            TenantCache(parts.cache_store, tenant_id)
            if parts.cache_store is not None
            else None
        ),
        meter=parts.meter,
        budget_resolver=None,  # M3 租户目录接 tenants 表；M1 静态配置兜底
        monthly_token_budget=settings.tenant_monthly_token_budget,
        request_token_budget=settings.request_token_budget,
        retry_policy=parts.retry_policy,
    )


def build_runtime_parts(
    session_factory: async_sessionmaker[AsyncSession],
    checkpointer: BaseCheckpointSaver,
) -> RuntimeParts:
    """账本会话工厂 → 事件 / 会话 / 审批单存取件（domain 实现 engine 协议靠结构匹配，两边互不 import）。"""
    return RuntimeParts(
        events=EventStore(session_factory),
        sessions=SessionStateStore(session_factory),
        approvals=ApprovalStore(session_factory),
        checkpointer=checkpointer,
    )


def build_runtime(
    gateway_parts: GatewayParts, runtime_parts: RuntimeParts
) -> AgentRuntime:
    """进程级 AgentRuntime：网关按租户由闭包装配（每 run 一个租户绑定的网关实例进图）。批准后前置校验 M3 注入（此处缺席 = 全通过）。"""
    return AgentRuntime(
        gateway_for=lambda tenant_id: gateway_for(gateway_parts, tenant_id),
        events=runtime_parts.events,
        sessions=runtime_parts.sessions,
        checkpointer=runtime_parts.checkpointer,
        approvals=runtime_parts.approvals,
    )
