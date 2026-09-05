"""候选环依赖的跨包协议：熔断入口判定、出站闸、计量、预算源。方法签名只用内建类型（M1 自律②）。

真件：自研三键 RedisBreaker/MemoryBreaker（ADR-007）、ProviderLimiter 包装 InMemoryRateLimiter（ADR-008）、
domain/usage.py 的 MeteringRecorder 靠结构匹配实现 MeterLike（不 import engine；deps.py 是唯一同时 import 二者的模块）。
key 一律是候选的 `provider:model`（熔断粒度）、provider（出站闸粒度）或 tenant_id（租户桶），由调用方决定。
缓存协议 CacheLike 不在此：它的签名带框架消息类型，且实现就在 engine 内（cache.py）。
"""

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol, runtime_checkable

Decision = Literal["allow", "probe", "deny"]

# 月度预算事实源：租户 → 预算 token 数；None 由调用方视为"读挂 fail-open"。
# M1 组合根传 None（静态配置兜底）；M3 租户目录接上 tenants 表。
BudgetResolver = Callable[[str], Awaitable[int | None]]


@runtime_checkable
class BreakerLike(Protocol):
    """入口判定 + 事后上报。probe=True 的上报同时归还试探锁；无裁决的结局只调 release_probe。"""

    async def allow(self, key: str) -> Decision: ...

    async def report_success(self, key: str, *, probe: bool) -> None: ...

    async def report_failure(self, key: str, *, probe: bool) -> None: ...

    async def release_probe(self, key: str) -> None: ...


@runtime_checkable
class LimiterLike(Protocol):
    """限时取令牌：max_wait=None 表示不受 deadline 约束（闸自有上限）；返回是否取得。"""

    async def acquire(self, key: str, max_wait: float | None) -> bool: ...


@runtime_checkable
class MeterLike(Protocol):
    """记账员：一调用一行；month_spend 是预算闸门的读路径。两个方法都可能抛（DB 挂），由候选环 fail-open 兜住。"""

    async def record(
        self,
        *,
        tenant_id: str,
        request_id: str,
        session_id: str | None,
        tier: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
        usage_missing: bool,
    ) -> None: ...

    async def month_spend(self, tenant_id: str) -> int: ...
