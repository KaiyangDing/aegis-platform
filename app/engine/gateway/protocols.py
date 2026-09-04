"""候选环依赖的协议：熔断入口判定与出站闸。方法签名只用内建类型（M1 自律②）。

替身先行（M1.4a 的 Stub），真件随 M1.4b 立（自研三键 RedisBreaker/MemoryBreaker，ADR-007；ProviderLimiter 包装 InMemoryRateLimiter）。
key 一律是候选的 `provider:model`（熔断粒度）或 provider（出站闸粒度），由调用方决定。
"""

from typing import Literal, Protocol, runtime_checkable

Decision = Literal["allow", "probe", "deny"]


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
