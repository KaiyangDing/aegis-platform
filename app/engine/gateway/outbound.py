"""出站闸：每 provider 一个 InMemoryRateLimiter，限时取令牌（ADR-008 出站部分）。

四约束（登记表）：进程内（多副本口径 = 全局 / 副本数）；按次数不按 token；初始 0 令牌——这里预填 burst
恢复 v1 的冷启动满桶；acquire 内嵌于 BaseChatModel.astream——所以不挂候选 rate_limiter=，由候选环经
complete_with_retry 的 acquire 缝在首块计时器之外显式限时取令牌（排队不冤枉供应商、不进熔断账）。
库无 max_wait/不换路/无等待预估：等待预算与"等不到就不等"的预判（v1 wait_take 语义）都是自建的——
按桶的补给模型算出下一枚令牌何时够，超出预算直接放弃，不白烧预算；预算内按 check_every 轮询、
到期前至少再试一次（不受库 blocking=True 的轮询粒度地板约束）。
"""

import asyncio
import time

from langchain_core.rate_limiters import InMemoryRateLimiter

from app.engine.gateway import utterances as u

_monotonic = time.monotonic  # 测试接缝


class ProviderLimiter:
    """实现 LimiterLike。key = provider。"""

    def __init__(
        self,
        *,
        rate: float,
        burst: float,
        max_wait: float,
        check_every_n_seconds: float = 0.02,
    ) -> None:
        # burst < 1 时库永远发不出令牌（要求 available_tokens >= 1）：启动即炸，别做成永久拒绝
        if rate <= 0 or burst < 1 or max_wait < 0 or check_every_n_seconds <= 0:
            raise ValueError(u.OUTBOUND_LIMITER_INVALID)
        self._rate = rate
        self._burst = burst
        self._max_wait = max_wait
        self._check_every = check_every_n_seconds
        self._buckets: dict[str, InMemoryRateLimiter] = {}

    def _bucket(self, key: str) -> InMemoryRateLimiter:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = InMemoryRateLimiter(
                requests_per_second=self._rate,
                check_every_n_seconds=self._check_every,
                max_bucket_size=self._burst,
            )
            bucket.available_tokens = (
                self._burst
            )  # 冷启动满桶：允许合理的开局突发（v1 语义）
            self._buckets[key] = bucket
        return bucket

    @staticmethod
    def _seconds_until_token(bucket: InMemoryRateLimiter) -> float:
        """按库的补给模型（elapsed × rps，封顶桶容量）估算下一枚令牌何时够。"""
        now = _monotonic()
        elapsed = now - (bucket.last if bucket.last is not None else now)
        projected = min(
            bucket.available_tokens + elapsed * bucket.requests_per_second,
            bucket.max_bucket_size,
        )
        return max(0.0, (1.0 - projected) / bucket.requests_per_second)

    async def acquire(self, key: str, max_wait: float | None) -> bool:
        bucket = self._bucket(key)
        if await bucket.aacquire(blocking=False):
            return True
        wait = self._max_wait if max_wait is None else min(self._max_wait, max_wait)
        if wait <= 0 or self._seconds_until_token(bucket) > wait:
            return False  # 预算内等不到：不排队，候选环换下一站（不进熔断账）
        deadline = _monotonic() + wait
        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(self._check_every, remaining))
            if await bucket.aacquire(blocking=False):
                return True
