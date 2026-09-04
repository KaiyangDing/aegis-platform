"""熔断：自研三键状态机，异步原生（ADR-007）。

每个候选 key（provider:model）三把键，TTL 即状态迁移，无后台任务：
  fails   连续失败计数；成功清零；自带窗口 TTL（fail_window：久未失败自动遗忘）
  open    存在 = open 态；TTL（reset_timeout）过期即进入"半开机会"
  probe   半开试探令牌（租约）；SET NX + TTL（probe_ttl）：全集群同一时刻只发一枚，持有方崩溃后到期自愈
原子性：读前态 + INCR + PEXPIRE 走 MULTI/EXEC；重复 SET open 幂等；试探互斥靠 SET NX。不需要 Lua；
allow 常态一次往返、试探路径三次，上报至多两次。
半开 = open 已过期且失败账仍在。失败账自身的过期是第二条自愈路径：半开期间试探方消失、fail_window 内再无任何裁决，
则整体遗忘回 closed；参数校验保证 open 到期后首枚令牌的自愈早于遗忘（fail_window > reset_timeout + probe_ttl）。
令牌是租约不是硬锁：试探的流长不设上限（ADR-006），流长超过 probe_ttl 时第二个试探可能并存，
旧持有者的归还也可能释放新令牌——后果有界（至多多一次试探），ADR-007 记账。

四个方法与 BreakerLike 协议同形：allow 入口判定 → allow/probe/deny；report_success/report_failure 事后上报；
release_probe 归还无裁决的试探令牌。进账谓词（只有 5xx/超时上报失败）由候选环持有，本模块只认 key。
迟到上报（跳闸前已在飞的请求）照常入账：失败续期 open，成功闭合；只有真实迁移才记日志。

RedisBreaker = 共享态主路 + 进程内备胎（同一状态机的 MemoryBreaker，每次上报双写，始终带着本进程的近期失败史）。
Redis 任一触点异常即降级，且降级粘滞：之后 probe_interval 内判定与上报只走备胎、不碰 Redis；顺路探针只在 allow 领
（上报路径降级期直接跳过 Redis，否则多个触点会互相续期、恢复时机不可预测），触点失败一律顺延窗口，探针成功即切回并记日志。
降级期承诺退化：全集群单探针失效（每副本各探一个）、备胎只看得见本进程的上报；不降级的是熔断能力本身。
"""

import math
import time
from dataclasses import dataclass

import redis.asyncio as aioredis

from app.core.logs import get_logger
from app.engine.gateway import utterances as u
from app.engine.gateway.protocols import Decision

logger = get_logger(__name__)

KEY_PREFIX = "aegis:cb"
_monotonic = time.monotonic  # 测试接缝：内存版与降级窗推进"现在"而不真等


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    fail_max: int = 5  # 连续入账失败达到即开路
    reset_timeout: float = 30.0  # open 持续时长；到期进入半开机会
    # 试探令牌租约：覆盖闸内等待 + 首块窗 + 典型流长；持有方崩溃后到期自愈
    probe_ttl: float = 120.0
    # 失败计数的遗忘窗：这么久没再失败就从零数；也是半开无裁决时的最长寿命
    fail_window: float = 300.0
    # 降级期顺路探针间隔：Redis 断了之后每隔这么久才再碰一次（只 RedisBreaker 用）
    probe_interval: float = 5.0

    def __post_init__(self) -> None:
        durations = (
            self.reset_timeout,
            self.probe_ttl,
            self.fail_window,
            self.probe_interval,
        )
        if (
            self.fail_max < 1
            or not all(math.isfinite(d) and d > 0 for d in durations)
            or self.fail_window <= self.reset_timeout + self.probe_ttl
        ):
            raise ValueError(u.BREAKER_POLICY_INVALID)


def _ms(seconds: float) -> int:
    return max(1, round(seconds * 1000))


def _log_trip(key: str, *, old: str, failures: int, store: str) -> None:
    logger.warning(
        u.LOG_BREAKER_STATE_CHANGED,
        breaker=key,
        old=old,
        new="open",
        failures=failures,
        store=store,
    )


def _log_close(key: str, *, old: str, store: str) -> None:
    logger.warning(
        u.LOG_BREAKER_STATE_CHANGED, breaker=key, old=old, new="closed", store=store
    )


@dataclass(slots=True)
class _Slot:
    """三键的进程内镜像：fails/open/probe 各自的到期时刻，0 表示不存在。"""

    fails: int = 0
    fails_until: float = 0.0
    open_until: float = 0.0
    probe_until: float = 0.0


class MemoryBreaker:
    """实现 BreakerLike：与 RedisBreaker 同一状态机的进程内实现（测试 / 无 Redis 的开发环境 / 降级备胎）。"""

    def __init__(self, policy: BreakerPolicy, *, quiet: bool = False) -> None:
        self._p = policy
        self.quiet = quiet  # 作备胎且主路健康时不喊日志：本进程的局部视野不代表集群
        self._slots: dict[str, _Slot] = {}

    def _slot(self, key: str) -> _Slot:
        return self._slots.setdefault(key, _Slot())

    def _fails(self, slot: _Slot, now: float) -> int:
        return slot.fails if slot.fails_until > now else 0

    async def allow(self, key: str) -> Decision:
        slot, now = self._slot(key), _monotonic()
        if slot.open_until > now:
            return "deny"
        if self._fails(slot, now) < self._p.fail_max:
            return "allow"
        if slot.probe_until > now:
            return "deny"  # 令牌互斥：只放一个试探（检查与写入之间无 await，事件循环内天然原子）
        slot.probe_until = now + self._p.probe_ttl
        return "probe"

    async def report_failure(self, key: str, *, probe: bool) -> None:
        slot, now = self._slot(key), _monotonic()
        was_open = slot.open_until > now
        slot.fails = self._fails(slot, now) + 1
        slot.fails_until = now + self._p.fail_window
        if slot.fails < self._p.fail_max:
            if probe:
                slot.probe_until = 0.0
            return
        slot.open_until = now + self._p.reset_timeout
        slot.probe_until = 0.0
        if not was_open and not self.quiet:
            old = "half-open" if probe or slot.fails > self._p.fail_max else "closed"
            _log_trip(key, old=old, failures=slot.fails, store="memory")

    async def report_success(self, key: str, *, probe: bool) -> None:
        slot, now = self._slots.pop(key, None), _monotonic()
        if slot is None or self.quiet:
            return
        if slot.open_until > now:
            _log_close(key, old="open", store="memory")
        elif self._fails(slot, now) >= self._p.fail_max:
            _log_close(key, old="half-open", store="memory")

    async def release_probe(self, key: str) -> None:
        self._slot(key).probe_until = 0.0

    async def state(self, key: str) -> str:
        slot, now = self._slot(key), _monotonic()
        if slot.open_until > now:
            return "open"
        return "half-open" if self._fails(slot, now) >= self._p.fail_max else "closed"


class RedisBreaker:
    """实现 BreakerLike：Redis 共享态主路，API 与 worker 多副本同视野；断连时备胎接手。"""

    def __init__(self, client: aioredis.Redis, policy: BreakerPolicy) -> None:
        self._r = client
        self._p = policy
        # TTL 预先换算成毫秒：换算错误在构造期炸，不会被降级分支吞成"存储不可用"
        self._reset_ms = _ms(policy.reset_timeout)
        self._probe_ms = _ms(policy.probe_ttl)
        self._window_ms = _ms(policy.fail_window)
        # 常温备胎：双写保温，降级期才开口
        self._local = MemoryBreaker(policy, quiet=True)
        self._degraded = False
        self._degraded_until = 0.0  # 单调时刻：此前不碰 Redis

    @staticmethod
    def _keys(key: str) -> tuple[str, str, str]:
        base = f"{KEY_PREFIX}:{key}"
        return f"{base}:fails", f"{base}:open", f"{base}:probe"

    # ---------------------------------------------------------------- 降级与恢复

    def _probe_due(self) -> bool:
        """降级期是否轮到放顺路探针：领取即续窗，检查与写入之间无 await，并发调用者继续走备胎。"""
        now = _monotonic()
        if now < self._degraded_until:
            return False
        self._degraded_until = now + self._p.probe_interval
        return True

    def _note_degraded(self) -> bool:
        """记降级并顺延窗口；首次降级返回 True，由调用方在 except 块内记日志（带 traceback）。"""
        # 首次降级和探针失败都要顺延窗口，否则故障期会连环撞 Redis
        self._degraded_until = _monotonic() + self._p.probe_interval
        if self._degraded:
            return False
        self._degraded = True
        self._local.quiet = False  # 备胎开口：降级期它就是熔断器
        return True

    def _note_recovered(self, key: str) -> None:
        self._degraded = False
        self._local.quiet = True
        logger.warning(u.LOG_BREAKER_RECOVERED, breaker=key)

    # ---------------------------------------------------------------- 协议四方法

    async def allow(self, key: str) -> Decision:
        """放行判定：allow = 正常 / probe = 你是全集群唯一的试探 / deny = 秒拒。降级期由备胎裁决。"""
        if self._degraded and not self._probe_due():
            return await self._local.allow(key)
        try:
            decision = await self._allow_redis(key)
        except Exception:
            if self._note_degraded():
                logger.warning(
                    u.LOG_BREAKER_DEGRADED, breaker=key, op="allow", exc_info=True
                )
            return await self._local.allow(key)
        if self._degraded:
            self._note_recovered(key)
        return decision

    async def _allow_redis(self, key: str) -> Decision:
        fails_key, open_key, probe_key = self._keys(key)
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.exists(open_key)
            pipe.get(fails_key)
            is_open, fails = await pipe.execute()  # 一次往返读两键
        if is_open:
            return "deny"
        if int(fails or 0) < self._p.fail_max:
            return "allow"
        # open 已过期但失败账未清 → 半开：SET NX 抢全集群唯一的试探令牌
        if not await self._r.set(probe_key, "1", nx=True, px=self._probe_ms):
            return "deny"
        # 读与抢令牌之间别处可能刚试探失败重开（它的 DEL probe 让我们抢到了）：回查一次
        if await self._r.exists(open_key):
            await self._r.delete(probe_key)
            return "deny"
        return "probe"

    async def report_failure(self, key: str, *, probe: bool) -> None:
        await self._local.report_failure(key, probe=probe)  # 双写：备胎始终是热的
        if self._degraded:
            return  # 探针只在 allow 领：上报路径降级期不撞 Redis
        fails_key, open_key, probe_key = self._keys(key)
        try:
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.exists(open_key)
                pipe.incr(fails_key)
                pipe.pexpire(fails_key, self._window_ms)
                # MULTI/EXEC：读前态、计数、续窗一次往返且原子
                was_open, fails, _ = await pipe.execute()
            if fails < self._p.fail_max:
                if probe:  # 失败账已被遗忘的迟到试探：从 1 计，只归还令牌
                    await self._r.delete(probe_key)
                return
            async with self._r.pipeline(transaction=False) as pipe:
                pipe.set(open_key, "1", px=self._reset_ms)  # 已 open 则续期
                # 试探失败：令牌作废、重新计时；普通失败到阈值也顺手清
                pipe.delete(probe_key)
                await pipe.execute()
            if not was_open:  # 续期不是跳闸，不记
                old = "half-open" if probe or fails > self._p.fail_max else "closed"
                _log_trip(key, old=old, failures=fails, store="redis")
        except Exception:
            if self._note_degraded():
                logger.warning(
                    u.LOG_BREAKER_DEGRADED,
                    breaker=key,
                    op="report_failure",
                    exc_info=True,
                )

    async def report_success(self, key: str, *, probe: bool) -> None:
        """成功 = 彻底闭合：三把键一并清掉（probe 只为协议同形，记不记日志看前态）。"""
        await self._local.report_success(key, probe=probe)
        if self._degraded:
            return
        fails_key, open_key, probe_key = self._keys(key)
        try:
            async with self._r.pipeline(transaction=False) as pipe:
                pipe.exists(open_key)
                pipe.get(fails_key)
                pipe.delete(fails_key, open_key, probe_key)
                was_open, fails, _ = await pipe.execute()  # 一次往返：读前态并清三键
            if was_open:
                _log_close(key, old="open", store="redis")
            elif int(fails or 0) >= self._p.fail_max:
                _log_close(key, old="half-open", store="redis")
        except Exception:
            if self._note_degraded():
                logger.warning(
                    u.LOG_BREAKER_DEGRADED,
                    breaker=key,
                    op="report_success",
                    exc_info=True,
                )

    async def release_probe(self, key: str) -> None:
        """归还未获裁决的试探令牌——试探没打出去或结局不构成裁决（429/闸满/弃流/取消）时调用。"""
        await self._local.release_probe(key)
        if self._degraded:
            return
        try:
            await self._r.delete(self._keys(key)[2])
        except Exception:
            if self._note_degraded():
                logger.warning(
                    u.LOG_BREAKER_DEGRADED,
                    breaker=key,
                    op="release_probe",
                    exc_info=True,
                )

    async def state(self, key: str) -> str:
        """只读观测：closed / open / half-open（open 已过期但失败账未清）。不在协议内；降级期看备胎。"""
        if self._degraded:
            return await self._local.state(key)
        fails_key, open_key, _ = self._keys(key)
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.exists(open_key)
            pipe.get(fails_key)
            is_open, fails = await pipe.execute()
        if is_open:
            return "open"
        return "half-open" if int(fails or 0) >= self._p.fail_max else "closed"
