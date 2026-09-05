"""入站限流原语（C14；ADR-010）：Redis 固定窗计数、即问即答 429、未挂载 fail-open、Redis 故障降级本地窗且粘滞。

与出站闸的分工：出站闸保护供应商，可以短排队（请求已在处理中，平滑突发优于失败）；入站闸保护自己，
排队等于替攻击者保管请求（连接/内存/任务全占着），所以即问即答：超限立刻 429 + Retry-After。
两本账数的不是同一种东西：入站数 HTTP 请求（门口扣 1），出站数上游调用（每次尝试扣 1），比例不可互推。

现成件：只借 fastapi-limiter 0.1.6 的固定窗 Lua（抄入本模块，依赖移除）。不借它的装配形态——
RateLimiter 依赖按 route_index 扫 app.routes 拼键，FastAPI 0.141 经 include_router 挂载的端点一碰即崩（探针⑲），
键里的路由序号随路由增删漂移，默认 identifier 无条件信 X-Forwarded-For；它的 init 把客户端/前缀/sha 存成类属性
（进程级全局态），与本仓"真实依赖只在组合根聚合、其余靠注入"相悖。
本仓形态：InboundLimiter 是进程级共享件，lifespan 建一个挂在 app.state.inbound_limiter；rate_limit(scope, times, seconds)
依赖在请求时从 request.app.state 取它，没挂 = fail-open。键 = {prefix}:{identity}:{scope}，identity ∈ tenant:/user:/ip: 三族，
稳定、可读、在 Redis 里能按租户 SCAN。阈值由挂载处给（M3 各端点常量），身份函数由挂载处注入（M3 接 JWT）。

fail-open 两层：未挂载（无 Redis 配置 / 测试）直接放行——限流是护栏不是核心功能的前置依赖；
运行期 Redis 触点异常 → 降级为进程内固定窗（Lua 算法的 Python 直译）且粘滞 probe_interval，
每副本各数各的（降级期口径 = times × 副本数），恢复时本地窗作废——与熔断/缓存同款粘滞规则（ADR-007 决策 6）。
话术例外（计划 §7 登记）：429 串与两条告警串住本模块，core 不得 import engine。
"""

import math
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Literal

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, status
from redis.exceptions import NoScriptError

from app.core.logs import get_logger

logger = get_logger(__name__)

PREFIX = "aegis-limit"
DEFAULT_DETAIL = "请求过于频繁，请稍后再试"
DEFAULT_PROBE_INTERVAL_S = 5.0
LOCAL_WINDOWS_MAX = 10_000  # 降级期本地窗 LRU 上限：身份含 ip，基数无界
SCOPE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # 键段：不许冒号与空串
RATE_LIMIT_PARAMS_INVALID = "入站限流参数非法：scope 只允许 [A-Za-z0-9_.-]{1,64}，times 与 seconds 须为 ≥ 1 的整数"
LIMITS_PROBE_INTERVAL_INVALID = "入站限流参数非法：probe_interval 须为有限正数"
LOG_LIMITS_DEGRADED = "入站限流存储不可用，降级为进程内固定窗（粘滞 probe_interval；降级期各副本各数各的）"
LOG_LIMITS_RECOVERED = "入站限流存储恢复，切回共享计数"

# 固定窗计数，抄自 fastapi-limiter 0.1.6 FastAPILimiter.lua_script，只改一处：已满分支给 PTTL 加下限 1——
# 键在过期那一毫秒 PTTL 为 0，原版会被调用方当成放行（0 = 放行哨兵）。
# KEYS[1] 计数键，ARGV[1] 上限，ARGV[2] 窗口毫秒；返回 0 = 放行，>0 = 本窗剩余毫秒。
# 三个分支：键不存在 → SET PX 开窗；未满 → INCR 不续期；已满 → 返回剩余毫秒。
FIXED_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire_time = ARGV[2]

local current = tonumber(redis.call('get', key) or "0")
if current > 0 then
  if current + 1 > limit then
    return math.max(1, redis.call("PTTL", key))
  else
    redis.call("INCR", key)
    return 0
  end
else
  redis.call("SET", key, 1, "px", expire_time)
  return 0
end
"""

_monotonic = time.monotonic  # 测试接缝：降级粘滞窗与本地窗推进"现在"而不真等

Identify = Callable[[Request], Awaitable[str]]
IdentityKind = Literal["tenant", "user", "ip"]


# ---------------------------------------------------------------- 身份


def identity(kind: IdentityKind, value: str) -> str:
    """键身份段的唯一拼装点：M3 的 JWT 身份函数返回 identity("tenant", principal.tenant_id)。"""
    return f"{kind}:{value}"


async def client_ip(request: Request) -> str:
    """M1 默认身份：直连对端地址。反代场景由部署面的 ProxyHeaders 中间件按可信代理改写 client，
    这里不信任 X-Forwarded-For（探针⑲：库默认无条件信它，每个请求都能自选身份）。"""
    return identity("ip", request.client.host if request.client else "unknown")


def retry_after_seconds(remaining_ms: int) -> int:
    """HTTP Retry-After 是整数秒且 ≥ 1：剩余毫秒向上取整，异常值也给 1。"""
    return max(1, math.ceil(remaining_ms / 1000))


# ---------------------------------------------------------------- 本地固定窗（降级形态）


class _LocalWindows:
    """Lua 三分支的 Python 直译（对照着读），有界 LRU；只管计数，不认识 Redis 与降级态。"""

    def __init__(self) -> None:
        self.windows: OrderedDict[str, tuple[int, float]] = (
            OrderedDict()
        )  # key -> (计数, 窗末时刻)

    def clear(self) -> None:
        self.windows.clear()

    def check(self, key: str, times: int, ms: int) -> int:
        """返回 0 = 放行；>0 = 超限，值为本窗剩余毫秒（与 Lua 返回同义）。"""
        now = _monotonic()
        count, ends_at = self.windows.get(key, (0, 0.0))
        if now >= ends_at:  # 无窗或窗已过期：开新窗（Lua 的 SET ... PX 分支）
            self.windows[key] = (1, now + ms / 1000)
            self.windows.move_to_end(key)
            if len(self.windows) > LOCAL_WINDOWS_MAX:
                self.windows.popitem(last=False)
            return 0
        # 已满（Lua 的 PTTL 分支）：ends_at > now 故 ceil ≥ 1，绝不返回放行哨兵 0；
        # 被拒也算"最近用过"：淘汰热键等于给刷子重开新窗
        if count + 1 > times:
            self.windows.move_to_end(key)
            return math.ceil((ends_at - now) * 1000)
        self.windows[key] = (count + 1, ends_at)  # 未满（Lua 的 INCR 分支）：不续期
        self.windows.move_to_end(key)
        return 0


# ---------------------------------------------------------------- 共享件


class InboundLimiter:
    """进程级共享件：Redis 固定窗计数 + 降级本地窗；lifespan 建一个挂到 app.state，与熔断/缓存共用同一个快速失败客户端。

    粘滞规则与熔断/缓存同款：任一 Redis 触点异常即降级，probe_interval 内不碰 Redis；顺路探针领取即续窗
    （检查与写入之间无 await）；探针失败顺延；只有被指派的那次探针成功才恢复（健康期发出、降级后才返回的
    迟到成功不算数），恢复时清空本地窗（旧窗作废，下次再降级从空窗开始，与 Lua 冷启动语义一致）。
    """

    def __init__(
        self, redis: aioredis.Redis, *, probe_interval: float = DEFAULT_PROBE_INTERVAL_S
    ) -> None:
        if not (math.isfinite(probe_interval) and probe_interval > 0):
            raise ValueError(LIMITS_PROBE_INTERVAL_INVALID)
        self._redis = redis
        self._sha: str | None = None  # None = 脚本尚未预载（start 时 Redis 不可达）
        self._probe_interval = probe_interval
        self._degraded = False
        self._degraded_until = 0.0  # 单调时刻：此前不碰 Redis
        self._local = _LocalWindows()

    @property
    def degraded(self) -> bool:
        return self._degraded

    @staticmethod
    def key(identity_: str, scope: str) -> str:
        return f"{PREFIX}:{identity_}:{scope}"

    # ---------------------------------------------------------------- Redis 触点

    async def _load(self) -> None:
        self._sha = await self._redis.script_load(FIXED_WINDOW_LUA)

    async def _eval(self, key: str, times: int, ms: int) -> int:
        return int(await self._redis.evalsha(self._sha, 1, key, str(times), str(ms)))

    async def _shared_check(self, key: str, times: int, ms: int) -> int:
        if self._sha is None:  # start 时 Redis 不可达：由首个成功探针加载脚本
            await self._load()
        try:
            return await self._eval(key, times, ms)
        except NoScriptError:  # Redis 重启 / SCRIPT FLUSH 后脚本缓存失效：重载再试一次
            await self._load()
            return await self._eval(key, times, ms)

    async def start(self) -> None:
        """lifespan 调一次：预载 Lua。Redis 此刻不可达不炸进程：直接进入降级态，脚本由首个成功探针加载
        （探针⑳：坏 sha 与从未加载都是 NoScriptError，但 sha=None 是 DataError，故 None 在 _shared_check 单独判）。"""
        try:
            await self._load()
        except Exception:
            self._note_degraded()
            logger.warning(LOG_LIMITS_DEGRADED, op="script_load", exc_info=True)

    # ---------------------------------------------------------------- 降级态

    def _probe_due(self) -> bool:
        """降级期是否轮到放顺路探针：领取即续窗（检查与写入之间无 await）。"""
        now = _monotonic()
        if now < self._degraded_until:
            return False
        self._degraded_until = now + self._probe_interval
        return True

    def _note_degraded(self) -> bool:
        """记降级并顺延窗口；首次降级返回 True，由调用方在 except 块内记日志（带 traceback）。"""
        self._degraded_until = _monotonic() + self._probe_interval
        if self._degraded:
            return False
        self._degraded = True
        return True

    def _note_recovered(self) -> None:
        self._degraded = False
        self._local.clear()
        logger.warning(LOG_LIMITS_RECOVERED, op="evalsha")

    async def check(self, key: str, times: int, ms: int) -> int:
        """一次取额：0 = 放行；>0 = 超限，值为本窗剩余毫秒。降级期由本地窗裁决；被指派的探针成功即切回共享计数。"""
        probing = False
        if self._degraded:
            if not self._probe_due():
                return self._local.check(key, times, ms)
            probing = True
        try:
            remaining = await self._shared_check(key, times, ms)
        except Exception:
            if self._note_degraded():
                logger.warning(
                    LOG_LIMITS_DEGRADED, op="evalsha", key=key, exc_info=True
                )
            return self._local.check(key, times, ms)
        if probing:
            self._note_recovered()
        return remaining


def limiter_of(app: FastAPI) -> InboundLimiter | None:
    """从应用状态取限流器；未挂载（无 Redis 配置 / 测试 / 已关停）= None = fail-open。"""
    return getattr(app.state, "inbound_limiter", None)


# ---------------------------------------------------------------- 依赖工厂


def rate_limit(
    scope: str,
    times: int,
    seconds: int,
    *,
    identify: Identify = client_ip,
    detail: str = DEFAULT_DETAIL,
) -> Callable[[Request], Awaitable[None]]:
    """频率闸依赖：`dependencies=[Depends(auth), Depends(rate_limit("chat", 20, 60, identify=...))]`。

    依赖按列表顺序解析（探针⑳）：身份函数可读前面的认证依赖放进 request.state 的主体；
    429 抛出后后续依赖与 handler 都不执行——超限探测不出单据存在性。
    参数在挂载时校验（启动即炸）：times=0 会被 Lua 的 SET 分支放过首个请求（探针⑳）；
    seconds 必须是整数——浮点会让 PX 参数变成 "60000.0"，Redis 拒收，每个请求都被当成存储故障。
    """
    if (
        not SCOPE_PATTERN.fullmatch(scope)
        or type(times) is not int
        or type(seconds) is not int
        or times < 1
        or seconds < 1
    ):
        raise ValueError(RATE_LIMIT_PARAMS_INVALID)
    milliseconds = seconds * 1000

    async def dependency(request: Request) -> None:
        limiter = limiter_of(request.app)
        if limiter is None:
            return  # 未挂载：fail-open
        key = limiter.key(await identify(request), scope)
        remaining = await limiter.check(key, times, milliseconds)
        if remaining > 0:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
                headers={"Retry-After": str(retry_after_seconds(remaining))},
            )

    return dependency
