"""网关异常契约：三组六类 + 内部 ProviderError 家族 + 翻译表 + 消毒。

L2 只见六类：
  请求级（首块前，可整体降级）：GatewayExhausted / BudgetExceeded / TenantQuotaExceeded / GatewayOverloadedError
  请求级（确定性拒绝，不降级）：GatewayRejected
  流级（首块后，进恢复语义）：GatewayStreamInterrupted
ProviderError 家族只在网关内部流转（重试白名单、熔断记账、换路判定），永远不穿出网关。

翻译源（v1 是 httpx 状态码，v2 是 SDK 异常对象）：
  openai.APIStatusError 按 status_code 分段；APITimeoutError / APIConnectionError 按 __cause__ 链
  区分本地连接池排队（PoolTimeout）与真超时；流内 error 事件是无状态码的裸 openai.APIError；
  langchain-openai 的 StreamChunkTimeoutError 是 TimeoutError 子类。
未知异常返回 None 让调用方裸抛——把编程错误伪装成上游故障会藏起 bug。
"""

import json
import re
import time
from email.utils import parsedate_to_datetime

import httpx2
import openai

from app.engine.gateway import utterances as u

# ---------------------------------------------------------------- 异常类树


class GatewayError(Exception):
    """网关所有错误的基类。L2 绝不 except 它——ProviderError 泄漏是 bug 信号，必须裸炸。"""


class ProviderError(GatewayError):
    """上游/传输层错误的内部家族。消息统一带 [provider] 前缀。"""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RateLimitedError(ProviderError):
    """429 被限流。retry_after 取自响应头（秒），解析不出为 None。"""

    def __init__(
        self, provider: str, message: str, retry_after: float | None = None
    ) -> None:
        super().__init__(provider, message)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """连接或读取超时。注意：请求可能已在上游执行（重复计费风险）。"""


class ProviderServerError(ProviderError):
    """5xx / 连接失败 / 流内错误——上游故障，可重试。"""


class BadRequestError(ProviderError):
    """4xx（除 429/401/403）与 501——请求本身有问题，重试无意义。"""


class AuthError(ProviderError):
    """401/403——该修配置，不该重试。"""


class GatewayOverloadedError(GatewayError):
    """本地连接池排队超时。三个"不"：不记熔断账（供应商无辜）、不重试（加剧争抢）、不换路（共用一个池）。"""


class GatewayExhausted(GatewayError):
    """重试与 fallback 全部用尽（首块之前）。暂时不可用，降级合理。"""


class BudgetExceeded(GatewayError):
    """token 预算闸门触发（月度或单请求）。"""


class TenantQuotaExceeded(GatewayError):
    """租户级出站配额耗尽——换供应商无解。"""


class GatewayRejected(GatewayError):
    """全部候选均为确定性拒绝且无暂时性因素。重试和降级都无意义，L2 不走兜底话术。"""


class GatewayStreamInterrupted(GatewayError):
    """流已开始后中断。原始死因保留在 __cause__ 上；绝不重放。"""


# ---------------------------------------------------------------- 消毒与 Retry-After

_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def sanitize_error_text(text: str, limit: int = 200) -> str:
    """所有上游错误文本进异常前必过：先打码再截断。

    401 响应体惯例回显 key 片段；异常文本会进日志与 __cause__ 链，是展示层打码罩不住的旁路。
    先打码再截断，避免截断点落在 key 中间留下半截明文。
    """
    return _KEY_PATTERN.sub(u.KEY_MASK, text)[:limit]


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After 双格式：秒数或 HTTP-date；解析失败退化 None 走指数退避。外部输入永不裸穿 ValueError。"""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    return max(0.0, when.timestamp() - time.time())


def retry_after_from(exc: BaseException) -> float | None:
    """从带 response 的 SDK 异常读 Retry-After；retry-after-ms 优先（SDK 同样约定）。"""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    return parse_retry_after(headers.get("retry-after"))


# ---------------------------------------------------------------- 翻译表


def _root_cause(exc: BaseException) -> BaseException:
    seen: set[int] = set()
    cur = exc
    while cur.__cause__ is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.__cause__
    return cur


def _has_cause(exc: BaseException, cls: type[BaseException]) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, cls):
            return True
        seen.add(id(cur))
        cur = cur.__cause__
    return False


def _snippet(exc: openai.APIError) -> str:
    """结构化字段优先：body 是 dict 就压成紧凑 JSON；否则退回 message。"""
    body = exc.body
    if isinstance(body, dict):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    elif isinstance(body, str) and body:
        raw = body
    else:
        raw = exc.message
    return sanitize_error_text(raw)


def _describe(exc: BaseException) -> str:
    return sanitize_error_text(repr(_root_cause(exc)))


def classify(provider: str, exc: BaseException) -> GatewayError | None:
    """把上游/传输层异常翻译成网关内部家族；未知异常返回 None（调用方裸抛）。

    唯一的非 ProviderError 输出是 GatewayOverloadedError（本地池排队），它直接穿出候选环。
    """
    if isinstance(exc, ProviderError | GatewayOverloadedError):
        return exc

    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        snippet = _snippet(exc)
        if status == 429:
            return RateLimitedError(
                provider, snippet, retry_after=retry_after_from(exc)
            )
        if status in (401, 403):
            return AuthError(provider, snippet)
        if status == 408:
            return ProviderTimeoutError(
                provider, u.HTTP_STATUS.format(status=status, snippet=snippet)
            )
        if status == 501:
            return BadRequestError(
                provider, u.HTTP_STATUS.format(status=status, snippet=snippet)
            )
        if status >= 500:
            return ProviderServerError(
                provider, u.HTTP_STATUS.format(status=status, snippet=snippet)
            )
        return BadRequestError(
            provider, u.HTTP_STATUS.format(status=status, snippet=snippet)
        )

    if isinstance(exc, openai.APITimeoutError):
        if _has_cause(exc, httpx2.PoolTimeout):
            return GatewayOverloadedError(
                f"[{provider}] {u.POOL_TIMEOUT.format(detail=_describe(exc))}"
            )
        return ProviderTimeoutError(provider, u.TIMEOUT.format(detail=_describe(exc)))

    if isinstance(exc, openai.APIConnectionError):
        return ProviderServerError(
            provider, u.CONNECT_FAILED.format(detail=_describe(exc))
        )

    if isinstance(exc, openai.APIError):
        code = exc.code or exc.type or "?"
        return ProviderServerError(
            provider, u.STREAM_ERROR_EVENT.format(code=code, detail=_snippet(exc))
        )

    if isinstance(exc, httpx2.PoolTimeout):
        return GatewayOverloadedError(
            f"[{provider}] {u.POOL_TIMEOUT.format(detail=_describe(exc))}"
        )

    if isinstance(exc, httpx2.TimeoutException | TimeoutError):
        return ProviderTimeoutError(provider, u.TIMEOUT.format(detail=_describe(exc)))

    if isinstance(exc, httpx2.TransportError):
        return ProviderServerError(
            provider, u.CONNECT_FAILED.format(detail=_describe(exc))
        )

    return None
