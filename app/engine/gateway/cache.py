"""租户前缀精确缓存：完全相同的请求直接回放上次的块序列（零上游成本）。ADR-009。

自建而不用框架缓存的三个理由：框架缓存对 astream 无效、对 ainvoke 内部流式无租户维度、无完整性守卫。

key 三原则（v1 平移）：
- tenant_id 明文前缀：跨租户绝不共享，且可按租户 SCAN 清理（字符集守卫见 tenancy.py）；
- 只哈希请求的语义本体：档位 + 消息白名单字段 + stop + 全部调用参数（tools/tool_choice/temperature/...）——
  request_id/session_id/deadline 是传输参数不是语义，混入则永不命中且静默烧钱；
- canonical JSON（sort_keys、紧凑分隔符）：字段顺序差异不产生不同 key。
别人的类型必须白名单：消息的 id / response_metadata / usage_metadata / additional_kwargs 每次都变，
只取 type/content/name/tool_calls/tool_call_id/status（探针⑮）。调用参数是我们自己的契约，反过来全量入 key：
凡影响上游调用的参数都必须影响 key，漏一个就是误命中。
key 前缀带 schema 版本号：值格式升级后旧缓存天然全体 miss，不会出现"新代码解析旧数据"。

值 = 自家最小 chunk schema（content / tool_call_chunks / usage / finish），不存框架对象：
块 id 由框架外壳按 run 赋值（lc_run--...），回放时框架会重新赋；回放块每块盖章
response_metadata["aegis_cached"]=True（同值可安全合并，探针⑱），末块标 chunk_position="last"（否则外壳补空块，探针⑯）。
入库标准：含 finish_reason 且含实质内容（至少一块 content 或 tool_call_chunks）——半截、失败、空洞流绝不入库。
脏数据自愈：解析不了的条目当场删除、按 miss 处理。

故障只告警 + 粘滞降级（与熔断同款，ADR-007 决策 6）：CacheStore 是进程级单例，任一 Redis 触点异常即降级为直通，
probe_interval 内不再碰 Redis；get/put 共用一个探针窗口（同一请求内 get 探针成功即切回，紧随其后的 put 走正常路径）；
探针失败顺延窗口；成功即恢复并记日志。TenantCache 是按租户装配的薄视图（组合根每请求构造）。
"""

import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import redis.asyncio as aioredis
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.messages.utils import message_chunk_to_message
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.core.logs import get_logger
from app.engine.gateway import utterances as u
from app.engine.gateway.tenancy import validate_tenant_id

logger = get_logger(__name__)

KEY_PREFIX = "aegis:cache"
SCHEMA_VERSION = "v1"
MESSAGE_FIELDS = frozenset(
    {"type", "content", "name", "tool_calls", "tool_call_id", "status"}
)
CACHED_MARK = "aegis_cached"

_monotonic = time.monotonic  # 测试接缝：降级粘滞窗


# ---------------------------------------------------------------- key


def _message_essence(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, BaseMessageChunk):
        # 历史里混入的块与整消息同 key（两者 type 不同，探针⑮）
        message = message_chunk_to_message(message)
    dumped = message.model_dump(include=MESSAGE_FIELDS)
    return {k: v for k, v in dumped.items() if v not in (None, "", [])}


def request_digest(
    tier: str,
    messages: Sequence[BaseMessage],
    *,
    stop: Sequence[str] | None = None,
    **kwargs: Any,
) -> str:
    """请求语义本体的 sha256（不含租户：前缀由 TenantCache 加）。

    非 JSON 值以 str 兜底：只会让 key 不可复现（miss），不会让两个不同请求撞成同一 key。
    """
    essence = {
        "tier": tier,
        "messages": [_message_essence(m) for m in messages],
        "stop": list(stop) if stop else None,
        "kwargs": kwargs,
    }
    blob = json.dumps(
        essence, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 值 schema


class _ChunkPayload(BaseModel):
    content: str | list[Any] = ""
    tool_call_chunks: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    finish: str | None = None


class _ReplyPayload(BaseModel):
    provider: str
    model: str
    chunks: list[_ChunkPayload]


_payload_adapter: TypeAdapter[_ReplyPayload] = TypeAdapter(_ReplyPayload)


@dataclass(frozen=True, slots=True)
class CachedReply:
    """一次完整回复：来源候选 + 块序列。命中回放时 provider/model 供零成本记账行使用。"""

    provider: str
    model: str
    chunks: list[AIMessageChunk]


def _encode(chunk: AIMessageChunk) -> _ChunkPayload:
    return _ChunkPayload(
        content=chunk.content,
        tool_call_chunks=[
            {"name": t["name"], "args": t["args"], "id": t["id"], "index": t["index"]}
            for t in chunk.tool_call_chunks
        ],
        usage=dict(chunk.usage_metadata) if chunk.usage_metadata else None,
        finish=chunk.response_metadata.get("finish_reason"),
    )


def _decode(payload: _ChunkPayload, *, last: bool) -> AIMessageChunk:
    metadata: dict[str, Any] = {CACHED_MARK: True}
    if payload.finish is not None:
        metadata["finish_reason"] = payload.finish
    return AIMessageChunk(
        content=payload.content,
        tool_call_chunks=[
            tool_call_chunk(
                name=t["name"], args=t["args"], id=t["id"], index=t["index"]
            )
            for t in payload.tool_call_chunks
        ],  # 工厂不收 type 键（探针⑯）
        usage_metadata=payload.usage,  # type: ignore[arg-type]
        response_metadata=metadata,
        chunk_position="last" if last else None,
    )


def _finished(chunk: AIMessageChunk) -> bool:
    return chunk.response_metadata.get("finish_reason") is not None


def _substantial(chunk: AIMessageChunk) -> bool:
    return bool(chunk.content) or bool(chunk.tool_call_chunks)


def is_complete(chunks: Sequence[AIMessageChunk]) -> bool:
    """入库标准：见到终止信号，且至少一块有实质内容。半截/失败/空洞流都不满足。"""
    return any(_finished(c) for c in chunks) and any(_substantial(c) for c in chunks)


# ---------------------------------------------------------------- 协议与实现


@runtime_checkable
class CacheLike(Protocol):
    """候选环看到的缓存：key 是 request_digest 的结果；实现永不抛异常（故障在内部降级为 miss / no-op）。"""

    async def get(self, key: str) -> CachedReply | None: ...

    async def put(self, key: str, reply: CachedReply) -> None: ...


class CacheStore:
    """进程级 Redis 存取 + fail-open + 粘滞降级；只认字符串 key/value。"""

    def __init__(
        self,
        client: aioredis.Redis,
        *,
        ttl_seconds: int,
        probe_interval: float = 5.0,
    ) -> None:
        if ttl_seconds <= 0 or probe_interval <= 0:
            raise ValueError(u.CACHE_PARAMS_INVALID)
        self._client = client
        self._ttl = ttl_seconds
        self._probe_interval = probe_interval
        self._degraded = False
        self._degraded_until = 0.0

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _probe_due(self) -> bool:
        """降级期是否轮到放顺路探针：领取即续窗（检查与写入之间无 await）。"""
        now = _monotonic()
        if now < self._degraded_until:
            return False
        self._degraded_until = now + self._probe_interval
        return True

    def _note_degraded(self) -> bool:
        """触点失败：顺延窗口；首次降级返回 True（由调用处带 exc_info 记日志）。"""
        self._degraded_until = _monotonic() + self._probe_interval
        if self._degraded:
            return False
        self._degraded = True
        return True

    async def _touch(self, op: str, call: Callable[[], Awaitable[Any]]) -> Any | None:
        if self._degraded and not self._probe_due():
            return None
        try:
            result = await call()
        except Exception:
            if self._note_degraded():
                logger.warning(u.LOG_CACHE_DEGRADED, op=op, exc_info=True)
            return None
        if self._degraded:
            self._degraded = False
            logger.warning(u.LOG_CACHE_RECOVERED, op=op)
        return result

    async def get(self, key: str) -> str | None:
        return await self._touch("get", lambda: self._client.get(key))

    async def put(self, key: str, value: str) -> None:
        await self._touch("put", lambda: self._client.set(key, value, ex=self._ttl))

    async def delete(self, key: str) -> None:
        await self._touch("delete", lambda: self._client.delete(key))


class TenantCache:
    """实现 CacheLike：租户前缀 + 值编解码 + 完整性守卫 + 脏条目自愈；存取交给共享的 CacheStore。"""

    def __init__(self, store: CacheStore, tenant_id: str) -> None:
        self._store = store
        self._tenant_id = validate_tenant_id(tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def key(self, digest: str) -> str:
        return f"{KEY_PREFIX}:{SCHEMA_VERSION}:{self._tenant_id}:{digest}"

    async def get(self, digest: str) -> CachedReply | None:
        key = self.key(digest)
        raw = await self._store.get(key)
        if raw is None:
            return None
        try:
            payload = _payload_adapter.validate_json(raw)
        except ValidationError:
            logger.warning(u.LOG_CACHE_DIRTY, key=key)
            await self._store.delete(key)  # 自愈：脏条目当场清除，本次按 miss 处理
            return None
        last = len(payload.chunks) - 1
        return CachedReply(
            provider=payload.provider,
            model=payload.model,
            chunks=[_decode(c, last=i == last) for i, c in enumerate(payload.chunks)],
        )

    async def put(self, digest: str, reply: CachedReply) -> None:
        if not is_complete(reply.chunks):
            return  # 防御：半截/空洞流不入库（候选环只在流耗尽处调用，这里是第二道）
        payload = _ReplyPayload(
            provider=reply.provider,
            model=reply.model,
            chunks=[_encode(c) for c in reply.chunks],
        )
        await self._store.put(self.key(digest), payload.model_dump_json())
