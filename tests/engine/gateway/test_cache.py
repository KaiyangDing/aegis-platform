"""租户前缀精确缓存（M1.5a）：key 三原则（纯函数）/ 值往返与盖章 / 完整性守卫 / 跨租对抗（真 Redis db1）/
故障降级粘滞（FlakyClient 驱动，零真等）。"""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.messages.utils import message_chunk_to_message
from structlog.testing import capture_logs

pytest.importorskip(
    "app.engine.gateway.cache", reason="M1.5a 未敲：app/engine/gateway/cache.py 不存在"
)

from app.engine.gateway import cache as cache_mod
from app.engine.gateway import utterances as u
from app.engine.gateway.cache import (
    CACHED_MARK,
    CachedReply,
    CacheLike,
    CacheStore,
    TenantCache,
    is_complete,
    request_digest,
)
from tests.engine.gateway.doubles import finish, text

H = [HumanMessage("你好")]

# ---------------------------------------------------------------- key（纯函数，零 Redis）


def test_digest_ignores_volatile_message_fields():
    a = [
        AIMessage(content="x", id="run-1", response_metadata={"finish_reason": "stop"})
    ]
    b = [
        AIMessage(
            content="x",
            id="run-2",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            additional_kwargs={"refusal": None},
        )
    ]
    assert request_digest("fast", a) == request_digest("fast", b)
    assert request_digest("fast", [HumanMessage("q", id="a")]) == request_digest(
        "fast", [HumanMessage("q", id="b")]
    )


def test_digest_changes_with_semantics():
    base = request_digest("fast", H)
    assert request_digest("fast", [HumanMessage("你好呀")]) != base
    assert request_digest("strong", H) != base
    assert request_digest("fast", [SystemMessage("sys"), *H]) != base
    assert request_digest("fast", H, stop=["\n"]) != base
    assert request_digest("fast", H, temperature=0.7) != base
    assert request_digest("fast", H, tools=[{"type": "function"}]) != base
    assert request_digest("fast", H, tool_choice="auto") != base


def test_digest_is_independent_of_kwarg_and_field_order():
    assert request_digest("fast", H, temperature=0.1, max_tokens=5) == request_digest(
        "fast", H, max_tokens=5, temperature=0.1
    )


def test_digest_includes_tool_calls_and_tool_results():
    call = {"name": "f", "args": {"a": 1}, "id": "c1", "type": "tool_call"}
    with_call = [AIMessage(content="", tool_calls=[call])]
    without = [AIMessage(content="")]
    assert request_digest("fast", with_call) != request_digest("fast", without)
    r1 = [ToolMessage(content="ok", tool_call_id="c1")]
    r2 = [ToolMessage(content="ok", tool_call_id="c2")]
    assert request_digest("fast", r1) != request_digest("fast", r2)
    err = [ToolMessage(content="ok", tool_call_id="c1", status="error")]
    assert request_digest("fast", r1) != request_digest("fast", err)


def test_digest_treats_history_chunk_like_its_message():
    # agent 累加出的 AIMessageChunk 进历史时，与等价的 AIMessage 同 key（两者 type 不同，探针⑮）
    assert request_digest("fast", [AIMessageChunk(content="x")]) == request_digest(
        "fast", [AIMessage(content="x")]
    )


def test_digest_survives_unserializable_kwarg():
    request_digest("fast", H, weird=object())  # 只会 miss，不会炸


def test_tenant_key_has_prefix_and_version(store):
    digest = request_digest("fast", H)
    key = TenantCache(store, "tA").key(digest)
    assert key.startswith("aegis:cache:v1:tA:")
    assert key.endswith(digest)
    assert TenantCache(store, "tB").key(digest) != key


def test_tenant_cache_rejects_illegal_tenant(store):
    with pytest.raises(ValueError, match="tenant_id"):
        TenantCache(store, "tA:evil")


def test_tenant_cache_is_cache_like(store):
    assert isinstance(TenantCache(store, "tA"), CacheLike)


# ---------------------------------------------------------------- 完整性守卫


def test_is_complete_requires_finish_and_substance():
    assert is_complete([text("好"), finish()])
    assert not is_complete([text("好")])  # 无终止信号：半截
    assert not is_complete([finish()])  # 只有尾块：空洞流
    assert not is_complete([])


# ---------------------------------------------------------------- 真 Redis db1


class FakeClient:
    """内存 dict 版 redis 客户端（get/set/delete 三命令），可开关故障并计触点数：粘滞期"一次都没碰"靠计数证明。"""

    def __init__(self) -> None:
        self.dead = False
        self.touches = 0
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        self.touches += 1
        if self.dead:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.touches += 1
        if self.dead:
            raise ConnectionError("redis down")
        self.data[key] = value
        return True

    async def delete(self, key: str) -> int:
        self.touches += 1
        if self.dead:
            raise ConnectionError("redis down")
        return int(self.data.pop(key, None) is not None)


@pytest.fixture
def store(redis_async) -> CacheStore:
    return CacheStore(redis_async, ttl_seconds=60)


def tool_chunks() -> list[AIMessageChunk]:
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                tool_call_chunk(name="f", args='{"a":', id="call_1", index=0)
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[tool_call_chunk(name=None, args=" 1}", id=None, index=0)],
        ),
    ]


def merged(chunks: list[AIMessageChunk]) -> AIMessage:
    acc = chunks[0]
    for c in chunks[1:]:
        acc = acc + c
    return message_chunk_to_message(acc)


async def test_roundtrip_preserves_content_tool_chunks_usage_finish(store, namespace):
    original = [text("你好"), *tool_chunks(), finish("tool_calls")]
    cache = TenantCache(store, "tA")
    digest = request_digest("fast", [HumanMessage(namespace)])
    await cache.put(digest, CachedReply("p1", "m1", original))
    hit = await cache.get(digest)
    assert hit is not None
    assert (hit.provider, hit.model) == ("p1", "m1")
    assert len(hit.chunks) == len(original)
    a, b = merged(original), merged(hit.chunks)
    assert (a.content, a.tool_calls, a.usage_metadata) == (
        b.content,
        b.tool_calls,
        b.usage_metadata,
    )
    assert b.response_metadata["finish_reason"] == "tool_calls"
    assert all(c.response_metadata[CACHED_MARK] is True for c in hit.chunks)  # 每块盖章
    assert b.response_metadata[CACHED_MARK] is True  # 合并后仍在（探针⑱）
    assert hit.chunks[-1].chunk_position == "last"  # 否则框架外壳补空块（探针⑯）
    assert all(c.id is None for c in hit.chunks)  # 不存框架 run id，回放时外壳重新赋


async def test_miss_returns_none(store, namespace):
    assert (
        await TenantCache(store, "tA").get(
            request_digest("fast", [HumanMessage(namespace)])
        )
        is None
    )


async def test_ttl_is_applied(store, redis_async, namespace):
    cache = TenantCache(store, "tA")
    digest = request_digest("fast", [HumanMessage(namespace)])
    await cache.put(digest, CachedReply("p1", "m1", [text("好"), finish()]))
    assert 0 < await redis_async.ttl(cache.key(digest)) <= 60


async def test_incomplete_stream_never_stored(store, namespace):
    cache = TenantCache(store, "tA")
    digest = request_digest("fast", [HumanMessage(namespace)])
    await cache.put(digest, CachedReply("p1", "m1", [text("半")]))  # 没有 finish
    assert await cache.get(digest) is None


async def test_hollow_stream_never_stored(store, namespace):
    cache = TenantCache(store, "tA")
    digest = request_digest("fast", [HumanMessage(namespace)])
    await cache.put(digest, CachedReply("p1", "m1", [finish()]))  # 只有 usage+finish
    assert await cache.get(digest) is None


async def test_cross_tenant_isolation_end_to_end(store, namespace):
    """跨租对抗（DoD#2）：租户 A 写入后，租户 B 的同一请求必 miss；A 自己必命中。"""
    digest = request_digest("fast", [HumanMessage(namespace)])
    await TenantCache(store, "tA").put(
        digest, CachedReply("p1", "m1", [text("A 的答案"), finish()])
    )
    assert await TenantCache(store, "tB").get(digest) is None
    hit = await TenantCache(store, "tA").get(digest)
    assert hit is not None and hit.chunks[0].content == "A 的答案"


async def test_same_tenant_different_request_metadata_hits(store, namespace):
    cache = TenantCache(store, "tA")
    await cache.put(
        request_digest("fast", [HumanMessage(namespace, id="req-1")]),
        CachedReply("p1", "m1", [text("好"), finish()]),
    )
    assert (
        await cache.get(request_digest("fast", [HumanMessage(namespace, id="req-2")]))
        is not None
    )


async def test_dirty_entry_is_deleted_and_treated_as_miss(
    store, redis_async, namespace
):
    cache = TenantCache(store, "tA")
    digest = request_digest("fast", [HumanMessage(namespace)])
    await redis_async.set(cache.key(digest), '{"not": "a reply"}')
    with capture_logs() as logs:
        assert await cache.get(digest) is None
    assert await redis_async.exists(cache.key(digest)) == 0  # 自愈：当场删除
    assert [log["event"] for log in logs] == [u.LOG_CACHE_DIRTY]


# ---------------------------------------------------------------- 故障降级（FakeClient，零真等）


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 1000.0}
    monkeypatch.setattr(cache_mod, "_monotonic", lambda: state["now"])

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return advance


def flaky() -> tuple[FakeClient, CacheStore]:
    client = FakeClient()
    return client, CacheStore(client, ttl_seconds=60, probe_interval=5.0)


def test_store_rejects_bad_params():
    with pytest.raises(ValueError, match="缓存参数"):
        CacheStore(FakeClient(), ttl_seconds=0)
    with pytest.raises(ValueError, match="缓存参数"):
        CacheStore(FakeClient(), ttl_seconds=60, probe_interval=0)


async def test_get_failure_degrades_to_miss_and_logs_once(clock):
    client, store = flaky()
    client.dead = True
    with capture_logs() as logs:
        assert await store.get("k") is None
        assert await store.get("k") is None
    assert store.degraded
    assert [log["event"] for log in logs] == [u.LOG_CACHE_DEGRADED]  # 只喊一次


async def test_put_failure_does_not_raise(clock):
    client, store = flaky()
    client.dead = True
    await store.put("k", "v")  # 已经成功的请求绝不能因为"写缓存失败"以异常收尾
    assert store.degraded


async def test_degradation_is_sticky_and_probes_after_interval(clock):
    client, store = flaky()
    client.dead = True
    await store.get("k")
    assert client.touches == 1
    for _ in range(5):
        await store.get("k")
        await store.put("k", "v")
    assert client.touches == 1  # 粘滞期一次都没碰
    clock(5.1)
    await store.get("k")  # 到点放一个探针
    assert client.touches == 2
    await store.get("k")
    assert client.touches == 2  # 探针失败 → 窗口顺延，绝不连撞


async def test_probe_success_recovers_and_logs(clock):
    client, store = flaky()
    client.dead = True
    await store.get("k")
    client.dead = False
    client.data["k"] = "v"
    clock(5.1)
    with capture_logs() as logs:
        assert await store.get("k") == "v"
    assert not store.degraded
    assert [log["event"] for log in logs] == [u.LOG_CACHE_RECOVERED]


async def test_get_and_put_share_one_probe_window(clock):
    """同一请求内 get 探针成功即切回，紧随其后的 put 走正常路径（不各自领探针）。"""
    client, store = flaky()
    client.dead = True
    await store.get("k")
    client.dead = False
    clock(5.1)
    await store.get("k")
    await store.put("k", "v")
    assert client.touches == 3
    assert client.data == {"k": "v"}


async def test_healthy_store_roundtrips_through_fake_client():
    _client, store = flaky()
    await store.put("k", "v")
    assert await store.get("k") == "v"
    await store.delete("k")
    assert await store.get("k") is None
    assert not store.degraded
