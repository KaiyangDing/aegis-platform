"""出站闸的有界 LRU（M1.5c，租户桶 A′）：超过 max_keys 淘汰最久未用的 key，淘汰再回来的 key 从满桶重新开始；
max_keys 非法启动即炸。供应商闸不传 max_keys，行为不变（test_outbound.py 一字未改）。"""

import inspect

import pytest

from app.engine.gateway.outbound import ProviderLimiter

if "max_keys" not in inspect.signature(ProviderLimiter.__init__).parameters:
    pytest.skip("M1.5c 未敲：ProviderLimiter 尚无 max_keys", allow_module_level=True)


def limiter(**kw) -> ProviderLimiter:
    # burst=1 且补给极慢：第一次取到、第二次取不到，桶"新鲜与否"一试便知
    params = {"rate": 0.01, "burst": 1.0, "max_wait": 0.0}
    params.update(kw)
    return ProviderLimiter(**params)


async def test_least_recently_used_key_is_evicted_and_comes_back_fresh():
    lim = limiter(max_keys=2)
    assert await lim.acquire("a", 0) is True
    assert await lim.acquire("b", 0) is True
    assert await lim.acquire("a", 0) is False  # a 空了，但被"用过"——比 b 更新
    assert await lim.acquire("c", 0) is True  # 第三个 key：淘汰最久未用的 b
    assert (
        await lim.acquire("b", 0) is True
    )  # b 回来是一只新桶（满桶）——近似，记账；此时淘汰 a
    assert await lim.acquire("c", 0) is False  # c 还在，仍是空桶
    assert await lim.acquire("a", 0) is True  # a 已被淘汰，回来也是新桶


async def test_unbounded_by_default():
    lim = limiter()
    for i in range(50):
        assert await lim.acquire(f"k{i}", 0) is True
    assert await lim.acquire("k0", 0) is False  # 没被淘汰：仍是那只空桶


def test_invalid_max_keys_fails_loud():
    with pytest.raises(ValueError):
        limiter(max_keys=0)
