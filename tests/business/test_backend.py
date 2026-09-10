"""模拟后端（S2；命运表 A8 降级形态）：租户过滤、幂等键去重零第二次副作用、业务拒绝三条逐字且零副作用、
工单去重、写挂起接缝（副作用已落 / 响应未回 → 同一把钥匙重放拿回首次结果）。"""

import asyncio
from decimal import Decimal

import pytest

from app.business import utterances as bu
from app.business.backend import BusinessRejected, demo_backend


def test_seed_and_tenant_filter():
    backend = demo_backend()
    assert backend.get_order("tenant-a", "AZ-1001").user_id == "u-a1"
    assert (
        backend.get_order("tenant-b", "AZ-1001") is None
    )  # 跨租：同订单号在别的租户下不存在
    assert backend.get_order("tenant-a", "AZ-9999") is None


async def test_refund_then_replay_with_same_key_has_one_side_effect():
    backend = demo_backend()
    kw = {
        "tenant_id": "tenant-a",
        "user_id": "u-a1",
        "order_id": "AZ-1001",
        "amount": Decimal(100),
    }
    first = await backend.apply_refund(idempotency_key="k1", **kw)
    again = await backend.apply_refund(idempotency_key="k1", **kw)
    assert first == {
        "order_id": "AZ-1001",
        "refunded": "100",
        "status": "refunded",
        "duplicate": False,
    }
    assert again == {**first, "duplicate": True}
    assert backend.write_calls == 1
    assert backend.get_order("tenant-a", "AZ-1001").status == "refunded"
    assert backend.replay("k1") == again and backend.replay("never") is None


async def test_business_rejections_are_verbatim_and_side_effect_free():
    backend = demo_backend()
    cases = [
        ({"order_id": "AZ-1001", "amount": Decimal(0)}, bu.REFUND_NOT_POSITIVE),
        ({"order_id": "AZ-1002", "amount": Decimal(1)}, bu.REFUND_ALREADY),
        ({"order_id": "AZ-1001", "amount": Decimal("350.01")}, bu.REFUND_OVER_PAID),
        (
            {"order_id": "AZ-2001", "amount": Decimal(1)},
            bu.DENIED_TEXT,
        ),  # 他人订单：防御性复核
    ]
    for i, (kw, text) in enumerate(cases):
        with pytest.raises(BusinessRejected) as info:
            await backend.apply_refund(
                tenant_id="tenant-a", user_id="u-a1", idempotency_key=f"k{i}", **kw
            )
        assert str(info.value) == text
    assert backend.write_calls == 0
    assert backend.get_order("tenant-a", "AZ-1001").status == "shipped"


async def test_ticket_dedupe():
    backend = demo_backend()
    kw = {"tenant_id": "tenant-a", "user_id": "u-a1", "title": "投诉", "detail": ""}
    first = await backend.create_ticket(idempotency_key="k", **kw)
    again = await backend.create_ticket(idempotency_key="k", **kw)
    assert first == {
        "ticket_id": "T-0001",
        "title": "投诉",
        "by": "u-a1",
        "duplicate": False,
    }
    assert again == {**first, "duplicate": True}
    assert backend.write_calls == 1


async def test_write_hang_happens_after_side_effect():
    """结果不明的真实形态：副作用已落、响应未回——同一把钥匙重放拿回首次结果而不是第二次退款。"""
    backend = demo_backend(write_hang_s=5.0)
    kw = {
        "tenant_id": "tenant-a",
        "user_id": "u-a1",
        "order_id": "AZ-1001",
        "amount": Decimal(10),
    }
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await backend.apply_refund(idempotency_key="k", **kw)
    assert backend.write_calls == 1
    backend.write_hang_s = 0.0
    replayed = await backend.apply_refund(idempotency_key="k", **kw)
    assert replayed["duplicate"] is True and replayed["refunded"] == "10"
    assert backend.write_calls == 1
