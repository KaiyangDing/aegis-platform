"""三工具（S2）：注册事实（读 / 需审批写 / 豁免写、schema 剔除 ctx）、越权三路逐字节同话术、审批谓词阈值与缺省 fail-closed、
退款幂等键 = ctx.tool_call_id 透传后端且重放零第二次副作用、业务拒绝以 dict 回填、工单豁免直通。每测一个干净后端经 set_backend 注入。"""

import uuid
from typing import Any

import pytest

from app.business import utterances as bu
from app.business.backend import demo_backend, set_backend
from app.business.tools import (
    ALL_TOOLS,
    order_query,
    refund_apply,
    refund_needs_approval,
    ticket_create,
)
from app.engine.runtime.tools import SideEffect, ToolContext


@pytest.fixture
def backend():
    instance = demo_backend()
    set_backend(instance)
    try:
        yield instance
    finally:
        set_backend(None)


def _ctx(
    user_id: str = "u-a1", tenant_id: str = "tenant-a", key: str | None = None
) -> ToolContext:
    return ToolContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id="s1",
        run_id="r1",
        tool_call_id=key or str(uuid.uuid4()),
    )


def _args(amount: Any) -> Any:
    assert refund_apply.args_model is not None
    return refund_apply.args_model(order_id="x", amount=amount)


def test_registration_facts():
    assert list(ALL_TOOLS) == ["order_query", "refund_apply", "ticket_create"]
    assert (
        order_query.side_effect is SideEffect.READ and order_query.risk_policy is None
    )
    assert refund_apply.side_effect is SideEffect.WRITE
    assert refund_apply.risk_policy is refund_needs_approval
    assert ticket_create.side_effect is SideEffect.WRITE and ticket_create.risk_exempt
    # ctx 不在 LLM 可见的 schema 里：身份由运行时注入
    assert set(refund_apply.parameters_schema["properties"]) == {"order_id", "amount"}
    assert set(order_query.parameters_schema["properties"]) == {"order_id"}
    assert all(t.description for t in ALL_TOOLS.values())


async def test_order_query_own_order(backend):
    out = await order_query.handler(_ctx(), order_id="AZ-1001")
    assert out["status"] == "shipped" and out["paid_amount"] == "350.00"
    assert out["items"] == [{"sku": "杉木书架", "qty": 1}]
    assert "tenant_id" not in out and "user_id" not in out


async def test_three_denials_are_byte_identical(backend):
    missing = await order_query.handler(_ctx(), order_id="AZ-9999")
    other_user = await order_query.handler(_ctx(user_id="u-a2"), order_id="AZ-1001")
    other_tenant = await order_query.handler(
        _ctx(user_id="u-b1", tenant_id="tenant-b"), order_id="AZ-1001"
    )
    assert missing == other_user == other_tenant == {"error": bu.DENIED_TEXT}


def test_refund_predicate_threshold_and_fail_closed_default():
    assert refund_needs_approval(_args(200), {"approval_threshold": 200}) is False
    assert refund_needs_approval(_args(200.01), {"approval_threshold": 200}) is True
    assert refund_needs_approval(_args(300), {"approval_threshold": 200}) is True
    assert refund_needs_approval(_args(0.01), {}) is True  # 缺省 0：任意正金额都要人批


async def test_refund_passes_idempotency_key_and_replays(backend):
    ctx = _ctx(key="tool-call-event-id-1")
    first = await refund_apply.handler(ctx, order_id="AZ-1001", amount=100)
    assert first == {
        "order_id": "AZ-1001",
        "refunded": "100",
        "status": "refunded",
        "duplicate": False,
    }
    assert backend.replay("tool-call-event-id-1") == {**first, "duplicate": True}
    again = await refund_apply.handler(ctx, order_id="AZ-1001", amount=100)
    assert again["duplicate"] is True and backend.write_calls == 1


async def test_refund_denials_and_business_rejections_are_dicts(backend):
    assert await refund_apply.handler(
        _ctx(user_id="u-a2"), order_id="AZ-1001", amount=10
    ) == {"error": bu.DENIED_TEXT}
    assert await refund_apply.handler(_ctx(), order_id="AZ-1002", amount=10) == {
        "error": bu.REFUND_ALREADY
    }
    assert await refund_apply.handler(_ctx(), order_id="AZ-1001", amount=1000) == {
        "error": bu.REFUND_OVER_PAID
    }
    assert backend.write_calls == 0


async def test_ticket_create_is_exempt_write_with_key(backend):
    ctx = _ctx(key="k-ticket")
    out = await ticket_create.handler(ctx, title="投诉物流")
    assert out == {
        "ticket_id": "T-0001",
        "title": "投诉物流",
        "by": "u-a1",
        "duplicate": False,
    }
    assert (await ticket_create.handler(ctx, title="投诉物流"))["duplicate"] is True
    assert backend.write_calls == 1
