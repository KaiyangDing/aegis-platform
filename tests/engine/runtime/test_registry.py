"""ToolRegistry——dispatch 表、重名拒绝、specs() 喂注入面、演示工具集体检。"""

import pytest

pytest.importorskip(
    "app.engine.runtime.tools", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.runtime.spec import AgentSpec
from app.engine.runtime.tools import (
    SideEffect,
    ToolContext,
    ToolDef,
    ToolRegistrationError,
    ToolRegistry,
    tool,
)


def test_registry_builds_dispatch(demo_registry: ToolRegistry):
    t = demo_registry.get("demo_order_query")
    assert isinstance(t, ToolDef)
    assert t.side_effect is SideEffect.READ


def test_duplicate_name_rejected():
    @tool(side_effect=SideEffect.READ, name="dup")
    async def a(ctx: ToolContext) -> None:
        """甲。"""

    @tool(side_effect=SideEffect.READ, name="dup")
    async def b(ctx: ToolContext) -> None:
        """乙。"""

    with pytest.raises(ToolRegistrationError, match="dup"):
        ToolRegistry([a, b])


def test_add_after_construction_still_rejects_duplicates(
    demo_registry: ToolRegistry,
):
    with pytest.raises(ToolRegistrationError, match="demo_order_query"):
        demo_registry.add(demo_registry.get("demo_order_query"))  # type: ignore[arg-type]


def test_get_unknown_returns_none(demo_registry: ToolRegistry):
    """模型幻觉工具名是运行期常态——注册表只管查表，处置政策归 M2.4 闸门 #5。"""
    assert demo_registry.get("模型幻觉出来的工具") is None


def test_specs_feeds_agent_spec(demo_registry: ToolRegistry):
    """注册表的产出直接填满注入面。"""
    spec = AgentSpec(system_prompt="演示客服", tools=demo_registry.specs())
    assert len(spec.tools) == 3
    assert {t.name for t in spec.tools} == {
        "demo_order_query",
        "demo_refund_apply",
        "demo_ticket_create",
    }


def test_specs_preserves_registration_order(demo_registry: ToolRegistry):
    """顺序确定：工具顺序进 bind_tools，是行为轨迹确定性的一环。"""
    names = [t.name for t in demo_registry.specs()]
    assert names == ["demo_order_query", "demo_refund_apply", "demo_ticket_create"]


def test_demo_toolset_covers_three_risk_shapes(demo_registry: ToolRegistry):
    """演示工具集的角色分工：读 / 写带闸门 / 写显式豁免——C15 的三种合法形态各占一席。"""
    read = demo_registry.get("demo_order_query")
    gated = demo_registry.get("demo_refund_apply")
    exempt = demo_registry.get("demo_ticket_create")
    assert read is not None and gated is not None and exempt is not None
    assert read.risk_policy is None and not read.risk_exempt
    assert gated.side_effect is SideEffect.WRITE and gated.risk_policy is not None
    assert exempt.side_effect is SideEffect.WRITE and exempt.risk_exempt


def test_demo_refund_policy_thresholds(demo_registry: ToolRegistry):
    """风险闸门谓词：同一工具，金额不同命运不同——args_model 与 risk_policy 首次联动。"""
    gated = demo_registry.get("demo_refund_apply")
    assert gated is not None
    assert gated.args_model is not None and gated.risk_policy is not None
    big = gated.args_model.model_validate({"order_id": "1024", "amount": 350})
    small = gated.args_model.model_validate({"order_id": "1024", "amount": 80})
    cfg = {"approval_threshold": 200}
    assert gated.risk_policy(big, cfg) is True
    assert gated.risk_policy(small, cfg) is False


async def test_demo_handler_receives_ctx(demo_registry: ToolRegistry):
    """handler 两路参数的汇合点：业务参数走校验，身份与幂等键走 ctx 注入。"""
    ctx = ToolContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-1",
        run_id="r-1",
        tool_call_id="evt-9",
    )
    gated = demo_registry.get("demo_refund_apply")
    assert gated is not None
    result = await gated.handler(ctx, order_id="1024", amount=80)
    assert result == {"refunded": 80, "idempotency_key": "evt-9"}
