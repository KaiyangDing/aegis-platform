"""@tool 装饰器——签名即事实源、ctx 剔除、schema 与 args_model 同源、注册期防呆在 import 时爆炸。"""

import pytest
from pydantic import ValidationError

pytest.importorskip(
    "app.engine.runtime.tools", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.runtime.tools import (
    SideEffect,
    ToolContext,
    ToolDef,
    ToolRegistrationError,
    tool,
)


def _make_order_query() -> ToolDef:
    @tool(side_effect=SideEffect.READ, timeout_s=15.0, retries=2)
    async def order_query(
        ctx: ToolContext, order_id: str, verbose: bool = False
    ) -> dict:
        """按订单号查询订单状态。"""
        return {"order_id": order_id}

    return order_query


def test_decorator_returns_tooldef_with_metadata():
    t = _make_order_query()
    assert isinstance(t, ToolDef)
    assert t.name == "order_query"
    assert t.description == "按订单号查询订单状态。"
    assert t.side_effect is SideEffect.READ
    assert (t.timeout_s, t.retries) == (15.0, 2)
    assert t.risk_policy is None and t.risk_exempt is False


def test_schema_excludes_ctx_and_lists_params():
    """ctx 是运行时注入的身份——模型连它的存在都不许知道。"""
    t = _make_order_query()
    props = t.parameters_schema["properties"]
    assert "ctx" not in props
    assert set(props) == {"order_id", "verbose"}


def test_schema_required_optional_and_types():
    t = _make_order_query()
    schema = t.parameters_schema
    assert schema["required"] == ["order_id"]  # verbose 有默认值 → 可选
    assert schema["properties"]["order_id"]["type"] == "string"
    assert schema["properties"]["verbose"]["type"] == "boolean"
    assert schema["properties"]["verbose"]["default"] is False
    assert schema["additionalProperties"] is False


def test_schema_and_args_model_share_one_source():
    t = _make_order_query()
    assert t.args_model is not None
    assert t.args_model.model_json_schema() == dict(t.parameters_schema)


def test_args_model_forbids_extra_params():
    """LLM 幻觉出的多余参数必须响亮拒绝——静默丢弃等于掩盖模型行为异常。"""
    t = _make_order_query()
    assert t.args_model is not None
    with pytest.raises(ValidationError):
        t.args_model.model_validate({"order_id": "1024", "bogus": 1})


def test_args_model_validates_happy_path():
    t = _make_order_query()
    assert t.args_model is not None
    args = t.args_model.model_validate({"order_id": "1024"})
    assert args.order_id == "1024"  # type: ignore[attr-defined]
    assert args.verbose is False  # type: ignore[attr-defined]


def test_handler_is_the_original_function():
    t = _make_order_query()
    assert t.handler.__name__ == "order_query"


def test_name_override_and_empty_params():
    @tool(side_effect=SideEffect.READ, name="orders_lookup")
    async def whatever(ctx: ToolContext) -> None:
        """查订单。"""

    assert whatever.name == "orders_lookup"
    assert whatever.parameters_schema["properties"] == {}


def test_risk_policy_and_exemption_pass_through():
    def needs_approval(args, cfg) -> bool:
        return True

    @tool(side_effect=SideEffect.WRITE, risk_policy=needs_approval)
    async def refund(ctx: ToolContext, amount: int) -> None:
        """退款。"""

    @tool(side_effect=SideEffect.WRITE, risk_exempt=True)
    async def ticket(ctx: ToolContext, title: str) -> None:
        """工单。"""

    assert refund.risk_policy is needs_approval
    assert ticket.risk_exempt is True


def test_missing_ctx_rejected():
    with pytest.raises(ToolRegistrationError, match="ctx"):

        @tool(side_effect=SideEffect.READ)
        async def bad(order_id: str) -> None:
            """没 ctx。"""


def test_ctx_with_wrong_annotation_rejected():
    with pytest.raises(ToolRegistrationError, match="ctx"):

        @tool(side_effect=SideEffect.READ)
        async def bad(ctx: str, order_id: str) -> None:
            """ctx 注解不是 ToolContext。"""


def test_missing_annotation_rejected():
    with pytest.raises(ToolRegistrationError, match="order_id"):

        @tool(side_effect=SideEffect.READ)
        async def bad(ctx: ToolContext, order_id) -> None:
            """业务参数缺类型注解。"""


def test_varargs_rejected():
    with pytest.raises(ToolRegistrationError, match="args"):

        @tool(side_effect=SideEffect.READ)
        async def bad(ctx: ToolContext, *items: str) -> None:
            """可变位置参数。"""


def test_kwargs_rejected():
    with pytest.raises(ToolRegistrationError, match="kwargs"):

        @tool(side_effect=SideEffect.READ)
        async def bad(ctx: ToolContext, **extra: str) -> None:
            """可变关键字参数。"""


def test_empty_docstring_rejected():
    with pytest.raises(ToolRegistrationError, match="description"):

        @tool(side_effect=SideEffect.READ)
        async def bad(ctx: ToolContext) -> None: ...


def test_write_without_policy_rejected_at_decoration():
    """C15 从装饰器这扇门看：沉默的危险按钮在 import 时就炸，且换装成注册期异常。"""
    with pytest.raises(ToolRegistrationError, match="risk_policy") as info:

        @tool(side_effect=SideEffect.WRITE)
        async def refund(ctx: ToolContext, amount: int) -> None:
            """退款。"""

    assert isinstance(info.value.__cause__, ValueError)


def test_bad_name_rejected_at_decoration():
    with pytest.raises(ToolRegistrationError, match="工具名"):

        @tool(side_effect=SideEffect.READ, name="有中文")
        async def bad(ctx: ToolContext) -> None:
            """坏名字。"""
