"""工具契约：身份注入、读写标记、"写不重试"不变量、C15 三层防呆、工具名与参数防呆。"""

from dataclasses import FrozenInstanceError, replace

import pytest

pytest.importorskip(
    "app.engine.runtime.tools", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.runtime.tools import SideEffect, ToolContext, ToolDef


async def _echo(ctx: ToolContext, text: str) -> str:
    return text


def _read_tool(**overrides: object) -> ToolDef:
    """合法读工具基线，各测试按需覆写单个字段。"""
    base = ToolDef(
        name="order_query",
        description="按订单号查询订单状态。",
        handler=_echo,
        side_effect=SideEffect.READ,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_side_effect_values_are_stable():
    """值将进事件 / 审计 payload 与恢复期判定，快照钉死。"""
    assert {s.value for s in SideEffect} == {"read", "write"}


def test_tool_context_fields_and_frozen():
    ctx = ToolContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-1",
        run_id="r-1",
        tool_call_id="evt-1",
    )
    assert ctx.tool_call_id == "evt-1"
    with pytest.raises(FrozenInstanceError):
        ctx.user_id = "u-2"  # type: ignore[misc]


@pytest.mark.parametrize(
    "blank", ["tenant_id", "user_id", "session_id", "run_id", "tool_call_id"]
)
def test_tool_context_rejects_blank_ids(blank: str):
    kwargs = {
        "tenant_id": "t",
        "user_id": "u",
        "session_id": "s",
        "run_id": "r",
        "tool_call_id": "e",
    }
    kwargs[blank] = ""
    with pytest.raises(ValueError, match=blank):
        ToolContext(**kwargs)


def test_tool_def_defaults():
    t = _read_tool()
    assert t.risk_policy is None
    assert t.args_model is None
    assert t.timeout_s is None  # None = 继承 LoopPolicy.tool_step_timeout_s
    assert t.retries == 0
    assert dict(t.parameters_schema) == {}


def test_tool_def_is_frozen():
    with pytest.raises(FrozenInstanceError):
        _read_tool().retries = 5  # type: ignore[misc]


def test_write_tool_never_retries():
    """写操作绝不自动重试——幂等靠 write-ahead 键，不靠再试一次。"""
    with pytest.raises(ValueError, match="禁止自动重试"):
        _read_tool(side_effect=SideEffect.WRITE, retries=1)


def test_read_tool_may_retry_and_write_zero_is_legal():
    assert _read_tool(retries=2).retries == 2
    assert _read_tool(side_effect=SideEffect.WRITE, risk_exempt=True).retries == 0


def test_write_tool_without_policy_rejected():
    """C15 类型层：写工具裸奔（无闸门无豁免）连实例都造不出来。"""
    with pytest.raises(ValueError, match="C15"):
        _read_tool(side_effect=SideEffect.WRITE)


def test_write_tool_with_exemption_is_legal():
    assert _read_tool(side_effect=SideEffect.WRITE, risk_exempt=True).risk_exempt


def test_write_tool_with_policy_is_legal():
    t = _read_tool(side_effect=SideEffect.WRITE, risk_policy=lambda args, cfg: True)
    assert t.risk_policy is not None


def test_exempt_with_policy_is_contradiction():
    with pytest.raises(ValueError, match="互斥"):
        _read_tool(
            side_effect=SideEffect.WRITE,
            risk_policy=lambda a, c: True,
            risk_exempt=True,
        )


def test_exempt_on_read_tool_rejected():
    with pytest.raises(ValueError, match="risk_exempt"):
        _read_tool(risk_exempt=True)


@pytest.mark.parametrize(
    "bad_name", ["", "带中文", "has space", "a" * 65, "semi;colon"]
)
def test_tool_def_rejects_bad_names(bad_name: str):
    with pytest.raises(ValueError, match="工具名"):
        _read_tool(name=bad_name)


def test_tool_def_accepts_boundary_names():
    assert _read_tool(name="a").name == "a"
    assert _read_tool(name="A-z_09" + "x" * 58).name.startswith("A-z_09")


def test_tool_def_rejects_blank_description():
    with pytest.raises(ValueError, match="description"):
        _read_tool(description="   ")


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_tool_def_rejects_bad_timeout(bad: float):
    with pytest.raises(ValueError, match="timeout_s"):
        _read_tool(timeout_s=bad)


def test_tool_def_rejects_negative_retries():
    with pytest.raises(ValueError, match="retries"):
        _read_tool(retries=-1)


def test_write_retry_error_precedes_c15_error():
    """两条不变量同时违反时先报"写禁重试"——C15 检查排在其后，不遮蔽既有报错。"""
    with pytest.raises(ValueError, match="禁止自动重试"):
        _read_tool(side_effect=SideEffect.WRITE, retries=1)
