"""M2.1 注入面口径：枚举值稳定（事件 payload / 行为轨迹兼容）、"六道闸门"术语口径、
策略与预算类型的冻结语义与防呆、model_tier 与 L1 档位同一事实源。"""

import json
from dataclasses import FrozenInstanceError, replace
from typing import get_args

import pytest

pytest.importorskip(
    "app.engine.runtime.spec", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.gateway.routing import TIERS, Tier
from app.engine.runtime.spec import (
    TERMINATION_GATES,
    AgentSpec,
    ContextConfig,
    LoopPolicy,
    SubAgentPolicy,
    TerminationReason,
)
from app.engine.runtime.tools import SideEffect, ToolDef


def test_termination_reason_values_are_stable():
    """值快照：改任何一个值 = 破坏历史事件的行为轨迹断言，本测试必须先红。"""
    assert {r.value for r in TerminationReason} == {
        "completed",
        "max_iterations",
        "step_timeout",
        "token_budget_exceeded",
        "repeated_calls",
        "protocol_violation",
        "cancelled",
        "gateway_rejected",
    }


def test_termination_reason_serializes_as_plain_string():
    """StrEnum：str() 与 json 序列化直接得到裸字符串，可径直进 payload。"""
    assert str(TerminationReason.GATEWAY_REJECTED) == "gateway_rejected"
    assert json.dumps(TerminationReason.COMPLETED) == '"completed"'


def test_termination_gates_are_exactly_six():
    """术语口径：六道闸门 = 7 类 - 正常完成；gateway_rejected 不算闸门。"""
    assert len(TERMINATION_GATES) == 6
    assert TerminationReason.COMPLETED not in TERMINATION_GATES
    assert TerminationReason.GATEWAY_REJECTED not in TERMINATION_GATES


def test_loop_policy_defaults_match_design_doc():
    p = LoopPolicy()
    assert p.max_iterations == 10
    assert p.llm_step_timeout_s == 90.0
    assert p.tool_step_timeout_s == 30.0
    assert p.session_token_budget == 50_000
    assert p.repeat_call_limit == 3
    assert p.protocol_retry_limit == 2
    assert p.approval_ttl_s == 3600.0


def test_loop_policy_is_frozen():
    with pytest.raises(FrozenInstanceError):
        LoopPolicy().max_iterations = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("max_iterations", 0),
        ("llm_step_timeout_s", 0.0),
        ("tool_step_timeout_s", -1.0),
        ("session_token_budget", 0),
        ("repeat_call_limit", 0),
        ("protocol_retry_limit", -1),
        ("approval_ttl_s", 0.0),
    ],
)
def test_loop_policy_rejects_invalid(field: str, bad: float):
    with pytest.raises(ValueError, match=field):
        replace(LoopPolicy(), **{field: bad})  # type: ignore[arg-type]


def test_context_config_defaults_match_design_doc():
    c = ContextConfig()
    assert c.system_budget == 1_500
    assert c.memory_budget == 1_000
    assert c.history_budget == 4_000
    assert c.retrieval_budget == 3_000
    assert c.tool_results_budget == 3_000
    assert c.output_reserve == 4_000


def test_context_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        ContextConfig().output_reserve = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("system_budget", 0),
        ("output_reserve", 0),
        ("memory_budget", -1),
        ("history_budget", -1),
        ("retrieval_budget", -1),
        ("tool_results_budget", -1),
    ],
)
def test_context_config_rejects_invalid(field: str, bad: int):
    with pytest.raises(ValueError, match=field):
        replace(ContextConfig(), **{field: bad})


def test_context_config_allows_zero_optional_layers():
    """非对称零值规则：中间四层可显式关闭（=0）——例如无工具、无 RAG 的纯对话 Agent。"""
    c = ContextConfig(
        memory_budget=0, history_budget=0, retrieval_budget=0, tool_results_budget=0
    )
    assert c.input_total == c.system_budget


def test_context_config_input_total():
    assert ContextConfig().input_total == 12_500


def test_sub_agent_policy_is_locked():
    """恒 DISABLED。想加成员，先让这条红、另立 ADR。"""
    assert [p.value for p in SubAgentPolicy] == ["disabled"]


def test_agent_spec_defaults():
    spec = AgentSpec(system_prompt="你是云杉电商的客服助手。")
    assert spec.tools == ()
    assert spec.policy == LoopPolicy()
    assert spec.context_config == ContextConfig()
    assert spec.model_tier == "standard"
    assert spec.sub_agent_policy is SubAgentPolicy.DISABLED
    assert dict(spec.tenant_config) == {}
    assert spec.owned_values == ()
    assert spec.entry_classifier is False


def test_agent_spec_is_frozen():
    spec = AgentSpec(system_prompt="x")
    with pytest.raises(FrozenInstanceError):
        spec.model_tier = "fast"  # type: ignore[misc]


def test_agent_spec_rejects_blank_prompt():
    with pytest.raises(ValueError, match="system_prompt"):
        AgentSpec(system_prompt="   ")


def test_agent_spec_rejects_unknown_tier():
    """Literal 只防静态；L3 从租户配置读出的裸字符串靠这道运行时防线。"""
    with pytest.raises(ValueError, match="model_tier"):
        AgentSpec(system_prompt="x", model_tier="turbo")  # type: ignore[arg-type]


def test_model_tier_shares_gateway_tier_literal():
    """档位语义两层同一事实源：L1 的 Tier / TIERS 就是 L2 的合法值全集。"""
    assert get_args(Tier) == TIERS == ("fast", "standard", "strong")
    for tier in TIERS:
        assert AgentSpec(system_prompt="x", model_tier=tier).model_tier == tier


def test_agent_spec_rejects_duplicate_tool_names():
    async def _h() -> None: ...

    dup = ToolDef(
        name="dup", description="占位", handler=_h, side_effect=SideEffect.READ
    )
    with pytest.raises(ValueError, match="dup"):
        AgentSpec(system_prompt="x", tools=(dup, dup))
