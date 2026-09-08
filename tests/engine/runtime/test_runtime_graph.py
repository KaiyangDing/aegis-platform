"""图工厂（M2.3；M2.4 / M2.6 随栈更新）：栈序与节点快照、按栈推导的 recursion_limit、按 (tenant, spec 指纹) 缓存、指纹敏感性。
推导式的紧致性在 test_runtime_gates（最长路径 -1 即炸）。本文件钉 M2.6 定稿的栈；M2.6 未敲时整体跳过。"""

import pytest

pytest.importorskip(
    "app.engine.runtime.middleware.summarization",
    reason="M2.6 未敲：middleware/summarization.py 不存在（栈快照按 M2.6 定稿）",
)

from app.engine.runtime.middleware.gates import Gates
from app.engine.runtime.middleware.model_call import ModelCall
from app.engine.runtime.middleware.run_events import RunEvents
from app.engine.runtime.middleware.summarization import AegisSummarization
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.runtime import (
    MIDDLEWARE_STACK,
    build_middleware,
    recursion_limit_for,
    spec_fingerprint,
)
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy
from tests.engine.gateway.doubles import scripted
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import make_runtime, scripted_gateway_factory


def test_stack_order_and_graph_nodes_snapshot():
    assert MIDDLEWARE_STACK == (
        RunEvents,
        AegisSummarization,
        Gates,
        ModelCall,
        ToolExec,
    )
    spec = AgentSpec(system_prompt="x", tools=build_registry().specs())
    gateway = scripted_gateway_factory(scripted())("t-a")
    assert [type(m) for m in build_middleware(gateway, spec)] == list(MIDDLEWARE_STACK)
    rt, _, _, _ = make_runtime()
    agent = rt.build_agent("t-a", spec)
    assert sorted(agent.get_graph().nodes) == [
        "AegisSummarization.before_model",
        "Gates.after_model",
        "Gates.before_model",
        "RunEvents.after_agent",
        "RunEvents.before_agent",
        "__end__",
        "__start__",
        "model",
        "tools",
    ]


@pytest.mark.parametrize(("max_iterations", "expected"), [(1, 10), (3, 20), (10, 55)])
def test_recursion_limit_formula_for_current_stack(max_iterations: int, expected: int):
    """M2.6 栈：外圈 2 + 每轮 (2 before_model + model + after_model + tools) 5 + 终止那一遍 2 个 before_model + 1 → 5·M + 5。"""
    assert recursion_limit_for(LoopPolicy(max_iterations=max_iterations)) == expected


def test_agent_cache_keys_on_tenant_and_spec_fingerprint():
    rt, _, _, _ = make_runtime()
    spec = AgentSpec(system_prompt="甲")
    same = AgentSpec(system_prompt="甲")
    other = AgentSpec(system_prompt="乙")
    a = rt.build_agent("t-a", spec)
    assert rt.build_agent("t-a", same) is a  # 等价 spec 命中
    assert rt.build_agent("t-a", other) is not a  # prompt 变即换图
    assert rt.build_agent("t-b", spec) is not a  # 租户不同即换图（网关按租户）


def test_cache_is_bounded_lru():
    rt, _, _, _ = make_runtime()
    rt._cache_size = 2  # type: ignore[attr-defined]
    first = rt.build_agent("t-a", AgentSpec(system_prompt="1"))
    rt.build_agent("t-a", AgentSpec(system_prompt="2"))
    rt.build_agent("t-a", AgentSpec(system_prompt="1"))  # 触碰：1 变最新
    rt.build_agent("t-a", AgentSpec(system_prompt="3"))  # 淘汰 2
    assert rt.build_agent("t-a", AgentSpec(system_prompt="1")) is first
    assert len(rt._agents) == 2  # type: ignore[attr-defined]


def test_fingerprint_is_stable_and_sensitive():
    base = AgentSpec(system_prompt="x", tools=build_registry().specs())
    assert spec_fingerprint(base) == spec_fingerprint(
        AgentSpec(system_prompt="x", tools=build_registry().specs())
    )
    variants = [
        AgentSpec(system_prompt="y", tools=base.tools),
        AgentSpec(system_prompt="x"),
        AgentSpec(system_prompt="x", tools=base.tools, model_tier="strong"),
        AgentSpec(
            system_prompt="x", tools=base.tools, policy=LoopPolicy(max_iterations=2)
        ),
        AgentSpec(
            system_prompt="x",
            tools=base.tools,
            context_config=ContextConfig(output_reserve=1),
        ),
        AgentSpec(
            system_prompt="x", tools=base.tools, tenant_config={"approval_threshold": 1}
        ),
        AgentSpec(system_prompt="x", tools=base.tools, owned_values=("138",)),
        AgentSpec(system_prompt="x", tools=base.tools, entry_classifier=True),
    ]
    prints = {spec_fingerprint(v) for v in variants}
    assert spec_fingerprint(base) not in prints and len(prints) == len(variants)
