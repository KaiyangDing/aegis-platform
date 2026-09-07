"""图工厂（M2.3）：recursion_limit 推导式与紧致性（探针⑻）、栈序与节点快照、按 (tenant, spec 指纹) 缓存、指纹敏感性。"""

import pytest

pytest.importorskip(
    "app.engine.runtime.runtime",
    reason="M2.3 未敲：app/engine/runtime/runtime.py 不存在",
)

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from app.engine.runtime.middleware.model_call import ModelCall
from app.engine.runtime.middleware.run_events import RunEvents
from app.engine.runtime.middleware.tool_exec import ToolExec
from app.engine.runtime.runtime import (
    MIDDLEWARE_STACK,
    recursion_limit_for,
    spec_fingerprint,
)
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy
from app.engine.runtime.state import RunContext
from app.engine.runtime.tools import ToolRegistry
from tests.engine.runtime.demo_tools import build_registry
from tests.engine.runtime.doubles import (
    MemoryEventStore,
    collect,
    make_runtime,
    text_turn,
    tool_turn,
)


def test_stack_order_and_graph_nodes_snapshot():
    assert MIDDLEWARE_STACK == (RunEvents, ModelCall, ToolExec)
    rt, _, _, _ = make_runtime()
    agent = rt.build_agent(
        "t-a", AgentSpec(system_prompt="x", tools=build_registry().specs())
    )
    assert sorted(agent.get_graph().nodes) == [
        "RunEvents.after_agent",
        "RunEvents.before_agent",
        "__end__",
        "__start__",
        "model",
        "tools",
    ]


@pytest.mark.parametrize(("max_iterations", "expected"), [(1, 5), (3, 9), (10, 23)])
def test_recursion_limit_formula_for_current_stack(max_iterations: int, expected: int):
    """M2.3 栈：外圈 2 节点 + 每轮 (model + tools) 2 节点 + 1 余量 → 2·max_iterations + 3。"""
    assert recursion_limit_for(LoopPolicy(max_iterations=max_iterations)) == expected


async def test_recursion_limit_is_tight_against_probe():
    """探针⑻：k 轮工具循环最少要 2k+4；推导值 2·M+3（M=k+1）恰好多 1，再减 2 就撞 GraphRecursionError。"""
    spec = AgentSpec(
        system_prompt="x",
        model_tier="fast",
        tools=build_registry().specs(),
        policy=LoopPolicy(max_iterations=3),
    )
    acts = [
        tool_turn(("demo_order_query", {"order_id": "1"}, "c1")),
        tool_turn(("demo_order_query", {"order_id": "2"}, "c2")),
        text_turn("完成"),
    ]
    rt, _, _, sessions = make_runtime(*acts)
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input="x", spec=spec
    )
    assert got[-1].payload == {
        "reason": "completed",
        "iteration": 3,
        "detail": "stop_reason=stop",
    }

    rt2, _, _, _ = make_runtime(*acts)
    agent = rt2.build_agent("t-a", spec)
    ctx = RunContext(
        tenant_id="t-a",
        user_id="u-1",
        session_id="s-2",
        run_id="r-2",
        spec=spec,
        registry=ToolRegistry(spec.tools),
        events=MemoryEventStore(),
    )
    with pytest.raises(GraphRecursionError):
        await agent.ainvoke(
            {"messages": [HumanMessage("x")]},
            {
                "configurable": {"thread_id": "s-2"},
                "recursion_limit": recursion_limit_for(spec.policy) - 2,
            },
            context=ctx,
        )


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
