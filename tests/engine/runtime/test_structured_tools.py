"""ToolDef → StructuredTool 只作 schema 载体（探针⑴）：名字 / 说明 / 参数同源、ctx 不出现、哨兵 handler 裸穿、
无 args_model 拒绝、顺序保持；再用 create_agent + 剧本模型证明哨兵在图里同样炸——没挂 ToolExec 就不许执行工具。"""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

pytest.importorskip(
    "app.engine.runtime.tools", reason="M2.1 未敲：app/engine/runtime/ 不存在"
)

from app.engine.fakes import scripted_model
from app.engine.runtime.tools import (
    SideEffect,
    ToolContext,
    ToolDef,
    ToolRegistrationError,
    to_structured_tools,
)
from tests.engine.runtime.demo_tools import build_registry


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def test_carries_name_description_and_schema_without_ctx():
    tools = to_structured_tools(build_registry().specs())
    assert [t.name for t in tools] == [
        "demo_order_query",
        "demo_refund_apply",
        "demo_ticket_create",
    ]
    assert all(isinstance(t, StructuredTool) for t in tools)
    refund = convert_to_openai_tool(tools[1])["function"]
    assert refund["description"] == "为演示订单发起退款（超租户阈值走人工审批）。"
    params = refund["parameters"]
    assert "ctx" not in params["properties"]
    assert set(params["properties"]) == {"order_id", "amount"}
    assert params["required"] == ["order_id", "amount"]
    assert params["properties"]["amount"]["type"] == "integer"


def test_llm_facing_parameters_derive_from_parameters_schema():
    """探针⑴：框架剥掉 title（各层）与顶层 additionalProperties，其余与 parameters_schema 逐键同源。"""
    t = build_registry().get("demo_order_query")
    assert t is not None
    (st,) = to_structured_tools([t])
    expected = _strip_titles(dict(t.parameters_schema))
    expected.pop("additionalProperties")
    assert convert_to_openai_tool(st)["function"]["parameters"] == expected


async def test_sentinel_handler_fails_loud():
    t = build_registry().get("demo_order_query")
    assert t is not None
    (st,) = to_structured_tools([t])
    with pytest.raises(RuntimeError, match="demo_order_query"):
        await st.ainvoke({"order_id": "1"})


def test_requires_args_model():
    async def _h(ctx: ToolContext, **kwargs: Any) -> None: ...

    bare = ToolDef(
        name="bare", description="占位", handler=_h, side_effect=SideEffect.READ
    )
    with pytest.raises(ToolRegistrationError, match="args_model"):
        to_structured_tools([bare])


def test_empty_input_gives_empty_list():
    assert to_structured_tools(()) == []


async def test_sentinel_propagates_out_of_agent_run():
    """图里的哨兵：ToolNode 默认只把校验错误转 ToolMessage，其余异常重抛穿出 run（探针⑴ / F）。"""
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "demo_order_query", "args": {"order_id": "1"}, "id": "c1"}
            ],
        ),
        AIMessage("done"),
    )
    agent = create_agent(model, tools=to_structured_tools(build_registry().specs()))
    with pytest.raises(RuntimeError, match="ToolExec"):
        await agent.ainvoke({"messages": [HumanMessage("查订单")]})
