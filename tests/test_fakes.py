"""fake 层自测：fake 本身先可信，其余一切测试才有根基（Argus test_fakes 纪律）。

非流式三条断言事实有 ADR-002 探针实证背书（core 1.6.1）；流式工具块与 langgraph
messages 模式由探针 probe_m12_b 实证。
"""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    UsageMetadata,
)
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.engine import fakes
from app.engine.fakes import FAKE_USAGE, FakeReplyChatModel, scripted_model

USAGE = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
TOOL_CALL = AIMessage(
    content="",
    tool_calls=[{"name": "order_query", "args": {"order_id": "o1"}, "id": "call_1"}],
    usage_metadata=USAGE,
)


def merge(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    full = chunks[0]
    for c in chunks[1:]:
        full = full + c
    return full


# ---------------------------------------------------------------- 剧本回放（非流式）


def test_scripted_reply_replays_in_order():
    model = scripted_model(AIMessage(content="你好"), AIMessage(content="再见"))
    assert model.invoke("第一轮").content == "你好"
    assert model.invoke("第二轮").content == "再见"


def test_bind_tools_returns_self_and_still_replays():
    def order_query(order_id: str) -> str:
        """查询订单（仅作 bind 参数，schema 被 fake 忽略）。"""
        return "ok"

    model = scripted_model(AIMessage(content="绑定后仍回放"))
    bound = model.bind_tools([order_query])
    assert bound is model
    assert bound.invoke("x").content == "绑定后仍回放"


def test_tool_call_script_passes_through():
    (call,) = scripted_model(TOOL_CALL).invoke("x").tool_calls
    assert call["name"] == "order_query"
    assert call["args"] == {"order_id": "o1"}
    assert call["id"] == "call_1"


def test_exhausted_script_raises():
    model = scripted_model(AIMessage(content="仅一条"))
    model.invoke("x")
    with pytest.raises(StopIteration):
        model.invoke("剧本已尽")


# ---------------------------------------------------------------- 剧本回放（流式）


async def test_tool_script_streams_tool_call_chunks_that_merge_back():
    chunks = [c async for c in scripted_model(TOOL_CALL).astream("查订单 o1")]
    assert len(chunks) == 2  # 工具块 + 末块；末块标 last，框架不再补空块
    assert chunks[0].tool_call_chunks[0]["name"] == "order_query"
    assert chunks[-1].chunk_position == "last"
    full = merge(chunks)
    assert full.tool_calls == [
        {
            "name": "order_query",
            "args": {"order_id": "o1"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]
    assert full.usage_metadata == USAGE
    assert full.response_metadata["finish_reason"] == "tool_calls"


async def test_text_script_streams_in_pieces_with_usage_on_last_chunk():
    model = scripted_model(AIMessage(content="您的订单已发货", usage_metadata=USAGE))
    chunks = [c async for c in model.astream("x")]
    assert [c.content for c in chunks] == ["您的订单", "已发货", ""]
    assert chunks[-1].usage_metadata == USAGE
    assert chunks[-1].response_metadata["finish_reason"] == "stop"
    assert merge(chunks).content == "您的订单已发货"


def test_sync_stream_matches_async_shape():
    chunks = list(scripted_model(TOOL_CALL).stream("x"))
    assert len(chunks) == 2
    assert merge(chunks).tool_calls[0]["args"] == {"order_id": "o1"}


@pytest.mark.filterwarnings("ignore:create_react_agent has been moved")
async def test_langgraph_messages_mode_consumes_tool_script_end_to_end():
    @tool
    def order_query(order_id: str) -> str:
        """查询订单状态。"""
        return f"订单 {order_id}：已发货"

    model = scripted_model(
        TOOL_CALL, AIMessage(content="您的订单已发货", usage_metadata=USAGE)
    )
    agent = create_react_agent(model, [order_query])
    kinds = [
        type(msg).__name__
        async for msg, _ in agent.astream(
            {"messages": [HumanMessage("查订单 o1")]}, stream_mode="messages"
        )
    ]
    # 工具块 → 末块 → 工具结果 → 文本块×2 → 末块：ainvoke 内部走了 _astream（探针 D）
    assert kinds == ["AIMessageChunk"] * 2 + ["ToolMessage"] + ["AIMessageChunk"] * 3
    out = await create_react_agent(
        scripted_model(TOOL_CALL, AIMessage(content="您的订单已发货")), [order_query]
    ).ainvoke({"messages": [HumanMessage("查订单 o1")]})
    assert [m.content for m in out["messages"] if isinstance(m, ToolMessage)] == [
        "订单 o1：已发货"
    ]
    assert out["messages"][-1].content == "您的订单已发货"


# ---------------------------------------------------------------- 默认回复 fake（线 B）


async def test_fake_reply_carries_nonzero_usage_on_both_paths():
    model = FakeReplyChatModel()
    reply = await model.ainvoke("x")
    assert reply.usage_metadata == FAKE_USAGE
    assert FAKE_USAGE["total_tokens"] > 0
    chunks = [c async for c in model.astream("x")]
    assert merge(chunks).content == reply.content == fakes.FAKE_REPLY
    assert chunks[-1].usage_metadata == FAKE_USAGE
    assert chunks[-1].chunk_position == "last"


async def test_fake_reply_delays_once_before_first_chunk(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(fakes, "_sleep", fake_sleep)
    chunks = [c async for c in FakeReplyChatModel(delay_s=0.5).astream("x")]
    assert slept == [0.5]  # 只在首块前睡一次，不是每块都睡
    assert len(chunks) > 1


def test_fake_reply_bind_tools_returns_self_and_ignores_schema():
    model = FakeReplyChatModel()
    assert model.bind_tools([{"type": "function", "function": {"name": "f"}}]) is model
    assert model.invoke("x").content == fakes.FAKE_REPLY
