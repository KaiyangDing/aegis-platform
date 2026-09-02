"""fake 层自测：fake 本身先可信，其余一切测试才有根基（Argus test_fakes 纪律）。

三条断言事实均有 ADR-002 探针实证背书（core 1.6.1）。
"""

import pytest
from langchain_core.messages import AIMessage

from app.engine.fakes import scripted_model


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
    model = scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "order_query", "args": {"order_id": "o1"}, "id": "call_1"}
            ],
        )
    )
    (call,) = model.invoke("x").tool_calls
    assert call["name"] == "order_query"
    assert call["args"] == {"order_id": "o1"}
    assert call["id"] == "call_1"


def test_exhausted_script_raises():
    model = scripted_model(AIMessage(content="仅一条"))
    model.invoke("x")
    with pytest.raises(StopIteration):
        model.invoke("剧本已尽")
