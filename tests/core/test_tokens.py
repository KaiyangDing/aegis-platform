"""token 尺口径（v1 C25 沿用）：CJK 1 字=1 token，其余 4 字符=1 token 向上取整。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.tokens import (
    CJK_FIRST,
    CJK_LAST,
    estimate_messages_tokens,
    estimate_tokens,
    message_text,
)


def test_empty_is_zero():
    assert estimate_tokens("") == 0


def test_cjk_counts_one_per_char():
    assert estimate_tokens("退款申请") == 4


def test_ascii_counts_four_chars_per_token_rounding_up():
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_mixed_text_adds_both_parts():
    # 2 个 CJK + "refund" 6 字符 → 2 + ceil(6/4)=2
    assert estimate_tokens("退款refund") == 4


def test_cjk_range_is_basic_block_only():
    assert CJK_FIRST == "一" and CJK_LAST == "鿿"
    # 假名不在基本区：按 4 字符/token 计
    assert estimate_tokens("こんにちは") == 2


def test_message_text_takes_string_content_and_text_blocks():
    assert message_text(HumanMessage("你好")) == "你好"
    blocks = HumanMessage(
        [{"type": "text", "text": "你好"}, {"type": "image_url", "image_url": "x"}]
    )
    assert message_text(blocks) == "你好"


def test_message_text_includes_tool_call_name_and_args():
    ai = AIMessage(content="", tool_calls=[{"name": "f", "args": {"a": 1}, "id": "c1"}])
    assert message_text(ai) == 'f{"a": 1}'


def test_messages_estimate_sums_messages_and_tool_schemas():
    messages = [SystemMessage("abcd"), HumanMessage("退款申请")]
    assert estimate_messages_tokens(messages) == 1 + 4
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    with_tools = estimate_messages_tokens(messages, tools=[tool])
    assert with_tools > 5
    assert with_tools == 5 + estimate_tokens(
        '{"type": "function", "function": {"name": "f", "parameters": {}}}'
    )
