"""token 尺：护栏用估算，账单用实测。

口径：CJK 基本区（U+4E00–U+9FFF）1 字≈1 token；其余字符 4 字符≈1 token，向上取整。
扩展区/假名/韩文按 4 字符/token 计，误差由预算数字自带的余量消化。
不引 tiktoken（OpenAI 词表对 Qwen 中文有系统性偏差）；不用框架的
count_tokens_approximately（按 4 字符/token 对 CJK 低估，ADR-005 实证）。
L1 单请求预算闸门与 L2 上下文预算共用这一把尺。
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

CJK_FIRST = "一"
CJK_LAST = "鿿"


def estimate_tokens(text: str) -> int:
    """纯函数：同输入同结果，是预算闸门可确定性断言的前提。"""
    cjk = sum(1 for ch in text if CJK_FIRST <= ch <= CJK_LAST)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def message_text(message: BaseMessage) -> str:
    """取消息的可计费文本。

    字符串 content 原样；分块 content 只取 text 块；AI 消息附带的 tool_calls
    以"名字 + 参数 JSON"计入——它们同样占上游的输入窗口。
    """
    content = message.content
    if isinstance(content, str):
        parts = [content]
    else:
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            parts.append(call["name"])
            parts.append(json.dumps(call["args"], ensure_ascii=False))
    return "".join(parts)


def estimate_messages_tokens(
    messages: Sequence[BaseMessage], tools: Sequence[Mapping[str, Any]] = ()
) -> int:
    """只估 prompt 侧：消息全文 + 工具 schema（整个 OpenAI 格式 dict 的 JSON）。"""
    total = sum(estimate_tokens(message_text(message)) for message in messages)
    for tool in tools:
        total += estimate_tokens(json.dumps(tool, ensure_ascii=False))
    return total
