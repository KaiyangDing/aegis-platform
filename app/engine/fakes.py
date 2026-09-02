"""测试与压测线 B 共用的 fake 层：零真实 LLM 调用的根基。

剧本回放：模型按预写顺序吐出 AIMessage，内容与 tool_calls 完全由测试掌控。
开关（AEGIS_FAKE_LLM）已在 Settings 就位，接线在 M1 组合根；respx 属网关测试，M1 登场。
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolChatModel(GenericFakeChatModel):
    """支持 bind_tools 的剧本回放模型。

    GenericFakeChatModel.bind_tools 抛 NotImplementedError（core 1.6.1 探针实证，
    ADR-002），而 create_agent 必须调用它——覆写为返回自身：工具 schema 被忽略，
    模型行为只由剧本决定。
    """

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeToolChatModel:
        return self


def scripted_model(*messages: AIMessage) -> FakeToolChatModel:
    """从剧本构造 fake 模型：scripted_model(AIMessage(...), AIMessage(...))。"""
    return FakeToolChatModel(messages=iter(messages))
