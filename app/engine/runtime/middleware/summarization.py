"""AegisSummarization：子类化框架 SummarizationMiddleware（契约 C9；ADR-012 决策 9）。

现成件的触发判定 / 切点算法（不拆 AI/Tool 对）/ RemoveMessage(REMOVE_ALL) 替换机制照用，换掉四样：
  token 尺 → 自家 estimate_tokens（框架版对中文低估 3–4 倍）；英文包装 → 中文 SUMMARY_WRAPPER；
  摘要模型的 with_retry → 关闭（重试权威唯一在网关，ADR-006；探针⒁：基类三次尝试、子类一次即返）；
  失败即抛 → fail-open 提取式降级（绝不二次调 LLM），只接网关六类公开异常、ProviderError 泄漏照样裸炸。
并在同一钩子写 summary_updated 事件（摘要是 LLM 产物、不可确定重算，不留痕回放重建不出模型视界；原文在事件流与旧 checkpoint，探针⒂）。
触发线 0.8 × history_budget（自家尺计 state 消息）、保留 history_budget // 2 的最新消息；history_budget < 2 即关层。
摘要模型 = 该租户网关：tier="fast" / session_id / deadline_s 经 ainvoke kwargs 到网关 _astream（探针 M2.6-Q1），内部调用打标不进外流。
before_model 按列表序在 Gates 之前（压缩先于预算预检）；注定终止的 run（已终止 / 已取消 / 轮数已满）不再压缩——增强层不为将死的 run 花钱。
"""

import json
from collections.abc import Iterable, Sequence
from typing import Any

from langchain.agents.middleware.internal_call_transformer import (
    internal_call_metadata,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from app.core.logs import get_logger
from app.core.tokens import estimate_tokens, message_text
from app.engine.runtime import utterances as u
from app.engine.runtime.context import (
    SUMMARY_SOURCE,
    clip_to_budget,
    is_summary,
    is_user_input,
)
from app.engine.runtime.events import EventType
from app.engine.runtime.spec import ContextConfig
from app.engine.runtime.state import (
    GATEWAY_FAILURES,
    INTERNAL_CALL_TAG,
    RunContext,
    RunState,
    emit,
)

logger = get_logger(__name__)

PREWARM_RATIO = 0.8
"""接近阈值提前压（v1 2026-07-11 拍板项 2）。"""
_HEAD_CHARS = 60


def count_message_tokens(messages: Iterable[object]) -> int:
    """框架 token_counter 形状 + 自家尺口径。形参按 object 接（框架的 MessageLikeRepresentation 是裸 Union 赋值、
    不带 TypeAlias，IDE 不认它是类型；object 比它更宽，作 token_counter 传入仍类型相容），非消息项不计。"""
    return sum(
        estimate_tokens(message_text(m)) for m in messages if isinstance(m, BaseMessage)
    )


def render_transcript(messages: Sequence[BaseMessage]) -> str:
    """喂给摘要模型的对话稿：逐条中文标注，确定性（同输入同文本）。"""
    lines: list[str] = []
    for message in messages:
        if is_summary(message):
            lines.append(f"此前摘要：{message.text}")
        elif isinstance(message, HumanMessage):
            lines.append(f"用户：{message.text}")
        elif isinstance(message, AIMessage):
            calls = "；".join(
                f"调用 {c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                for c in message.tool_calls
            )
            body = message.text.strip()
            if body and calls:
                lines.append(f"助手：{body}（{calls}）")
            elif calls:
                lines.append(f"助手：（{calls}）")
            elif body:
                lines.append(f"助手：{body}")
        elif isinstance(message, ToolMessage):
            lines.append(f"工具 {message.name or ''} 返回：{message.text}")
    return "\n".join(lines)


def _head(text: str) -> str:
    return text if len(text) <= _HEAD_CHARS else text[:_HEAD_CHARS] + "…"


def extractive_fallback(messages: Sequence[BaseMessage], budget_tokens: int) -> str:
    """摘要 LLM 失败时的确定性降级：按轮截取用户原话与助手回答的开头，正文裁进预算（前缀不裁——它是给模型看的降级声明）；
    绝不二次调 LLM。"""
    lines: list[str] = []
    for message in messages:
        if is_summary(message):
            lines.append(f"此前摘要：{_head(message.text)}")
        elif is_user_input(message):
            lines.append(f"用户：{_head(message.text)}")
        elif isinstance(message, AIMessage) and message.text.strip():
            lines.append(f"助手：{_head(message.text)}")
    body = clip_to_budget("\n".join(lines), budget_tokens) if lines else ""
    return u.SUMMARY_FALLBACK_PREFIX + (f"\n{body}" if body else "")


class AegisSummarization(SummarizationMiddleware):
    state_schema = RunState

    def __init__(self, model: BaseChatModel, *, config: ContextConfig) -> None:
        budget = config.history_budget
        trigger = (
            ("tokens", max(1, int(budget * PREWARM_RATIO))) if budget >= 2 else None
        )
        super().__init__(
            model,
            trigger=trigger,
            keep=("tokens", max(1, budget // 2)),
            token_counter=count_message_tokens,
            trim_tokens_to_summarize=None,
        )
        # 关掉框架的 with_retry：本类不经此属性调模型（走 _summarize），置回裸模型防误用（ADR-006 重试权威唯一在网关）
        self._summary_model = self.model
        self._config = config

    async def abefore_model(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        ctx = runtime.context
        if self._doomed(state, ctx):
            return None
        messages = state["messages"]
        self._ensure_message_ids(messages)
        if not self._should_summarize(messages, self.token_counter(messages)):
            return None
        cutoff = self._determine_cutoff_index(messages)
        if cutoff <= 0:
            return None
        to_summarize, preserved = self._partition_messages(messages, cutoff)
        summary, fallback = await self._summarize(to_summarize, ctx)
        await emit(
            runtime,
            EventType.SUMMARY_UPDATED,
            {
                "summary": summary,
                "covered": len(to_summarize),
                "kept": len(preserved),
                "fallback": fallback,
            },
            hook="summary_updated",
        )
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *self._build_new_messages(summary),
                *preserved,
            ]
        }

    @staticmethod
    def _doomed(state: RunState, ctx: RunContext) -> bool:
        """已终止（工具检查点）/ 已取消 / 轮数已满：Gates.before_model 紧接着就 jump end，压缩只会白花一次 LLM 调用。"""
        if state.get("termination") is not None:
            return True
        if ctx.cancel is not None and ctx.cancel.is_set():
            return True
        return state.get("iteration", 0) >= ctx.spec.policy.max_iterations

    async def _summarize(
        self, messages: list[BaseMessage], ctx: RunContext
    ) -> tuple[str, bool]:
        """fast 档摘要；网关六类公开异常 → 提取式降级（fallback=True）。空产物也降级：空摘要等于把历史蒸发。"""
        prompt = f"{u.SUMMARIZE_PROMPT}\n\n{render_transcript(messages)}"
        try:
            response = await self.model.ainvoke(
                [HumanMessage(prompt)],
                config={
                    "tags": [INTERNAL_CALL_TAG],
                    "metadata": {
                        "lc_source": SUMMARY_SOURCE,
                        **internal_call_metadata(),
                    },
                },
                tier="fast",
                session_id=ctx.session_id,
                deadline_s=ctx.spec.policy.llm_step_timeout_s,
            )
        except GATEWAY_FAILURES as exc:
            logger.warning(u.LOG_SUMMARY_FALLBACK, error=type(exc).__name__)
            return extractive_fallback(messages, self._fallback_budget()), True
        text = response.text.strip()
        if not text:
            logger.warning(u.LOG_SUMMARY_FALLBACK, error="empty")
            return extractive_fallback(messages, self._fallback_budget()), True
        return text, False

    def _fallback_budget(self) -> int:
        return max(1, self._config.history_budget // 2)

    @staticmethod
    def _build_new_messages(summary: str) -> list[HumanMessage]:
        """中文包装替换框架写死的英文 "Here is a summary of the conversation to date:"；lc_source 沿用框架值。"""
        return [
            HumanMessage(
                content=u.SUMMARY_WRAPPER.format(summary=summary),
                additional_kwargs={"lc_source": SUMMARY_SOURCE},
            )
        ]
