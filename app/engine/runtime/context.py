"""上下文编译（契约 C8）：prompt 是按预算编译出来的，不是把 state 原样拼上去——"prompt 是 state 的有损投影"。

只改 ModelRequest、不改 state：ModelCall.awrap_model_call 在调网关前调用 compile_prompt，把 request 的 system_message / messages
换成编译产物；checkpoint 里的对话原文一字不动。六层（v1 03 §3）在 v2 的落点：
  system：spec.system_prompt + 不可信数据声明（M2.8 再加入口打标）——超 system_budget 即 ValueError fail-loud（固定层没有合法降级，那是 L3 配置 bug）；
  长期记忆 / 本轮检索：槽位恒 None（M3 RAG）；
  会话历史：摘要消息（AegisSummarization 的产物，已在 state 里）+ 旧轮（每轮压成 user 原话 + 最终 assistant 文本；旧轮的工具往返不进 prompt）
    按 history_budget − 当前 user 输入 从最新往回装、装不下即停；有旧轮排队时摘要至多占一半版面（肥摘要不许挤掉最新轮）；
  当前 user 原文：恒保留、绝不裁剪——挤掉它 = 答非所问；
  工具结果层：本轮工作序列（AI(tool_calls) / ToolMessage / 纠错提示）超 tool_results_budget 时从最老的 ToolMessage 起整条折叠为
    带 tool_call_id 的标注（原文在事件流），AI 的 tool_calls 不动（协议结构）；
  输出余量：max_tokens = output_reserve（ModelCall 经 model_settings 下发）。
确定性红线：不读时钟、不随机；同 state 同 spec ⇒ 同产物（行为轨迹断言的前提）。
空 AIMessage（零话术终止 / 协议违规留下的）在此丢弃（M2.3 登记 L2）。
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.core.logs import get_logger
from app.core.tokens import estimate_tokens, message_text
from app.engine.runtime import utterances as u
from app.engine.runtime.spec import AgentSpec
from app.engine.runtime.state import AEGIS_SOURCE

logger = get_logger(__name__)

SUMMARY_SOURCE = "summarization"
"""框架 SummarizationMiddleware 给摘要消息打的 lc_source 值（additional_kwargs），本仓子类沿用——编译器据此认出摘要消息。"""
SUMMARY_PROMPT_SHARE = 0.5
"""有旧轮排队时摘要在历史层版面的最大份额（v1 复盘补丁三）：最新轮的席位是结构保证；无人排队则不设限。"""


def is_summary(message: BaseMessage) -> bool:
    return (
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source") == SUMMARY_SOURCE
    )


def is_user_input(message: BaseMessage) -> bool:
    """用户原话：HumanMessage 且不是运行时注入（纠错提示）也不是摘要。"""
    return (
        isinstance(message, HumanMessage)
        and not is_summary(message)
        and not message.additional_kwargs.get(AEGIS_SOURCE)
    )


def _is_empty_assistant(message: BaseMessage) -> bool:
    return (
        isinstance(message, AIMessage)
        and not message.tool_calls
        and not message.text.strip()
    )


def _tokens(message: BaseMessage) -> int:
    """自家尺口径：content + AI 的 tool_calls 名字与参数（真实进 prompt 的负载；角色 / 结构开销由预算余量消化）。"""
    return estimate_tokens(message_text(message))


def clip_to_budget(text: str, budget_tokens: int) -> str:
    """确定性截断：0.8 倍循环缩短进预算，尾部标注去向（原文在事件流）；预算内原样返回。"""
    if estimate_tokens(text) <= budget_tokens:
        return text
    while estimate_tokens(text) > budget_tokens and len(text) > 1:
        text = text[: max(1, int(len(text) * 0.8))]
    return text + u.CLIP_SUFFIX


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """编译产物：system + 其余消息（顺序即层序）；折叠 / 丢弃留痕供测试与审计。"""

    system: SystemMessage
    messages: list[BaseMessage]
    folded: tuple[str, ...]  # 被折叠的 ToolMessage 的 tool_call_id（模型侧 id）
    dropped_turns: int  # 历史层装不下而丢弃的旧轮数
    summary_clipped: bool

    @property
    def all_messages(self) -> list[BaseMessage]:
        return [self.system, *self.messages]


def compile_prompt(messages: Sequence[BaseMessage], spec: AgentSpec) -> CompiledPrompt:
    """按层编译（v1 D12 次序）：system → [摘要 → 旧轮] → 当前 user → 本轮工作序列（工具结果层）。"""
    config = spec.context_config
    # ① system 层：固定不可挤占——超预算没有合法降级
    system_text = f"{spec.system_prompt}\n\n{u.UNTRUSTED_NOTICE}"
    system_cost = estimate_tokens(system_text)
    if system_cost > config.system_budget:
        raise ValueError(
            f"system_prompt 估算 {system_cost} token 超出 system_budget={config.system_budget}"
            "——固定层不可挤占（03 §3）"
        )
    system = SystemMessage(content=system_text)
    summary, turns = _split(messages)
    if not turns:
        return CompiledPrompt(system, [], (), 0, False)
    *older, current = turns
    # ⑤ 当前 user 原文恒保留；⑥ 工具结果层确定性折叠（空 AIMessage 丢弃）
    user, working = current[0], [m for m in current[1:] if not _is_empty_assistant(m)]
    working, folded = _fold(working, config.tool_results_budget)
    # ③ 会话历史层：摘要 + 旧轮，预算先扣当前 user 的份额
    history, dropped, clipped = _history(
        summary, older, budget=config.history_budget - _tokens(user)
    )
    return CompiledPrompt(system, [*history, user, *working], folded, dropped, clipped)


def _split(
    messages: Sequence[BaseMessage],
) -> tuple[HumanMessage | None, list[list[BaseMessage]]]:
    """摘要消息单独摘出（框架 REMOVE_ALL 后至多一条）；其余按"用户原话起一轮"分组，摘要切点留下的无 user 前导段自成一轮。"""
    summary: HumanMessage | None = None
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if is_summary(message):
            summary = message  # type: ignore[assignment]
        elif is_user_input(message) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return summary, turns


def _fold(
    working: list[BaseMessage], budget: int
) -> tuple[list[BaseMessage], tuple[str, ...]]:
    """从最老一条 ToolMessage 起整条替换为折叠标注，直至层内 ≤ 预算；AI 的 tool_calls 不折（协议字段）。"""
    out = list(working)
    if sum(_tokens(m) for m in out) <= budget:
        return out, ()
    folded: list[str] = []
    for index, message in enumerate(out):
        if not isinstance(message, ToolMessage):
            continue
        out[index] = ToolMessage(
            content=u.FOLDED_TOOL_TEMPLATE.format(tool_call_id=message.tool_call_id),
            tool_call_id=message.tool_call_id,
            name=message.name,
            status=message.status,
        )
        folded.append(message.tool_call_id)
        if sum(_tokens(m) for m in out) <= budget:
            break
    else:  # 全部折叠仍超预算：照放 + 响亮留痕，余量消化
        logger.warning(u.LOG_TOOL_FOLD_OVER_BUDGET, budget=budget)
    return out, tuple(folded)


def _reduce_turn(turn: list[BaseMessage]) -> list[BaseMessage]:
    """旧轮压成 [user 原话?, 最终 assistant 文本?]：同轮多条 assistant 取最后一条有文字的；工具往返不进历史层。"""
    out: list[BaseMessage] = []
    if is_user_input(turn[0]):
        out.append(HumanMessage(content=turn[0].text))
    final = next(
        (m for m in reversed(turn) if isinstance(m, AIMessage) and m.text.strip()),
        None,
    )
    if final is not None:
        out.append(AIMessage(content=final.text))
    return out


def _history(
    summary: HumanMessage | None,
    older: list[list[BaseMessage]],
    *,
    budget: int,
) -> tuple[list[BaseMessage], int, bool]:
    """摘要（份额裁剪）+ 旧轮从最新往回装（装不下即停：保住最新轮的连续后缀）。返回 (消息, 丢弃轮数, 摘要是否被裁)。"""
    if budget < 0:
        logger.warning(u.LOG_HISTORY_CLEARED)
        budget = 0
    reduced = [r for r in (_reduce_turn(t) for t in older) if r]
    out: list[BaseMessage] = []
    remaining = budget
    clipped = False
    if summary is not None and remaining > 0:
        allowed = int(remaining * SUMMARY_PROMPT_SHARE) if reduced else remaining
        text = summary.text
        if estimate_tokens(text) > allowed:
            text = clip_to_budget(
                text, max(0, allowed - estimate_tokens(u.CLIP_SUFFIX))
            )
            clipped = True
        message = HumanMessage(
            content=text, additional_kwargs=dict(summary.additional_kwargs)
        )
        if (
            _tokens(message) <= remaining
        ):  # 连"截断标注"都装不下 ⇒ 摘要退出 prompt（事件仍在）
            out.append(message)
            remaining -= _tokens(message)
    kept: list[list[BaseMessage]] = []
    for turn in reversed(reduced):
        cost = sum(_tokens(m) for m in turn)
        if cost > remaining:
            break
        kept.append(turn)
        remaining -= cost
    for turn in reversed(kept):
        out.extend(turn)
    return out, len(reduced) - len(kept), clipped
