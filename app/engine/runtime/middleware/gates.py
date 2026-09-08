"""Gates：六道闸门里住在钩子节点的四道——#1 轮数 / #6 取消（before_model）、#5 协议 / #4 重复（after_model）。
#3 会话预算与 #2 的 LLM 半边在 ModelCall（wrap 内，M2.3）；#2 的工具半边与 #6 的工具检查点在 ToolExec（M2.5）。ADR-012 决策 2。

两个钩子都显式 hook_config（决策 4）：before_model 只跳 end；after_model 跳 end（终止）或 model（纠错重试）。
终止 = 写 termination 通道 + 追加兜底 AIMessage（零话术不追加）+ jump end；loop_terminated 仍由 RunEvents.after_agent 单点写（决策 3）。
打断 / 拒绝 / 终止时最后一条 AIMessage 的每个 tool_call 都配对 ToolMessage（决策 5）：after_model 返回的 ToolMessage 经 reducer 追加，
模型→工具边只把仍未配对的调用送进 tools，全部配对则回到模型（探针⑼）。幻觉工具名在这里先于 ToolNode 判定、回填中文（决策 10）。
after_model 在列表里排在 Approvals（M2.7）之前——反序运行时它先跑，终止后的 jump end 绕过审批钩子（探针⑽）。
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from app.engine.runtime import utterances as u
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import (
    AEGIS_SOURCE,
    SOURCE_PROTOCOL_RETRY,
    RunContext,
    RunState,
    discard_note,
    terminated,
)

Kind = Literal["text", "tools", "violation"]


def canonical_key(name: str, args: Mapping[str, Any]) -> str:
    """闸门 #4 的规范形：工具名 + sort_keys 紧凑 JSON——键序 / 空白抖动不算换参数（v1 canonical_json 同口径）。"""
    return (
        name
        + ":"
        + json.dumps(
            args, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        )
    )


def classify(message: AIMessage) -> Kind:
    """三分支判定（v1 D7 / D18）：宣告工具停却没给调用 → 违规；有调用 → tools；非空文本 → text
    （max_tokens 截断也算完成：截断是预算现实不是协议错误）；否则违规。幻觉工具名不在此判——那要查注册表，归 _screen_calls。"""
    stop = message.response_metadata.get("finish_reason")
    if stop == "tool_calls" and not message.tool_calls:
        return "violation"
    if message.tool_calls:
        return "tools"
    if message.text.strip():
        return "text"
    return "violation"


def _not_executed(call: Mapping[str, Any], content: str) -> ToolMessage:
    return ToolMessage(
        content=content, tool_call_id=call["id"], name=call["name"], status="error"
    )


class Gates(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        if state.get("termination") is not None:
            # 工具检查点已终止（ToolExec 经 Command 写通道，M2.5）：不再调模型，直奔末事件
            return {"jump_to": "end"}
        ctx = runtime.context
        # 闸门 #6：取消信号在每次 LLM 调用前检查（每个工具前的检查点在 ToolExec）
        if ctx.cancel is not None and ctx.cancel.is_set():
            return self._end(
                terminated(TerminationReason.CANCELLED, detail="收到取消信号")
            )
        # 闸门 #1：按 LLM 调用计、调用前查（wrap 内的作废重发另有一道）
        iteration = state.get("iteration", 0)
        if iteration >= ctx.spec.policy.max_iterations:
            return self._end(
                terminated(
                    TerminationReason.MAX_ITERATIONS,
                    detail=f"已完成 {iteration} 次 LLM 调用，达 max_iterations 上限",
                )
            )
        return None

    @hook_config(can_jump_to=["end", "model"])
    async def aafter_model(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        if state.get("termination") is not None:
            # wrap 已终止（#1 重发上限 / #3 预检 / L1 四组）：绕过后续 after_model 钩子（审批）
            return {"jump_to": "end"}
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        policy = runtime.context.spec.policy
        kind = classify(last)
        if kind == "violation":
            # 闸门 #5：连续违规计数，超纠错上限终止；否则纠错提示以 user 消息注入并回到模型（system 层固定不可挤占）
            violations = state.get("violations", 0) + 1
            if violations > policy.protocol_retry_limit:
                return self._end(
                    terminated(
                        TerminationReason.PROTOCOL_VIOLATION,
                        detail=f"连续 {violations} 次协议违规，超过纠错上限",
                    ),
                    violations=violations,
                )
            return {
                "violations": violations,
                "messages": [
                    HumanMessage(
                        u.PROMPT_PROTOCOL_RETRY,
                        additional_kwargs={AEGIS_SOURCE: SOURCE_PROTOCOL_RETRY},
                    )
                ],
                "jump_to": "model",
            }
        if kind == "text":
            return {"violations": 0}  # 合法输出清零（连续计数语义）
        return self._screen_calls(last, state, runtime.context)

    def _screen_calls(
        self, last: AIMessage, state: RunState, ctx: RunContext
    ) -> dict[str, Any]:
        """工具轮：逐调用按声明序过闸门 #4（重复）与 #5（幻觉名）——v1 _run_tools 的判定半边；执行半边在 ToolExec。"""
        policy = ctx.spec.policy
        repeat = state.get("repeat") or {}
        key, streak = repeat.get("key"), repeat.get("streak", 0)
        violations = 0  # 工具轮本身是合法输出：先清零，再按幻觉名计
        paired: list[ToolMessage] = []
        calls = last.tool_calls
        total = len(calls)
        for index, call in enumerate(calls):
            # 闸门 #4：(工具名, 参数规范形) 连续计数，换 key 即重置
            current = canonical_key(call["name"], call["args"])
            if current == key:
                streak += 1
            else:
                key, streak = current, 1
            if streak > policy.repeat_call_limit:
                # 打断后原样再犯 → 终止；本调用与其后的调用全部配对"未执行"
                paired.extend(
                    _not_executed(c, u.TOOL_NOT_EXECUTED) for c in calls[index:]
                )
                return self._end(
                    terminated(
                        TerminationReason.REPEATED_CALLS,
                        detail=discard_note(
                            f"打断后仍第 {streak} 次重复调用 {call['name']}",
                            total,
                            index,
                        ),
                    ),
                    messages=paired,
                    violations=violations,
                    repeat={"key": key, "streak": streak},
                )
            if streak == policy.repeat_call_limit:
                # 达阈值：该次不执行（无 write-ahead 即无 tool_call 事件），打断话术配对回填；打断不清零
                paired.append(
                    _not_executed(
                        call,
                        u.PROMPT_REPEAT_BREAK.format(limit=policy.repeat_call_limit),
                    )
                )
                continue
            # 闸门 #5 的幻觉半边：语法是调用、语义非法——计违规，中文点名可用工具（先于 ToolNode 的英文回填）
            if ctx.registry.get(call["name"]) is None:
                violations += 1
                available = "、".join(t.name for t in ctx.registry.specs())
                paired.append(
                    _not_executed(
                        call,
                        u.TOOL_UNKNOWN.format(name=call["name"], available=available),
                    )
                )
                if violations > policy.protocol_retry_limit:
                    paired.extend(
                        _not_executed(c, u.TOOL_NOT_EXECUTED)
                        for c in calls[index + 1 :]
                    )
                    return self._end(
                        terminated(
                            TerminationReason.PROTOCOL_VIOLATION,
                            detail=discard_note(
                                f"幻觉工具名 {call['name']}，连续违规第 {violations} 次",
                                total,
                                index,
                            ),
                        ),
                        messages=paired,
                        violations=violations,
                        repeat={"key": key, "streak": streak},
                    )
        return {
            "violations": violations,
            "repeat": {"key": key, "streak": streak},
            "messages": paired,
        }

    @staticmethod
    def _end(
        term: dict[str, Any],
        *,
        messages: Sequence[ToolMessage] = (),
        **channels: Any,
    ) -> dict[str, Any]:
        """钩子内唯一的终止出口：termination 通道 + 配对 ToolMessage + 兜底 AIMessage（零话术不追加）+ jump end。"""
        out: dict[str, Any] = {"termination": term, "jump_to": "end", **channels}
        tail: list[Any] = list(messages)
        if term["fallback"] is not None:
            tail.append(AIMessage(content=term["fallback"]))
        if tail:
            out["messages"] = tail
        return out
