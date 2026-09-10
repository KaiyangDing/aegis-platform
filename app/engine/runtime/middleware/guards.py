"""Guards：入口守卫中间件（契约 C13 入口段；ADR-012 栈序判据①之后：RunEvents 写完首事件才轮到它，压缩与闸门都在它之后）。

before_agent：规则库全量扫描 + （租户开通 entry_classifier 时）fast 档分类器合成裁决，只抬不压——
  HIGH → guardrail_triggered(refused) 事件 + 拒答 AIMessage + jump end：模型零调用；after_agent 仍运行（探针⒇）、以 COMPLETED 收尾
    （拒答是平台的正常回答，不是第七道闸门；loop_terminated.detail 带 stop_reason=guardrail_refused）；
  MEDIUM → guardrail_triggered(tagged) + 打标提醒进 entry_notice 通道（ModelCall 编译进 system 层，本 run 有效；user_message 事件保持原文）；
  分类器 fail-open → guardrail_triggered(classifier_fail_open) 审计后照常；规则零命中且无分类器 → 零事件。
分类器按 spec.entry_classifier 开通（默认关）：build_middleware 只在开通时把租户网关交给本中间件，每 run 按 session_id 构造分类器
（deadline 经 kwargs 传播到网关）；测试可直接注入 classifier。声明 can_jump_to=["end"]（未声明即静默无效，ADR-012 决策 4）。
出口守卫（挂点③）在 ModelCall wrap 内、包裹（挂点②）在 ToolExec——本中间件只管入口。
"""

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.core.tokens import message_text
from app.engine.runtime import utterances as u
from app.engine.runtime.events import EventType
from app.engine.runtime.guards import (
    INJECTION_RULES_V1,
    Classifier,
    Guardrails,
    InjectionRule,
    build_classifier,
    entry_audit_payload,
)
from app.engine.runtime.state import RunContext, RunState, emit

REFUSED_STOP = "guardrail_refused"
"""拒答 AIMessage 的 finish_reason：after_agent 据此写 loop_terminated.detail = stop_reason=guardrail_refused（前作 detail 带规则名，这里规则名在审计事件里）。"""


class Guards(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    def __init__(
        self,
        model: BaseChatModel | None = None,
        *,
        classifier: Classifier | None = None,
        rules: Sequence[InjectionRule] = INJECTION_RULES_V1,
    ) -> None:
        """model：租户网关（entry_classifier 开通时由 build_middleware 注入，None = 仅规则库）；classifier：测试直接注入的分类器（优先）。"""
        super().__init__()
        self._model = model
        self._classifier = classifier
        self._rules = tuple(rules)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        ctx = runtime.context
        user_input = message_text(state["messages"][-1])  # 本 run 的用户输入（列表尾）
        guards = Guardrails(rules=self._rules, classify=self._classifier_for(ctx))
        verdict = await guards.check_input(user_input)
        payload = entry_audit_payload(verdict)
        if payload is not None:
            await emit(
                runtime, EventType.GUARDRAIL_TRIGGERED, payload, hook="guardrail_entry"
            )
        if verdict.refuse:
            # 不写 termination：这不是闸门终止——after_agent 按末 AIMessage 走 COMPLETED 路径写 assistant_message + loop_terminated
            return {
                "messages": [
                    AIMessage(
                        content=u.REFUSAL_TEMPLATE,
                        response_metadata={"finish_reason": REFUSED_STOP},
                    )
                ],
                "jump_to": "end",
            }
        return {"entry_notice": verdict.notice}

    def _classifier_for(self, ctx: RunContext) -> Classifier | None:
        if self._classifier is not None:
            return self._classifier
        if self._model is None:
            return None
        return build_classifier(self._model, session_id=ctx.session_id)
