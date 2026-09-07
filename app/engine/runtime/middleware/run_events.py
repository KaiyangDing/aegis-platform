"""RunEvents：栈首中间件——首事件与末事件的唯一写入者（ADR-012 判据①）。

before_agent（列表序最先）：user_message 首事件 + run 级通道归零（tokens_used 从 D8 种子起）。
after_agent（列表序反转后最后）：assistant_message（终答或兜底话术）+ loop_terminated 末事件（八值 reason、iteration、detail、
cause?），然后 T4 running→idle（CAS 失败只告警：状态机被旁路是响亮留痕不掀收尾的事）+ 恢复计数清零。
终止事实只从 termination 通道读：谁终止谁写通道并追加兜底 AIMessage，这里不选话术、不加消息。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.core.logs import get_logger
from app.core.tokens import message_text
from app.engine.runtime import utterances as u
from app.engine.runtime.events import EventType
from app.engine.runtime.protocols import SessionRunState
from app.engine.runtime.spec import TerminationReason
from app.engine.runtime.state import RunContext, RunState, emit, run_channels_reset

logger = get_logger(__name__)


class RunEvents(AgentMiddleware[RunState, RunContext]):
    state_schema = RunState

    async def abefore_agent(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        content = message_text(state["messages"][-1])  # 本 run 的用户输入（列表尾）
        await emit(
            runtime, EventType.USER_MESSAGE, {"content": content}, hook="user_message"
        )
        return run_channels_reset(runtime.context.token_seed)

    async def aafter_agent(
        self, state: RunState, runtime: Runtime[RunContext]
    ) -> dict[str, Any] | None:
        ctx = runtime.context
        term = state.get("termination")
        iteration = state.get("iteration", 0)
        if term is None:
            last = state["messages"][-1]
            usage = last.usage_metadata if isinstance(last, AIMessage) else None
            stop = (
                last.response_metadata.get("finish_reason")
                if isinstance(last, AIMessage)
                else None
            )
            await emit(
                runtime,
                EventType.ASSISTANT_MESSAGE,
                {
                    "content": message_text(last),
                    "token_usage": usage["output_tokens"] if usage else None,
                },
                hook="assistant_message",
            )
            payload: dict[str, Any] = {
                "reason": TerminationReason.COMPLETED.value,
                "iteration": iteration,
                "detail": f"stop_reason={stop}",
            }
        else:
            if term.get("fallback") is not None:
                await emit(
                    runtime,
                    EventType.ASSISTANT_MESSAGE,
                    {"content": term["fallback"]},
                    hook="assistant_message",
                )
            payload = {
                "reason": term["reason"],
                "iteration": iteration,
                "detail": term["detail"],
            }
            if term.get("cause") is not None:
                payload["cause"] = term["cause"]
        await emit(runtime, EventType.LOOP_TERMINATED, payload, hook="loop_terminated")
        if ctx.sessions is not None:
            flipped = await ctx.sessions.transition(
                ctx.session_id,
                expected=SessionRunState.RUNNING.value,
                to=SessionRunState.IDLE.value,
            )
            if not flipped:
                logger.warning(u.LOG_RUN_STATE_FLIP_FAILED, session_id=ctx.session_id)
            await ctx.sessions.reset_recovery(ctx.session_id)
        return None
