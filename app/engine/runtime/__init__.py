"""L2 运行时：注入面（AgentSpec / ToolDef）、事件事实、话术、跨包协议、图状态与门面。不被 gateway import（import-linter 第四条契约）。"""

from app.engine.runtime.events import (
    EVENT_ID_NAMESPACE,
    SCHEMA_VERSION,
    AgentEvent,
    EventType,
    event_id,
    normalize_event,
    normalize_events,
)
from app.engine.runtime.protocols import (
    EventSink,
    EventSource,
    EventStoreLike,
    SessionRunState,
    SessionStateLike,
)
from app.engine.runtime.runtime import (
    MIDDLEWARE_STACK,
    AgentRuntime,
    SessionBusy,
    recursion_limit_for,
    spec_fingerprint,
)
from app.engine.runtime.spec import (
    TERMINATION_GATES,
    AgentSpec,
    ContextConfig,
    LoopPolicy,
    SubAgentPolicy,
    TerminationReason,
)
from app.engine.runtime.state import RunContext, RunState
from app.engine.runtime.tools import (
    RiskPolicy,
    SideEffect,
    ToolContext,
    ToolDef,
    ToolRegistrationError,
    ToolRegistry,
    to_structured_tools,
    tool,
)

__all__ = [
    "EVENT_ID_NAMESPACE",
    "MIDDLEWARE_STACK",
    "SCHEMA_VERSION",
    "TERMINATION_GATES",
    "AgentEvent",
    "AgentRuntime",
    "AgentSpec",
    "ContextConfig",
    "EventSink",
    "EventSource",
    "EventStoreLike",
    "EventType",
    "LoopPolicy",
    "RiskPolicy",
    "RunContext",
    "RunState",
    "SessionBusy",
    "SessionRunState",
    "SessionStateLike",
    "SideEffect",
    "SubAgentPolicy",
    "TerminationReason",
    "ToolContext",
    "ToolDef",
    "ToolRegistrationError",
    "ToolRegistry",
    "event_id",
    "normalize_event",
    "normalize_events",
    "recursion_limit_for",
    "spec_fingerprint",
    "to_structured_tools",
    "tool",
]
