"""L2 运行时：注入面（AgentSpec / ToolDef）、事件事实、话术。不被 gateway import（import-linter 第四条契约）。"""

from app.engine.runtime.events import (
    EVENT_ID_NAMESPACE,
    SCHEMA_VERSION,
    AgentEvent,
    EventType,
    event_id,
    normalize_event,
    normalize_events,
)
from app.engine.runtime.protocols import EventSink
from app.engine.runtime.spec import (
    TERMINATION_GATES,
    AgentSpec,
    ContextConfig,
    LoopPolicy,
    SubAgentPolicy,
    TerminationReason,
)
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
    "SCHEMA_VERSION",
    "TERMINATION_GATES",
    "AgentEvent",
    "AgentSpec",
    "ContextConfig",
    "EventSink",
    "EventType",
    "LoopPolicy",
    "RiskPolicy",
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
    "to_structured_tools",
    "tool",
]
