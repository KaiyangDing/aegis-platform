"""SSE 帧编码（M3.2；ADR-014）。

帧 = `id: {seq}` + `event: {事件类型}` + `data: {单行 JSON}`：id 就是 events.seq——断线重订阅的游标来自事实源而不是另起一套。
同一个编码器服务 POST 活流（AgentEvent）与 GET 回放（EventStore.read 的行字典）：两者字段名一致（as_row 归一）。
合成帧（done / error）不是事实源里的事件：不写 id: 行，重订阅不该拿它们当游标。
角色可见性：终端用户不可见完整 trace——内部事件（模型调用 / 摘要 / 守卫 / 前置校验）对 USER 过滤，坐席 / 管理员全量；
清单在此单点（ADR-014 决策 4），活流与回放共用。
"""

import json
from collections.abc import Mapping
from typing import Any

from app.core.auth import STAFF_ROLES, Role
from app.engine.runtime.events import AgentEvent

MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
INTERNAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "llm_call",
        "llm_result",
        "summary_updated",
        "guardrail_triggered",
        "precheck_vetoed",
    }
)


def as_row(event: AgentEvent | Mapping[str, Any]) -> dict[str, Any]:
    """AgentEvent 或行字典 → 帧的 data 视图（不带 tenant_id：客户端本来就知道自己是谁）。"""
    if isinstance(event, AgentEvent):
        return {
            "id": event.id,
            "seq": event.seq,
            "run_id": event.run_id,
            "type": event.type.value,
            "payload": dict(event.payload),
        }
    event_type = event["type"]
    return {
        "id": event["id"],
        "seq": event["seq"],
        "run_id": event["run_id"],
        "type": getattr(event_type, "value", event_type),
        "payload": dict(event["payload"]),
    }


def visible(event_type: str, role: Role) -> bool:
    return role in STAFF_ROLES or event_type not in INTERNAL_EVENT_TYPES


def encode(event: str, data: Mapping[str, Any], *, seq: int | None = None) -> bytes:
    body = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {event}\ndata: {body}\n\n".encode()


def event_frame(row: Mapping[str, Any]) -> bytes:
    return encode(row["type"], row, seq=row["seq"])


def done_frame(**data: Any) -> bytes:
    return encode("done", data)


def error_frame(detail: str, error: str) -> bytes:
    return encode("error", {"detail": detail, "error": error})
