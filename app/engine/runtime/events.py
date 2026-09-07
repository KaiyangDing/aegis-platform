"""事件类型、AgentEvent、事件 id 派生与行为轨迹归一化（契约 C5 类型面 / C14）。

事件是审计 / 回放 / SSE 游标的事实源（ADR-011 双写：checkpoint 管图恢复，events 表管其余三件）。
本文件只定义运行时侧类型与纯函数；表、seq 播种、去重落盘随 M2.2（domain/events.py，经 protocols.EventSink 注入）。
AgentEvent 是"已落盘事实"的镜像——能流出门面的事件必然已有 seq；不带时间戳（墙钟由 DB 落盘赋值，
运行时逻辑不读墙钟）。
v2 升级两处：id 不再是 uuid4，而是由崩溃前就确定的稳定任务身份派生（event_id）——重放命中即同 id，
落盘 ON CONFLICT DO NOTHING 就是去重；行上多 tenant_id（全表 tenant_id 纪律）与 task_id / checkpoint_id
（回指框架 checkpoint）。
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1
"""当前事件 payload 的 schema 版本。跨里程碑重构 payload 时版本 +1 并保留旧解析器，老事件永远可读。"""


class EventType(StrEnum):
    """事件类型全集（v1 17 类逐字快照）。值进 events 表与行为轨迹断言，测试钉死；加成员先让快照红。"""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    LLM_CALL = "llm_call"
    LLM_RESULT = "llm_result"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    APPROVAL_CANCELLED = "approval_cancelled"
    APPROVAL_EXPIRED = "approval_expired"
    SUMMARY_UPDATED = "summary_updated"
    LOOP_TERMINATED = "loop_terminated"
    HANDOFF = "handoff"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    RECOVERY_ABANDONED = "recovery_abandoned"
    PRECHECK_VETOED = "precheck_vetoed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """一条"步"级事实。粒度到步为止——逐 token 传输是 SSE 瞬态通道的事（M3）。

    payload 存原文（tool_result 完整结果进 payload，摘要只进上下文注入）。seq 由单写者在事务内递增、从 1 起；
    (session_id, seq) 唯一约束是并发写入的最后防线。task_id / checkpoint_id 是框架侧坐标（可为 None：
    图外写入的事件没有任务身份），审计时用来回指 checkpoint。
    """

    id: str
    tenant_id: str
    session_id: str
    run_id: str
    seq: int
    type: EventType
    payload: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION
    task_id: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id 不许为空——事件身份即幂等键，要出境到下游去重")
        if not self.tenant_id:
            raise ValueError("tenant_id 不许为空——事件必须归属租户（全表 tenant_id）")
        if not self.session_id:
            raise ValueError(
                "session_id 不许为空——事件必须归属会话（trace_id ≡ session_id）"
            )
        if not self.run_id:
            raise ValueError("run_id 不许为空——恢复计数与重放边界依赖它")
        if self.seq < 1:
            raise ValueError(f"seq 须 ≥1（单写者从 1 起递增），得到 {self.seq}")
        if self.schema_version < 1:
            raise ValueError(f"schema_version 须 ≥1，得到 {self.schema_version}")


# ---------------------------------------------------------------- 事件 id 派生

EVENT_ID_NAMESPACE = uuid.UUID("9aa11ec8-940c-5bf7-b377-cbf8faafe45d")
"""uuid5 命名空间，定了不动：改它 = 历史事件的 id 全部换算不出来。"""


def event_id(thread_id: str, task_id: str, hook: str, ordinal: str | int) -> str:
    """事件幂等键：uuid5(命名空间, [thread_id, task_id, hook, ordinal])。

    四段都来自崩溃前就确定的事实——thread_id = session_id；task_id = 框架 `__pregel_task_id`
    （崩溃重放与中断重放中与首次执行相同，探针 G2）；hook = 写入点名（"user_message" / "tool_call" …）；
    ordinal = 同一钩子内的序号或模型侧 tool_call id。于是同一步骤重放两次派生同一个 id，
    落盘 ON CONFLICT DO NOTHING 即去重；幂等键先于副作用存在（契约 C6 ④）。
    空段拒绝：任一段为空会让不同步骤的键塌缩到一起（去重误伤）。
    """
    parts = (thread_id, task_id, hook, str(ordinal))
    for label, value in zip(("thread_id", "task_id", "hook", "ordinal"), parts):
        if not value:
            raise ValueError(
                f"event_id 的 {label} 不许为空——空段会让不同步骤的幂等键塌缩"
            )
    return str(uuid.uuid5(EVENT_ID_NAMESPACE, json.dumps(parts, separators=(",", ":"))))


# ---------------------------------------------------------------- 行为轨迹归一化（C31）

_EXEMPT_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "latency_ms",
        "duration_ms",
        "usage",
        "prompt_tokens",
        "completion_tokens",
        "expires_at",
    }
)
"""豁免键（只滴 payload 顶层——递归滴除会误伤 result 原文里的同名业务字段）：
墙钟产物（latency / duration / expires_at）与供应商实测 usage 数值，重跑必然波动；
自家尺的估算值不豁免（确定性，是断言的一部分）。"""


def normalize_event(
    event: AgentEvent, *, id_aliases: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """单事件归一化。

    参与比较：type / schema_version / payload 其余全部键值；豁免：顶层墙钟与 usage 键。
    tool_call_id / event_id 命中别名表 → 替换（幂等引用结构保真）；不命中 → 保留原值
    （引用流外事件 = bug，让断言响亮失败）。payload 先做 canonical JSON 往返——
    "刚落盘的事件"（含 Decimal 等原生对象）与"DB 读回的事件"（JSONB 已 JSON 化）才可比。
    """
    aliases: Mapping[str, str] = id_aliases or {}
    payload: dict[str, Any] = json.loads(
        json.dumps(dict(event.payload), ensure_ascii=False, default=str)
    )
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _EXEMPT_PAYLOAD_KEYS:
            continue
        if key in ("tool_call_id", "event_id") and isinstance(value, str):
            normalized[key] = aliases.get(value, value)
        else:
            normalized[key] = value
    return {
        "type": event.type.value,
        "schema_version": event.schema_version,
        "payload": normalized,
    }


def normalize_events(events: Sequence[AgentEvent]) -> list[dict[str, Any]]:
    """流级归一化：`normalize_events(A) == normalize_events(B)` 即行为轨迹等价（Q7 降级方案的断言本体）。

    纯函数（不读时钟 / 全局状态）。事件 id 按流序别名 e1..eN；approval_id 按首现顺序别名 a1..aM。
    tenant_id / session_id / run_id / seq / task_id / checkpoint_id 不进输出——相对序由列表顺序承载，
    "seq 连续合法"与"task_id 可回指"是独立的不变量断言，不混进等价性。
    """
    id_aliases = {e.id: f"e{i + 1}" for i, e in enumerate(events)}
    out = [normalize_event(e, id_aliases=id_aliases) for e in events]
    approval_aliases: dict[str, str] = {}
    for item in out:
        value = item["payload"].get("approval_id")
        if isinstance(value, str):
            item["payload"]["approval_id"] = approval_aliases.setdefault(
                value, f"a{len(approval_aliases) + 1}"
            )
    return out
