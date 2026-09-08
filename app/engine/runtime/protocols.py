"""运行时依赖的跨包协议：事件落盘与读取（M2.1 / M2.2）、会话调度状态（M2.2）、取消信号（M2.4）；审批单随 M2.7 补。

方法签名只用内建类型（M1 自律②）：domain 靠结构匹配实现，不 import engine；
deps.py 是唯一同时 import 二者的模块（ADR-003 分层 + M2 第四条契约 gateway ↛ runtime）。
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """事件落盘：一条一事务，事件写在钩子返回之前（先事实后 checkpoint，ADR-011）。

    event_id 由调用方派生（events.event_id）；重放命中既有 id 时不新建行、返回既有 seq。
    返回 (seq, created)：created=False 即去重命中——ToolExec 据此进 reexecute 分支（M2.5）。
    三岔口异常由实现定义（M2.2：围栏 EventWriteFenced / 不可用 EventStoreUnavailable），运行时裸穿不接。
    """

    async def append(
        self,
        *,
        event_id: str,
        tenant_id: str,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        task_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> tuple[int, bool]: ...


@runtime_checkable
class EventSource(Protocol):
    """按 seq 升序读会话事件（after_seq 游标语义）；每项是 AgentEvent 同名字段的字典（id 为 uuid 字符串）。

    tenant_id 与 session_id 双过滤：会话不属于该租户时得到空列表。消费者：D8 token 种子（M2.3）、恢复分诊（M2.9）、SSE（M3）。
    """

    async def read(
        self,
        tenant_id: str,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class EventStoreLike(EventSink, EventSource, Protocol):
    """门面要的完整事实源：既写又读（domain 的 EventStore 同时满足两者）。"""


class SessionRunState(StrEnum):
    """sessions.run_state 的四个值（domain 侧 RUN_STATES 与之同值，测试互钉）。

    五翻转全 CAS：T1 idle→running（起跑）/ T2 running→awaiting_approval（挂起）/ T3 awaiting→running（恢复）/
    T4 running→idle（终止）/ T5 running→failed（恢复放弃）。
    """

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    FAILED = "failed"


@runtime_checkable
class SessionStateLike(Protocol):
    """会话行的原语：读身份 / CAS 翻转（返回是否翻成）/ 恢复计数。建行不在协议里（API 与测试的事）。"""

    async def get(self, session_id: str) -> dict[str, Any] | None: ...

    async def transition(self, session_id: str, *, expected: str, to: str) -> bool: ...

    async def bump_recovery(self, session_id: str) -> int | None: ...

    async def reset_recovery(self, session_id: str) -> None: ...


@runtime_checkable
class CancelSignal(Protocol):
    """闸门 #6 的取消源：只要 is_set()（asyncio.Event 天然满足；M3 的 API 层把客户端断连 / 用户取消翻译成它）。

    检查点两处：每次 LLM 调用前（Gates.before_model）与每个工具调用前（ToolExec，M2.5）；命中即 cancelled 终止，零话术。
    """

    def is_set(self) -> bool: ...
