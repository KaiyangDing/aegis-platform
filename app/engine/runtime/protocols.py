"""运行时依赖的跨包协议：事件落盘（M2.1 立）；会话状态 / 审批单 / 取消信号随 M2.2 / M2.4 / M2.7 补。

方法签名只用内建类型（M1 自律②）：domain 靠结构匹配实现，不 import engine；
deps.py 是唯一同时 import 二者的模块（ADR-003 分层 + M2 第四条契约 gateway ↛ runtime）。
"""

from collections.abc import Mapping
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
