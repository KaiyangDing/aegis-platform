"""事件循环：Windows 开发进程用 Selector 循环（psycopg 异步拒绝 Proactor；ADR-011 决策 9）。

uvicorn：`uvicorn app.main:app --loop app.core.loops:selector_loop_factory`（非内建名的 --loop 串被
import 后直接当零参工厂用）；pytest：根 conftest 的 pytest_asyncio_loop_factories 钩子整套切换；
容器 Linux 默认就是 Selector，工厂在那里与默认等价。
psycopg 在 Proactor 下不是立刻报错：连接池后台反复重连、最后以 PoolTimeout（默认 30s）收场——
require_selector_loop 把它变成启动期的一句明白话。
"""

import asyncio
import sys

LOOP_INCOMPATIBLE = (
    "当前事件循环是 ProactorEventLoop，psycopg 异步不支持：Windows 开发进程请以 "
    "`--loop app.core.loops:selector_loop_factory` 启动（测试经 conftest 钩子已切换）"
)


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """零参工厂：uvicorn `--loop` 与 asyncio.run(loop_factory=…) 共用。"""
    return asyncio.SelectorEventLoop()


def require_selector_loop() -> None:
    """在运行中的循环里调用；Windows 的 Proactor 循环即抛 RuntimeError（其它平台无此类型，恒通过）。"""
    loop = asyncio.get_running_loop()
    if sys.platform == "win32" and isinstance(loop, asyncio.ProactorEventLoop):
        raise RuntimeError(LOOP_INCOMPATIBLE)
