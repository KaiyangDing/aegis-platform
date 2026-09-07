"""事件循环（M2.2；ADR-011 决策 9）：工厂给 Selector 循环；全仓测试已跑在 Selector 上；Proactor 被明白话拒绝。"""

import asyncio
import sys

import pytest

pytest.importorskip("app.core.loops", reason="M2.2 未敲：app/core/loops.py 不存在")

from app.core.loops import require_selector_loop, selector_loop_factory


def test_factory_returns_selector_loop():
    loop = selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


async def test_suite_runs_on_selector_loop():
    """conftest 钩子生效的证据：本测试自己就跑在 Selector 循环上，require 静默通过。"""
    assert isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop)
    require_selector_loop()


@pytest.mark.skipif(sys.platform != "win32", reason="Proactor 循环只有 Windows 有")
def test_require_rejects_proactor_loop():
    async def probe() -> None:
        require_selector_loop()

    with pytest.raises(RuntimeError, match="selector_loop_factory"):
        asyncio.run(probe(), loop_factory=asyncio.ProactorEventLoop)
