"""runtime 子目录夹具：演示工具注册表；网关退避不真睡。延迟导入——M2.1 未敲时 conftest 不能把整个目录的收集炸掉。"""

import pytest


@pytest.fixture
def demo_registry():
    from tests.engine.runtime.demo_tools import build_registry

    return build_registry()


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """网关受控重试的退避在测试里不真睡（与 gateway 测试同款接缝）。"""
    from app.engine.gateway import resilience

    async def nosleep(delay: float) -> None: ...

    monkeypatch.setattr(resilience, "_sleep", nosleep)
