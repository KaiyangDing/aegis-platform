"""熔断测试的假时钟：MemoryBreaker 用 breakers._monotonic 判到期，推进它而不真等。"""

from app.engine.gateway import breakers


def fake_clock(monkeypatch):
    state = {"now": 1000.0}
    monkeypatch.setattr(breakers, "_monotonic", lambda: state["now"])

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return advance
