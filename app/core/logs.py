"""结构化日志：全仓"告警"的唯一落点。

价目表缺失、记账失败、缓存降级、熔断状态变化——凡 fail-open 之处都必须在这里喊一声，
静默降级等于事故没人知道。M1 只做进程内 JSON 行输出到 stdout；采集栈在 M4 裁决。
"""

import logging
import sys

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: int | str = logging.INFO, *, json: bool = True) -> None:
    """进程入口调用一次。json=False 时输出人类可读的控制台格式（dev 用）。"""
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    return structlog.get_logger(name)
