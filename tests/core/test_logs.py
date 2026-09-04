"""日志落点：告警必须真的喊出来，且中文不被转义成 \\uXXXX。"""

import json
import logging

import pytest
import structlog

from app.core.logs import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_structlog():
    """configure_logging 改的是进程级全局配置：用完恢复默认，别让 WARNING 门槛泄漏到别的测试。"""
    yield
    structlog.reset_defaults()


def test_warning_is_emitted_as_json_line(capsys):
    configure_logging(logging.WARNING, json=True)
    get_logger("t").warning("模型不在价目表中", model="qwen-x", cost=0)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "模型不在价目表中"
    assert record["level"] == "warning"
    assert record["model"] == "qwen-x"
    assert "模型" in line  # ensure_ascii=False：中文原样


def test_levels_below_threshold_are_filtered(capsys):
    configure_logging(logging.WARNING, json=True)
    get_logger("t").info("不该出现")
    assert "不该出现" not in capsys.readouterr().out
