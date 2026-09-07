"""alembic 环境：异步引擎跑迁移；模型元数据由"导入即注册"收集。

连接串优先级：调用方 set_main_option("sqlalchemy.url") > Settings.database_url（.env）。
测试用前者指向 aegis_test（迁移=被测物，不用 create_all）。
LangGraph checkpointer 的四张表不在这里：由框架 setup() 自管（ADR-011 决策 8），漂移检查按名过滤。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# 导入即注册（四张业务表；只有最后一行需要 noqa，前几行的 app 绑定被后面的 import 复用）
import app.domain.approvals  # M2.2 审批单
import app.domain.events  # M2.2 事件事实源
import app.domain.sessions  # M2.2 会话调度状态
import app.domain.usage  # noqa: F401  —— M1.5b 计量账本
from app.core.config import get_settings
from app.core.db import Base

config = context.config
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只用 URL 生成 SQL 脚本，不建引擎。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
