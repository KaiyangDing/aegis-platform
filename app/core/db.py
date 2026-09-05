"""数据库层：ORM 公共基类 + 引擎/会话工厂。M1 只为计量账本服务；RLS 低权角色与租户钩子 M3 立。

不在模块级建引擎：引擎绑定创建时的事件循环（asyncpg 连接不能跨 loop 复用，探针⑰），
由进程入口（lifespan）在自己的 loop 里建、关停时 dispose；测试每测各建一个。
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

POOL_SIZE = 5  # 常驻连接数（账本写入低并发；对话主链路不占 DB 连接）
MAX_OVERFLOW = 10  # 高峰可临时再借的连接数


class Base(DeclarativeBase):
    """所有 ORM 模型的公共基类——alembic 靠它的 metadata 发现表结构。"""


def make_engine(url: str) -> AsyncEngine:
    # pool_pre_ping：借出前探活，挡住数据库重启后的"半死连接"
    return create_async_engine(
        url, pool_size=POOL_SIZE, max_overflow=MAX_OVERFLOW, pool_pre_ping=True
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False：commit 后对象属性仍可读（async 下访问过期属性会隐式 IO，禁）
    return async_sessionmaker(engine, expire_on_commit=False)
