"""计量账本：usage_ledger 的 ORM 模型、价目表与记账员（C10）。不 import engine。

四条纪律（v1 逐条平移，各自的反例见设计稿）：
- 钱用 Numeric/Decimal，永远不用 float——浮点误差在账本里是事故不是笑话；价目表 Decimal(str(p)) 中转；
- created_at 用数据库时钟（server_default）：多副本时钟会漂移，账本认一个报时员；月度聚合同样在 DB 端 date_trunc；
- 价目表缺失记 0 且告警：计费不崩溃，但静默记零是财务事故；
- 记账是独立工作单元（自己开会话自己提交）；失败由调用方兜住绝不拖垮请求，缺口留给对账脚本。
一调用一行；全表 tenant_id；上游缺 usage 以 usage_missing 标记（比合成 0 行诚实：0 是账，缺是未知）；
缓存回放记行但零成本，月度聚合排除 cached。
MeterLike 协议（engine/gateway/protocols.py）签名只用内建类型：本模块靠结构匹配实现，不认识 engine。
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base
from app.core.logs import get_logger

logger = get_logger(__name__)

# 话术例外登记（计划 §7）：domain 不得 import engine 的 utterances，本模块自带这一条告警文案
LOG_PRICE_MISSING = "模型不在价目表中，本行成本记 0（请尽快补配置）"
CACHE_PROVIDER = "cache"  # 命中回放行的 provider 值：命中率统计的分母在账本里
EMBEDDING_TIER = "embedding"  # embedding 通道行的 tier（自由字符串，不入档位枚举）

PriceTable = dict[str, tuple[Decimal, Decimal]]  # model → (输入价, 输出价)，元/千 token
_PER_THOUSAND = Decimal(1000)


class UsageRecord(Base):
    """usage_ledger：每次 LLM 调用一行——成本治理的原始账本。"""

    __tablename__ = "usage_ledger"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tier: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    cached: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # 缓存回放：记行但零成本
    # 上游没给 usage：两个 token 列记 0，但这一列说明"0 是缺失不是免费"
    usage_missing: Mapped[bool] = mapped_column(Boolean, default=False)
    cost: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal(0))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # 租户月度预算闸门与成本视图的主查询路径（tenant_id 单列查询走同一索引的前缀）
    __table_args__ = (Index("ix_usage_tenant_created", "tenant_id", "created_at"),)


def price_table(raw: Mapping[str, Sequence[float]]) -> PriceTable:
    """Settings 的 float 报价一次性转 Decimal（str 中转防二进制尾差）——组装边缘唯一转换点。"""
    return {
        model: (Decimal(str(prompt)), Decimal(str(completion)))
        for model, (prompt, completion) in raw.items()
    }


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached: bool,
    prices: PriceTable,
) -> Decimal:
    """一次调用的成本。纯函数：好测，且把"钱怎么算"钉死在一处。"""
    if cached:
        return Decimal(0)  # 缓存回放不打上游，一分钱不花——命中盖的章在此兑现
    pair = prices.get(model)
    if pair is None:
        # 新模型忘补价目表：计费不崩溃，但必须在日志里喊
        logger.warning(LOG_PRICE_MISSING, model=model)
        return Decimal(0)
    prompt_price, completion_price = pair
    return (
        Decimal(prompt_tokens) * prompt_price
        + Decimal(completion_tokens) * completion_price
    ) / _PER_THOUSAND


class MeteringRecorder:
    """记账员：把一次调用写成 usage_ledger 的一行；并为预算闸门提供月度聚合读路径。实现 MeterLike。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        prices: PriceTable,
    ) -> None:
        self._sf = session_factory
        self._prices = prices

    async def _insert(self, row: UsageRecord) -> None:
        # 工厂的 begin()：新会话 + 事务，正常退出即 commit 并关闭会话，异常即 rollback
        async with self._sf.begin() as session:
            session.add(row)

    async def record(
        self,
        *,
        tenant_id: str,
        request_id: str,
        session_id: str | None,
        tier: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
        usage_missing: bool,
    ) -> None:
        cost = compute_cost(
            model, prompt_tokens, completion_tokens, cached=cached, prices=self._prices
        )
        await self._insert(
            UsageRecord(
                request_id=request_id,
                tenant_id=tenant_id,
                session_id=session_id,
                tier=tier,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached=cached,
                usage_missing=usage_missing,
                cost=cost,
            )
        )

    async def month_spend(self, tenant_id: str) -> int:
        """该租户本月真实消耗的 token 总量（缓存回放不计——预算管的是花钱）。

        月初由数据库端 date_trunc 计算：账本认谁的钟，预算就认谁的钟。
        查询走 (tenant_id, created_at) 复合索引。
        """
        stmt = select(
            func.coalesce(
                func.sum(UsageRecord.prompt_tokens + UsageRecord.completion_tokens), 0
            )
        ).where(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.created_at >= func.date_trunc("month", func.now()),
            UsageRecord.cached.is_(False),
        )
        async with self._sf() as session:
            return int((await session.execute(stmt)).scalar_one())

    async def record_embedding(
        self,
        *,
        tenant_id: str,
        request_id: str,
        model: str,
        prompt_tokens: int,
        provider: str,
        session_id: str | None = None,
    ) -> None:
        """embedding 通道的一行账（协议缝，M3 接通道）：无输出侧计费、无缓存语义，计入月度预算。"""
        cost = compute_cost(model, prompt_tokens, 0, cached=False, prices=self._prices)
        await self._insert(
            UsageRecord(
                request_id=request_id,
                tenant_id=tenant_id,
                session_id=session_id,
                tier=EMBEDDING_TIER,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                cached=False,
                usage_missing=False,
                cost=cost,
            )
        )
