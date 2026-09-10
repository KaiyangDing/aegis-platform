"""模拟业务后端（S2；命运表 A8 降级：进程内类 + 幂等键字典判重；v1 为 PG 两表 + ON CONFLICT 同事务）。

它是两条防线的"下游端"：
- 归属校验的事实源：订单行带 tenant_id / user_id，get_order 先按租户过滤，工具内再双比对 user——判定权在工具，后端只交出行；
- write-ahead 幂等键的去重台账：写操作以 idempotency_key（= 透传的 tool_call 事件 id）判重，同一把钥匙第二次开门返回首次结果
  并标 duplicate=True、零第二次副作用——恢复期"凭原键重执行"在下游安全的物质基础。
绝不挂主 app；只经 get_backend() 单点取用，测试 set_backend 注入干净实例。钱用 Decimal：float 进门即定形。
write_hang_s 是测试接缝：写操作在副作用之后、返回之前挂起，配合 ToolExec 的超时演示"结果不明"（RESULT_UNKNOWN）——
副作用已发生而响应丢失，正是幂等键存在的理由。
"""

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.business.utterances import (
    DENIED_TEXT,
    REFUND_ALREADY,
    REFUND_NOT_POSITIVE,
    REFUND_OVER_PAID,
)


class BusinessRejected(Exception):
    """业务拒绝（越权 / 已退款 / 超额 / 非正金额）：不是故障——工具以 {"error": 话术} 回填模型，不进连败账。"""


@dataclass(slots=True)
class Order:
    id: str
    tenant_id: str
    user_id: str
    status: str  # paid / shipped / delivered / refunded
    paid_amount: Decimal
    items: tuple[dict[str, Any], ...] = ()
    created_at: str = "2026-09-01T10:00:00+08:00"


@dataclass(slots=True)
class MockBackend:
    write_hang_s: float = 0.0
    write_calls: int = 0  # 真实执行的写次数；去重命中不计——"零第二次副作用"的证人
    _orders: dict[tuple[str, str], Order] = field(default_factory=dict)
    _write_ops: dict[str, dict[str, Any]] = field(default_factory=dict)
    _tickets: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- 订单

    def add_order(self, order: Order) -> None:
        self._orders[(order.tenant_id, order.id)] = order

    def get_order(self, tenant_id: str, order_id: str) -> Order | None:
        """按 (tenant_id, order_id) 取行：跨租得 None——租户过滤在数据访问层，用户归属由工具双比对。"""
        return self._orders.get((tenant_id, order_id))

    # ---------------------------------------------------------------- 写操作（幂等键去重）

    async def _hang(self) -> None:
        if self.write_hang_s > 0:
            await asyncio.sleep(self.write_hang_s)

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        """同一把钥匙的首次结果（去重台账读面）；None = 没开过门。"""
        stored = self._write_ops.get(idempotency_key)
        return {**stored, "duplicate": True} if stored is not None else None

    async def apply_refund(
        self,
        *,
        tenant_id: str,
        user_id: str,
        order_id: str,
        amount: Decimal,
        idempotency_key: str,
    ) -> dict[str, Any]:
        replayed = self.replay(idempotency_key)
        if replayed is not None:
            return replayed
        order = self.get_order(tenant_id, order_id)
        if order is None or order.user_id != user_id:
            raise BusinessRejected(DENIED_TEXT)  # 防御性复核：工具已先校验归属
        if amount <= 0:
            raise BusinessRejected(REFUND_NOT_POSITIVE)
        if order.status == "refunded":
            raise BusinessRejected(REFUND_ALREADY)
        if amount > order.paid_amount:
            raise BusinessRejected(REFUND_OVER_PAID)
        self.write_calls += 1
        order.status = "refunded"
        result = {
            "order_id": order_id,
            "refunded": str(amount),
            "status": "refunded",
            "duplicate": False,
        }
        self._write_ops[idempotency_key] = result
        await self._hang()  # 副作用已落、响应未回：结果不明的真实形态
        return dict(result)

    async def create_ticket(
        self,
        *,
        tenant_id: str,
        user_id: str,
        title: str,
        detail: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        replayed = self.replay(idempotency_key)
        if replayed is not None:
            return replayed
        self.write_calls += 1
        ticket_id = f"T-{len(self._tickets) + 1:04d}"
        self._tickets.append(
            {
                "ticket_id": ticket_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "title": title,
                "detail": detail,
            }
        )
        result = {
            "ticket_id": ticket_id,
            "title": title,
            "by": user_id,
            "duplicate": False,
        }
        self._write_ops[idempotency_key] = result
        await self._hang()
        return dict(result)

    # ---------------------------------------------------------------- 演示种子

    def seed_demo(self) -> None:
        """两租户四单：A 的 u-a1 有一单可退（350）与一单已退；u-a2 一单；B 的 u-b1 一单——足够演示越权三路与审批阈值。"""
        for order in (
            Order(
                "AZ-1001",
                "tenant-a",
                "u-a1",
                "shipped",
                Decimal("350.00"),
                ({"sku": "杉木书架", "qty": 1},),
            ),
            Order(
                "AZ-1002",
                "tenant-a",
                "u-a1",
                "refunded",
                Decimal("120.00"),
                ({"sku": "台灯", "qty": 2},),
            ),
            Order(
                "AZ-2001",
                "tenant-a",
                "u-a2",
                "paid",
                Decimal("899.00"),
                ({"sku": "升降桌", "qty": 1},),
            ),
            Order(
                "BZ-1001",
                "tenant-b",
                "u-b1",
                "delivered",
                Decimal("59.90"),
                ({"sku": "滤芯", "qty": 3},),
            ),
        ):
            self.add_order(order)


def demo_backend(**kwargs: Any) -> MockBackend:
    backend = MockBackend(**kwargs)
    backend.seed_demo()
    return backend


_current: MockBackend | None = None


def get_backend() -> MockBackend:
    """进程级单点：首次取用即建演示种子（组合根 / 测试可先 set_backend 注入）。"""
    global _current
    if _current is None:
        _current = demo_backend()
    return _current


def set_backend(backend: MockBackend | None) -> None:
    """注入缝：测试每测一个干净实例，finally 归还 None。"""
    global _current
    _current = backend
