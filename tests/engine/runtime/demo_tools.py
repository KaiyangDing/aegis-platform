"""runtime 测试共用的演示工具集（不是测试文件）。

运行时对"客服"一无所知——用这套假业务工具排练注入：一个读工具、一个带风险闸门的写工具、
一个显式豁免的低危写工具（C15 的三种合法形态各占一席）。M2.3+ 的循环测试共用这一套道具。
"""

from collections.abc import Mapping
from typing import Any

from app.engine.runtime.tools import SideEffect, ToolContext, ToolRegistry, tool


@tool(side_effect=SideEffect.READ)
async def demo_order_query(ctx: ToolContext, order_id: str) -> dict:
    """查询演示订单的状态与实付金额。"""
    return {"order_id": order_id, "status": "已发货", "paid": 350}


def _demo_refund_needs_approval(args: Any, tenant_config: Mapping[str, Any]) -> bool:
    return bool(args.amount > tenant_config.get("approval_threshold", 200))


@tool(side_effect=SideEffect.WRITE, risk_policy=_demo_refund_needs_approval)
async def demo_refund_apply(ctx: ToolContext, order_id: str, amount: int) -> dict:
    """为演示订单发起退款（超租户阈值走人工审批）。"""
    return {"refunded": amount, "idempotency_key": ctx.tool_call_id}


@tool(side_effect=SideEffect.WRITE, risk_exempt=True)
async def demo_ticket_create(ctx: ToolContext, title: str) -> dict:
    """创建演示工单（低危写：显式豁免审批）。"""
    return {"ticket": title, "by": ctx.user_id}


def build_registry() -> ToolRegistry:
    """每次调用一个全新注册表——测试之间不共享可变状态。"""
    return ToolRegistry([demo_order_query, demo_refund_apply, demo_ticket_create])
