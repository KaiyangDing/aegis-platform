"""三工具（S2；命运表 B3 降级五 → 三）：读 / 需审批写 / 豁免写各一，C15 三种合法形态各占一席。

身份全部取自 ctx（运行时注入，模型不可控）：归属判定权在工具内（fetch_owned_order 双比对 tenant + user），后端只交出行。
三种失败（不存在 / 他租 / 他人）逐字节同一话术 DENIED_TEXT——泄露"存在但无权"= 泄露他人订单号的有效性。
写工具一律把 ctx.tool_call_id（write-ahead 落盘的 tool_call 事件 id）作为幂等键透传后端（契约 C6 ④ 的下游端）。
归属是权限（handler 内 fail-closed），风险是闸门（放行但要人批）：refund_needs_approval 里没有 ctx，两道防线不混。
docstring 是给模型的说明书（@tool 机制）——机制注释一律写在函数体内。
"""

from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from app.business.backend import BusinessRejected, Order, get_backend
from app.business.utterances import DENIED_TEXT
from app.engine.runtime.tools import SideEffect, ToolContext, ToolDef, tool


async def fetch_owned_order(ctx: ToolContext, order_id: str) -> Order | None:
    """查单 + 归属校验（fail-closed）：任何一环不过一律 None，调用方回统一话术。
    后端 get_order 已按租户过滤；这里再显式双比对 tenant + user——谓词里没有任何 LLM 参数。"""
    order = get_backend().get_order(ctx.tenant_id, order_id)
    if (
        order is None
        or order.tenant_id != ctx.tenant_id
        or order.user_id != ctx.user_id
    ):
        return None
    return order


def refund_needs_approval(args: Any, tenant_config: Mapping[str, Any]) -> bool:
    """超过租户审批阈值挂 HITL。缺省 0 = fail-closed：租户漏配阈值时任意正金额都要人批，
    而不是"某个数以下静默直退"——同类闸门对同一种故障（少一个键）必须给同一个方向的答案。"""
    threshold = Decimal(str(tenant_config.get("approval_threshold", 0)))
    return bool(Decimal(str(args.amount)) > threshold)


@tool(side_effect=SideEffect.READ)
async def order_query(ctx: ToolContext, order_id: str) -> dict[str, Any]:
    """查询订单详情：状态、实付金额、商品列表。order_id 为订单号。"""
    # 归属校验 fail-closed；回给模型的视图剔除 tenant_id / user_id 身份列
    order = await fetch_owned_order(ctx, order_id)
    if order is None:
        return {"error": DENIED_TEXT}
    return {
        "order_id": order.id,
        "status": order.status,
        "paid_amount": str(order.paid_amount),
        "items": list(order.items),
        "created_at": order.created_at,
    }


@tool(side_effect=SideEffect.WRITE, risk_policy=refund_needs_approval)
async def refund_apply(
    ctx: ToolContext, order_id: str, amount: float
) -> dict[str, Any]:
    """为订单发起退款。order_id 为订单号，amount 为退款金额（元）。金额超过审批阈值时需人工批准。"""
    dec = Decimal(str(amount))  # float 进门立即定形，钱不过 float
    order = await fetch_owned_order(ctx, order_id)
    if order is None:
        return {"error": DENIED_TEXT}
    try:
        return await get_backend().apply_refund(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            order_id=order_id,
            amount=dec,
            idempotency_key=ctx.tool_call_id,  # write-ahead 事件 id 就是下游去重键
        )
    except BusinessRejected as e:
        # 业务拒绝：dict 话术回填、不抛异常——不进连败账、不禁用工具
        return {"error": str(e)}


# risk_exempt 理由（C15 要求豁免可审计）：建工单无资金面、无状态破坏面；转人工靠它无审批直通，
# 挂闸门会让"转人工"卡在待人批里
@tool(side_effect=SideEffect.WRITE, risk_exempt=True)
async def ticket_create(
    ctx: ToolContext, title: str, detail: str = ""
) -> dict[str, Any]:
    """创建人工工单：用于投诉、复杂问题上报或需要人工跟进的事项。title 为标题，detail 为详情。"""
    return await get_backend().create_ticket(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        title=title,
        detail=detail,
        idempotency_key=ctx.tool_call_id,  # 写工具一律带键
    )


ALL_TOOLS: Mapping[str, ToolDef] = MappingProxyType(
    {t.name: t for t in (order_query, refund_apply, ticket_create)}
)
"""工具注册清单（name → ToolDef）。租户白名单从这里点名取用，点不到的名字 = 配置错误启动炸；dict 保插入序。"""
