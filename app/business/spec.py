"""AgentSpec 装配器（S2）：依赖倒置的落点——prompt / 工具集 / 策略 / 租户配置全部在此注入，运行时对"客服"一无所知。

工具面 = 攻击面：按租户配置的 tools 点名、一个不多（给没有退款业务的租户装 refund_apply"反正闸门会拦"是错的）；
点不到的名字 = 配置错误，preheat_specs 在启动期就炸而不是第一次请求时炸（命运表 B8 保留）；dict.fromkeys 去重保序。
owned_values 留空：指纹含它，按用户注入会让图缓存按用户分裂（命运表 B29 降级）。
memory / retrieval 两层预算显式 0：v2.0 无长期记忆、无检索，层关闭而不是留默认值假装存在。
"""

from app.business.config import tenant_config, tenant_ids
from app.business.tools import ALL_TOOLS
from app.business.utterances import SYSTEM_PROMPT_TEMPLATE
from app.engine.runtime.spec import AgentSpec, ContextConfig, LoopPolicy


def build_spec(tenant_id: str) -> AgentSpec:
    """按租户静态配置装配 AgentSpec。UnknownTenant 裸穿（API 层译 403）；未知工具名 / 非法档位 ValueError。"""
    cfg = tenant_config(tenant_id)
    names = [str(n) for n in cfg.get("tools", ())]
    unknown = sorted(set(names) - set(ALL_TOOLS))
    if unknown:
        raise ValueError(
            f"租户 {tenant_id} 配置了未知工具：{unknown}——可用：{sorted(ALL_TOOLS)}"
        )
    defaults = LoopPolicy()
    return AgentSpec(
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(tenant_name=cfg["name"]),
        tools=tuple(ALL_TOOLS[n] for n in dict.fromkeys(names)),
        policy=LoopPolicy(
            session_token_budget=int(
                cfg.get("session_token_budget", defaults.session_token_budget)
            ),
            approval_ttl_s=float(cfg.get("approval_ttl_s", defaults.approval_ttl_s)),
        ),
        context_config=ContextConfig(memory_budget=0, retrieval_budget=0),
        model_tier=cfg.get("model_tier", "standard"),
        tenant_config=cfg,
        owned_values=(),
        entry_classifier=bool(cfg.get("entry_classifier", False)),
    )


def preheat_specs() -> dict[str, AgentSpec]:
    """启动期把每个静态租户的 spec 构造一遍：工具白名单点名、档位、预算的配置错误在启动时炸。返回值供组合根缓存。"""
    return {tid: build_spec(tid) for tid in tenant_ids()}
