"""L2 注入面类型：终止原因 + 循环策略 + 上下文预算 + AgentSpec（契约 C1 与 C2 的类型面）。

运行时对"客服"一无所知——prompt / 工具 / 策略 / 租户配置全部由 L3 经这些类型注入（v1 spec.py 逐字平移）。
校验强度跟着信任边界走：这里是受信代码的配置，frozen dataclass + 构造期防呆即可；
LLM 生成的工具参数才需要 pydantic 严校验（tools.py 的 args_model，M2.5 消费）。
v2 只改一处：model_tier 复用 L1 的 Tier 字面量（app.engine.gateway.routing）——档位语义两层同一事实源。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.engine.gateway.routing import TIERS, Tier
from app.engine.runtime.tools import ToolDef


class TerminationReason(StrEnum):
    """循环终止原因全集（v1 03 §2 七类 + 七类之外的 gateway_rejected）。

    值是稳定的 snake_case 字符串：进 loop_terminated 事件 payload 与行为轨迹断言，
    历史事件一旦落盘，改值 = 破坏重放——test_spec 的值快照会先红。
    GATEWAY_REJECTED 在"七类终止条件"之外：不是循环闸门，而是 L1 上抛的确定性拒绝
    （配置 / 协议 bug 信号），终止时不走兜底话术（契约 C4）。
    """

    COMPLETED = "completed"  # 0 正常完成
    MAX_ITERATIONS = "max_iterations"  # 闸门 1 最大轮数
    STEP_TIMEOUT = "step_timeout"  # 闸门 2 单步超时
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"  # 闸门 3 会话 token 预算
    REPEATED_CALLS = "repeated_calls"  # 闸门 4 重复调用
    PROTOCOL_VIOLATION = "protocol_violation"  # 闸门 5 协议违规
    CANCELLED = "cancelled"  # 闸门 6 取消 / HITL 拒绝或超时
    GATEWAY_REJECTED = "gateway_rejected"  # 七类之外：L1 确定性拒绝


TERMINATION_GATES: frozenset[TerminationReason] = frozenset(TerminationReason) - {
    TerminationReason.COMPLETED,
    TerminationReason.GATEWAY_REJECTED,
}
"""六道终止闸门（术语口径：7 类里除正常完成外的 6 项防护）。
全集减法而不是手写六个：新增成员时集合自动变化、测试 len == 6 立刻红，防术语漂移。
gateway_rejected 不在七类内，自然不算闸门。"""


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    """循环约束——终止条件表"默认阈值"列的家（闸门 1–5 的阈值 + 闸门 6 的审批时限）。

    frozen：策略在一次 run 内不许中途改动，要变换新实例（行为轨迹一致性依赖此语义）。
    approval_ttl_s 是审批单 expires_at 的生成依据（审批时限本质是租户级策略，M3 起经租户配置注入）；
    闸门 6 的触发源（取消信号 / 审批终局）仍在外部。
    llm_step_timeout_s 即传给网关的 deadline——与 L1 三段超时的嵌套约束由 deadline 传播保证，
    不做人肉算术校验（ADR-006）。session_token_budget 的生产值由 L3 从租户配置注入，默认值只服务
    运行时测试 / 演示；计数用 core/tokens.py 的估算值（护栏用估算、账单用实测）。
    tool_step_timeout_s 是循环级默认上限，单工具 ToolDef.timeout_s 更严时取更严（M2.5）。
    max_iterations 还决定图的 recursion_limit（M2.3 推导式）。
    """

    max_iterations: int = 10
    llm_step_timeout_s: float = 90.0
    tool_step_timeout_s: float = 30.0
    session_token_budget: int = 50_000
    repeat_call_limit: int = 3
    protocol_retry_limit: int = 2
    approval_ttl_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations 须 ≥1，得到 {self.max_iterations}")
        if self.llm_step_timeout_s <= 0:
            raise ValueError(
                f"llm_step_timeout_s 须 >0，得到 {self.llm_step_timeout_s}"
            )
        if self.tool_step_timeout_s <= 0:
            raise ValueError(
                f"tool_step_timeout_s 须 >0，得到 {self.tool_step_timeout_s}"
            )
        if self.session_token_budget < 1:
            raise ValueError(
                f"session_token_budget 须 ≥1，得到 {self.session_token_budget}"
            )
        if self.repeat_call_limit < 1:
            raise ValueError(f"repeat_call_limit 须 ≥1，得到 {self.repeat_call_limit}")
        if self.protocol_retry_limit < 0:
            raise ValueError(
                f"protocol_retry_limit 须 ≥0，得到 {self.protocol_retry_limit}"
            )
        if self.approval_ttl_s <= 0:
            raise ValueError(f"approval_ttl_s 须 >0，得到 {self.approval_ttl_s}")


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """六层上下文预算。单位 token，估算口径同 LoopPolicy。

    system 与 output_reserve 不许为 0（system 固定不可挤占；没有输出余量的循环无意义）；
    中间四层允许 0 = 显式关闭该层——非对称零值规则。长期记忆与本轮检索两层在 M2 只有槽位
    （恒 None），实现随 M3 RAG。
    """

    system_budget: int = 1_500
    memory_budget: int = 1_000
    history_budget: int = 4_000
    retrieval_budget: int = 3_000
    tool_results_budget: int = 3_000
    output_reserve: int = 4_000

    def __post_init__(self) -> None:
        if self.system_budget < 1:
            raise ValueError(f"system_budget 须 ≥1，得到 {self.system_budget}")
        if self.output_reserve < 1:
            raise ValueError(f"output_reserve 须 ≥1，得到 {self.output_reserve}")
        for name in (
            "memory_budget",
            "history_budget",
            "retrieval_budget",
            "tool_results_budget",
        ):
            value: int = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} 须 ≥0，得到 {value}")

    @property
    def input_total(self) -> int:
        """输入侧五层合计（不含输出余量），默认 12_500——编译器与预算对账用。"""
        return (
            self.system_budget
            + self.memory_budget
            + self.history_budget
            + self.retrieval_budget
            + self.tool_results_budget
        )


class SubAgentPolicy(StrEnum):
    """恒 DISABLED——前作为"只读子 Agent 并行调查"预留的接口位，v2 原样保留。

    只有一个成员是有意的：测试钉死 len==1，想加成员先让测试红、另立 ADR。
    """

    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """L3 注入运行时的全部内容——运行时对"客服"一无所知（依赖倒置的落点）。

    tools 用 tuple 不用 list：注入面是冻结的（一次 run 内不可变；M2.3 按 spec 指纹缓存编译好的图）。
    tenant_config 对运行时不透明：只透传给 risk_policy 等注入点，解释权在 L3——
    运行时不知道 approval_threshold 是什么。
    model_tier 复用 L1 的 Tier 字面量：Literal 只防静态，TIERS 运行时防线拦 L3 从配置读出的裸字符串。
    """

    system_prompt: str
    tools: tuple[ToolDef, ...] = ()
    policy: LoopPolicy = LoopPolicy()
    context_config: ContextConfig = ContextConfig()
    model_tier: Tier = "standard"
    sub_agent_policy: SubAgentPolicy = SubAgentPolicy.DISABLED
    tenant_config: Mapping[str, Any] = field(default_factory=dict)
    # 用户本人 PII 允许清单（手机 / 地址等），出口守卫据此区分"本人数据 vs 他人泄漏"（M2.8）；
    # L3 每会话装配时注入真实值，M2 测试用演示值
    owned_values: tuple[str, ...] = ()
    # 入口 LLM 分类器按租户开通、默认关：规则库是无条件底座，分类是增强层（fail-open），
    # 开关归注入面（与 sub_agent_policy 同型）；M3 从租户配置读
    entry_classifier: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt 不许为空——没有平台规则的 Agent 不许起跑")
        if self.model_tier not in TIERS:
            raise ValueError(f"model_tier 须为 {TIERS} 之一，得到 {self.model_tier!r}")
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"工具名重复：{dupes}——dispatch 表将无法唯一路由")
