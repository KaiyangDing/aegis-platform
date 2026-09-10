"""工具契约（契约 C7）：ToolDef（工具的完整说明书）、ToolContext（运行时注入的身份）、@tool 装饰器、
ToolRegistry、执行结局 OutcomeKind / ToolOutcome（M2.5）、批准后前置校验挂点 PrecheckVeto / PrecheckHook（M2.7），
以及 ToolDef → 框架 StructuredTool 的载体转换。

核心安全分野：LLM 只能提供业务参数（order_id 这类"查询条件"），身份（tenant_id / user_id）由运行时注入 ctx、
模型不可控——水平越权的第一道防线在类型签名上就成立。
注解即事实源：schema 与校验模型同源生成；一切注册期防呆在 import 时爆炸（ToolRegistrationError）。
v2 新增 to_structured_tools：框架只拿说明书（名字 / 说明 / args_model），执行由 ToolExec 中间件接管，
框架自己的 handler 是"不该被调用"的哨兵（ADR-012）。
"""

import inspect
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, get_type_hints

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, create_model


class SideEffect(StrEnum):
    """读写标记：恢复期"仅读可重发"、执行期"写恒单次"由此机器判定，不靠人读文档。"""

    READ = "read"
    WRITE = "write"


_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""OpenAI 兼容 tool schema 对函数名的硬约束（线格式现实），构造期就拦住。"""


class ToolRegistrationError(ValueError):
    """注册期防呆：工具定义的配置 bug 在 import 时就炸——启动时炸好过凌晨三点炸。"""


@dataclass(frozen=True, slots=True)
class ToolContext:
    """运行时注入给工具实现的身份与关联 id——全部 LLM 不可控。

    tool_call_id 即 write-ahead 落盘的 tool_call 事件 id（契约 C6 ④）：工具实现把它作为幂等键透传给下游
    （M3 退款服务按键去重）。它与模型侧的 tool_call id 是两种 id，严禁混用：模型侧 id 进对话配对 ToolMessage，
    事件 id 进 ctx 与事件。
    """

    tenant_id: str
    user_id: str
    session_id: str
    run_id: str
    tool_call_id: str

    def __post_init__(self) -> None:
        for name in ("tenant_id", "user_id", "session_id", "run_id", "tool_call_id"):
            if not getattr(self, name):
                raise ValueError(
                    f"{name} 不许为空——空租户/空幂等键意味着隔离或去重已失效"
                )


RiskPolicy = Callable[[Any, Mapping[str, Any]], bool]
"""风险闸门谓词：(已校验的工具参数, 租户配置) -> 是否需要 HITL 审批。
参数的真实类型是装饰器为各工具生成的 args 模型，运行时无法静态枚举，故 Any。
谓词自身崩溃 = 阻断（fail-closed，M2.5 / M2.7 消费）。"""


@dataclass(frozen=True, slots=True)
class PrecheckVeto:
    """批准后前置校验的否决（TOCTOU 挂点，ADR-013 决策 5）：审批的是数小时前的参数快照，执行前重跑业务校验。

    observation 回填模型：拿不到身份的层说出口的话必须对所有身份安全（统一话术）；detail 是审计细节（具体状态 / 金额上限），
    只进 precheck_vetoed 事件 payload 与日志——绝不进模型上下文与用户面。
    """

    observation: str
    detail: str | None = None


PrecheckHook = Callable[[str, Mapping[str, Any]], Awaitable[PrecheckVeto | None]]
"""(tool_name, 已校验的参数快照) -> None = 通过 / PrecheckVeto = 否决。校验逻辑（订单状态 / 可退余额）M3 注入；M2 缺席 = 全通过。
否决不终止：工具不执行（无 write-ahead，没有副作用要保护），observation 作为观察结果回填模型（ToolExec ③ 之后 ④ 之前）。"""


class OutcomeKind(StrEnum):
    """工具调用的五种结局（v1 逐字）。值进 ToolMessage.status 的映射与测试断言，快照钉死。"""

    OK = "ok"  # 成功：结果已入事件流
    ERROR = "error"  # 失败：错误文本回填给模型，它通常能自我修正
    RESULT_UNKNOWN = "result_unknown"  # 写工具超时 / 结果不明：禁止重试话术
    NEEDS_APPROVAL = "needs_approval"  # 风险闸门命中而无通行证：不执行（挂起由 Approvals 在 tools 之前接管，M2.7）
    DISABLED = "disabled"  # 本轮连败禁用：改道提示


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """一次工具调用的结局（ToolExec 内部值对象）。content 是回填给模型的观察结果——它是对话的一部分。"""

    kind: OutcomeKind
    tool_name: str
    content: str
    tool_call_id: str | None = (
        None  # write-ahead 之后才有：事件 id（幂等键），不是模型侧 id
    )


@dataclass(frozen=True, slots=True)
class ToolDef:
    """一个工具的完整说明书：给 LLM 看的、给执行器用的、给恢复期读的——单一事实源。

    side_effect 无默认值：是读是写必须显式声明（C15 防呆的类型层）。
    timeout_s=None 表示继承 LoopPolicy.tool_step_timeout_s，显式值与循环级上限取更严（M2.5）。
    写工具 retries 恒为 0：写操作绝不自动重试，幂等靠 write-ahead 键透传而不是"再试一次"。
    C15 三层防呆：写工具须有 risk_policy 或显式豁免；两者互斥；读工具不许豁免。
    """

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    side_effect: SideEffect
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)
    # 严格校验用；与 parameters_schema 同源生成（只有 @tool 会填）
    args_model: type[BaseModel] | None = None
    risk_policy: RiskPolicy | None = None
    risk_exempt: bool = False  # 豁免开关：写工具明示"我不需要审批"，留档可审计
    timeout_s: float | None = None
    retries: int = 0

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"工具名不合法（须匹配 LLM tool schema 硬约束），得到 {self.name!r}"
            )
        if not self.description.strip():
            raise ValueError(
                f"{self.name}: description 不许为空——它是给模型的说明书，空说明书=盲选工具"
            )
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(
                f"{self.name}: timeout_s 须 >0 或 None（继承循环级默认），得到 {self.timeout_s}"
            )
        if self.retries < 0:
            raise ValueError(f"{self.name}: retries 须 ≥0，得到 {self.retries}")
        if self.side_effect is SideEffect.WRITE and self.retries > 0:
            raise ValueError(f"{self.name}: 写工具禁止自动重试，retries 须为 0")
        # ——C15 注册期防呆（放在写禁重试之后，不遮蔽既有不变量的报错）——
        if self.risk_exempt and self.side_effect is not SideEffect.WRITE:
            raise ValueError(
                f"{self.name}: risk_exempt 仅对写工具有意义——读工具本就不过审批"
            )
        if self.risk_exempt and self.risk_policy is not None:
            raise ValueError(
                f"{self.name}: risk_policy 与 risk_exempt 互斥——要么有闸门要么显式豁免"
            )
        if (
            self.side_effect is SideEffect.WRITE
            and self.risk_policy is None
            and not self.risk_exempt
        ):
            raise ValueError(
                f"{self.name}: 写工具必须声明 risk_policy 或 risk_exempt=True"
                "（C15——沉默的危险按钮不许注册）"
            )


def _build_args_model(
    fn: Callable[..., Awaitable[Any]], tool_name: str
) -> type[BaseModel]:
    """从函数签名构建参数模型：注解即事实源，schema 与校验模型同源生成。"""
    sig = inspect.signature(fn)
    # get_type_hints 把字符串注解（工具模块若开了 from __future__ import annotations）解析回真类型
    hints = get_type_hints(fn)
    params = list(sig.parameters.values())
    if not params or params[0].name != "ctx" or hints.get("ctx") is not ToolContext:
        raise ToolRegistrationError(
            f"{tool_name}: 第一个参数必须是 ctx: ToolContext——身份由运行时注入，模型不可见"
        )
    fields: dict[str, Any] = {}
    for p in params[1:]:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolRegistrationError(
                f"{tool_name}: 不支持 *args/**kwargs——JSON Schema 表达不了"
            )
        if p.name not in hints:
            raise ToolRegistrationError(
                f"{tool_name}: 参数 {p.name} 缺类型注解——注解即 schema 的事实源"
            )
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (hints[p.name], default)
    # extra="forbid"：LLM 幻觉出的多余参数响亮拒绝，静默丢弃=掩盖模型行为异常
    return create_model(
        f"{tool_name}_args", __config__=ConfigDict(extra="forbid"), **fields
    )


def tool(
    *,
    side_effect: SideEffect,
    risk_policy: RiskPolicy | None = None,
    risk_exempt: bool = False,
    timeout_s: float | None = None,
    retries: int = 0,
    name: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], ToolDef]:
    """把 async 函数变成 ToolDef：docstring + 类型注解自动生成 schema（单一事实源）。

    装饰后模块级名字指向 ToolDef 而非函数（函数在 .handler 里）——工具从此是"说明书"，
    LLM 看 schema、执行器用 args_model、恢复期读 side_effect，各取所需。
    一切防呆在 import 时爆炸，统一抛 ToolRegistrationError。
    """

    def register(fn: Callable[..., Awaitable[Any]]) -> ToolDef:
        tool_name = name or fn.__name__
        model = _build_args_model(fn, tool_name)
        try:
            return ToolDef(
                name=tool_name,
                description=inspect.getdoc(fn) or "",
                handler=fn,
                side_effect=side_effect,
                parameters_schema=model.model_json_schema(),
                args_model=model,
                risk_policy=risk_policy,
                risk_exempt=risk_exempt,
                timeout_s=timeout_s,
                retries=retries,
            )
        except ValueError as e:
            # ToolDef 的构造期校验（空 description / C15 / 坏名字）统一换装成注册期异常
            raise ToolRegistrationError(str(e)) from e

    return register


class ToolRegistry:
    """工具注册表：把一组说明书排上架，自动建 dispatch 表（name → ToolDef），重名即拒。

    不做自动发现——注入是唯一入口（依赖倒置）：同一个运行时，生产喂真工具、测试喂演示工具集，
    换武器不换枪手。dict 保插入序 = specs() 顺序确定（工具顺序进 bind_tools，是行为轨迹确定性的一环）。
    """

    def __init__(self, tools: Iterable[ToolDef] = ()) -> None:
        self._by_name: dict[str, ToolDef] = {}
        for t in tools:
            self.add(t)

    def add(self, t: ToolDef) -> None:
        if t.name in self._by_name:
            raise ToolRegistrationError(
                f"工具名重复：{t.name}——dispatch 表无法唯一路由"
            )
        self._by_name[t.name] = t

    def get(self, name: str) -> ToolDef | None:
        """按名字取说明书。查不到返回 None 而非抛错——模型幻觉工具名是运行期常态，
        怎么处置（回填纠错、闸门 #5 计数）是 M2.4 的政策，注册表只管查表。"""
        return self._by_name.get(name)

    def specs(self) -> tuple[ToolDef, ...]:
        """产出喂给 AgentSpec.tools 的元组（冻结注入面）。"""
        return tuple(self._by_name.values())


def _bypassed_handler(name: str) -> Callable[..., Awaitable[Any]]:
    """框架侧 handler 哨兵：被调用 = 中间件栈没挂上 ToolExec，RuntimeError 裸穿 run（探针⑴：ToolNode 只把
    校验错误转 ToolMessage，其余异常重抛）——宁可整条 run 炸，也不让工具绕过七步执行。"""

    async def sentinel(**kwargs: Any) -> Any:
        raise RuntimeError(
            f"工具 {name} 的框架 handler 被调用——执行应由 ToolExec 中间件接管，"
            "检查中间件栈是否挂上 ToolExec"
        )

    return sentinel


def to_structured_tools(tools: Iterable[ToolDef]) -> list[StructuredTool]:
    """ToolDef → 框架 StructuredTool，只作 schema 载体喂给 create_agent / bind_tools。

    名字 / 说明 / args_model 三样进框架；LLM 看到的 parameters 由框架从 args_model 派生（探针⑴：
    剥掉 title 与顶层 additionalProperties，properties / required / 类型 / 默认值与 parameters_schema 同源）。
    只接受 @tool 生成的 ToolDef（有 args_model）：没有严格校验模型的工具不许暴露给模型。
    """
    out: list[StructuredTool] = []
    for t in tools:
        if t.args_model is None:
            raise ToolRegistrationError(
                f"{t.name}: 缺 args_model——只有 @tool 生成的 ToolDef 才能进图"
            )
        out.append(
            StructuredTool(
                name=t.name,
                description=t.description,
                args_schema=t.args_model,
                coroutine=_bypassed_handler(t.name),
            )
        )
    return out
