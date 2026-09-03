"""档位路由：调用方只声明 fast/standard/strong，永不写模型名。

路由表在进程启动时整体校验四项（provider:model 形式 / provider 已知 / 链非空 / 三档齐全）——
配置错误要在启动时炸，不许拖到凌晨第一个 strong 请求用 KeyError 告诉你。
能力断崖（strong 链退到 fast 级模型）不做硬校验：是配置纪律与设计稿审查项。
Tier 只在 engine 定义：core 不认识档位与候选。
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from app.engine.gateway import utterances as u

Tier = Literal["fast", "standard", "strong"]
TIERS: tuple[str, ...] = get_args(Tier)


@dataclass(frozen=True, slots=True)
class Candidate:
    """候选 = 供应商 + 模型名。frozen：可作 dict 键（候选实例表、熔断器表）。"""

    provider: str
    model: str

    @property
    def key(self) -> str:
        """`provider:model`——熔断 namespace、故障注入点名、日志字段共用的同一把钥匙。"""
        return f"{self.provider}:{self.model}"


def parse_routes(
    raw: Mapping[str, Sequence[str]], known_providers: Collection[str]
) -> dict[str, list[Candidate]]:
    """启动即校验：路由配置错误在进程启动时炸，不许拖到运行时。"""
    routes: dict[str, list[Candidate]] = {}
    for tier, entries in raw.items():
        chain: list[Candidate] = []
        for entry in entries:
            provider, sep, model = entry.partition(":")
            if not sep or not provider or not model or provider not in known_providers:
                raise ValueError(u.ROUTE_ENTRY_INVALID.format(tier=tier, entry=entry))
            chain.append(Candidate(provider, model))
        if not chain:
            raise ValueError(u.ROUTE_CHAIN_EMPTY.format(tier=tier))
        routes[tier] = chain
    # 齐档校验：MODEL_ROUTES 被环境变量整体覆盖时最容易漏档
    missing = set(TIERS) - routes.keys()
    if missing:
        raise ValueError(u.ROUTE_TIERS_MISSING.format(missing=sorted(missing)))
    return routes


def unique_candidates(routes: Mapping[str, Iterable[Candidate]]) -> list[Candidate]:
    """三档链条里出现过的候选去重（保持首次出现顺序）：一个候选一个模型实例、一把熔断器。"""
    seen: dict[Candidate, None] = {}
    for chain in routes.values():
        for candidate in chain:
            seen.setdefault(candidate)
    return list(seen)
