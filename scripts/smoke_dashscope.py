"""联网冒烟：只核对 DashScope 兼容端点的 SSE 真实形态，预算护栏写死，结果写进 reports/。

三验（每个路由表里的候选各打一次，max_tokens 极小）：
  1. 模型存在且能流式回复（不是"幻影候选"）；
  2. 请求体带 enable_thinking=false 是否被接受；
  3. usage 块的位置与形态（是否单独一块、choices 是否为空）、finish_reason 的位置与取值、[DONE] 哨兵是否在场。
另打一次 enable_thinking=true（便宜模型 + thinking_budget 上限）看 reasoning_content 的原始形态；
再经本仓网关（fake 关、缓存关、账本关）走一次 fast 档，核对框架侧 usage_metadata / finish_reason 与首块延迟。

分账：本脚本不建账本会话工厂（meter=None），真实花费只出现在报告里；账本里的 fake 记账与真实花费永不混在一张表。
护栏：BUDGET_CNY 写死；每次调用前按自家 token 尺估算输入 + max_tokens 上限预扣，超预算直接跳过；
调用后按真实 usage 记账；缺 usage 按预扣计。密钥只从 Settings 读（本脚本是 app/ 之外唯一另一处取真值的地方）。

运行：uv run python scripts/smoke_dashscope.py [--dry-run] [--out reports/xxx.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx2
from langchain_core.messages import HumanMessage

from app.core.config import Settings
from app.core.tokens import estimate_tokens
from app.deps import build_gateway_parts, gateway_for
from app.domain.usage import compute_cost, price_table
from app.engine.gateway.candidates import make_http_client
from app.engine.gateway.routing import parse_routes, unique_candidates

BUDGET_CNY = Decimal("0.05")  # 硬上限：预扣 + 实付都不得超过
PROMPT = "只回复两个字：收到"
MAX_TOKENS = 16
THINKING_MODEL = "qwen-plus"  # 便宜且支持思考的模型：看 reasoning_content 原始形态
THINKING_BUDGET = 64
THINKING_MAX_TOKENS = 32
ABORT_CHARS = 2000  # 客户端侧保险：流里累计文本超过即断开
TIMEOUT = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


@dataclass
class Budget:
    spent: Decimal = Decimal(0)

    def reserve(self, projected: Decimal) -> bool:
        return self.spent + projected <= BUDGET_CNY

    def charge(self, actual: Decimal) -> None:
        self.spent += actual


@dataclass
class RawResult:
    model: str
    variant: str
    status: int | None = None
    error: str | None = None
    chunks: int = 0
    content: str = ""
    reasoning_chars: int = 0
    first_chunk_s: float | None = None
    usage: dict[str, int] | None = None
    usage_index: int | None = None
    usage_choices_empty: bool | None = None
    finish_reason: str | None = None
    finish_index: int | None = None
    done_sentinel: bool = False
    delta_keys: set[str] = field(default_factory=set)
    cost: Decimal = Decimal(0)
    skipped: str | None = None


def projected_cost(model: str, prompt: str, max_tokens: int, prices) -> Decimal:
    return compute_cost(
        model, estimate_tokens(prompt) + 8, max_tokens, cached=False, prices=prices
    )


async def raw_stream(
    client: httpx2.AsyncClient,
    *,
    base_url: str,
    key: str,
    model: str,
    variant: str,
    extra: dict[str, Any],
    max_tokens: int,
    prices,
    budget: Budget,
) -> RawResult:
    r = RawResult(model=model, variant=variant)
    projected = projected_cost(model, PROMPT, max_tokens, prices)
    if not budget.reserve(projected):
        r.skipped = f"预算不足：预扣 {projected} 会超过 {BUDGET_CNY}"
        return r
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        **extra,
    }
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {key}"},
        ) as resp:
            r.status = resp.status_code
            if resp.status_code != 200:
                raw = await resp.aread()
                r.error = raw.decode("utf-8", "replace")[:300]
                budget.charge(Decimal(0))
                return r
            index = -1
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    r.done_sentinel = True
                    continue
                index += 1
                r.chunks += 1
                if r.first_chunk_s is None:
                    r.first_chunk_s = time.perf_counter() - t0
                obj = json.loads(data)
                choices = obj.get("choices") or []
                if obj.get("usage"):
                    r.usage = {
                        k: obj["usage"].get(k)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                    r.usage_index = index
                    r.usage_choices_empty = not choices
                for ch in choices:
                    delta = ch.get("delta") or {}
                    r.delta_keys |= {k for k, v in delta.items() if v}
                    r.content += delta.get("content") or ""
                    r.reasoning_chars += len(delta.get("reasoning_content") or "")
                    if ch.get("finish_reason"):
                        r.finish_reason = ch["finish_reason"]
                        r.finish_index = index
                if len(r.content) + r.reasoning_chars > ABORT_CHARS:
                    r.error = "客户端保险触发：文本超过上限，主动断开"
                    break
    except httpx2.HTTPError as e:
        r.error = f"{type(e).__name__}: {e}"[:300]
    if r.usage:
        r.cost = compute_cost(
            model,
            r.usage["prompt_tokens"] or 0,
            r.usage["completion_tokens"] or 0,
            cached=False,
            prices=prices,
        )
    else:
        r.cost = projected  # 缺 usage 按预扣计，宁多勿少
    budget.charge(r.cost)
    return r


@dataclass
class GatewayResult:
    chunks: int = 0
    content: str = ""
    first_chunk_s: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    provider_model: str | None = None
    error: str | None = None
    cost: Decimal = Decimal(0)
    skipped: str | None = None


async def via_gateway(settings: Settings, prices, budget: Budget) -> GatewayResult:
    g = GatewayResult()
    model = settings.model_routes["fast"][0].split(":", 1)[1]
    projected = projected_cost(model, PROMPT, MAX_TOKENS, prices)
    if not budget.reserve(projected):
        g.skipped = f"预算不足：预扣 {projected}"
        return g
    http_client = make_http_client(max_connections=4, max_keepalive_connections=2)
    try:
        parts = build_gateway_parts(
            settings, http_client=http_client, redis=None, session_factory=None
        )  # redis=None：熔断进程内、缓存关；session_factory=None：不记账（分账）
        gw = gateway_for(parts, "smoke")
        assert gw.meter is None and gw.reply_cache is None
        t0 = time.perf_counter()
        last = None
        async for chunk in gw.astream(
            [HumanMessage(PROMPT)], tier="fast", max_tokens=MAX_TOKENS
        ):
            g.chunks += 1
            if g.first_chunk_s is None:
                g.first_chunk_s = time.perf_counter() - t0
            g.content += str(chunk.content)
            if chunk.usage_metadata:
                g.usage = dict(chunk.usage_metadata)
            if chunk.response_metadata.get("finish_reason"):
                g.finish_reason = chunk.response_metadata["finish_reason"]
            last = chunk
        if last is not None:
            meta = last.response_metadata
            g.provider_model = str(meta.get("model_name") or meta.get("model") or "")
    except Exception as e:  # noqa: BLE001 —— 冒烟要把死因写进报告
        g.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        await http_client.aclose()
    if g.usage:
        g.cost = compute_cost(
            model,
            int(g.usage.get("input_tokens") or 0),
            int(g.usage.get("output_tokens") or 0),
            cached=False,
            prices=prices,
        )
    else:
        g.cost = projected
    budget.charge(g.cost)
    return g


def fmt_money(x: Decimal) -> str:
    return f"¥{x:.6f}"


def render(
    *,
    settings: Settings,
    raws: list[RawResult],
    gw: GatewayResult | None,
    budget: Budget,
    argv: list[str],
    dry_run: bool,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# DashScope 联网冒烟报告（{now}）",
        "",
        f"- 命令：`uv run python {' '.join(argv)}`",
        (
            f"- 环境：openai {version('openai')} / langchain-openai {version('langchain-openai')} / "
            f"httpx2 {version('httpx2')}；fake 模式关；端点 `{settings.providers['bailian']}`"
        ),
        (
            f"- 口径：每候选一次流式调用，`max_tokens={MAX_TOKENS}`，提示词“{PROMPT}”；"
            f"思考形态用 `{THINKING_MODEL}` + `enable_thinking=true, thinking_budget={THINKING_BUDGET}, "
            f"max_tokens={THINKING_MAX_TOKENS}`；成本 = 真实 usage × 配置价目表（元/千 token），缺 usage 按预扣计"
        ),
        f"- 预算护栏：写死 {fmt_money(BUDGET_CNY)}；本次实付 **{fmt_money(budget.spent)}**",
        "- 分账：本脚本不接账本（meter=None），账本里没有本次调用；fake 记账与真实花费分离",
        "",
    ]
    if dry_run:
        lines.append("**dry-run：未联网，以下为计划与预扣。**")
        lines.append("")
    lines += [
        "## 原始 SSE 三验",
        "",
        "| 模型 | 变体 | HTTP | 块数 | 首块 s | usage 块位置 | usage 块 choices 空 | finish_reason（位置） | [DONE] | delta 键 | 思考字符 | tokens 入/出 | 花费 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in raws:
        pos = "-" if r.usage_index is None else f"第 {r.usage_index + 1}/{r.chunks} 块"
        fin = (
            "-"
            if r.finish_reason is None
            else f"{r.finish_reason}（第 {r.finish_index + 1} 块）"
        )
        toks = (
            "-"
            if not r.usage
            else f"{r.usage['prompt_tokens']}/{r.usage['completion_tokens']}"
        )
        note = (
            r.skipped or r.error or (f"回复“{r.content.strip()}”" if r.content else "")
        )
        lines.append(
            f"| {r.model} | {r.variant} | {r.status or '-'} | {r.chunks} | "
            f"{'-' if r.first_chunk_s is None else f'{r.first_chunk_s:.2f}'} | {pos} | "
            f"{'-' if r.usage_choices_empty is None else ('是' if r.usage_choices_empty else '否')} | {fin} | "
            f"{'是' if r.done_sentinel else '否'} | {', '.join(sorted(r.delta_keys)) or '-'} | {r.reasoning_chars} | "
            f"{toks} | {fmt_money(r.cost)} | {note} |"
        )
    lines += ["", "## 网关路径（fast 档，fake 关、缓存关、账本关）", ""]
    if gw is None:
        lines.append("未执行。")
    elif gw.skipped:
        lines.append(f"跳过：{gw.skipped}")
    elif gw.error:
        lines.append(f"失败：{gw.error}")
    else:
        lines += [
            f"- 块数 {gw.chunks}，首块 {gw.first_chunk_s:.2f}s，回复“{gw.content.strip()}”",
            f"- 末块 usage_metadata：`{gw.usage}`",
            f"- finish_reason：`{gw.finish_reason}`；上游 model_name：`{gw.provider_model}`",
            f"- 花费 {fmt_money(gw.cost)}",
        ]
    lines += ["", "## 结论", ""]
    ok = [r for r in raws if r.status == 200 and not r.error]
    lines.append(
        f"- 入池三验：{len(ok)}/{len([r for r in raws if not r.skipped])} 个调用 HTTP 200 且流完整；"
        "逐行看上表的 usage 位置、finish_reason 位置与 [DONE] 列。"
    )
    rejected = [r for r in raws if r.status and r.status != 200]
    if rejected:
        lines.append(
            "- 非 200："
            + "；".join(
                f"{r.model}/{r.variant} → {r.status} {r.error}" for r in rejected
            )
        )
    lines.append(
        "- 本报告的数字只用于 README 数字表“真实上游冒烟花费”一行；定性结论进 ADR 实证节。"
    )
    return "\n".join(lines) + "\n"


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不联网，只打印计划与预扣")
    ap.add_argument(
        "--out", default=None, help="报告路径（默认 reports/<日期>-smoke-dashscope.md）"
    )
    args = ap.parse_args(argv[1:])
    settings = Settings(aegis_fake_llm=False)  # 其余从 .env 读；密钥只从这里取
    key = settings.dashscope_api_key.get_secret_value()
    if not key:
        print("DASHSCOPE_API_KEY 为空：填进本机 .env 再跑", file=sys.stderr)
        return 2
    prices = price_table(settings.model_prices)
    routes = parse_routes(settings.model_routes, set(settings.providers))
    candidates = unique_candidates(routes)
    base_url = settings.providers["bailian"]
    budget = Budget()
    raws: list[RawResult] = []
    plan = [
        (c.model, "enable_thinking=false", {"enable_thinking": False}, MAX_TOKENS)
        for c in candidates
    ]
    plan.append(
        (
            THINKING_MODEL,
            "enable_thinking=true",
            {"enable_thinking": True, "thinking_budget": THINKING_BUDGET},
            THINKING_MAX_TOKENS,
        )
    )
    if args.dry_run:
        for model, variant, _, max_tokens in plan:
            r = RawResult(model=model, variant=variant)
            r.cost = projected_cost(model, PROMPT, max_tokens, prices)
            r.skipped = "dry-run 预扣"
            budget.charge(r.cost)
            raws.append(r)
        gw = None
    else:
        async with httpx2.AsyncClient(timeout=TIMEOUT) as client:
            for model, variant, extra, max_tokens in plan:
                raws.append(
                    await raw_stream(
                        client,
                        base_url=base_url,
                        key=key,
                        model=model,
                        variant=variant,
                        extra=extra,
                        max_tokens=max_tokens,
                        prices=prices,
                        budget=budget,
                    )
                )
        gw = await via_gateway(settings, prices, budget)
    report = render(
        settings=settings,
        raws=raws,
        gw=gw,
        budget=budget,
        argv=argv,
        dry_run=args.dry_run,
    )
    out = Path(args.out or f"reports/{datetime.now(UTC):%Y-%m-%d}-smoke-dashscope.md")
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"报告已写入 {out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
