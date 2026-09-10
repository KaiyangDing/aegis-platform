"""端到端演示 / 凭证脚本（M3.3）：进程内起真 lifespan（真 PG / Redis / checkpointer），经 HTTP 走完一轮客服对话并写报告。

两种模式：
- 默认 fake（AEGIS_FAKE_LLM=1，零真实调用）：签发 token → POST /v1/chat 一轮 → GET 回放快照。fake 候选不发 tool_calls，
  审批闭环由 tests/routers 与 tests/test_e2e_pg.py 以剧本网关覆盖；本模式是 README 可复现步骤的机器版。
- --real（AEGIS_FAKE_LLM=0，打 DashScope）：同一条 HTTP 链路，消息引导模型查单并申请退款 300（超租户 A 阈值 200）→
  期望 approval_requested 挂起 → 坐席 token 批准 → 续跑 → GET 回放；真实花费从 usage_ledger 按本会话 session_id 读出（账本可复算）。

预算护栏（写死）：租户 A 的 max_iterations 覆盖为 MAX_ITERATIONS（每个 run 最多这么多次 LLM 调用，两个 run 合计 ≤ 2×）；
实付超过 BUDGET_CNY 报告标"超预算"并以非零退出；生产环境禁故障注入由 Settings 校验器兜底。
分账：只读本会话的账本行——fake 模式的行 session_id 不同，不会混进真实花费；报告明示本次是 fake 记账还是真实花费。
运行（仓库根；.env 需 DATABASE_URL / REDIS_URL / CHECKPOINT_DATABASE_URL，--real 另需 DASHSCOPE_API_KEY；JWT_SECRET 缺席时本脚本用临时密钥）：

    uv run python scripts/e2e_flow.py [--real] [--out reports/xxx.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if (
    str(ROOT) not in sys.path
):  # 以 `python scripts/e2e_flow.py` 直接运行时把仓库根加进导入路径
    sys.path.insert(0, str(ROOT))

BUDGET_CNY = Decimal("0.05")
MAX_ITERATIONS = 4
REAL_MESSAGE = (
    "我的订单 AZ-1001 已经签收，但商品到手有破损。请先查询这笔订单的状态和实付金额，"
    "然后直接调用退款工具为它申请退款 300 元；如果需要人工审批，请提交审批。"
)
FAKE_MESSAGE = "你好，请介绍一下你能帮我做什么。"


def _bootstrap_env(real: bool) -> None:
    os.environ["AEGIS_FAKE_LLM"] = "0" if real else "1"
    os.environ.setdefault("JWT_SECRET", secrets.token_urlsafe(32))


def parse_sse(text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        frame: dict[str, Any] = {"id": None, "event": None, "data": None}
        for line in block.split("\n"):
            key, _, value = line.partition(": ")
            if key == "id":
                frame["id"] = int(value)
            elif key == "event":
                frame["event"] = value
            elif key == "data":
                frame["data"] = json.loads(value)
        frames.append(frame)
    return frames


def _excerpt(obj: Any, limit: int = 160) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _money(value: Decimal) -> str:
    return f"¥{value:.6f}"


async def flow(real: bool) -> dict[str, Any]:
    import httpx
    from sqlalchemy import select

    from app.core.auth import Role, issue_token
    from app.core.config import get_settings
    from app.core.db import make_engine, make_session_factory
    from app.domain.usage import UsageRecord
    from app.main import app

    get_settings.cache_clear()
    settings = get_settings()
    if real and not settings.dashscope_api_key.get_secret_value():
        raise SystemExit("--real 需要 .env 里的 DASHSCOPE_API_KEY")
    mode = "real" if real else "fake"
    session_id = f"e2e-{mode}-{uuid.uuid4().hex[:8]}"
    steps: list[dict[str, Any]] = []
    async with app.router.lifespan_context(app):
        specs = app.state.specs
        if real:
            spec = specs["tenant-a"]
            app.state.specs = {
                **specs,
                "tenant-a": replace(
                    spec, policy=replace(spec.policy, max_iterations=MAX_ITERATIONS)
                ),
            }
        secret = settings.jwt_secret.get_secret_value()

        def bearer(uid: str, role: Role) -> dict[str, str]:
            token = issue_token(
                user_id=uid, tenant_id="tenant-a", role=role, ttl_s=600, secret=secret
            )
            return {"Authorization": f"Bearer {token}"}

        user, staff = bearer("u-a1", Role.USER), bearer("op-a1", Role.OPERATOR)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://e2e", timeout=180
        ) as client:
            t0 = time.monotonic()
            resp = await client.post(
                "/v1/chat",
                json={
                    "session_id": session_id,
                    "message": REAL_MESSAGE if real else FAKE_MESSAGE,
                },
                headers=user,
            )
            frames = parse_sse(resp.text) if resp.status_code == 200 else []
            steps.append(
                {
                    "step": "POST /v1/chat（终端用户）",
                    "status": resp.status_code,
                    "seconds": round(time.monotonic() - t0, 2),
                    "frames": frames,
                    "raw": None if resp.status_code == 200 else resp.text[:400],
                }
            )
            done = frames[-1]["data"] if frames else {}
            if done.get("reason") == "awaiting_approval":
                approval_id = done["approvals"][0]["approval_id"]
                t1 = time.monotonic()
                decided = await client.post(
                    f"/v1/approvals/{approval_id}",
                    json={"decision": "approve"},
                    headers=staff,
                )
                steps.append(
                    {
                        "step": "POST /v1/approvals/{id}（坐席批准）",
                        "status": decided.status_code,
                        "seconds": round(time.monotonic() - t1, 2),
                        "summary": decided.json()
                        if decided.status_code == 200
                        else decided.text[:400],
                    }
                )
            snap = await client.get(f"/v1/sessions/{session_id}/events", headers=staff)
            steps.append(
                {
                    "step": "GET /v1/sessions/{id}/events（坐席全量回放）",
                    "status": snap.status_code,
                    "seconds": 0.0,
                    "frames": parse_sse(snap.text) if snap.status_code == 200 else [],
                }
            )
    engine = make_engine(settings.database_url)
    try:
        session_factory = make_session_factory(engine)
        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(UsageRecord)
                        .where(UsageRecord.session_id == session_id)
                        .order_by(UsageRecord.id)
                    )
                )
                .scalars()
                .all()
            )
            ledger = [
                {
                    "tier": r.tier,
                    "provider": r.provider,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "cached": r.cached,
                    "usage_missing": r.usage_missing,
                    "cost": Decimal(r.cost),
                }
                for r in rows
            ]
    finally:
        await engine.dispose()
    return {
        "mode": mode,
        "session_id": session_id,
        "steps": steps,
        "ledger": ledger,
        "total_cost": sum((r["cost"] for r in ledger), Decimal(0)),
        "fake_mode": not real,
        "settings": {
            "model_routes": settings.model_routes,
            "app_env": settings.app_env,
            "fault_injection_rate": settings.fault_injection_rate,
        },
    }


def render(result: dict[str, Any], command: str) -> str:
    now = (
        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    )  # 本机时区：与 reports/ 既有日期口径一致
    real = not result["fake_mode"]
    lines = [
        f"# HTTP 端到端{'（真实上游 DashScope）' if real else '（fake 模式）'}运行记录（{now}）",
        "",
        f"- 命令：`{command}`",
        (
            f"- 环境：langchain {version('langchain')} / langgraph {version('langgraph')} / fastapi {version('fastapi')} / "
            f"openai {version('openai')}；app_env={result['settings']['app_env']}；"
            f"fault_injection_rate={result['settings']['fault_injection_rate']}；"
            f"AEGIS_FAKE_LLM={'0' if real else '1'}；档位路由 {json.dumps(result['settings']['model_routes'], ensure_ascii=False)}"
        ),
        (
            "- 口径：进程内起真 lifespan（真 PG / Redis / checkpointer），经 httpx ASGITransport 走 HTTP；帧 = text/event-stream 原文解析；"
            "花费 = usage_ledger 中本会话（session_id）全部行的 cost 之和（真实 usage × 配置价目表，元/千 token），缓存命中行零成本"
        ),
        (
            f"- 预算护栏：BUDGET_CNY = {_money(BUDGET_CNY)} 硬上限（超过即报告标超预算并非零退出）；"
            f"租户 A 的 max_iterations 覆盖为 {MAX_ITERATIONS}（每 run 至多 {MAX_ITERATIONS} 次 LLM 调用）；生产环境禁故障注入由 Settings 校验器兜底"
        ),
        f"- 分账：只读 session_id = `{result['session_id']}` 的账本行；"
        + (
            "本次为 **真实花费**。"
            if real
            else "本次为 **fake 记账（不是真实花费，模型是 FakeReplyChatModel）**。"
        ),
        "",
        "## 步骤",
        "",
    ]
    for step in result["steps"]:
        lines.append(
            f"### {step['step']} → HTTP {step['status']}（{step['seconds']}s）"
        )
        lines.append("")
        if step.get("raw"):
            lines.append(f"```\n{step['raw']}\n```")
        if "summary" in step:
            lines.append("```json")
            lines.append(
                json.dumps(step["summary"], ensure_ascii=False, indent=1, default=str)
            )
            lines.append("```")
        if step.get("frames"):
            lines.append("| id | event | data（节选） |")
            lines.append("|---|---|---|")
            for f in step["frames"]:
                payload = (
                    f["data"].get("payload", f["data"])
                    if isinstance(f["data"], dict)
                    else f["data"]
                )
                lines.append(
                    f"| {f['id'] if f['id'] is not None else '—'} | {f['event']} | {_excerpt(payload).replace('|', '\\|')} |"
                )
        lines.append("")
    lines += [
        "## 账本（usage_ledger，本会话）",
        "",
        "| # | tier | provider | model | prompt | completion | cached | usage_missing | cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(result["ledger"], 1):
        lines.append(
            f"| {i} | {r['tier']} | {r['provider']} | {r['model']} | {r['prompt_tokens']} | {r['completion_tokens']} | "
            f"{r['cached']} | {r['usage_missing']} | {_money(r['cost'])} |"
        )
    total = result["total_cost"]
    over = total > BUDGET_CNY
    lines += [
        "",
        f"- 本会话调用 {len(result['ledger'])} 次，合计 **{_money(total)}**"
        + ("（fake 记账）" if not real else "")
        + ("；**超预算**" if over else "；在预算内"),
        "",
        "## 结论",
        "",
    ]
    chat = result["steps"][0]
    kinds = [f["event"] for f in chat.get("frames", [])]
    lines.append(f"- POST /v1/chat 帧序列：{kinds}")
    approval = next((s for s in result["steps"] if "approvals" in s["step"]), None)
    if approval is not None:
        summary = approval.get("summary")
        lines.append(
            f"- 审批闭环：{'已走通' if isinstance(summary, dict) and summary.get('status') == 'done' else '未走通'}"
            f"（摘要 status={summary.get('status') if isinstance(summary, dict) else summary}）"
        )
    else:
        lines.append(
            "- 审批闭环：本次未触发挂起（fake 模式不发 tool_calls，或真实模型未申请退款）"
        )
    snapshot = result["steps"][-1]
    if snapshot.get("frames"):
        tail = snapshot["frames"][-1]["data"]
        lines.append(
            f"- 回放快照：run_state={tail.get('run_state')}，count={tail.get('count')}，next_seq={tail.get('next_seq')}"
        )
    lines.append("- 本报告的数字只用于 README 数字表对应行；定性结论进 ADR 实证节。")
    return "\n".join(lines) + "\n"


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="HTTP 端到端演示 / 凭证")
    parser.add_argument(
        "--real", action="store_true", help="打真实上游（需 DASHSCOPE_API_KEY）"
    )
    parser.add_argument(
        "--out", default=None, help="报告路径（默认 reports/<日期>-e2e-<模式>.md）"
    )
    args = parser.parse_args(argv)
    result = await flow(args.real)
    command = "uv run python scripts/e2e_flow.py" + (" --real" if args.real else "")
    report = render(result, command)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d")
    out = Path(args.out or f"reports/{stamp}-e2e-{result['mode']}.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {out}")
    return 1 if result["total_cost"] > BUDGET_CNY else 0


if __name__ == "__main__":
    real_mode = "--real" in sys.argv[1:]
    _bootstrap_env(real_mode)
    from app.core.loops import selector_loop_factory

    raise SystemExit(
        asyncio.run(main(sys.argv[1:]), loop_factory=selector_loop_factory)
    )
