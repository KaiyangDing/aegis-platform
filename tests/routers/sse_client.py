"""routers 测试共用（不是测试文件）：SSE 帧解析、演示 JWT 签发与请求头。"""

import json
from typing import Any

from app.core.auth import Role, issue_token

SECRET = "routers-test-secret-0123456789abcdef"


def auth(
    role: Role = Role.USER, *, uid: str = "u-a1", tid: str = "tenant-a"
) -> dict[str, str]:
    token = issue_token(user_id=uid, tenant_id=tid, role=role, ttl_s=600, secret=SECRET)
    return {"Authorization": f"Bearer {token}"}


def parse_sse(text: str) -> list[dict[str, Any]]:
    """把 text/event-stream 正文拆成帧：{id: int | None, event: str, data: 解析后的 JSON}。"""
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
