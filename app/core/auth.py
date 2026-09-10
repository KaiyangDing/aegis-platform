"""JWT 认证与 RBAC 依赖（S2；ADR-014）。

凭证形态：终端用户与坐席 / 管理员同为 HS256 短期 JWT（claims: sub / tid / role / iat / exp），差别只在 TTL 两档
（ttl_for）与发放形态——无登录端点，scripts/mint_token.py 签发；生产接 IdP 只换签发方，验签面不变。
密钥托管：Settings.jwt_secret（环境变量 JWT_SECRET）。
失败分层：空钥 / 弱钥 = 服务端配置 bug，ValueError fail-loud（不许混进 401 掩盖错配）；token 验证失败 = 客户端问题，
InvalidToken → 依赖层映射 401。decode 显式锁 algorithms 且强制 claim 清单：alg 混淆（none / 换头）是 JWT 第一攻击面，
缺 exp 的票 = 永不过期票，一并拒收。
矩阵执行器 = require_roles 依赖工厂：端点用 Annotated 声明角色面，handler 体内零散 if。"身份合法但无权"403 与
"身份无效"401 严格分家。
身份写进 request.state.principal 供后续依赖读——入站限流的 tenant_identity 由此拼键；依赖按声明序解析，
认证必须排在限流之前（ADR-010 决策 6），排反了 tenant_identity 直接 RuntimeError 而不是静默退化成 IP 键。
core 不 import 兄弟包：租户段字符集与网关守卫（engine.gateway.tenancy.TENANT_ID_RE）同一条规则，以常量互钉（测试对照）。
"""

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.core.limits import identity

ALGORITHM = "HS256"
REQUIRED_CLAIMS: tuple[str, ...] = ("exp", "sub", "tid", "role")
MIN_SECRET_BYTES = 32
"""HS256 密钥硬下限（RFC 7518 §3.2：MUST ≥ 256 bit）。PyJWT 对短钥只发告警——按"配置错误启动时炸"口径升为硬错误。"""
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""与 app.engine.gateway.tenancy.TENANT_ID_RE 同一条规则：非法租户段不许进入任何共享件的 key（缓存前缀 / 账本列 / 限流键）。"""

# 依赖层话术（core 登记例外：与入站限流串同一分野，不进 engine 话术表）
AUTH_MISSING = "缺少或格式错误的 Authorization: Bearer 凭证"
ROLE_FORBIDDEN = "当前角色无权访问该端点"


class Role(StrEnum):
    """三角色：终端用户 / 坐席 / 管理员。值进 JWT role claim（字符串稳定，改值 = 在途票全部失效）。"""

    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"


STAFF_ROLES: frozenset[Role] = frozenset({Role.OPERATOR, Role.ADMIN})


class InvalidToken(Exception):
    """token 验证失败（签名 / 过期 / claims 缺失 / 角色或租户段非法）——依赖层映射 401。
    与 ValueError（空钥弱钥 = 服务端配置 bug，fail-loud 不映射 401）刻意分家。"""


@dataclass(frozen=True, slots=True)
class Principal:
    """一次请求的已验证身份：端点矩阵、归属校验、限流键的唯一输入。"""

    user_id: str
    tenant_id: str
    role: Role


def _check_secret(secret: str) -> None:
    if not secret:
        raise ValueError("JWT_SECRET 未配置——空密钥上不许签发 / 验签")
    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise ValueError(
            f"JWT_SECRET 不足 {MIN_SECRET_BYTES} 字节（RFC 7518 HS256 下限）——用 secrets.token_urlsafe(32) 生成"
        )


def _epoch_now() -> int:
    return int(time.time())


def issue_token(
    *,
    user_id: str,
    tenant_id: str,
    role: Role,
    ttl_s: int,
    secret: str,
    now: Callable[[], int] = _epoch_now,
) -> str:
    """签发 HS256 JWT。now 可注入：过期测试构造旧票，不做真实等待。"""
    _check_secret(secret)
    if not user_id:
        raise ValueError("user_id 不许为空——身份三元组是矩阵与归属校验的根")
    if not TENANT_ID_RE.fullmatch(tenant_id or ""):
        raise ValueError(
            f"tenant_id 非法：{tenant_id!r}（须匹配 {TENANT_ID_RE.pattern}）"
        )
    if ttl_s <= 0:
        raise ValueError(f"ttl_s 须 >0，得到 {ttl_s}")
    iat = now()
    claims = {
        "sub": user_id,
        "tid": tenant_id,
        "role": role.value,
        "iat": iat,
        "exp": iat + ttl_s,
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def ttl_for(role: Role, settings: Settings) -> int:
    """按角色取票时长单点：USER 短窗 / OPERATOR·ADMIN 员工窗——方向由测试钉死，签发面一律走这里。"""
    return settings.jwt_user_ttl_s if role is Role.USER else settings.jwt_staff_ttl_s


def _principal_from_claims(claims: dict[str, Any]) -> Principal:
    sub, tid, role_raw = (
        claims["sub"],
        claims["tid"],
        claims["role"],
    )  # require 清单已拦缺席
    if not isinstance(sub, str) or not sub:
        raise InvalidToken("token 的 sub 非法")
    if not isinstance(tid, str) or not TENANT_ID_RE.fullmatch(tid):
        raise InvalidToken("token 的 tid 非法")
    try:
        role = Role(role_raw)
    except ValueError as e:
        raise InvalidToken(f"token 角色非法：{role_raw!r}") from e
    return Principal(user_id=sub, tenant_id=tid, role=role)


def decode_token(token: str, *, secret: str) -> Principal:
    """验签并提取身份。只回显异常类型名不回显 token 内容（源头打码纪律）。"""
    _check_secret(secret)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            options={"require": list(REQUIRED_CLAIMS)},
        )
    except jwt.PyJWTError as e:
        raise InvalidToken(f"token 无效：{type(e).__name__}") from e
    return _principal_from_claims(claims)


async def current_principal(request: Request) -> Principal:
    """FastAPI 依赖：解析 Authorization: Bearer 头。401 三态：缺头 / 格式错 / 验签失败，均带 WWW-Authenticate。"""
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=AUTH_MISSING,
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings: Settings = request.app.state.settings
    try:
        principal = decode_token(
            token.strip(), secret=settings.jwt_secret.get_secret_value()
        )
    except InvalidToken as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    request.state.principal = principal
    return principal


def require_roles(*roles: Role) -> Callable[..., Awaitable[Principal]]:
    """依赖工厂：端点 × 角色矩阵的执行器。
    用法：principal: Annotated[Principal, Depends(require_roles(Role.OPERATOR, Role.ADMIN))]。"""
    allowed = frozenset(roles)
    if not allowed:
        raise ValueError("require_roles 至少点名一个角色——空矩阵行等于把端点关掉")

    async def dependency(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=ROLE_FORBIDDEN
            )
        return principal

    return dependency


async def tenant_identity(request: Request) -> str:
    """入站限流的身份函数（rate_limit(identify=tenant_identity)）：键身份段 = 租户。
    读的是认证依赖写进 request.state 的主体；没有主体 = 依赖声明序排反了，fail-loud 而不是退化成 IP 键。"""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise RuntimeError(
            "tenant_identity 在认证之前被解析：路由 dependencies 必须先 current_principal / require_roles 再 rate_limit"
        )
    return identity("tenant", principal.tenant_id)
