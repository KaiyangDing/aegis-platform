"""JWT 签发 / 验签（S2；ADR-014）：往返、过期、坏签名、alg 混淆、缺 claim、角色 / 租户段非法、空钥弱钥 fail-loud、
TTL 两档方向、租户段规则与网关守卫互钉；依赖层：401 三态带 WWW-Authenticate、403 分工、request.state.principal 写入、
限流身份段读取、依赖声明序排反即 RuntimeError。过期用注入 now 构造旧票不真等；端点测试用最小 FastAPI 应用 + httpx ASGITransport，零网络。"""

import base64
import json
from typing import Annotated

import httpx
import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI, Request
from pydantic import SecretStr

from app.core.auth import (
    TENANT_ID_RE,
    InvalidToken,
    Principal,
    Role,
    current_principal,
    decode_token,
    issue_token,
    require_roles,
    tenant_identity,
    ttl_for,
)
from app.core.config import Settings
from app.engine.gateway.tenancy import TENANT_ID_RE as GATEWAY_TENANT_ID_RE

CURRENT = "unit-test-current-secret-0123456789ab"
OTHER = "unit-test-other-secret-0123456789abcd"
FAR_FUTURE = 33_000_000_000  # ≈ 公元 3015 年：手工伪造 claims 用的不过期 exp


def _token(
    role: Role = Role.USER,
    *,
    secret: str = CURRENT,
    ttl_s: int = 3600,
    uid: str = "u-a1",
    tid: str = "tenant-a",
) -> str:
    return issue_token(
        user_id=uid, tenant_id=tid, role=role, ttl_s=ttl_s, secret=secret
    )


def _b64(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


# ---------------------------------------------------------------- 纯函数层


def test_issue_decode_roundtrip():
    principal = decode_token(_token(Role.OPERATOR, uid="op-a1"), secret=CURRENT)
    assert principal == Principal(
        user_id="op-a1", tenant_id="tenant-a", role=Role.OPERATOR
    )


def test_expired_token_rejected():
    # 1970 年签发的 60s 票——过期判定交给 PyJWT 对真实时钟，余量五十余年，零时序敏感
    stale = issue_token(
        user_id="u",
        tenant_id="t",
        role=Role.USER,
        ttl_s=60,
        secret=CURRENT,
        now=lambda: 1_000_000,
    )
    with pytest.raises(InvalidToken, match="ExpiredSignatureError"):
        decode_token(stale, secret=CURRENT)


def test_bad_signature_rejected():
    with pytest.raises(InvalidToken, match="InvalidSignatureError"):
        decode_token(_token(secret=OTHER), secret=CURRENT)


def test_alg_none_rejected():
    """alg 混淆：手工拼一张 alg=none 的票（不经 PyJWT 签发路径），验签端只认 HS256。"""
    claims = {"sub": "u", "tid": "t", "role": "admin", "iat": 1, "exp": FAR_FUTURE}
    forged = f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(claims)}."
    with pytest.raises(InvalidToken):
        decode_token(forged, secret=CURRENT)


def test_missing_required_claim_rejected():
    no_exp = pyjwt.encode(
        {"sub": "u", "tid": "t", "role": "user"}, CURRENT, algorithm="HS256"
    )
    with pytest.raises(InvalidToken, match="MissingRequiredClaimError"):
        decode_token(no_exp, secret=CURRENT)


def test_bad_role_and_bad_tenant_segment_rejected():
    bad_role = pyjwt.encode(
        {"sub": "u", "tid": "t", "role": "root", "exp": FAR_FUTURE},
        CURRENT,
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken, match="角色非法"):
        decode_token(bad_role, secret=CURRENT)
    bad_tid = pyjwt.encode(
        {"sub": "u", "tid": "a:b", "role": "user", "exp": FAR_FUTURE},
        CURRENT,
        algorithm="HS256",
    )
    with pytest.raises(InvalidToken, match="tid 非法"):
        decode_token(bad_tid, secret=CURRENT)


def test_issue_rejects_bad_inputs():
    with pytest.raises(ValueError, match="tenant_id"):
        _token(tid="a:b")
    with pytest.raises(ValueError, match="user_id"):
        _token(uid="")
    with pytest.raises(ValueError, match="ttl_s"):
        _token(ttl_s=0)


def test_empty_or_short_secret_is_config_error_not_401():
    """空钥 / 弱钥 = 服务端错配，fail-loud ValueError——绝不装作"客户端 token 不对"混进 401。"""
    for bad in ("", "dev123"):
        with pytest.raises(ValueError):
            issue_token(
                user_id="u", tenant_id="t", role=Role.USER, ttl_s=60, secret=bad
            )
        with pytest.raises(ValueError):
            decode_token("whatever", secret=bad)


def test_ttl_for_direction():
    settings = Settings(_env_file=None, jwt_user_ttl_s=10, jwt_staff_ttl_s=20)
    assert ttl_for(Role.USER, settings) == 10
    assert ttl_for(Role.OPERATOR, settings) == 20
    assert ttl_for(Role.ADMIN, settings) == 20


def test_tenant_segment_rule_matches_gateway_guard():
    """core 不 import engine：两处正则以测试互钉，改一处不改另一处这里先红。"""
    assert TENANT_ID_RE.pattern == GATEWAY_TENANT_ID_RE.pattern


def test_require_roles_needs_at_least_one_role():
    with pytest.raises(ValueError):
        require_roles()


# ---------------------------------------------------------------- 依赖层


def _app() -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(_env_file=None, jwt_secret=SecretStr(CURRENT))

    @app.get("/me")
    async def me(
        request: Request, principal: Annotated[Principal, Depends(current_principal)]
    ) -> dict[str, str]:
        return {
            "user": principal.user_id,
            "tenant": principal.tenant_id,
            "role": principal.role.value,
            "identity": await tenant_identity(request),
        }

    @app.get("/staff")
    async def staff(
        principal: Annotated[
            Principal, Depends(require_roles(Role.OPERATOR, Role.ADMIN))
        ],
    ) -> dict[str, str]:
        return {"role": principal.role.value}

    @app.get("/misordered")
    async def misordered(request: Request) -> dict[str, str]:
        return {"identity": await tenant_identity(request)}

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_missing_or_malformed_header_is_401_with_challenge():
    async with _client(_app()) as client:
        for headers in (
            {},
            {"Authorization": "Basic abc"},
            {"Authorization": "Bearer "},
        ):
            resp = await client.get("/me", headers=headers)
            assert resp.status_code == 401, headers
            assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_invalid_token_is_401_without_echoing_token():
    forged = _token(secret=OTHER)
    async with _client(_app()) as client:
        resp = await client.get("/me", headers={"Authorization": f"Bearer {forged}"})
    assert resp.status_code == 401
    assert "InvalidSignatureError" in resp.json()["detail"]
    assert forged not in resp.text and OTHER not in resp.text


async def test_valid_token_yields_principal_and_tenant_identity():
    async with _client(_app()) as client:
        resp = await client.get("/me", headers={"Authorization": f"Bearer {_token()}"})
    assert resp.status_code == 200
    assert resp.json() == {
        "user": "u-a1",
        "tenant": "tenant-a",
        "role": "user",
        "identity": "tenant:tenant-a",
    }


async def test_role_matrix_403_vs_200():
    async with _client(_app()) as client:
        denied = await client.get(
            "/staff", headers={"Authorization": f"Bearer {_token(Role.USER)}"}
        )
        assert denied.status_code == 403
        for role in (Role.OPERATOR, Role.ADMIN):
            ok = await client.get(
                "/staff", headers={"Authorization": f"Bearer {_token(role)}"}
            )
            assert ok.status_code == 200 and ok.json()["role"] == role.value


async def test_tenant_identity_before_auth_is_a_programming_error():
    """依赖序排反（限流在认证之前）不许静默退化成 IP 键：直接 RuntimeError 穿出。"""
    async with _client(_app()) as client:
        with pytest.raises(RuntimeError, match="认证之前"):
            await client.get("/misordered")
