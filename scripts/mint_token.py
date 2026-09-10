"""签发演示 token（无登录端点：本脚本即凭证发放口；生产接 IdP 只换签发方，验签面不变）。

在仓库根执行（.env 相对 cwd 加载，需含 JWT_SECRET，≥32 字节；生成：python -c "import secrets;print(secrets.token_urlsafe(32))"）：

    uv run python scripts/mint_token.py --user u-a1 --tenant tenant-a --role user
    uv run python scripts/mint_token.py --user op-a1 --tenant tenant-a --role operator

租户须在静态表内（app/business/config.py）；TTL 按角色两档（ttl_for），--ttl 可覆盖。密钥只从 Settings 读。
"""

from __future__ import annotations

import argparse

from app.business.config import UnknownTenant, tenant_config, tenant_ids
from app.core.auth import Role, issue_token, ttl_for
from app.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="签发演示 JWT")
    parser.add_argument(
        "--user", required=True, help="user_id（如 u-a1 / op-a1 / admin-a1）"
    )
    parser.add_argument("--tenant", required=True, help="tenant_id（须在静态租户表内）")
    parser.add_argument(
        "--role",
        required=True,
        choices=[r.value for r in Role],
        help="user / operator / admin",
    )
    parser.add_argument("--ttl", type=int, default=None, help="覆盖角色默认 TTL（秒）")
    args = parser.parse_args()

    try:
        tenant_config(args.tenant)
    except UnknownTenant:
        raise SystemExit(
            f"租户 {args.tenant} 未开通——可用：{list(tenant_ids())}"
        ) from None
    settings = get_settings()
    role = Role(args.role)
    token = issue_token(
        user_id=args.user,
        tenant_id=args.tenant,
        role=role,
        ttl_s=args.ttl or ttl_for(role, settings),
        secret=settings.jwt_secret.get_secret_value(),
    )
    print(token)


if __name__ == "__main__":
    main()
