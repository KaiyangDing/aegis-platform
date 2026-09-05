"""租户标识守卫：tenant_id 是缓存 key 前缀、账本列、配额桶键共用的同一把钥匙。

字符集收紧到可安全拼入 Redis key（v1 schema 的纵深防御，迁至网关构造与组合根入口）：
空串/冒号/通配符会破坏租户隔离前缀与按前缀 SCAN 的运维；长度上限与账本 String(64) 对齐。
这是内部标识符不是显示名：认证 → 标识的映射归 API 层（M3）。
"""

import re

from app.engine.gateway import utterances as u

TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_tenant_id(tenant_id: str) -> str:
    """合法则原样返回；非法抛 ValueError（组合根入口与网关字段校验器共用同一条规则）。"""
    if not isinstance(tenant_id, str) or not TENANT_ID_RE.fullmatch(tenant_id):
        raise ValueError(u.TENANT_ID_INVALID)
    return tenant_id
