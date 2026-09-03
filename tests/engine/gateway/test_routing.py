"""档位路由：调用方只声明档位；路由表启动即校验四项；默认配置必须能过自己的校验。"""

import pytest

from app.core.config import Settings
from app.engine.gateway.routing import (
    TIERS,
    Candidate,
    parse_routes,
    unique_candidates,
)

FULL = {
    "fast": ["bailian:qwen-flash", "bailian:qwen-turbo"],
    "standard": ["bailian:qwen-plus"],
    "strong": ["bailian:qwen3.7-max", "bailian:qwen-plus"],
}


def test_parse_routes_builds_ordered_candidate_chains():
    routes = parse_routes(FULL, {"bailian"})
    assert set(routes) == set(TIERS) == {"fast", "standard", "strong"}
    assert routes["fast"] == [
        Candidate("bailian", "qwen-flash"),
        Candidate("bailian", "qwen-turbo"),
    ]
    assert routes["strong"][0].key == "bailian:qwen3.7-max"


@pytest.mark.parametrize(
    "entry",
    ["no-colon-here", ":qwen-flash", "bailian:", "ghost:qwen-flash"],
    ids=["无冒号", "空 provider", "空 model", "未知 provider"],
)
def test_parse_routes_rejects_bad_entry_forms(entry):
    with pytest.raises(ValueError, match="路由配置非法"):
        parse_routes({**FULL, "fast": [entry]}, {"bailian"})


def test_parse_routes_rejects_empty_chain():
    with pytest.raises(ValueError, match="候选链为空"):
        parse_routes({**FULL, "standard": []}, {"bailian"})


def test_parse_routes_requires_all_three_tiers():
    with pytest.raises(ValueError, match="缺少档位.*standard.*strong"):
        parse_routes({"fast": ["bailian:qwen-flash"]}, {"bailian"})


def test_unique_candidates_dedups_across_tiers_keeping_first_seen_order():
    routes = parse_routes(FULL, {"bailian"})
    assert unique_candidates(routes) == [
        Candidate("bailian", "qwen-flash"),
        Candidate("bailian", "qwen-turbo"),
        Candidate("bailian", "qwen-plus"),
        Candidate("bailian", "qwen3.7-max"),
    ]


def test_default_settings_routes_pass_their_own_validation():
    # 守"配置字段 → 路由表"的搬运：默认路由与默认供应商表必须互相认识
    s = Settings(_env_file=None)
    routes = parse_routes(s.model_routes, set(s.providers))
    assert routes["fast"][0] == Candidate("bailian", "qwen-flash")
    assert all(c.provider in s.providers for c in unique_candidates(routes))
