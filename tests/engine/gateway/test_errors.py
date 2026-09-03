"""六类异常契约、翻译表与消毒。

上游异常用 openai SDK 的真实类型构造（探针 probe_m11_a）；端到端一条用 ChatOpenAI +
httpx2.MockTransport 真走 SDK（探针 probe_m11_b），证明翻译源与生产路径一致。
"""

import httpx2
import openai
import pytest
from langchain_openai import ChatOpenAI, StreamChunkTimeoutError

from app.engine.gateway.errors import (
    AuthError,
    BadRequestError,
    BudgetExceeded,
    GatewayError,
    GatewayExhausted,
    GatewayOverloadedError,
    GatewayRejected,
    GatewayStreamInterrupted,
    ProviderError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitedError,
    TenantQuotaExceeded,
    classify,
    parse_retry_after,
    retry_after_from,
    sanitize_error_text,
)

REQ = httpx2.Request("POST", "http://test/v1/chat/completions")
L2_VISIBLE = (
    GatewayOverloadedError,
    GatewayExhausted,
    BudgetExceeded,
    TenantQuotaExceeded,
    GatewayRejected,
    GatewayStreamInterrupted,
)
PROVIDER_FAMILY = (
    RateLimitedError,
    ProviderTimeoutError,
    ProviderServerError,
    BadRequestError,
    AuthError,
)


def status_error(
    status: int, body: dict | str | None = None, headers: dict | None = None
) -> openai.APIStatusError:
    """按 SDK 自己的映射造异常：与真实响应路径同一张表（探针 probe_m11_a）。"""
    if isinstance(body, dict):
        resp = httpx2.Response(
            status, request=REQ, headers=headers or {}, json={"error": body}
        )
    else:
        resp = httpx2.Response(
            status, request=REQ, headers=headers or {}, text=body or ""
        )
    client = openai.OpenAI(api_key="fake", base_url="http://test/v1", max_retries=0)
    return client._make_status_error_from_response(resp)


# ---------------------------------------------------------------- 类树


def test_six_l2_visible_classes_are_gateway_errors_but_not_provider_errors():
    for cls in L2_VISIBLE:
        assert issubclass(cls, GatewayError)
        assert not issubclass(cls, ProviderError)


def test_provider_family_is_internal_and_prefixed_with_provider():
    for cls in PROVIDER_FAMILY:
        assert issubclass(cls, ProviderError)
    err = ProviderServerError("bailian", "HTTP 503: x")
    assert str(err) == "[bailian] HTTP 503: x"
    assert err.provider == "bailian"


def test_rate_limited_carries_retry_after():
    assert RateLimitedError("p", "m", retry_after=3.0).retry_after == 3.0
    assert RateLimitedError("p", "m").retry_after is None


# ---------------------------------------------------------------- 消毒


def test_sanitize_masks_api_keys():
    text = "Incorrect API key provided: sk-abc123DEF456ghi789 (request id: r1)"
    out = sanitize_error_text(text)
    assert "sk-abc123DEF456ghi789" not in out
    assert "sk-***" in out


def test_sanitize_truncates_to_limit():
    assert len(sanitize_error_text("x" * 10_000)) <= 200
    assert len(sanitize_error_text("x" * 10_000, limit=120)) <= 120


def test_sanitize_masks_before_truncating_so_no_half_key_survives():
    text = "y" * 195 + "sk-abcdefghijklmnop"
    out = sanitize_error_text(text, limit=200)
    assert "sk-abcde" not in out
    assert out.endswith("sk-**")  # 掩码本身被截断没关系，明文一个字符都不能剩


def test_sanitize_leaves_normal_text_alone():
    assert sanitize_error_text("模型不存在: qwen-x") == "模型不存在: qwen-x"


# ---------------------------------------------------------------- Retry-After


def test_parse_retry_after_seconds():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("2.5") == 2.5


def test_parse_retry_after_http_date_in_past_clamps_to_zero():
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_parse_retry_after_garbage_is_none():
    assert parse_retry_after("soon-ish") is None
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None


def test_retry_after_from_prefers_ms_header_and_tolerates_missing_response():
    err = status_error(
        429, {"message": "m"}, headers={"retry-after-ms": "1500", "retry-after": "9"}
    )
    assert retry_after_from(err) == 1.5
    assert retry_after_from(openai.APITimeoutError(REQ)) is None


# ---------------------------------------------------------------- 翻译矩阵


def test_429_maps_to_rate_limited_with_retry_after():
    out = classify(
        "p",
        status_error(
            429,
            {"message": "slow down", "code": "rate_limit_exceeded"},
            {"retry-after": "7"},
        ),
    )
    assert isinstance(out, RateLimitedError)
    assert out.retry_after == 7.0
    assert "rate_limit_exceeded" in str(out)


@pytest.mark.parametrize("status", [401, 403])
def test_401_403_map_to_auth(status):
    assert isinstance(
        classify("p", status_error(status, {"message": "nope"})), AuthError
    )


@pytest.mark.parametrize("status", [400, 402, 404, 409, 422])
def test_other_4xx_map_to_bad_request(status):
    out = classify("p", status_error(status, {"message": "bad"}))
    assert isinstance(out, BadRequestError)
    assert f"HTTP {status}" in str(out)


def test_408_maps_to_timeout():
    assert isinstance(
        classify("p", status_error(408, {"message": "slow"})), ProviderTimeoutError
    )


def test_501_is_bad_request_not_server_error():
    out = classify("p", status_error(501, {"message": "not implemented"}))
    assert isinstance(out, BadRequestError)
    assert not isinstance(out, ProviderServerError)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_maps_to_server_error(status):
    assert isinstance(
        classify("p", status_error(status, {"message": "down"})), ProviderServerError
    )


def test_html_body_is_sanitized_snippet_not_crash():
    out = classify("p", status_error(503, "<html>" + "x" * 500 + "</html>"))
    assert isinstance(out, ProviderServerError)
    assert len(str(out)) < 260


def test_key_in_error_body_is_masked_before_it_reaches_the_exception():
    out = classify(
        "p",
        status_error(
            401, {"message": "Incorrect API key provided: sk-abc123DEF456ghi789"}
        ),
    )
    assert isinstance(out, AuthError)
    assert "sk-abc123DEF456ghi789" not in str(out)
    assert "sk-***" in str(out)


def test_pool_timeout_in_cause_chain_is_local_overload():
    try:
        raise openai.APITimeoutError(REQ) from httpx2.PoolTimeout("pool", request=REQ)
    except openai.APITimeoutError as err:
        out = classify("p", err)
    assert isinstance(out, GatewayOverloadedError)
    assert not isinstance(out, ProviderError)
    assert "[p]" in str(out)


def test_read_timeout_in_cause_chain_is_provider_timeout():
    try:
        raise openai.APITimeoutError(REQ) from httpx2.ReadTimeout("read", request=REQ)
    except openai.APITimeoutError as err:
        out = classify("p", err)
    assert isinstance(out, ProviderTimeoutError)


def test_connection_error_maps_to_server_error():
    assert isinstance(
        classify("p", openai.APIConnectionError(request=REQ)), ProviderServerError
    )


def test_stream_error_event_without_status_maps_to_server_error():
    err = openai.APIError(
        "stream", REQ, body={"message": "boom", "type": "server_error", "code": "E1"}
    )
    out = classify("p", err)
    assert isinstance(out, ProviderServerError)
    assert "流内错误 E1" in str(out)


def test_stream_chunk_timeout_maps_to_provider_timeout():
    assert isinstance(
        classify("p", StreamChunkTimeoutError(30.0, chunks_received=2)),
        ProviderTimeoutError,
    )


def test_raw_httpx2_exceptions_are_translated():
    assert isinstance(
        classify("p", httpx2.PoolTimeout("x", request=REQ)), GatewayOverloadedError
    )
    assert isinstance(
        classify("p", httpx2.ReadTimeout("x", request=REQ)), ProviderTimeoutError
    )
    assert isinstance(
        classify("p", httpx2.ConnectError("x", request=REQ)), ProviderServerError
    )


def test_already_translated_errors_pass_through_unchanged():
    err = AuthError("p", "m")
    assert classify("p", err) is err


def test_unknown_exceptions_are_not_disguised_as_upstream_faults():
    assert classify("p", ValueError("programming error")) is None
    assert classify("p", KeyError("x")) is None


# ---------------------------------------------------------------- 端到端：真走 SDK


async def test_end_to_end_401_body_with_key_is_masked_through_real_sdk_path():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: sk-abc123DEF456ghi789",
                    "type": "invalid_request_error",
                }
            },
        )

    model = ChatOpenAI(
        model="qwen-plus",
        api_key="fake",
        base_url="http://test/v1",
        max_retries=0,
        stream_usage=True,
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(openai.AuthenticationError) as info:
        async for _ in model.astream("hi"):
            pass
    assert "sk-abc123DEF456ghi789" in str(info.value)  # SDK 原样回显：消毒不可省
    out = classify("bailian", info.value)
    assert isinstance(out, AuthError)
    assert "sk-abc123DEF456ghi789" not in str(out)
    assert "sk-***" in str(out)
