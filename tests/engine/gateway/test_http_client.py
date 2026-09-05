"""共享上游 httpx2 客户端工厂（M1.5c）：注入即放弃框架默认 keepalive，这里自设同一组；
候选工厂接受它；关停可 aclose。零网络。"""

import socket
import sys

import httpx2
import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.engine.gateway import candidates as candidates_mod
from app.engine.gateway.routing import Candidate

if not hasattr(candidates_mod, "make_http_client"):
    pytest.skip(
        "M1.5c 未敲：candidates.py 尚无 make_http_client", allow_module_level=True
    )

from app.engine.gateway.candidates import (
    keepalive_socket_options,
    make_candidate,
    make_http_client,
)


def test_keepalive_is_always_on_and_tuning_only_where_the_platform_exposes_it():
    options = keepalive_socket_options()
    assert options[0] == (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    assert (len(options) > 1) == (sys.platform == "linux")


async def test_client_is_shared_across_candidates_and_closable():
    client = make_http_client(max_connections=7, max_keepalive_connections=3)
    assert isinstance(client, httpx2.AsyncClient)
    assert client.follow_redirects is True
    a = make_candidate(
        Candidate("bailian", "qwen-plus"),
        base_url="http://test/v1",
        api_key=SecretStr("sk-x"),
        http_async_client=client,
    )
    b = make_candidate(
        Candidate("bailian", "qwen-flash"),
        base_url="http://test/v1",
        api_key=SecretStr("sk-x"),
        http_async_client=client,
    )
    assert isinstance(a, ChatOpenAI)
    assert a.http_async_client is client and b.http_async_client is client
    await client.aclose()
    assert client.is_closed
