"""候选模型工厂：全仓唯一构造 ChatOpenAI、唯一取密钥真值、唯一 fake 开关落点；附共享 httpx2 客户端工厂。

每个 kwarg 对应框架默认能力登记表的一行（ADR-005 / README）：
  max_retries=0          关闭 SDK 自带的 2 次重试——重试权威唯一在网关首块窗口
  timeout=CANDIDATE_TIMEOUT  三段超时的传输层部分：connect / read（块间闸）/ write / pool
  stream_usage=True      自定义 base_url 下框架不自动开启；不开则流式无 usage、记账全 0
  extra_body enable_thinking=False  DashScope 方言：不消费思考流，否则首块计时器被 reasoning 饿死
  cache=False            框架缓存显式关闭（ainvoke 内部流式会查缓存）
  stream_chunk_timeout   块间静默阈值，对首块也计时；取值 ≥ 首块窗口，归属由 ADR-006 裁决
  http_async_client      共享 httpx2 客户端注入：测试注入 MockTransport 的正门；注入即放弃框架默认
                         keepalive socket_options（由 make_http_client 自设）；不注入则每候选各建一池
fake 开关（AEGIS_FAKE_LLM）只在这里生效：组合根/网关/测试的注入点零改动统一换成 FakeReplyChatModel。
"""

import socket
import sys
from collections.abc import Iterable

import httpx2
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import Settings
from app.engine.fakes import FakeReplyChatModel
from app.engine.gateway import utterances as u
from app.engine.gateway.routing import Candidate

CANDIDATE_TIMEOUT = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
STREAM_CHUNK_TIMEOUT_S = 60.0


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """注入共享客户端即放弃框架默认的 TCP keepalive（登记表）：这里自设同一组。

    SO_KEEPALIVE 全平台；探测节奏（空闲 60s / 间隔 10s / 3 次）只有 Linux 暴露为 socket 常量，
    其它平台用内核默认（Windows 需 WSAIoctl，不做）。用途：长流中对端静默消失时不等到 read 超时才发现。
    """
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if sys.platform == "linux":
        options += [
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
        ]
    return options


def make_http_client(
    *, max_connections: int, max_keepalive_connections: int
) -> httpx2.AsyncClient:
    """所有候选共用的上游客户端（组合根建、关停 aclose）。

    连接池上限即 GatewayOverloadedError 的来源：池满等 CANDIDATE_TIMEOUT.pool 后抛 PoolTimeout。
    timeout 不在这里设：每个候选经 ChatOpenAI(timeout=...) 按请求传。
    """
    return httpx2.AsyncClient(
        transport=httpx2.AsyncHTTPTransport(socket_options=keepalive_socket_options()),
        limits=httpx2.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
        follow_redirects=True,
    )


def make_candidate(
    candidate: Candidate,
    *,
    base_url: str,
    api_key: SecretStr,
    http_async_client: httpx2.AsyncClient,
) -> ChatOpenAI:
    """构造一个真实候选。密钥为空启动即炸（SDK 也会炸，但话术是我们的）。"""
    key = api_key.get_secret_value()  # 全仓唯一取真值处
    if not key:
        raise ValueError(u.API_KEY_MISSING)
    return ChatOpenAI(
        model=candidate.model,
        base_url=base_url,
        api_key=key,
        max_retries=0,
        timeout=CANDIDATE_TIMEOUT,
        stream_usage=True,
        extra_body={"enable_thinking": False},
        cache=False,
        stream_chunk_timeout=STREAM_CHUNK_TIMEOUT_S,
        http_async_client=http_async_client,
    )


def build_candidates(
    settings: Settings,
    candidates: Iterable[Candidate],
    *,
    http_async_client: httpx2.AsyncClient,
) -> dict[Candidate, BaseChatModel]:
    """候选 → 模型实例表。fake 开关在此生效：开则每个候选都是确定性替身，不碰密钥不碰网络。"""
    if settings.aegis_fake_llm:
        return {
            c: FakeReplyChatModel(delay_s=settings.aegis_fake_llm_delay_s)
            for c in candidates
        }
    return {
        c: make_candidate(
            c,
            base_url=settings.providers[c.provider],
            api_key=settings.dashscope_api_key,
            http_async_client=http_async_client,
        )
        for c in candidates
    }
