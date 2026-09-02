# ADR-002: 版本基线

- 日期：2026-09-02
- 状态：已接受

## 背景

spike 仓的框架映射结论产自 langchain 1.3.14 / langchain-core 1.4.9 / langgraph 1.2.9 世代；本仓 2026-09-02 实装已是更新世代。依赖漂移与"凭旧世代记忆写码"是两类已知事故源，必须钉死基线并规定复核纪律。

## 决策

- **Python 3.14**（uv 管理；asyncpg/psycopg 对 3.14 的兼容已由 Argus 全线验证）。
- 精确版本权威=**uv.lock**。关键基线（2026-09-02 实装）：langchain-core 1.6.1、langchain-openai 1.6.0、langgraph 1.2.11、langgraph-checkpoint-postgres 3.1.2、fastapi 0.141.1、sqlalchemy 2.0.52、pydantic-settings 2.15.0、arq 0.28.0、pybreaker 1.4.1、redis 5.3.1、structlog 26.1.0；dev 组 pytest 9.1.1、pytest-asyncio 1.4.0、respx 0.23.1、ruff 0.16.5、import-linter 2.14。
- **fastapi-limiter 钉死 ==0.1.6**（沿用 Argus 决定；全表唯一用 == 的依赖）。
- PG 镜像=**pgvector/pgvector:pg16**：即 PG16 加装 pgvector 扩展，M3 做 RAG 时免重建数据卷。
- **暂缓清单**（非遗漏，届时随对应 ADR 加入）：pgvector(py)/jieba/langchain-text-splitters → M3 检索方案 ADR；locust → M4 压测；mypy → M4 CI 定档再议。
- 依赖变更唯一入口=**uv add/remove**（IDE 的图形化装包入口会绕过锁文件审慎直改 pyproject，禁用）。

## 后果

- spike 仓结论引用前**须在本仓版本上逐条复核**；框架 API 用前必 inspect/探针实测，不凭记忆写码。
  首例实证（2026-09-02 探针，core 1.6.1）：GenericFakeChatModel.bind_tools 仍抛 NotImplementedError（须子类覆写，研究仓结论成立）；tool_calls 剧本原样透传且框架附加 `type: "tool_call"` 字段（断言按字段比）；剧本耗尽抛 StopIteration。
- 任何依赖升级走新 ADR，并重跑受影响探针。
