# Aegis v2

多租户客服 Agent 平台的 LangChain / LangGraph 框架原生重写版。前作把网关、Agent 循环、治理层全部自研；本仓的规则是：凡框架能代劳的交给框架，代劳不了的自建，每一处取舍写进 ADR 并在本文的两张表里记账。

**当前状态**：M0 脚手架、M1 网关层已交付（ADR 001–010）；M2 起为 Agent 循环。测试零真实 LLM 调用。

## 一分钟架构

三域包加入口留根，依赖方向由 import-linter 钉死：

```
app/
  main.py / deps.py        入口与组合根：lifespan 建共享件，按租户装配网关
  core/                    横切件：config / logs / redis / db / limits / tokens
  engine/gateway/          网关：路由、异常契约、重试、熔断、出站闸、缓存、话术
  domain/usage.py          计量账本（不 import engine）
  routers/                 HTTP 层（M3 起）
```

网关是 `BaseChatModel` 的子类 `AegisGateway`，主实现是 `_astream`，九步顺序即设计：

1. 路由防御：调用方只声明 fast / standard / strong 档位，永不写模型名
2. deadline 换算
3. 租户前缀缓存：完整流才入库，命中零成本回放
4. 租户月度预算闸（fail-open）
5. 单请求预算闸（自家 token 尺估算）
6. 租户出站闸（进程内令牌桶）
7. 候选环：deadline 预检 → 熔断入口判定 → 受控重试（每次尝试先向供应商出站闸限时取令牌，再调用候选；候选可被故障注入器包装；只在首块前重试）→ 三待遇分流
8. 簿记在流尾：熔断上报、入缓存、记账全部在流耗尽处；半途弃流一概不记
9. 终局三段：预算耗尽 / 全员确定性拒绝 / 其余不可用

对外只有六类异常：四类可降级（Exhausted / Budget / TenantQuota / Overloaded）、一类确定性拒绝（Rejected，零兜底话术）、一类流级中断（StreamInterrupted，死因在 `__cause__`，半截不换路）。供应商异常永不穿出网关。

## 选型表：现成件 vs 自建

| 能力 | 前作 | 本仓 | 裁决 |
|---|---|---|---|
| 网关形态 | 自研统一协议 `LLMRequest / LLMChunk` | `BaseChatModel` 子类，契约锚在行为上 | ADR-005 |
| 上游客户端与 SSE 解析 | 自研 | langchain-openai + openai SDK（httpx2） | ADR-005 |
| 异常翻译 | 状态码白名单 | SDK 异常类型 + status_code 分段，自建 `classify()` 与消毒 | ADR-005 |
| 重试 | 自研首块窗口重试 | 自建于网关首块窗口；SDK `max_retries=0`；不用 `with_retry` / `with_fallbacks` | ADR-006 |
| 三段超时 | 自研 | connect 走 httpx2，首块自建，块间为 httpx2 字节级 read 超时与框架 `stream_chunk_timeout` 块级超时双闸并存，后者约束取值不小于首块窗 | ADR-006 |
| 熔断 | 自研三键 Redis 状态机 | 自研异步三键（pybreaker 经探针后翻案），Redis 挂降级进程内备胎且粘滞 | ADR-007 |
| 出站限流 | Redis Lua 令牌桶 | `InMemoryRateLimiter` + 自建限时取令牌包装，进程内近似 | ADR-008 |
| 租户配额事实源 | PG 账本 | PG 账本，事后闸口径 | ADR-008 |
| 租户缓存 | 自研 | 自建（框架缓存对 `astream` 无效、无租户维度、无完整性守卫） | ADR-009 |
| 计量账本 | Decimal 账本 | 保留，数据源改 `usage_metadata`，加 `usage_missing` 列 | ADR-009 |
| 组合根 | 一次装全 | 共享件进程级单例 + 网关按租户每请求装配 | ADR-009 |
| 入站限流 | Redis Lua 令牌桶依赖 | 固定窗 Lua（抄自 fastapi-limiter）+ 自建注入式限流器与依赖工厂；该库依赖移除 | ADR-010 |
| token 尺 | 自家（CJK 一字一 token） | 自家；框架尺对 CJK 低估 | ADR-005 |
| 测试拦截 | cassette 回放 | `httpx2.MockTransport` 注入 + 自建 fake 层；respx 只拦 httpx 0.28 路径，本仓无测试使用，仅留作开发依赖 | ADR-002 / 005 |

## 框架默认能力登记表

框架默认打开或默认可用的能力，本仓逐项决定接受、关闭或不用：

| 框架能力 | 默认态 | 处置 |
|---|---|---|
| openai SDK 自带重试 `max_retries` | 2 | 关闭（=0），重试权威唯一在网关 |
| `stream_usage` 自动开启 | 仅官方 base_url | 显式开启 |
| `stream_chunk_timeout` | 静默启用，对首块也计时 | 显式接受并约束取值不小于首块窗口 |
| `ChatOpenAI.cache` / 网关 `cache` / 全局 `set_llm_cache` | 未设 | 显式关闭两处 `cache=False`，不设全局缓存（`ainvoke` 内部流式会查缓存） |
| `with_retry` / `with_fallbacks` | 可用 | 不用（不重试流；首块后不切换） |
| `InMemoryRateLimiter` | 可用 | 接受为出站闸内核；不挂候选，经取令牌缝显式限时调用；预填桶容量 |
| `count_tokens_approximately` / tiktoken | 可用 | 不用，自家尺 |
| `include_response_headers` | False | 保持默认 |
| `disable_streaming` / `use_responses_api` / `streaming` | 默认 | 保持默认 |
| 默认 httpx 客户端（连接池、keepalive 套接字选项） | 自建 | 生产亦注入共享 `httpx2.AsyncClient`，连接池与 keepalive 自设，lifespan 收尾 |
| `ModelError.is_retryable` | 可用 | 不用；可重试判定由自建白名单与翻译表承担 |
| langchain-openai 异常类型化重包 | 自动 | 接受为翻译表输入端 |
| `extra_body` 透传 | 空 | 显式带 DashScope 的 `enable_thinking: false`：不消费思考流，配合首块判据；本仓保留的唯一一处供应商方言旋钮 |
| fastapi-limiter `RateLimiter` 依赖 / 默认 identifier / 默认回调 / `init` 类属性全局态 | 可用 | 不用，依赖移除；只抄其固定窗 Lua |
| 日志 | stdlib | structlog 最小配置 |

## 差距记账表：前作语义在本仓的命运

| 前作语义 | 命运 | 说明 |
|---|---|---|
| 统一协议判别联合 + 无损 JSON 往返 | 放弃 | 框架消息类型接管；tenant_id 字符集守卫保留并迁至组合根入口 |
| 六类异常契约 + 状态码翻译表 | 保留 | 翻译源改为 SDK 异常类型 + status_code 分段，含 501 归 BadRequest |
| 双重重试防线 | 降级为纪律 | `max_retries=0` 进审查清单，候选工厂单测断言 |
| tool-call 按 index 组装 / `[DONE]` 哨兵 / 流内 error 解析 | 黑盒化 | SDK 代劳；截断检测降级为 finish_reason 在场检查 |
| 跨供应商方言抹平 | 替代 | 框架集成包代劳；保留一处 DashScope 方言旋钮（见登记表 `extra_body` 行） |
| 首块窗口重试 + Retry-After 优先 + 满抖动 + 预算裸抛 | 保留（自建） | 探针证明两件现成件都不满足 |
| `PoolTimeout` 单独分类 | 保留 | 本地过载不重试、不换路、不进熔断账 |
| 三段超时 | 保留 | 块间为字节级与块级双闸并存 |
| deadline 传播 | 保留 | 经 `bind()` 作为具名参数传递 |
| 熔断三键 + 半开单探针 + 秒拒 + 失败窗 + Redis 挂本地镜像 + 粘滞降级 | 全部平移 | 自研异步三键；令牌是租约不是硬锁；降级期单探针退化为每副本各探；恢复只认被指派的探针 |
| 熔断粒度 provider | 升级为 provider:model | 一个候选三把键 |
| 出站 Lua 令牌桶（租户 + 供应商两维，多副本精确） | 降级近似 | 进程内桶，多副本口径为全局除以副本数；自建预判与限时等待 |
| 租户出站配额 `TenantQuotaExceeded` | 降级近似 | 每租户进程内桶，有界 LRU，非 HTTP 调用方同样受节流 |
| 出站限流按 HTTP 调用计 | 升级为按尝试计 | 每次尝试显式取令牌 |
| 入站令牌桶 + 即问即答 + Redis 挂降级本地桶且粘滞 | 算法降级近似，其余平移 | 固定窗：窗边可放两倍阈值突发、不管瞬时密度、Retry-After 为本窗剩余；降级期多副本口径为阈值乘以副本数 |
| fallback 矩阵不做能力断崖 | 保留为配置纪律 | 不做硬校验 |
| 租户前缀缓存 + 完整性守卫 + 自愈 + 命中零成本记账 | 保留（自建） | key = 消息白名单 + 调用参数全量 |
| Decimal 账本 + DB 时钟 + 价目表告警 + 记账 fail-open + 月度聚合 | 保留 | 上游缺 usage 以 `usage_missing` 列标记而不合成零行 |
| 三级预算 | 保留 | 会话级归 M2 |
| `GatewayLike` 协议 | 放弃 | 网关即 `BaseChatModel` |
| Anthropic 适配器 | 放弃 | 本仓只做 OpenAI 兼容族 |
| embedding 独立通道 | 移 M3 | 留协议缝 |
| 故障注入三模式 | 保留 | 包装对象换成 `BaseChatModel`，生产环境配置注入率即启动失败 |
| 密钥消毒 + SecretStr | 保留 | 消毒输入端换成 SDK 异常文本 |
| 中文话术单一事实源 | 保留 | 登记例外按类别：core 的入站限流串（429、降级与恢复告警、参数校验）、core 的配置校验串（生产禁注入、价目表形态）、domain 的价目表告警串；其余话术集中在 `engine/gateway/utterances.py` |
| cassette 确定性回放 | 待裁决 | M4 复核 |

## 数字表

对外引用的数字只来自本表，且每个数字都附口径与凭证；无实测不填。

| 指标 | 口径 | 数值 | 凭证 |
|---|---|---|---|
| 测试收集项数 | `uv run pytest --collect-only -q` 的收集项数，含跳过项，不设目标值 | 467（2026-09-05） | `reports/2026-09-05-gates.md` |
| 熔断跳闸后拒绝延迟 | open 态入口不触网络、不排队，定性结论 | 未实测 | ADR-007 实证节 |
| 缓存命中零上游调用 | 命中路径不经候选环，定性结论 | 未实测 | ADR-009 实证节 |
| 真实上游冒烟花费 | 一次冒烟六次调用（四候选各一次、思考形态一次、网关路径一次），真实 usage × 配置价目表，fake 记账与真实花费分账 | ¥0.000390（2026-09-05） | `reports/2026-09-05-smoke-dashscope.md` |

`reports/` 目录存放三门运行记录、探针日志与联网冒烟报告，每份报告注明日期、命令、口径与环境；本表数字一律回指其中一份报告，ADR 正文只写定性结论。

## 本地开发

```bash
docker compose up -d redis postgres
uv sync
uv run alembic upgrade head
```

三门：

```bash
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
uv run pytest -q
```

测试固定连 Redis db1（会话开始时清库）与 Postgres 测试库 `aegis_test`（由迁移建表，每测外层事务回滚）；两者任一不可达时对应测试组跳过并提示启动命令。`AEGIS_FAKE_LLM=1` 时候选工厂统一换成 fake 模型，全仓测试零真实调用。

## 文档

- `docs/adr/`：设计决策，只增不改。001 重写章程、002 版本基线、003 包结构、004 废弃探索仓、005 网关形态与异常契约、006 重试权威与超时、007 熔断语义、008 出站闸与租户配额、009 租户缓存与计量组合根、010 入站限流原语。
- 自 ADR-005 起，涉及框架事实的 ADR 带"实证（探针）"节：写复现方式与定性结论，量化数字进 `reports/`。
