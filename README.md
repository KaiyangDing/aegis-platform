# Aegis v2

![ci](https://github.com/KaiyangDing/aegis-platform/actions/workflows/ci.yml/badge.svg)

多租户客服 Agent 平台的 LangChain / LangGraph 框架原生重写版。前作把网关、Agent 循环、治理层全部自研；本仓的规则是：凡框架能代劳的交给框架，代劳不了的自建，每一处取舍写进 ADR 并在本文的三张表里记账。

**v2.0 范围**：L1 网关（ADR 005–010）、L2 运行时（ADR 011–013）、一层刚好让"发消息 → 工具 → 审批挂起 → 坐席批准 → 续跑 → 回放"闭环成立的薄业务面（ADR-014），以及 `reports/` 里可复算的凭证。未包含的范围见 [未覆盖范围](#未覆盖范围)。测试零真实 LLM 调用。

## 一分钟架构

四域包加入口留根，依赖方向由 import-linter 五条契约钉死（core 不依赖兄弟包；engine / domain 只依赖 core；gateway 不依赖 runtime；core / engine / domain 不依赖 business）：

```
app/
  main.py / deps.py        入口与组合根：lifespan 建共享件、预热租户 spec、按租户装配网关（含月度预算 resolver）
  core/                    横切件：config / logs / redis / db / limits / tokens / auth（JWT 三角色）/ checkpoint / loops
  engine/gateway/          网关：档位路由、异常契约、重试、熔断、出站闸、租户缓存、话术
  engine/runtime/          运行时：AgentSpec 注入面、事件、工具契约、七件中间件栈、守卫、上下文编译、门面 run / resume
  domain/                  数据契约：usage 账本 / events 事实源 / sessions 状态机 / approvals 审批单
  business/                业务薄片：租户静态配置、spec 装配、三工具、进程内模拟后端、业务话术
  routers/                 HTTP：POST /v1/chat（SSE）/ POST /v1/approvals/{id} / GET /v1/sessions/{id}/events
```

### 网关（L1）

网关是 `BaseChatModel` 的子类 `AegisGateway`，主实现是 `_astream`，九步顺序即设计：

1. 路由防御：调用方只声明 fast / standard / strong 档位，永不写模型名
2. deadline 换算
3. 租户前缀缓存：完整流才入库，命中零成本回放
4. 租户月度预算闸（fail-open；生产值由业务层 resolver 注入）
5. 单请求预算闸（自家 token 尺估算）
6. 租户出站闸（进程内令牌桶）
7. 候选环：deadline 预检 → 熔断入口判定 → 受控重试（每次尝试先向供应商出站闸限时取令牌，再调用候选；候选可被故障注入器包装；只在首块前重试）→ 三待遇分流
8. 簿记在流尾：熔断上报、入缓存、记账全部在流耗尽处；半途弃流一概不记
9. 终局三段：预算耗尽 / 全员确定性拒绝 / 其余不可用

对外只有六类异常：四类可降级（Exhausted / Budget / TenantQuota / Overloaded）、一类确定性拒绝（Rejected，零兜底话术）、一类流级中断（StreamInterrupted，死因在 `__cause__`，半截不换路）。供应商异常永不穿出网关。

### 运行时（L2）

`create_agent(model=AegisGateway, tools, middleware=[七件])`：一张按 `(tenant_id, spec 指纹)` 编译并缓存的图，网关直接作为 model，档位 / deadline / 会话经 `model_settings` 落到网关具名参数。七件中间件全部自建，**栈序即语义**：

```
[RunEvents, Guards, AegisSummarization, Approvals, Gates, ModelCall, ToolExec]
before_agent : RunEvents（user_message 首事件、run 级通道归零、T1）→ Guards（入口规则库 + 按租户开通的 fast 档分类器；HIGH 拒答以 COMPLETED 收尾）
before_model : AegisSummarization（历史超 0.8×预算先压缩）→ Gates（#6 取消 / #1 轮数）
wrap_model   : ModelCall（编译上下文 → #3 预算预检 → llm_call → 网关 → L1 异常四组映射 → llm_result → 出口守卫在 AIMessage 进 state 之前替换）
after_model  : Gates（#5 协议违规含幻觉工具名 / #4 重复调用）→ Approvals（风险闸门 → 幂等开单 → approval_requested → interrupt()）
wrap_tool    : ToolExec（七步：严校验 → 风险闸门 fail-closed / 通行证 → write-ahead 幂等键 → 身份注入 → 超时与重试策略 → 结果规范化 → 事件与连败禁用）
after_agent  : RunEvents（兜底 assistant_message → loop_terminated 末事件 → T4 / T5）
```

- **事实源双写不投影**（ADR-011）：`AsyncPostgresSaver`（`durability="sync"`）管图恢复，自建 `events` 表管审计 / 回放断言 / SSE 游标。事件 id 由框架任务身份派生（uuid5）并以 `ON CONFLICT DO NOTHING` 去重：失同步只可能"事件领先一步"，重放同一步派生同一 id 被吸收——write-ahead 幂等键、半截 LLM 作废重发、崩溃恢复 = 重放 + 去重都靠这一个机制。
- **六道终止闸门**与八值终止原因（completed / max_iterations / step_timeout / token_budget_exceeded / repeated_calls / protocol_violation / cancelled / gateway_rejected）进 `loop_terminated` 事件可断言；终止时最后一条 AIMessage 的每个 tool_call 都配对 ToolMessage。
- **HITL**（ADR-013）：审批单五态 CAS，`decide` 只翻"pending 且未过期"（到期 fail-closed）；一个 interrupt 载荷装本轮全部单，run 干净返回、进程可下线；`resume(session_id, approval_id)` 既是计划内续跑也是崩溃恢复入口，拒绝 / 撤回 / 超时 → `cancelled` 终止零 LLM 调用，双坐席并发靠 CAS 恰一赢家。
- **崩溃恢复**：半截工具凭原键重执行绝不产生第二把幂等键；半截 LLM 补 `llm_result(interrupted)` 后作废重发；恢复次数超限 → `failed` + `recovery_abandoned` 图外事件。无租约，"上一进程已死"由恢复调用方断言。

### 业务面与 HTTP（L3，薄）

| 端点 | 角色 | 要点 |
|---|---|---|
| `POST /v1/chat` | 任意角色，本人会话 | 依赖序 = 认证 → 租户级固定窗限流；会话首见即建；等审批 409 附单号；先取首帧再返回 `text/event-stream`（`SessionBusy` → 409 / 会话不存在 → 404 只在首帧前可译为状态码）；流末合成 `done{reason}` 或 `done{awaiting_approval, approvals}` |
| `POST /v1/approvals/{id}` | operator / admin，本租户 | 404 → 他租 403 → 惰性到期扫描 → `decide` CAS（输家 409 附赢家终态）→ 同步续跑并返回摘要 `done / awaiting_approval / no_op` |
| `GET /v1/sessions/{id}/events` | 终端用户本人 / 员工本租户 | 游标 = `max(after_seq, Last-Event-ID)` 一次性读事实源，帧 `id` 就是 `events.seq`；以 `done{snapshot, run_state, next_seq}` 收尾 |

终端用户不可见 `llm_call / llm_result / summary_updated / guardrail_triggered / precheck_vetoed`，员工全量。身份来自 HS256 JWT（`sub / tid / role`），租户配置是代码内静态表（工具白名单启动期点名校验），三工具（`order_query` 读 / `refund_apply` 写 + 阈值闸门 / `ticket_create` 写豁免）在工具内做租户与用户双重归属校验、三种失败同一话术，写工具把 write-ahead 事件 id 作为幂等键透传进程内模拟后端。

## 选型表：现成件 vs 自建

| 能力 | 前作 | 本仓 | 裁决 |
|---|---|---|---|
| 网关形态 | 自研统一协议 `LLMRequest / LLMChunk` | `BaseChatModel` 子类，契约锚在行为上 | ADR-005 |
| 上游客户端与 SSE 解析 | 自研 | langchain-openai + openai SDK（httpx2） | ADR-005 |
| 异常翻译 | 状态码白名单 | SDK 异常类型 + status_code 分段，自建 `classify()` 与消毒 | ADR-005 |
| 重试 | 自研首块窗口重试 | 自建于网关首块窗口；SDK `max_retries=0`；不用 `with_retry` / `with_fallbacks` | ADR-006 |
| 三段超时 | 自研 | connect 走 httpx2，首块自建，块间为 httpx2 字节级 read 超时与框架 `stream_chunk_timeout` 块级超时双闸并存 | ADR-006 |
| 熔断 | 自研三键 Redis 状态机 | 自研异步三键（pybreaker 经探针后翻案），Redis 挂降级进程内备胎且粘滞 | ADR-007 |
| 出站限流 | Redis Lua 令牌桶 | `InMemoryRateLimiter` + 自建限时取令牌包装，进程内近似 | ADR-008 |
| 租户配额事实源 | PG 账本 | PG 账本，事后闸口径；月度生产值由业务层 resolver 注入 | ADR-008 / 014 |
| 租户缓存 | 自研 | 自建（框架缓存对 `astream` 无效、无租户维度、无完整性守卫） | ADR-009 |
| 计量账本 | Decimal 账本 | 保留，数据源改 `usage_metadata`，加 `usage_missing` 列 | ADR-009 |
| 组合根 | 一次装全 | 共享件进程级单例 + 网关按租户每请求装配 | ADR-009 |
| 入站限流 | Redis Lua 令牌桶依赖 | 固定窗 Lua（抄自 fastapi-limiter）+ 自建注入式限流器与依赖工厂；该库依赖移除 | ADR-010 |
| token 尺 | 自家（CJK 一字一 token） | 自家；框架尺对 CJK 低估 | ADR-005 |
| 循环骨架 | 自研 AgentLoop | `create_agent` 图 + 七件自建 `AgentMiddleware`，栈序即语义 | ADR-012 |
| 事实源 | 自研事件溯源 + 同事务投影 | `AsyncPostgresSaver`（`durability="sync"`）管恢复 + 自建 events 表管审计 / 回放 / 游标，双写不投影 | ADR-011 |
| 终止闸门 | 自研六道 | 自建于 Gates / ModelCall；`ModelCallLimitMiddleware` 不用 | ADR-012 |
| 工具执行 | 自研七步 | `wrap_tool_call` 内自建七步；框架 ToolNode 只作载体、handler 是哨兵 | ADR-012 |
| 滚动摘要 | 自研 | 子类化 `SummarizationMiddleware`（自家尺、中文包装、关内部重试、fail-open 提取式降级） | ADR-012 |
| 人工审批 | 自研五态状态机 | 自建 Approvals 于 `interrupt()` 原语；`HumanInTheLoopMiddleware` 不用，借其载荷形态 | ADR-013 |
| 守卫 | 自研三段 | 自建三段；`PIIMiddleware` 不用 | ADR-012 |
| 崩溃恢复 | 四支分诊 + 租约 + reaper | 重放 + 去重；无租约 | ADR-011 / 012 |
| 身份 | JWT + users 表 | JWT 三角色，脚本签发，无用户目录 | ADR-014 |
| 租户配置 | tenants 表 + 种子脚本 | 代码内静态表，启动期预热校验 | ADR-014 |
| 模拟业务后端 | PG 两表 + `ON CONFLICT` 去重 | 进程内类按幂等键判重 | ADR-014 |
| 流式出口 | 自研双通道 SSE | 步级事件帧（`id` = `events.seq`），回放一次性读 | ADR-014 |
| 测试拦截 | cassette 回放 | `httpx2.MockTransport` 注入 + 自建 fake 层 + 剧本网关；respx 只拦 httpx 0.28 路径，本仓无测试使用，仅留作开发依赖 | ADR-002 / 005 |

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
| checkpoint `durability` | `"async"`（无持久化屏障） | 显式 `"sync"`；`"exit"` 禁用 |
| `create_agent` 钩子各成节点、每节点一次 checkpoint | 自动 | 接受（恢复粒度 = 节点） |
| `recursion_limit` | 25 | 显式由 `LoopPolicy` 推导（七件栈 6·max_iterations+6） |
| `stream_mode="messages"` 嵌套双发 | 每块两次 | 对外只走 `custom` 步级事件；token 级通道未接 |
| `jump_to` 未声明 `can_jump_to` | 静默无效 | 所有跳转钩子显式 `hook_config`，静态测试钉列表顺序与出边 |
| `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` | 可用 | 不用（英文话术不可注入、无终止事实、不配对悬空 tool_calls） |
| `HumanInTheLoopMiddleware` | 可用 | 不用，借载荷形态与决定类型名 |
| `SummarizationMiddleware` 与其内部 `with_retry(3)` | 可用 / 开 | 接受并子类化；关闭内部重试 |
| `RemoveMessage(REMOVE_ALL_MESSAGES)` 摘要替换 | 开 | 接受（原文在事件与历史 checkpoint） |
| `PIIMiddleware` / `ModelRetry` / `ModelFallback` / `ToolRetry` / `ToolError` / `ContextEditing` / `TodoList` 等 | 可用 | 不用 |
| ToolNode `handle_tool_errors` / 并行执行 / 同步工具走线程池 | 默认 | wrap 内接管全部结局；每 run 一把锁串行；工具实现一律 `async def` |
| ToolNode 幻觉工具名英文 ToolMessage | 自动 | after_model 先拦（#5 计违规、中文回填） |
| `interrupt()` 重放节点 | 结构事实 | 接受 + 幂等（开单 / 事件 id 派生） |
| `AgentState` 自定义通道出现在输出 | 开 | 全部 `PrivateStateAttr`，run 级计数在 `before_agent` 归零 |
| `AsyncPostgresSaver.setup()` 自带迁移；checkpoint 表无 tenant 列 | 手动 / 结构事实 | lifespan 调用；表在 alembic 之外；`thread_id = session_id` 全局唯一 + 应用层归属校验 |
| psycopg 拒绝 Windows Proactor 循环 | Windows 事实 | 开发进程 `--loop app.core.loops:selector_loop_factory`；测试 Selector；容器无事 |
| `context_schema` / `runtime.context`；`stream_mode="custom"` | 可用 | 接受为每 run 载体与事件外流通道 |
| LangSmith 追踪 | 环境变量关 | 保持关闭 |
| FastAPI `include_router` 结果为 `_IncludedRouter` | 0.141 事实 | 路径以 `app.openapi()["paths"]` 为准 |

## 差距记账表：前作语义在本仓的命运

| 前作语义 | 命运 | 说明 |
|---|---|---|
| 统一协议判别联合 + 无损 JSON 往返 | 放弃 | 框架消息类型接管；tenant_id 字符集守卫保留并迁至组合根入口 |
| 六类异常契约 + 状态码翻译表 | 保留 | 翻译源改为 SDK 异常类型 + status_code 分段，含 501 归 BadRequest |
| 双重重试防线 | 降级为纪律 | `max_retries=0` 进审查清单，候选工厂单测断言 |
| tool-call 按 index 组装 / `[DONE]` 哨兵 / 流内 error 解析 | 黑盒化 | SDK 代劳；截断检测降级为 finish_reason 在场检查 |
| 跨供应商方言抹平 | 替代 | 框架集成包代劳；保留一处 DashScope 方言旋钮 |
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
| 三级预算 | 保留 | 会话级在运行时闸门 #3；月度生产值由业务层 resolver 注入 |
| `GatewayLike` 协议 | 放弃 | 网关即 `BaseChatModel` |
| Anthropic 适配器 | 放弃 | 本仓只做 OpenAI 兼容族 |
| embedding 独立通道 | 未包含 | 无检索链路即无消费者；记账缝与价目表保留 |
| 故障注入三模式 | 保留 | 包装对象换成 `BaseChatModel`，生产环境配置注入率即启动失败 |
| 密钥消毒 + SecretStr | 保留 | 消毒输入端换成 SDK 异常文本 |
| 中文话术单一事实源 | 保留 | 登记例外按类别：core 的入站限流串、配置校验串与认证依赖串；domain 的价目表告警串；business 的业务话术与 HTTP detail 串；其余话术集中在 `engine/gateway/utterances.py` 与 `engine/runtime/utterances.py` |
| AgentLoop 循环骨架 + working 序列管理 | 替代 | `create_agent` 图 + `messages` 通道 |
| AgentSpec 注入面四类型 | 保留 | 逐字平移；`Tier` 引 gateway |
| 六道终止闸门 + 八值 + 兜底话术 | 保留（自建中间件） | `ModelCallLimitMiddleware` 只近似 #1 且英文话术 / 无事实 / 不配对 |
| 事件即事实源（17 类、seq、payload 原文） | 保留 + 升级 | 加 `task_id / checkpoint_id` 回指框架 checkpoint；事件 id 改为派生；与 checkpoint 双写 |
| 同事务投影（messages / tool_invocations / summary） | 放弃 | 消息读模型 = checkpoint state；工具审计 = events 直查 |
| 单写者 + 唯一约束 + 围栏三防线 | 保留（无锁） | 围栏靠 `(session_id, seq)` 唯一约束；并发 resume 由 CAS 裁决 |
| 工具七步 / 顺序执行 / write-ahead 幂等键 / 写超时 RESULT_UNKNOWN | 保留 | wrap 内自建；每 run 一把锁串行；重放命中既有 tool_call 事件即原键重执行 |
| 六层预算编译 + 确定性折叠 | 保留（只改 request 不改 state） | 记忆 / 检索两层本期预算为 0 |
| 滚动摘要 + `summary_updated` 事件 + fail-open 降级 | 保留（子类化现成件） | 关框架重试；中文包装；提取式降级 |
| 审批五态 CAS + 到期 fail-closed + TOCTOU 挂点 | 保留（自建） | 业务侧前置校验器未包含，挂点与 `precheck_vetoed` 事件保留 |
| reject / expire / cancel → cancelled 终止；挂起后进程可下线；计划内恢复 = 灾难恢复同路径 | 保留 | `interrupt()` + checkpoint |
| 崩溃恢复四支分诊 | 降级为"重放 + 去重" | 半截 LLM 补 `llm_result(interrupted, cause=replay)` |
| 会话锁三实现 / 租约 / reaper / 审批到期周期扫描 | 未包含 | 无锁：T1 / T3 CAS + 唯一约束；到期扫描在审批端点入口惰性调用 |
| 守卫三段 | 保留 | 出口守卫在 wrap 内替换、泄漏原文不进 checkpoint；流式滑动缓冲算法保留、token 流未接 |
| cassette 录制回放 + 四道游标 | 降级 | fake 剧本 + `normalize_events` 行为轨迹断言 |
| 逐 token 流（第二通道） | 未包含 | 对外只有步级事件帧 |
| JWT / RBAC 端点×角色矩阵 | 降级 | 三端点 × 三角色；单密钥；无登录端点与用户目录 |
| 三层租户隔离（Repository 强制 / RLS 兜底 / 归属校验在工具内） | 两层 | Store 签名强制 tenant_id 且读双过滤 + 工具内双比对；RLS 未包含（框架 checkpoint 表无 tenant 列） |
| 入站三件（租户限流 / 会话互斥 / awaiting_approval 准入） | 保留（形态降级） | 等审批的新消息 409 附单号，不做短流；用户侧撤回端点未包含（存储 CAS 与恢复分支保留） |
| 意图路由五值 + FAQ 直答 | 未包含 | 入口分类器（守卫）占用了 fast 档单次分类的位置 |
| RAG 全链（切块降级 / 断点续传 / 检索失败话术） | 未包含 | 纯工具客服；`retrieval_budget` 段保留 |
| 五工具 + 模拟后端 | 降级 | 三工具；进程内后端按键判重 |
| 越权三路逐字节同话术 | 保留 | `DENIED_TEXT` 单常量 |
| 审批 API 闭环（401 → 403 → 404 → 跨租 403 → CAS 输家 409） | 保留 | 员工两角色均为租户级身份 |
| 转人工三档降级取材 | 降级 | 转人工 = 模型调用工单工具；`handoff` 事件无生产者 |
| SSE 帧词汇 / after_seq 重订阅 | 保留 / 降级 | 帧 id = seq；回放一次性读，无活尾轮询与 LISTEN/NOTIFY |
| 租户配置治理（种子脚本唯一入口 / 工具白名单启动炸） | 降级 / 保留 | 代码内静态表；`preheat_specs` 启动期点名 |
| 终端用户不可见完整 trace | 降级 | 帧与回放按角色过滤；无 trace 还原器与展示层脱敏 |
| worker 进程形态（arq） | 未包含 | 周期任务改成幂等按需入口；依赖移除 |
| trace 还原 / metrics / 回放回归红绿 / 离线评测 / 成本对照实验 / 压测 / 演示前端 | 未包含 | 可观测底座 = events 表 + structlog |

## 数字表

对外引用的数字只来自本表，且每个数字都附口径与凭证；无实测不填。

| 指标 | 口径 | 数值 | 凭证 |
|---|---|---|---|
| 测试收集项数 | `uv run pytest --collect-only -q` 的收集项数，含跳过项，不设目标值 | 939（2026-09-09） | `reports/2026-09-09-gates.md` |
| 真实上游 HTTP 端到端花费 | 一次运行（查单 → 申请退款 → 审批挂起 → 坐席批准 → 续跑退款 → 终答）三次 qwen-plus 调用，`usage_ledger` 按 session_id 求和，配置价目表 | ¥0.002323（2026-09-09） | `reports/2026-09-09-e2e-real.md` |
| 真实上游冒烟花费 | 一次冒烟六次调用（四候选各一次、思考形态一次、网关路径一次），真实 usage × 配置价目表，fake 记账与真实花费分账 | ¥0.000390（2026-09-05） | `reports/2026-09-05-smoke-dashscope.md` |
| 熔断跳闸后拒绝延迟 | open 态入口不触网络、不排队，定性结论 | 未实测 | ADR-007 实证节 |
| 缓存命中零上游调用 | 命中路径不经候选环，定性结论 | 未实测 | ADR-009 实证节 |
| 崩溃恢复 | 半截工具凭原键重执行（一把钥匙、执行两次）、半截 LLM 作废重发，真 PG，定性结论 | 不报时长 | `reports/2026-09-09-recovery.md` |

`reports/` 目录存放三门运行记录、探针日志、联网冒烟与端到端报告，每份报告注明日期、命令、口径与环境；本表数字一律回指其中一份报告，ADR 正文只写定性结论。

## 未覆盖范围

v2.0 未包含：检索增强（RAG）与 embedding 通道、行级安全（RLS）、意图路由与 FAQ 直答、转人工三档摘要、逐 token 流式通道、活尾重订阅、用户侧撤回审批端点、批准后业务前置校验器、周期任务与 worker 进程、trace 还原与指标端点、回放回归与离线评测、成本对照实验与压测、演示前端。每一项在上方差距记账表有命运与说明，架构层面的裁决见 ADR-014。

## 本地开发

```bash
docker compose up -d redis postgres
uv sync
uv run alembic upgrade head
```

启动 API（Windows 须指定 Selector 事件循环；`.env` 需 `JWT_SECRET`（≥32 字节）与 `DASHSCOPE_API_KEY`，`AEGIS_FAKE_LLM=1` 时不打真实上游）：

```bash
uv run uvicorn app.main:app --loop app.core.loops:selector_loop_factory
```

签发演示 token 并走一遍闭环（默认 fake 模式；`--real` 打真实上游，预算护栏写死在脚本里）：

```bash
uv run python scripts/mint_token.py --user u-a1 --tenant tenant-a --role user
uv run python scripts/e2e_flow.py
```

三门：

```bash
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
uv run pytest -q
```

测试固定连 Redis db1（会话开始时清库）与 Postgres 测试库 `aegis_test`（由迁移建表，每测外层事务回滚）；两者任一不可达时对应测试组跳过并提示启动命令。`AEGIS_FAKE_LLM=1` 时候选工厂统一换成 fake 模型，全仓测试零真实调用。CI（`.github/workflows/ci.yml`）在 ubuntu 上以 services 起 PG + Redis 跑同一套三门。

## 文档

- `docs/adr/`：设计决策，只增不改。001 重写章程、002 版本基线、003 包结构、004 废弃探索仓、005 网关形态与异常契约、006 重试权威与超时、007 熔断语义、008 出站闸与租户配额、009 租户缓存与计量组合根、010 入站限流原语、011 事实源主权、012 闸门栈序与终止事实、013 人工审批加固、014 薄 HTTP 面与 v2.0 范围。
- 自 ADR-005 起，涉及框架事实的 ADR 带"实证（探针）"节：写复现方式与定性结论，量化数字进 `reports/`。
