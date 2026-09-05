# ADR-009: 租户缓存、计量账本与组合根

- 日期：2026-09-05
- 状态：已接受

## 背景

前作在网关内自建三件与租户绑定的组件：租户前缀精确缓存（Redis，key 三原则、完整流才入库、命中盖章零成本记账、故障粘滞降级）、Decimal 计量账本（一调用一行、DB 时钟、价目表缺失记 0 且告警、记账 fail-open、月度聚合排除缓存回放）、组合根（真实依赖只在一处聚合）。本仓网关是 `BaseChatModel` 子类（ADR-005），没有前作的统一请求对象，租户身份、缓存 key、计量数据源与装配方式都要重新落点；LangChain 自带的缓存（`BaseCache` / `set_llm_cache`）能否替代自建缓存需裁决。

本仓 venv 探针事实（langchain-core 1.6.1 / asyncpg 0.31 / SQLAlchemy 2.0.52 / alembic 1.19）：
- 框架缓存只在 `_generate_with_cache` 路径生效：默认 `cache=None` 加全局缓存时 `ainvoke` 两次只打一次上游（内部流式聚合后写缓存），`astream` 完全绕过；类级 `cache=False` 免疫。框架缓存 key 无租户维度、无完整性守卫。
- 消息对象：`HumanMessage.id` 默认 None，框架外壳不改入参的 id；流出块由外壳统一赋一个 run id；`model_dump` 字段集因消息类型而异（AI 多 `tool_calls/invalid_tool_calls/usage_metadata`，Tool 多 `tool_call_id/status/artifact`）；`AIMessageChunk.type` 与 `AIMessage.type` 不同。`convert_to_openai_tool` 结果可复现。
- 块合并：`response_metadata` 自定义键在 `+` 合并、外壳 `astream`、`ainvoke` 聚合三条链上都保留；同键同值可合并，同键异值（bool）合并抛 `TypeError`；回放流不标 `chunk_position="last"` 时外壳会补一个空块；`tool_call_chunk()` 工厂不接受 `type` 键。
- contextvars 经 `astream`/`ainvoke`、同步与异步回调（框架 `run_in_executor` 复制上下文）、`asyncio.to_thread`、`create_task` 全部传播；子任务内 `set` 不回流父任务；同一任务内异步生成器里的 `set` 会泄漏给消费者。
- asyncpg 在 Windows 默认事件循环下可用；`Numeric(12,6)` 与 Decimal 往返无损、`SUM` 返回 Decimal、`server_default=now()` 带时区且在事务内冻结；同一引擎的池化连接不能跨事件循环复用；外层连接事务加 `join_transaction_mode="create_savepoint"` 的会话工厂可让组件真实 `commit` 而外层 `rollback` 一笔勾销。
- alembic 以平台 locale 编码读取 `alembic.ini`。

## 决策

1. **缓存自建，不用框架缓存**（`engine/gateway/cache.py`）：框架缓存对 `astream` 无效、无租户维度、无完整性守卫，三条各自足以否决；`AegisGateway` 类级 `cache=False` 保留并以静态断言钉住，租户缓存另起字段 `reply_cache`。
2. **key = `aegis:cache:{schema 版本}:{tenant_id}:{sha256(canonical(语义本体))}`**。语义本体 = 档位 + 消息白名单字段（`type/content/name/tool_calls/tool_call_id/status`，历史中的块先转为整消息）+ `stop` + **全部调用参数**。两条相反方向的规则：消息类型是框架的，字段会随版本变化，只能白名单语义字段（黑名单会漏易变字段导致永不命中）；调用参数是网关自己的契约，凡影响上游调用的参数都必须影响 key（白名单会漏新参数导致误命中），非 JSON 值以字符串兜底只会 miss 不会误命中。档位、deadline、session_id 是 `_astream` 的具名参数，天然不进 key。canonical = 键排序 + 紧凑分隔符。版本前缀：值格式变更后旧条目全体 miss、随 TTL 消失。
3. **值 = 自家最小块 schema**（content / tool_call_chunks / usage / finish，附来源 provider 与 model），不存框架对象或 run id；回放时每块盖章 `response_metadata["aegis_cached"]=True`（只盖 True），末块标 `chunk_position="last"`。**入库标准**：见到终止信号且至少一块有实质内容；候选环只在流耗尽处入库，缓存层再守一次。脏条目解析失败即删除、按 miss 处理。TTL 进配置，0 = 组合根不装缓存。
4. **缓存故障 fail-open 且粘滞**（与 ADR-007 决策 6 同款）：进程级 `CacheStore` 任一 Redis 触点异常即降级为直通，`probe_interval` 内不再碰 Redis，get 与 put 共用一个探针窗口，探针失败顺延、成功即恢复并记日志；`CacheLike` 实现永不抛异常，候选环不捕获缓存调用。粘滞态必须住在进程级对象（网关实例与租户视图都是每请求构造的）。
5. **计量账本**（`domain/usage.py`，表 `usage_ledger`）：一调用一行；`Numeric(12,6)`/Decimal，价目表由 `Decimal(str(价格))` 转换且只在组合根一处；`created_at` 用数据库时钟，月度聚合在数据库端 `date_trunc('month', now())` 且排除 `cached`；价目表缺失记 0 并告警；**全表带 `tenant_id`**；上游缺 usage 以 `usage_missing` 列标记而不是合成 0 行。数据源 = 框架 `usage_metadata`（取流中最后一个带 usage 的块）。记账员自己开会话自己提交，失败不吞——"绝不拖垮请求"由候选环兜住、只告警。价目表告警文案住 domain（话术单一事实源的登记例外：domain 不得反向 import engine）。
6. **协议签名只用内建类型**：`MeterLike.record` 十个关键字参数（str/int/bool/None）、`month_spend(tenant_id) -> int`；`BudgetResolver = Callable[[str], Awaitable[int | None]]`。domain 靠结构匹配实现，`app/deps.py` 是唯一同时 import engine 与 domain 的模块。缓存协议 `CacheLike` 带框架消息类型，与实现同住 engine，不入跨包协议文件。
7. **迁移是被测物**：alembic 骨架入库，`usage_ledger` 迁移手写；测试库只经 `alembic downgrade base` / `upgrade head` 建表，不用 `create_all`，并以 `compare_metadata` 断言迁移与 ORM 零漂移。数据库引擎不做模块级单例：由持有事件循环的入口（lifespan、worker、测试夹具）创建并在同一循环内释放。`alembic.ini` 保持 ASCII。
8. **四项载体**：租户身份是 `AegisGateway` 的必填构造字段（缓存前缀、账本列、配额桶键共用），组合根每请求装配一个网关实例；档位、deadline、session_id 经 `bind()` 作为 `_astream` 的具名关键字参数传递（ADR-006/M1.4a 已用同路）；request_id 由网关每次调用生成（一调用一行）。
9. **tenant_id 字符集 `^[A-Za-z0-9_-]{1,64}$`** 守卫两道：组合根入口先校验（非法身份碰不到任何共享件），网关字段校验器第二道（绕过组合根直接构造同样过不了）；规则只有一份（`engine/gateway/tenancy.py`）。字符集是缓存前缀隔离与按租户 `SCAN` 运维成立的前提，长度与账本列对齐。
10. **预算闸（C11）**：月度预算三态（resolver 值 / 静态配置 / None 或异常 = 读挂 fail-open）、`budget ≤ 0` 不查账本、账本读挂放行并告警、超支产 `BudgetExceeded`；单请求预算以自家 token 尺估算 prompt 侧（含工具 schema），0 = 关闭。两道闸都在缓存之后、出站闸之前：命中不问预算，被拒的请求不消耗配额。
11. **`TenantQuotaExceeded` 生产者 = 每租户进程内桶**（ADR-008 决策 4 的落地）：复用出站闸实现按 tenant_id 分片，`max_keys` 有界 LRU，位于候选环之外，等待上限由 deadline 反推；被淘汰后再回来的租户从满桶开始（近似，记账）。
12. **组合根按租户装配**（`app/deps.py`）：进程级共享件 = 路由表、候选实例（含注入器包装、共享上游 HTTP 客户端）、熔断器、供应商出站闸、租户桶、缓存存取、记账员；租户绑定件 = 网关实例与租户缓存视图，每请求构造。判据：对象带租户身份的必须每请求装配，不带的必须共享（状态按 key 分片）。fake 开关在候选工厂生效，组合根不感知。共享上游客户端由入口创建、注入候选、关停时关闭；注入即放弃框架默认的 keepalive 套接字选项，由本仓自设同一组（全平台 `SO_KEEPALIVE`，Linux 另设探测节奏）；连接池上限进配置，池满超时即本地过载错误。

## 实证（探针）

复现方式：`FakeReplyChatModel` / 最小 `BaseChatModel` 子类经 `astream`/`ainvoke` 外壳；`set_llm_cache(InMemoryCache())` 后计数实际调用；自家块 schema 编码为 JSON 再解码、`+` 合并后 `message_chunk_to_message`；`contextvars.ContextVar` 在回调、线程池、子任务中读值；本机 Postgres 16 上 `Numeric(12,6)` 往返、`date_trunc`、外层事务加 savepoint 会话、同一引擎跨两个 `asyncio.run`。定性结论：
- 默认缓存配置加全局缓存时 `ainvoke` 第二次不打上游，`astream` 每次都打，`cache=False` 每次都打。
- 同语义消息在不同 id / response_metadata / usage_metadata 下白名单序列化相同；流出块 id 由外壳赋值且同一流内相同。
- 自家 schema 往返后合并的 content / tool_calls / usage 与原流相同；盖章键在合并与两条外壳路径上保留；同键异值 bool 合并抛错；不标末块外壳补空块。
- ContextVar 在 `_astream`、同步与异步回调、`run_in_executor`、`to_thread`、`create_task` 中均读到父值；子任务 `set` 不回流；同任务生成器内 `set` 泄漏给消费者。
- Decimal 往返无损，SUM 为 Decimal，`created_at` 带 UTC 且事务内冻结；第 7 位小数被按 scale 舍入，整数位溢出报错；外层回滚后 savepoint 内已提交的行消失；引擎在第二个事件循环复用连接抛错。
- 以上事实由 `tests/engine/gateway/test_cache.py`、`test_router_cache.py`、`test_router_budget.py`、`tests/domain/test_usage.py`、`tests/test_deps.py` 钉住。

## 后果

- 前作四项缓存语义（租户前缀、完整性守卫、脏条目自愈、命中零成本记账）与故障粘滞降级全部恢复；key 规则升级为"消息白名单 + 调用参数全量"，误命中的方向被封死，代价是任何新增调用参数都会自动分裂缓存（正确方向）。
- 账本以 `usage_missing` 消除"0 是免费还是缺失"的歧义；缓存回放行照记 token、成本归零、不计入月度预算——账本同时是账单与流量记录。
- 记账 fail-open 在行级安全策略（后续里程碑）下的失败形态是静默空账而非报错：网关构造必带租户身份在结构上封住"裸调用无身份"一类事故，行级策略拒绝写入的缺口仍由对账脚本暴露。
- 租户桶为进程内近似：多副本下配额口径为全局除以副本数；LRU 淘汰再回来的租户多得一次突发。精确多副本配额若成为需求应回到 Redis 侧实现（ADR-008 后果）。
- 网关实例每请求构造：开销为一次模型校验，字典类字段容器会被重建但共享件实例同一批；租户身份不可能被调用参数覆盖。
- 引擎、Redis 客户端、上游 HTTP 客户端均绑定事件循环：只在进程入口的 lifespan 中创建与释放，worker 进程另建一份。
- 依赖基线：alembic 由开发依赖转为运行时使用的迁移工具，无新增包。
