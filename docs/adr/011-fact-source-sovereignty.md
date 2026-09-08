# ADR-011: 事实源主权——checkpoint 管恢复、事件表管审计与游标

- 日期：2026-09-05
- 状态：已接受（M2.2 定稿）

## 背景

前作的运行时用一条事件流同时承担三种能力：崩溃恢复（重放事件重建状态）、审计与回放（payload 存原文、seq 逻辑时钟）、SSE 游标（`after_seq` 重订阅）。本仓把循环骨架交给 `create_agent`，而 LangGraph 自带 checkpointer：每个节点结束落一次 checkpoint，`ainvoke(None, config)` 从最后一个 checkpoint 之后重放。于是三种能力不再天然同源，必须裁决谁是权威：(a) 全押 checkpoint——审计靠相邻快照 diff、游标改 checkpoint_id、回放放弃；(b) 双写——checkpointer 管图恢复、自建 events 表管审计 / 回放 / 游标；(c) 自研事件溯源为权威、放弃框架恢复。选 (b) 必须回答两个前置问题：钩子写完事件之后 checkpoint 落盘有没有持久化屏障；两套记录失同步时以谁为准、如何弥合。

本仓 venv 探针事实（langchain 1.4.0 / langgraph 1.2.11 / langgraph-checkpoint 4.2.0 / langgraph-checkpoint-postgres 3.1.2 / psycopg 3.3.5）：
- `durability` 三态：默认 `"async"` 下 checkpoint 写入提交后**不等待完成**即进入下一节点（慢存储替身下多个节点在未完成的写入期间开跑）；`"sync"` 每步写完才开下一节点；`"exit"` 整条 run 只在结束落一次，崩溃期间零 checkpoint。真 Postgres 本机两轮循环九次 checkpoint，`sync` 比 `async` 慢定性上几毫秒每次。
- 每个中间件钩子是独立图节点、每个节点一次 checkpoint；每个任务完成即写待定写入（任务异常也落错误记录）。
- 工具节点中途死亡（异常或 BaseException）后 checkpoint 停在 tools 之前；恢复时**整个 tools 节点重放、工具函数再次执行**——框架用"重放节点"恢复，没有 write-ahead 概念。
- 崩溃重放与中断重放中，节点内读到的 `__pregel_task_id` 与 `checkpoint_id` **与首次执行相同**，下一轮才变。
- `AsyncPostgresSaver.setup()` 自带迁移（`checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations` 四表），独立于 alembic，**无 tenant 列**；接受连接或 `psycopg_pool.AsyncConnectionPool`。
- psycopg 异步拒绝 Windows 默认的 `ProactorEventLoop`（明示需 `SelectorEventLoop`）；asyncpg 不受影响。uvicorn 的 `--loop` 接受自定义 loop factory 的 import 串。

## 决策

1. **双写，各管一摊**：`AsyncPostgresSaver` 是图恢复的权威（`resume` = `ainvoke(None, config)`）；自建 `events` 表是审计、回放断言、SSE 游标的权威。两者不互相派生，也不做同事务投影。
2. **`durability="sync"` 必选**，每次 `astream / ainvoke` 显式传入；`"exit"` 禁用。默认 `"async"` 没有持久化屏障，"钩子里事件已提交、checkpoint 尚未落盘"的窗口在默认值下可以跨越多个节点，恢复语义无法成立。代价是每次 checkpoint 多几毫秒的等待，接受。
3. **事件 id 派生自稳定任务身份**：`id = uuid5(命名空间, [thread_id, task_id, hook, ordinal])`，四段分别是会话 id、框架任务 id、写入点名、钩子内序号或模型侧 tool_call id，任一段为空即拒绝。落盘 `INSERT … ON CONFLICT (id) DO NOTHING`，命中既有行时返回既有 seq 并标记"去重命中"。这是双写失同步的唯一弥合机制：重放同一步骤派生同一个 id，事件表不会多出第二条。
4. **失同步只可能"事件领先一步"**：事件在钩子内先提交、钩子返回后框架才写 checkpoint；反向（checkpoint 已落、事件未落）结构上不可能。领先的那一步在重放时被第 3 条去重吸收。写工具的 write-ahead 由此成立：`tool_call` 事件先于副作用落盘，重放命中既有事件即进入"凭原键重发"分支，绝不产生第二把幂等键。
5. **投影放弃**：前作的 messages / tool_invocations / sessions.summary 同事务投影不建。消息读模型 = checkpoint 里的 state；工具审计 = events 表直查；摘要在 state（框架 REMOVE_ALL 替换）+ `summary_updated` 事件保留原文。
6. **events 表形态**：`id uuid` 主键、`tenant_id`、`session_id`、`run_id`、`seq`、`type`、`payload jsonb`、`schema_version`、`task_id`、`checkpoint_id`、`created_at`（数据库时钟）；`(session_id, seq)` 唯一。seq 由 INSERT 同一语句内的标量子查询 `max(seq) + 1` 播种（无状态：写入点散在各钩子、跨崩溃没有写者对象可持有计数；单写者，会话锁归后续里程碑）。append 三岔口：id 冲突已被 `DO NOTHING` 吸收，剩下的唯一约束冲突只能是别的写者占了本会话的 seq → 围栏异常，终态不重试；连接类异常（含未经 SQLAlchemy 包装的 OSError 族）退避 0.1 / 0.2 / 0.4 秒三次耗尽 → 不可用异常；其余（编程错误、类型错误）裸抛。运行时对这些异常一概不接，让 run 炸出去。读取 `read(tenant_id, session_id, after_seq, limit)` 按 seq 升序、租户与会话双过滤。
7. **事件契约沿用前作**：17 类事件类型值稳定、payload 存原文、seq 从 1 起单调、事件不带应用侧时间戳、`user_message` 首事件 / `loop_terminated` 末事件、`schema_version` 随行；`tenant_id` 为必填（全表 tenant_id 纪律）；`task_id / checkpoint_id` 回指框架坐标，图外写入的事件可为空。
8. **框架表由 `setup()` 管**，在 lifespan 调用、不进 alembic，模型漂移检查过滤这四张表。checkpointer 构造为 `AsyncPostgresSaver` over `psycopg_pool.AsyncConnectionPool`（连接参数照抄框架自身：autocommit、`prepare_threshold=0`、`dict_row`；池 `open=False`，lifespan 内 `open(wait=True)` + `setup()`，关停关池，之后读写抛 `PoolClosed`）。`thread_id = session_id`（全局唯一 uuid）；框架表无 tenant 列、行级安全覆盖不到，租户归属靠应用层校验，记入差距记账表。
9. **Windows 开发进程使用 Selector 事件循环**：自家零参 loop factory 交给 uvicorn `--loop`（非内建名的串被 import 后直接作工厂）；测试经 pytest-asyncio 的 `pytest_asyncio_loop_factories` 钩子全仓统一 Selector；checkpointer 开池前守卫循环类型——psycopg 在 Proactor 下不是立刻报错，而是连接池后台反复重连直到超时，守卫把它变成启动期的一句明白话。容器 Linux 默认即 Selector，三处都是无事。
10. **SSE 游标 = `events.seq`**（`after_seq` 重订阅语义保留）；token 级帧走瞬态通道，不进事件表。

## 实证（探针）

复现方式：慢存储替身包裹 InMemorySaver 记录"写入完成"与"下一节点开始"的先后，三种 `durability` 各跑一遍；真 Postgres 上跑两轮工具循环计数 checkpoint；工具函数内在副作用之后抛 BaseException，再 `ainvoke(None, config)` 恢复并计数副作用；在崩溃重放与 `interrupt()` 中断重放的节点内打印 `__pregel_task_id / checkpoint_id`；`AsyncPostgresSaver.setup()` 后查表结构；Windows 上分别用 Proactor 与 Selector 循环连接。定性结论：
- 默认 `async` 下慢存储期间多个节点在未完成写入期开跑；`sync` 零违规；`exit` 崩溃期间无 checkpoint。
- 工具崩溃后恢复 = tools 节点整体重放，副作用两次。
- 重放节点内任务 id 与 checkpoint id 与首次相同。
- 框架四表由 `setup()` 建、无 tenant 列；Proactor 循环连接被拒。
- 事件 id 派生函数的确定性、逐段敏感、空段拒绝与命名空间钉死已由运行时类型测试钉住。
- 事实源落地（M2.2，真 Postgres）：`INSERT … ON CONFLICT (id) DO NOTHING RETURNING seq` 在命中既有 id 时不返回行，同事务二次查询取既有 seq；同 id 两次写入 = 一行、既有 seq、后续 seq 不跳号；两个并发写者由同一子查询算出同一 seq 时唯一约束裁决一赢一输，输家得围栏异常且零退避；连接级故障（含 `ConnectionRefusedError`）按 0.1 / 0.2 / 0.4 秒退避、耗尽即不可用异常，编程错误零重试裸抛；非 uuid 的 id 在语句层被拒。
- checkpointer over 连接池：`setup()` 重跑幂等（框架迁移十步）；两个 thread 各自三个 checkpoint（输入 + 起点 + 模型节点，每节点一次）、消息互不串；同 thread 第二次调用从上次 checkpoint 续跑；关池后 `PoolClosed`。Windows 上 Proactor 循环下开池：无守卫时后台反复重连 30 秒后 `PoolTimeout`，有守卫时立刻 `RuntimeError`。
- 以上由 `tests/domain/test_events.py`、`tests/domain/test_sessions.py`、`tests/core/test_checkpoint.py`、`tests/core/test_loops.py` 与迁移零漂移测试钉住。

## 后果

- 前作"恢复 / 审计 / 游标同源"拆成两源，恢复粒度从"事件"变为"节点"；审计与回放断言仍以事件为准，行为轨迹归一化（豁免墙钟与 usage、id 别名）保留。
- 每个 run 多一份存储写入（checkpoint 与事件各写各的）；`sync` 使每个节点多等一次落盘。
- 事件 id 的派生输入必须包含框架任务 id：图外写入的事件（例如恢复放弃审计）另行约定派生规则。
- 投影放弃后，"最近 N 条消息"这类读路径读 checkpoint state 而非查表；工具调用审计改为按事件类型查 events。
- 框架表在租户隔离之外：这是记账项，后续里程碑以应用层归属校验兜底。

## 增补（M2.5，2026-09-07）：write-ahead 与重放去重的落地

- 决策 4 的"凭原键重发"在工具执行中间件里的形态：`tool_call` 事件在严校验与风险闸门之后、副作用之前落盘，事件 id 派生自 (会话 id, 框架任务 id, "tool_call", 模型侧调用 id)，作为幂等键经运行时上下文注入工具实现；重放进入同一 wrap 时派生同一 id，`append` 返回"去重命中"，不产生第二把钥匙，只记一条日志后照常执行。`tool_result / tool_error` 同样派生自任务身份，重放命中的事件不再外流。
- 前作的 `reexecute` 窄入口（跳过校验与闸门）在本仓坍缩为同一条路径：同参数同通行证下重跑校验与闸门无害；"恢复期工具缺失"由执行前的注册表查询兜底回填。
- 实证：工具副作用之后以 BaseException 中断（checkpoint 停在 tools 之前）→ `astream(None)` 恢复 → 事件恰九条无重复、结果以原 id 闭合、下游只见一把钥匙、副作用执行两次（框架重放整节点）；直连 wrap 二次进入同任务身份 → 无第二条 `tool_call`。由 `tests/engine/runtime/test_runtime_tool_exec.py` 钉住。
- 摘要事件（M2.6）同样派生自 before_model 节点的任务身份：崩溃发生在"事件已写、节点未完成"的窗口时，重放会得到另一份摘要文本而事件保留首次文本——保事实不保字节，恢复分诊时复核。
