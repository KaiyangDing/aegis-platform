# ADR-007: 熔断语义裁决

- 日期：2026-09-04
- 状态：已接受（增补 ADR-002 依赖基线：移除 pybreaker）

## 背景

前作熔断器是自研的 Redis 三键状态机（fails / open / probe 各一把 key，TTL 即状态迁移，`SET NX` 保证全集群单探针），语义为：5xx/超时进账，429/Auth/BadRequest 不进账；open 秒拒不排队；半开一次定胜负、无裁决不闭合；失败计数带时间窗；Redis 不可用时降级为本地计数且降级粘滞。

本仓最初拟改用 pybreaker 1.4.1 + `CircuitRedisStorage`，以"事后上报薄壳"弥合库与前作语义的差距，并定下判据：**以设计阶段薄壳原型的体量与复杂度为准，若薄壳不小于自研实现则翻案**。原型复核：库全同步（属性与方法均不可 `await`，异步客户端无法接入），薄壳的每个存储触点都要经线程池下沉；消除跨副本重同步清账须覆盖库的私有成员；半开互斥只能在进程内、试探锁无时间上界、无失败时间窗三项降级不可消除。薄壳加装配已不小于前作实现，按判据翻案。

本仓 venv 探针事实（pybreaker 1.4.1 / redis-py 5.3.1 / Redis 7），保留为"为何不用"的证据：
- `calling()` 包不住会 `yield` 的流：块内一切 `BaseException` 交 `_handle_error`，谓词判为排除即走 `_handle_success` 清零；消费者首块后 `aclose()`、任务取消、半开试探被弃流、块内吞掉 429 再块外重抛——四种出口全部把失败账清零或直接闭合。
- 半开态无并发门；`exclude` 谓词把排除当成功清零；无失败时间窗；`call()` 全程持有实例内的 `threading.RLock`。
- 实例缓存的状态对象落后于共享存储时，`state` 属性以通知方式重建，重建为闭合态会清零共享失败计数。
- `CircuitRedisStorage` 只接受同步客户端且须 `decode_responses=False`；构造即写三条 `SETNX`，Redis 不可达时构造抛 `RedisError`；运行期不可达读状态回退 closed、计数读写静默吞错。

## 决策

1. **总裁决：自研异步三键状态机，pybreaker 从依赖移除。** `RedisBreaker`（Redis 共享态主路）与 `MemoryBreaker`（同一状态机的进程内实现，供测试、无 Redis 的开发环境与降级备胎）实现候选环的 `BreakerLike` 协议（`allow` / `report_success` / `report_failure` / `release_probe`），候选环零改动。其它熔断库（aiobreaker、circuitbreaker 等）未纳入评估，本决策只在 pybreaker 与自研之间取舍。
2. **三键与迁移**（每个候选 key 三把，前缀 `aegis:cb:{provider}:{model}`）：`fails` 连续失败计数，`INCR` 后 `PEXPIRE fail_window`，成功清零；`open` 存在即 open 态，`SET PX reset_timeout`，过期即进入半开机会；`probe` 半开试探令牌，`SET NX PX probe_ttl`。`allow`：`open` 在 → deny；`fails < fail_max` → allow；否则 `SET NX` 抢令牌，输 → deny，赢 → 回查一次 `open`（读与抢令牌之间别处可能刚试探失败重开）：在 → 删自己的令牌、deny；不在 → probe。`report_failure`：读前态 + 计数续窗；达阈值 → 写 `open`（已 open 则续期）、删 `probe`；未达阈值的试探失败只删 `probe`。`report_success`：读前态 + 删三键。`release_probe`：删 `probe`。**原子性**：读前态 + `INCR` + `PEXPIRE` 走事务管道（MULTI/EXEC）；读两键、开路的 `SET open` + `DEL probe`、闭合的读前态 + `DEL` 三键走普通管道（非事务，顺序执行；开路时 `SET` 在前，插队者先看到 `open` 即秒拒）；`release_probe` 为单条 `DEL`。无 Lua、无后台任务；往返次数：`allow` 常态一次、试探路径三次，`report_failure` 至多两次，`report_success` 与 `release_probe` 各一次。
3. **半开 = `open` 已过期且失败账仍在**；全集群同一时刻只发一枚试探令牌（`SET NX`）；令牌是带 `probe_ttl` 的**租约**，持有方崩溃后到期自愈；候选环对无裁决结局在 `finally` 归还令牌，上报路径自行清除。失败账自身的过期是第二条自愈路径：半开期间若试探方消失且 `fail_window` 内再无任何裁决，则整体遗忘、回到 closed。策略构造期校验 **`fail_window > reset_timeout + probe_ttl`**，保证 open 到期后首枚令牌的自愈早于遗忘；`fail_max ≥ 1`，四个时长为有限正数（`inf`/`nan` 拒绝：`inf` 能通过"大于零"校验，却会让 TTL 换算溢出）。
4. **进账谓词与三待遇**：只有 `ProviderServerError` / `ProviderTimeoutError` 上报失败；整流耗尽上报成功；429、出站闸拒绝、Auth、BadRequest、本地过载、消费者弃流、任务取消、未知异常一律不触碰断路器（不进账也不清账），只归还试探令牌。半开试探只尝试一次（`max_attempts=1`），无裁决不闭合。谓词由候选环持有，熔断模块只认 key。
5. **迟到上报照常入账**：跳闸前已在飞的请求在 open 期间上报——失败续期 `open` 并清令牌；成功删三键即闭合（成功是上游可用的最强证据）。不做"open 未到期忽略上报"。
6. **Redis 不可达 → 降级为进程内状态机，且降级粘滞**（前作语义平移，实现改为组合）：`RedisBreaker` 内持一个同语义的 `MemoryBreaker` 作备胎，每次上报**双写**，备胎始终带着本进程的近期失败史，Redis 一断即刻接手、不必重新累计失败。任一 Redis 触点异常即进入降级：之后 `probe_interval` 内所有判定与上报只走备胎、不碰 Redis；顺路探针**只在 `allow` 领**（上报路径降级期直接跳过 Redis，否则多个触点会互相续期、恢复时机不可预测），触点失败一律顺延窗口；探针成功即切回共享态并记日志。降级期承诺退化：全集群单探针失效（每副本各探一个）、备胎只看得见本进程的上报；降级的是共享状态，不是熔断能力。故障期代价 = 每副本每 `probe_interval` 一次快速失败超时（`core/redis.py`：短连接/读写超时、不重试）。连接池用 `BlockingConnectionPool`（建连在池锁之外、连接数封顶、池满限时等待后抛 `ConnectionError`，同样触发降级）——默认 `ConnectionPool` 在池锁内建连，N 个并发触点会排队付 N 次超时。构造期无 I/O。
7. **粒度 `provider:model`**：一个候选三把键，key 数随候选数线性，无实例成本。参数起点 `fail_max=5`、`reset_timeout=30s`、`probe_ttl=120s`、`fail_window=300s`、`probe_interval=5s`，进配置；计数单位 = 一次耗尽重试后的候选访问（重试次数变化不改变单位）。`probe_ttl` 是令牌租约而非试探上界：试探中有界的部分只有闸内排队与首块窗，首块之后整流不设上限（ADR-006 决策 3）；取值按"闸内等待 + 首块窗 + 典型流长"定，超过租约的试探见后果。
8. **只有真实状态迁移进结构化日志**（warning，带 `store=redis|memory`）：跳闸 closed / half-open → open（带失败数）、闭合 open / half-open → closed；open 上的续期、closed 上的成功等同态上报不记。备胎在主路健康时静默（本进程的局部视野不代表集群），降级期开口；降级与恢复各记一条。
9. **`MemoryBreaker` 与 `RedisBreaker` 同一状态机**：内存版以单调时钟接缝判到期（测试推进假时钟），Redis 版以 TTL 判到期；状态机测试在内存版上跑，只有共享存储才有的事实（跨实例同视野、全集群单探针、令牌自愈、失败窗、迟到上报）在真 Redis 上跑，降级与恢复用开关代理驱动。

## 实证（探针）

复现方式：`redis.asyncio.Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.2, socket_timeout=0.2, retry=Retry(NoBackoff(), 0), retry_on_timeout=False)`，本机 Redis 7 db1，随机键前缀；死端口用本机无人监听的端口。定性结论：
- `SET NX PX` 首次返回 True、再次 None，过期或 `DEL` 后可再领；同一客户端并发 8 次只有一个赢家。
- `INCR` + `PEXPIRE` 组成的失败窗过期后计数消失，再 `INCR` 从 1 起；`GET` 在 `decode_responses=True` 下返回 `str`。
- `SET PX` + `EXISTS` 在过期后返回 0；一次 `DEL` 三键返回 3。
- 事务管道 `INCR` + `PEXPIRE` 一次往返返回 `[n, True]`；普通管道 `EXISTS` + `GET` 一次往返返回 `[0, '1']`。管道内的命令调用返回管道自身，不能对中间结果分支，这是自研两次往返而非 Lua 一次往返的原因。
- 死端口上的命令与管道均在设定超时内抛 `redis.exceptions.RedisError`（Windows 回环闭端口不回 RST，表现为 `TimeoutError`；Linux 立即拒绝，表现为 `ConnectionError`）。
- 默认 `ConnectionPool` 下 8 个并发触点对死端口的耗时呈 0.5s 递增阶梯（在池锁内建连）；`BlockingConnectionPool` 下 8 个全部在一次超时内失败。
- 以上事实由 `tests/engine/gateway/test_breakers.py`（Redis 段）与 `tests/core/test_redis.py` 钉住。

## 后果

- 前作七项语义（全集群单探针、令牌时间上界、失败时间窗、秒拒、共享态、本地降级、粘滞降级）全部恢复；本地降级以组合已有的 `MemoryBreaker` 实现，不再手工镜像每一步。
- 令牌是租约：试探流长超过 `probe_ttl` 时会放出第二个试探，旧持有者的归还也可能释放新持有者的令牌；后果有界（至多多一次试探），先到的裁决清掉后者令牌。要保住硬保证需令牌带持有者身份或流中续租，均为协议变更，留待有观测数据再议。
- `allow` 的读与 `SET NX` 之间不原子：读到半开后别处恰好闭合，会多设一枚孤儿令牌，由 TTL 或该请求的上报清除；别处恰好重开的方向由抢到令牌后的回查关闭。跳闸是两次往返，两次之间的并发 `allow` 可能领到随即被删的令牌，即每次跳闸至多多放一个请求。
- `allow` 在 `SET NX` 往返中被取消：令牌已写入而无人持有，该候选至多被拒一个 `probe_ttl`，靠租约到期自愈。
- 降级期各副本各自为政：备胎只累计本进程看到的失败，多副本下跳闸晚于集群口径；切回时以 Redis 现状为准，降级期备胎的裁决不回写共享态。
- 迟到成功闭合、迟到失败续期是语义而非缺陷；跳闸由连续失败驱动，一次迟到成功之后若上游仍不可用会再次跳闸。
- 依赖基线变更：移除 `pybreaker`（ADR-002 依赖纪律）；Redis 只用异步客户端一种（`decode_responses=True`），限流/缓存共用。
- 事后上报把"尝试"与"记账"分离，簿记在流尾（缓存、计量）与熔断上报共用同一个流尾时机。
